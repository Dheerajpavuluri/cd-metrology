"""Critical-dimension measurement on cross-section images.

Steps for one image:

1. Estimate noise, and reject the image if its contrast-to-noise ratio (SNR)
   is too low to measure reliably.
2. Find the substrate interface: rows inside the substrate are uniform, rows
   that cut through the lines are not, so the row-wise standard deviation
   drops sharply at the interface.
3. Segment the lines above the interface (Otsu threshold + morphological
   opening + connected components) to get one region per line, then refine
   the SNR from the line and background intensity levels. The sub-pixel
   interface comes from the vertical intensity profile in the gaps.
4. For each line and each height, average a few rows into one horizontal
   profile, smooth it, and locate the left and right edges at the 50% level
   between the local background and line intensities, with linear
   interpolation for sub-pixel positions. CD = right edge - left edge.
5. Fit the edge positions over 20-80% of the height to get the sidewall angle.

`measure_image` never raises on a bad image; it returns a result whose status
is "rejected" (quality gate) or "failed"/"partial" (edges not found).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

HEIGHTS = {"bottom": 0.1, "mid": 0.5, "top": 0.9}
MIN_SNR = 5.0


@dataclass
class LineResult:
    index: int
    center_nm: float
    height_nm: float
    cd_nm: dict                 # "bottom"/"mid"/"top" -> CD in nm
    swa_deg: float              # mean sidewall angle, 90 = vertical
    edges_px: dict = field(default_factory=dict)   # for overlays: name -> (y, xl, xr)


@dataclass
class ImageResult:
    name: str
    pixel_nm: float
    contrast: float
    noise: float
    snr: float
    interface_px: float
    lines: list[LineResult]
    status: str = "ok"
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for ln in d["lines"]:
            ln.pop("edges_px", None)
        return d


class MeasurementError(RuntimeError):
    pass


def to_float(img: np.ndarray) -> np.ndarray:
    """8- or 16-bit image (grayscale or BGR) -> float32 grayscale in [0, 1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    raise MeasurementError(f"unsupported pixel type {img.dtype}")


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise MeasurementError(f"cannot read {path}")
    return to_float(img)


def _crossing(profile: np.ndarray, level: float, near: int, rising: bool, reach: int = 6) -> float:
    """Sub-pixel position where `profile` crosses `level`, closest to `near`."""
    lo, hi = max(near - reach, 0), min(near + reach, len(profile) - 1)
    best, best_dist = None, None
    for i in range(lo, hi):
        a, b = profile[i], profile[i + 1]
        hit = (a < level <= b) if rising else (a >= level > b)
        if hit:
            x = i + (level - a) / (b - a)
            dist = abs(x - near)
            if best is None or dist < best_dist:
                best, best_dist = x, dist
    if best is None:
        raise MeasurementError("edge not found")
    return float(best)


def _smooth1d(p: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (0, 0), sigma).ravel()


def find_interface(img: np.ndarray) -> int:
    """First row below the line structures (integer, refined later)."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    row_std = blur.std(axis=1)
    lo, hi = np.percentile(row_std, 5), np.percentile(row_std, 95)
    busy = row_std > (lo + hi) / 2
    rows = np.nonzero(busy)[0]
    if len(rows) == 0:
        raise MeasurementError("no line structures found")
    return int(rows.max()) + 1


def segment_lines(img: np.ndarray, interface: int, min_area_px: int = 200):
    """Bounding boxes (x, y, w, h) of the lines above the interface, left to right, and the mask."""
    region = cv2.GaussianBlur(img[: max(interface - 3, 1)], (0, 0), 1.0)
    u8 = np.clip(region * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(cc[i, :4]) for i in range(1, n) if cc[i, cv2.CC_STAT_AREA] >= min_area_px]
    return sorted(boxes, key=lambda b: b[0]), mask


def estimate_noise(img: np.ndarray) -> float:
    """Gaussian noise sigma (Immerkaer 1996), ignoring pixels on strong edges."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(cv2.filter2D(img, -1, kernel))[1:-1, 1:-1]
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)[1:-1, 1:-1]
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)[1:-1, 1:-1]
    mag = np.hypot(gx, gy)
    flat = mag <= np.percentile(mag, 80)
    return float(np.sqrt(np.pi / 2) / 6 * resp[flat].mean())


def measure_image(img: np.ndarray, pixel_nm: float, name: str = "image",
                  band: int = 2, min_snr: float = MIN_SNR) -> ImageResult:
    """Measure every line in a float image. `band` = rows averaged on each side of a cut."""
    h, w = img.shape
    noise = estimate_noise(img)

    def snr_of(contrast: float) -> float:
        return float(contrast / max(noise, 1e-6))

    def reject(contrast: float, why: str) -> ImageResult:
        snr = snr_of(contrast)
        return ImageResult(name, pixel_nm, contrast, noise, snr, 0.0, [],
                           "rejected", f"{why} (SNR {snr:.1f})")

    smooth = cv2.GaussianBlur(img, (0, 0), 3.0)
    rough_contrast = float(np.percentile(smooth, 99) - np.percentile(smooth, 1))
    if snr_of(rough_contrast) < min_snr:
        return reject(rough_contrast, "contrast too low to measure")
    try:
        interface = find_interface(img)
        boxes, mask = segment_lines(img, interface)
    except MeasurementError as e:
        return reject(rough_contrast, str(e))
    if not boxes:
        return reject(rough_contrast, "no lines segmented")

    # line and background intensity levels from the segmentation
    upper = img[: interface - 3]
    line_px = upper[mask > 0]
    bg_px = upper[cv2.erode(255 - mask, np.ones((7, 7), np.uint8)) > 0]
    line_level, bg_level = float(np.median(line_px)), float(np.median(bg_px))
    contrast = line_level - bg_level
    snr = snr_of(contrast)
    if snr < min_snr:
        return reject(contrast, "line/background contrast too low")

    # sub-pixel interface from gap columns (between lines)
    covered = np.zeros(w, bool)
    for x, _, bw, _ in boxes:
        covered[max(x - 6, 0): x + bw + 6] = True
    gap_cols = np.nonzero(~covered)[0]

    def fail(why: str) -> ImageResult:
        return ImageResult(name, pixel_nm, contrast, noise, snr, 0.0, [], "failed", why)

    if len(gap_cols) < 3:
        return fail("no gap between lines to locate the interface")
    vprof = _smooth1d(img[:, gap_cols].mean(axis=1))
    sub_level = float(np.median(img[min(interface + 5, h - 1):, gap_cols]))
    try:
        interface_y = _crossing(vprof, (bg_level + sub_level) / 2, interface,
                                rising=sub_level > bg_level, reach=8)
    except MeasurementError:
        return fail("substrate interface not found")

    lines, skipped = [], []
    for idx, box in enumerate(boxes):
        try:
            lines.append(_measure_line(img, idx, box, interface_y, bg_level,
                                       line_level, pixel_nm, band))
        except MeasurementError as e:
            skipped.append(f"line {idx}: {e}")

    res = ImageResult(name=name, pixel_nm=pixel_nm, contrast=contrast, noise=noise,
                      snr=snr, interface_px=float(interface_y), lines=lines)
    if skipped:
        res.status = "partial" if lines else "failed"
        res.reason = "; ".join(skipped)
    return res


def _measure_line(img, idx, box, interface_y, bg_level, line_level, pixel_nm, band) -> LineResult:
    w = img.shape[1]
    x, y, bw, _ = box
    cx = x + bw // 2
    # top of the line from a vertical profile through its centre
    cols = img[:, max(cx - 3, 0): cx + 4].mean(axis=1)
    top_y = _crossing(_smooth1d(cols), (bg_level + line_level) / 2, y, rising=True, reach=8)
    height_px = interface_y - top_y

    x0, x1 = max(x - 10, 0), min(x + bw + 10, w)

    def edges_at(frac: float):
        yc = interface_y - frac * height_px
        r0, r1 = int(round(yc)) - band, int(round(yc)) + band + 1
        prof = _smooth1d(img[max(r0, 0): r1, x0:x1].mean(axis=0))
        bgl = float(np.median(np.r_[prof[:4], prof[-4:]]))
        mid = len(prof) // 2
        fl = float(np.median(prof[mid - 2: mid + 3]))
        level = (bgl + fl) / 2
        grad = np.gradient(prof)
        left_peak = int(np.argmax(grad[:mid]))
        right_peak = mid + int(np.argmin(grad[mid:]))
        xl = _crossing(prof, level, left_peak, rising=True)
        xr = _crossing(prof, level, right_peak, rising=False)
        return yc, x0 + xl, x0 + xr

    cd, edges = {}, {}
    for key, frac in HEIGHTS.items():
        yc, xl, xr = edges_at(frac)
        cd[key] = (xr - xl) * pixel_nm
        edges[key] = (yc, xl, xr)

    # sidewall angle from a straight-line fit of both walls over 20-80% height
    ys, ls, rs = [], [], []
    for frac in np.linspace(0.2, 0.8, 13):
        yc, xl, xr = edges_at(float(frac))
        ys.append(yc); ls.append(xl); rs.append(xr)
    dl = np.polyfit(ys, ls, 1)[0]      # px of x per px of y
    dr = np.polyfit(ys, rs, 1)[0]
    swa = 90.0 - np.degrees(np.arctan((abs(dl) + abs(dr)) / 2))

    mid_edges = edges["mid"]
    return LineResult(
        index=idx,
        center_nm=(mid_edges[1] + mid_edges[2]) / 2 * pixel_nm,
        height_nm=height_px * pixel_nm,
        cd_nm=cd,
        swa_deg=float(swa),
        edges_px=edges,
    )


def measure_file(path: Path, pixel_nm: float | None = None) -> tuple[ImageResult, dict]:
    """Measure one image file; pixel size comes from the JSON sidecar unless given.

    Problems with the file itself (unreadable, no pixel size) come back as a
    "failed" result so one bad file does not stop a batch.
    """
    path = Path(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    px = float(pixel_nm or meta.get("pixel_nm") or 0)
    if px <= 0:
        return ImageResult(path.stem, 0.0, 0, 0, 0, 0, [], "failed",
                           "pixel size unknown (no sidecar, no --pixel-nm)"), meta
    try:
        img = load_image(path)
    except MeasurementError as e:
        return ImageResult(path.stem, px, 0, 0, 0, 0, [], "failed", str(e)), meta
    return measure_image(img, px, name=path.stem), meta


def overlay(img: np.ndarray, res: ImageResult) -> np.ndarray:
    """Colour image with the measured edges drawn on it (for reports)."""
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    colours = {"bottom": (60, 170, 255), "mid": (80, 220, 80), "top": (255, 120, 60)}   # BGR: orange, green, blue
    for ln in res.lines:
        for key, (yc, xl, xr) in ln.edges_px.items():
            y = int(round(yc))
            cv2.line(out, (int(round(xl)), y), (int(round(xr)), y), colours[key], 1, cv2.LINE_AA)
            for xe in (xl, xr):
                cv2.circle(out, (int(round(xe)), y), 2, colours[key], -1, cv2.LINE_AA)
    cv2.line(out, (0, int(round(res.interface_px))), (out.shape[1] - 1, int(round(res.interface_px))),
             (200, 200, 200), 1)
    return out
