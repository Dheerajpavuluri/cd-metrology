"""Synthetic cross-section images of line arrays with known dimensions.

Real TEM cross-sections are not something I can publish, so the pipeline is
developed and validated against rendered images where every edge position is
known exactly. Each image shows a row of trapezoid-shaped lines standing on a
substrate, blurred by a point-spread function and corrupted with noise, so the
measurement code has to deal with the same problems it would on a real image:
soft edges, noise, sloped sidewalls and slightly rough walls.

Each image is written as ``<name>.png`` (16-bit grayscale) next to a
``<name>.json`` sidecar holding the pixel size, wafer/site labels and the
ground-truth geometry of every line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from .measure import HEIGHTS

SUPERSAMPLE = 4          # render at 4x then area-average down: anti-aliased edges
SHIFT_BITS = 4           # cv2.fillPoly fixed-point precision (1/16 px)


@dataclass
class Recipe:
    """Imaging and geometry settings for one synthetic image."""
    width_px: int = 512
    height_px: int = 256
    pixel_nm: float = 0.5
    pitch_nm: float = 48.0
    n_lines: int = 5
    line_height_nm: float = 60.0
    bottom_cd_nm: float = 24.0
    top_cd_nm: float = 16.0
    interface_row: int = 200          # y of the substrate surface (px); lines stand on it
    bg_level: float = 0.25            # fill material between lines
    line_level: float = 0.80
    substrate_level: float = 0.55
    psf_sigma_px: float = 1.0         # Gaussian blur standing in for the microscope PSF
    noise_sigma: float = 0.04
    cd_sigma_nm: float = 0.4          # line-to-line CD variation inside one image
    wall_roughness_nm: float = 0.3    # smooth random wobble along each sidewall
    cd_offset_nm: float = 0.0         # shifts every CD in the image (process drift)
    pinch: dict | None = None         # {"line": i, "depth_nm": d}: a necked line (defect)


@dataclass
class LineTruth:
    center_nm: float
    height_nm: float
    cd_nm: dict = field(default_factory=dict)   # "bottom"/"mid"/"top" -> CD in nm


def _smooth_noise(rng: np.random.Generator, n: int, amp: float) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, n + 8)
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    smooth = np.convolve(raw, kernel, mode="valid")[:n]
    std = smooth.std()
    return smooth / std * amp if std > 0 else smooth


def render(recipe: Recipe, rng: np.random.Generator) -> tuple[np.ndarray, list[LineTruth]]:
    """Render one image; returns float image in [0, 1] and the ground truth."""
    r = recipe
    s = SUPERSAMPLE
    canvas = np.full((r.height_px * s, r.width_px * s), r.bg_level, np.float32)
    # supersampled row j has its centre at output y = (j - (s-1)/2) / s; the
    # substrate fills every row whose centre lies at or below interface_row
    canvas[int(np.ceil(r.interface_row * s + (s - 1) / 2)):, :] = r.substrate_level

    px = 1.0 / r.pixel_nm                          # px per nm
    base_y = float(r.interface_row)                 # bottom of every line, in px
    k = 48                                          # vertices per sidewall
    fr = np.linspace(0.0, 1.0, k)                   # fraction of height, 0 = bottom

    first = (r.width_px / px - r.pitch_nm * (r.n_lines - 1)) / 2.0
    truths: list[LineTruth] = []
    for i in range(r.n_lines):
        center = first + i * r.pitch_nm + rng.normal(0.0, 0.3)   # small placement error
        delta = r.cd_offset_nm + rng.normal(0.0, r.cd_sigma_nm)
        half = (r.bottom_cd_nm + (r.top_cd_nm - r.bottom_cd_nm) * fr + delta) / 2.0
        if r.pinch and r.pinch.get("line") == i:
            # a neck centred at mid-height, Gaussian in shape
            half = half - r.pinch["depth_nm"] / 2.0 * np.exp(-((fr - 0.5) / 0.12) ** 2)
        left = center - half + _smooth_noise(rng, k, r.wall_roughness_nm)
        right = center + half + _smooth_noise(rng, k, r.wall_roughness_nm)

        ys = base_y - fr * r.line_height_nm * px
        poly = np.concatenate([
            np.stack([left * px, ys], 1),
            np.stack([right[::-1] * px, ys[::-1]], 1),
        ])
        # output pixel i averages supersampled pixels s*i .. s*i+s-1, whose centres
        # average to s*i + (s-1)/2; offset so output pixel centres sit at integers
        ss = poly * s + (s - 1) / 2
        # cv2.fillPoly fills both boundary pixels; pull each wall in by half a
        # supersampled pixel so the filled width equals the requested width
        ss[:k, 0] += 0.5
        ss[k:, 0] -= 0.5
        ss[k - 1, 1] += 0.5          # same for the top edge (y grows downwards)
        ss[k, 1] += 0.5
        pts = np.round(ss * (1 << SHIFT_BITS)).astype(np.int32)
        cv2.fillPoly(canvas, [pts], float(r.line_level), lineType=cv2.LINE_8, shift=SHIFT_BITS)

        cd = {key: float(np.interp(f, fr, right - left)) for key, f in HEIGHTS.items()}
        truths.append(LineTruth(center_nm=float(center), height_nm=r.line_height_nm, cd_nm=cd))

    img = cv2.resize(canvas, (r.width_px, r.height_px), interpolation=cv2.INTER_AREA)
    if r.psf_sigma_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), r.psf_sigma_px)
    img = img + rng.normal(0.0, r.noise_sigma, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0), truths


def save(img: np.ndarray, meta: dict, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    cv2.imwrite(str(png), np.round(img * 65535).astype(np.uint16))
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return png


def make_lot(out_dir: Path, wafers: int = 4, sites: int = 9, seed: int = 7) -> list[Path]:
    """A small 'lot' of images with known structure to recover.

    Sites are numbered row by row on a 3x3 grid (site 5 is the centre).

    * each wafer has its own CD offset (wafer-to-wafer process drift)
    * corner sites print 0.8 nm wider and edge sites 0.3 nm wider than the
      centre (radial signature); sites beyond 9 get no radial offset
    * wafer 3 / site 5 has a necked line (should be flagged as an outlier)
    * wafer 2 / site 8 is a low-contrast, noisy image (should fail quality check)
    """
    rng = np.random.default_rng(seed)
    wafer_offsets = rng.normal(0.0, 0.8, wafers)
    paths = []
    for w in range(1, wafers + 1):
        for site in range(1, sites + 1):
            radial = 0.8 if site in (1, 3, 7, 9) else (0.3 if site in (2, 4, 6, 8) else 0.0)
            recipe = Recipe(cd_offset_nm=float(wafer_offsets[w - 1] + radial))
            note = ""
            if (w, site) == (3, 5):
                recipe.pinch = {"line": 2, "depth_nm": 6.0}
                note = "necked line 2"
            if (w, site) == (2, 8):
                recipe.line_level = 0.31
                recipe.noise_sigma = 0.06
                note = "low contrast"
            img, truths = render(recipe, rng)
            name = f"W{w:02d}_S{site:02d}"
            meta = {
                "pixel_nm": recipe.pixel_nm,
                "wafer": f"W{w:02d}",
                "site": site,
                "note": note,
                "recipe": asdict(recipe),
                "truth": [asdict(t) for t in truths],
            }
            paths.append(save(img, meta, out_dir, name))
    return paths
