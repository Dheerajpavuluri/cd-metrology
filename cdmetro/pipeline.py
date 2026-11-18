"""Batch pipeline: folder of images -> measurements -> outlier flags -> SQLite + CSV."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import db, stats
from .measure import measure_file

IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def _one(args):
    path, pixel_nm = args
    res, meta = measure_file(path, pixel_nm)
    return res.to_dict(), meta


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run(folder: Path, db_path: Path, csv_path: Path | None = None,
        pixel_nm: float | None = None, workers: int = 0) -> dict:
    images = find_images(folder)
    if not images:
        raise FileNotFoundError(f"no images in {folder}")
    stems = [p.stem for p in images]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        raise ValueError(f"image names must be unique without the extension: {', '.join(dupes)}")
    jobs = [(p, pixel_nm) for p in images]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, jobs))
    else:
        results = [_one(j) for j in jobs]

    # outliers are judged against every line from every usable image in the run,
    # so a run should hold one structure type (one nominal CD)
    usable = [(im, meta) for im, meta in results if im["status"] in ("ok", "partial")]
    all_lines = [ln for im, _ in usable for ln in im["lines"]]
    z, out = stats.flag_outliers([ln["cd_nm"]["mid"] for ln in all_lines])
    for ln, zi, oi in zip(all_lines, z, out):
        ln["robust_z"], ln["outlier"] = float(zi), bool(oi)

    # attach ground truth when the sidecar carries it (synthetic data)
    for im, meta in usable:
        truth = meta.get("truth") or []
        if len(truth) == len(im["lines"]):
            for ln, t in zip(im["lines"], truth):
                ln["true_cd_mid_nm"] = t["cd_nm"]["mid"]

    conn = db.connect(db_path)
    run_id = db.save_run(conn, str(folder), results)
    conn.close()

    if csv_path:
        write_csv(csv_path, results)

    counts = {}
    for im, _ in results:
        counts[im["status"]] = counts.get(im["status"], 0) + 1
    return {
        "run_id": run_id,
        "images": len(results),
        "status": counts,
        "lines": len(all_lines),
        "outliers": int(out.sum()),
        "cd_mid": stats.summarize([ln["cd_nm"]["mid"] for ln in all_lines]),
    }


def write_csv(path: Path, results: list[tuple[dict, dict]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image", "wafer", "site", "status", "line", "cd_bottom_nm", "cd_mid_nm", "cd_top_nm",
            "height_nm", "swa_deg", "robust_z", "outlier", "true_cd_mid_nm"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for im, meta in results:
            if not im["lines"]:
                w.writerow([im["name"], meta.get("wafer"), meta.get("site"), im["status"]] + [""] * 9)
            for ln in im["lines"]:
                w.writerow([
                    im["name"], meta.get("wafer"), meta.get("site"), im["status"], ln["index"],
                    f"{ln['cd_nm']['bottom']:.3f}", f"{ln['cd_nm']['mid']:.3f}", f"{ln['cd_nm']['top']:.3f}",
                    f"{ln['height_nm']:.3f}", f"{ln['swa_deg']:.2f}", f"{ln['robust_z']:.2f}",
                    int(ln["outlier"]),
                    "" if ln.get("true_cd_mid_nm") is None else f"{ln['true_cd_mid_nm']:.3f}",
                ])
