"""Command line entry point: cdmetro {synth,measure,report,serve}."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cdmetro", description="Critical-dimension metrology on cross-section images")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic lot with ground truth")
    s.add_argument("--out", type=Path, default=Path("data/lot1"))
    s.add_argument("--wafers", type=int, default=4)
    s.add_argument("--sites", type=int, default=9)
    s.add_argument("--seed", type=int, default=7)

    m = sub.add_parser("measure", help="measure every image in a folder")
    m.add_argument("folder", type=Path)
    m.add_argument("--db", type=Path, default=Path("results/cd.sqlite"))
    m.add_argument("--csv", type=Path, default=Path("results/cd.csv"))
    m.add_argument("--pixel-nm", type=float, help="pixel size if images have no JSON sidecar")
    m.add_argument("--workers", type=int, default=0, help="processes to use (0 = serial)")

    r = sub.add_parser("report", help="charts + HTML report for a run")
    r.add_argument("--db", type=Path, default=Path("results/cd.sqlite"))
    r.add_argument("--out", type=Path, default=Path("results/report"))
    r.add_argument("--images", type=Path, help="image folder, for edge overlays")
    r.add_argument("--run", type=int)
    r.add_argument("--baseline-wafer", help="compute control limits from this wafer only")

    v = sub.add_parser("serve", help="start the REST API")
    v.add_argument("--db", type=Path, default=Path("results/cd.sqlite"))
    v.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    if a.cmd == "synth":
        from .synth import make_lot
        paths = make_lot(a.out, a.wafers, a.sites, a.seed)
        print(f"wrote {len(paths)} images to {a.out}")
    elif a.cmd == "measure":
        from .pipeline import run
        a.db.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            summary = run(a.folder, a.db, a.csv, a.pixel_nm, a.workers)
        except FileNotFoundError as e:
            p.error(str(e))
        summary["seconds"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(summary, indent=2))
    elif a.cmd == "report":
        from .report import build
        try:
            print(build(a.db, a.out, a.images, a.run, a.baseline_wafer))
        except ValueError as e:
            p.error(str(e))
    elif a.cmd == "serve":
        import uvicorn
        os.environ["CDMETRO_DB"] = str(a.db)
        uvicorn.run("cdmetro.api:app", port=a.port)
    return 0
