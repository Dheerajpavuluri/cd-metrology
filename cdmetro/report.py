"""Charts and an HTML summary for one measurement run."""

from __future__ import annotations

import html
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import db, stats  # noqa: E402
from .measure import load_image, measure_image  # noqa: E402
from .measure import overlay as draw_overlay  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cfe0f6"
CRITICAL = "#d03b3b"


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(w=7.0, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def build(db_path: Path, out_dir: Path, image_dir: Path | None = None,
          run_id: int | None = None, baseline_wafer: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db_path)
    run_id = run_id or db.latest_run(conn)
    if run_id is None:
        raise ValueError("database has no runs")
    found = db.rows(conn, "SELECT * FROM runs WHERE id = ?", (run_id,))
    if not found:
        raise ValueError(f"run {run_id} not found")
    run = found[0]
    images = db.rows(conn, "SELECT * FROM image_summary WHERE run_id = ? ORDER BY wafer, site, name", (run_id,))
    lines = db.rows(conn, """SELECT l.*, i.name, i.wafer, i.site FROM lines l JOIN images i ON i.id = l.image_id
                              WHERE i.run_id = ? ORDER BY i.wafer, i.site, l.idx""", (run_id,))
    ok_images = [r for r in images if r["status"] in ("ok", "partial")]
    charts = []

    # 1. accuracy against ground truth (synthetic data only)
    with_truth = [ln for ln in lines if ln["true_cd_mid_nm"] is not None]
    acc = None
    if with_truth:
        meas = np.array([ln["cd_mid_nm"] for ln in with_truth])
        true = np.array([ln["true_cd_mid_nm"] for ln in with_truth])
        acc = stats.accuracy(meas, true)
        acc["bias"] = round(acc["bias"], 3) + 0.0       # avoid printing "-0.000"
        fig, ax = _fig(6.0, 3.4)
        err = meas - true
        ax.hist(err, bins=24, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
        ax.axvline(0, color=INK_2, linewidth=1)
        _style(ax, f"Mid-height CD error vs ground truth  (n={acc['n']}, bias {acc['bias']:+.3f} nm, "
                   f"σ {acc['std']:.3f} nm)", "measured − true CD (nm)", "lines")
        _save(fig, out_dir / "accuracy.png")
        charts.append(("accuracy.png", "Measurement error on synthetic images with known edges."))

    # 2. control chart of the per-image mean mid CD, in wafer/site order
    if ok_images:
        vals = np.array([r["mean_cd_mid_nm"] for r in ok_images])
        # images are sorted by wafer, so the baseline wafer's images are a prefix
        # only when it is the first wafer; otherwise use every image
        base = None
        if baseline_wafer and ok_images[0]["wafer"] == baseline_wafer:
            base = sum(1 for r in ok_images if r["wafer"] == baseline_wafer)
        lim = stats.control_limits(vals, baseline=base)
        fig, ax = _fig(7.5, 3.6)
        x = np.arange(len(vals))
        ax.plot(x, vals, color=BLUE, linewidth=2, zorder=2)
        inside = (vals <= lim["ucl"]) & (vals >= lim["lcl"])
        ax.scatter(x[inside], vals[inside], s=22, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(x[~inside], vals[~inside], s=40, color=CRITICAL, edgecolor=SURFACE, linewidth=1.5,
                   zorder=4, label="outside limits")
        for key, style in (("centre", "-"), ("ucl", "--"), ("lcl", "--")):
            ax.axhline(lim[key], color=INK_2 if key == "centre" else MUTED, linestyle=style, linewidth=1)
            ax.annotate(f"{key.upper() if key != 'centre' else 'CL'} {lim[key]:.2f}", (len(vals) - 0.5, lim[key]),
                        color=INK_2, fontsize=8, va="bottom", ha="right")
        wafers = [r["wafer"] for r in ok_images]
        for i in range(1, len(wafers)):
            if wafers[i] != wafers[i - 1]:
                ax.axvline(i - 0.5, color=GRID, linewidth=1)
        starts = {}
        for i, wf in enumerate(wafers):
            starts.setdefault(wf, []).append(i)
        ax.set_xticks([np.mean(v) for v in starts.values()], list(starts.keys()))
        title = "Image mean CD (mid height) — individuals chart"
        if base:
            title += f", limits from {baseline_wafer}"
        _style(ax, title, "", "CD (nm)")
        ax.xaxis.grid(False)            # the wafer separators replace vertical grid lines
        if (~inside).any():
            ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8, labelcolor=INK_2,
                      loc="upper center")
        _save(fig, out_dir / "control_chart.png")
        charts.append(("control_chart.png", "Per-image mean CD in wafer/site order with ±3σ limits."))

        # 3. per-wafer distribution of line CDs; boxes from normal lines, flagged
        # outliers drawn separately so the chart agrees with the outlier table
        by_wafer: dict[str, list[float]] = {}
        flagged: list[tuple[str, float]] = []
        for ln in lines:
            by_wafer.setdefault(ln["wafer"], [])
            if ln["outlier"]:
                flagged.append((ln["wafer"], ln["cd_mid_nm"]))
            else:
                by_wafer[ln["wafer"]].append(ln["cd_mid_nm"])
        fig, ax = _fig(6.0, 3.4)
        names = list(by_wafer)
        bp = ax.boxplot([by_wafer[n] for n in names], tick_labels=names, widths=0.5, patch_artist=True,
                        whis=(0, 100), medianprops={"color": INK, "linewidth": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE_LIGHT)
            patch.set_edgecolor(BLUE)
        if flagged:
            ax.scatter([names.index(w) + 1 for w, _ in flagged], [v for _, v in flagged], s=40, color=CRITICAL,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label="flagged outlier line")
            ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8, labelcolor=INK_2,
                      loc="upper right")
        _style(ax, "Line CD by wafer (mid height), whiskers = min/max", "", "CD (nm)")
        _save(fig, out_dir / "by_wafer.png")
        charts.append(("by_wafer.png", "Every measured line, grouped by wafer; flagged outliers in red."))

        # 4. site signature: mean CD per site position (3x3 grid) across wafers,
        # leaving out flagged lines so one defect does not shift a whole site
        sites: dict[int, list[float]] = {}
        for ln in lines:
            if ln["site"] and not ln["outlier"]:
                sites.setdefault(int(ln["site"]), []).append(ln["cd_mid_nm"])
        if sites and max(sites) <= 9:
            grid = np.full((3, 3), np.nan)
            for s, v in sites.items():
                grid[(s - 1) // 3, (s - 1) % 3] = np.mean(v)
            fig, ax = _fig(4.2, 3.6)
            # start the colour scale a little below the minimum so no cell is white
            im = ax.imshow(grid, cmap="Blues", vmin=np.nanmin(grid) - 0.3, vmax=np.nanmax(grid))
            for (i, j), v in np.ndenumerate(grid):
                if not np.isnan(v):
                    dark = v > (np.nanmin(grid) + np.nanmax(grid)) / 2
                    ax.text(j, i, f"S{i * 3 + j + 1}\n{v:.2f}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if dark else INK)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Mean CD by site (nm), all wafers", loc="left", color=INK, fontsize=11, pad=10)
            ax.set_xlabel("outlier lines excluded", color=INK_2, fontsize=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors=MUTED, labelsize=8)
            _save(fig, out_dir / "site_map.png")
            charts.append(("site_map.png", "Site-to-site signature averaged over wafers, outliers excluded."))

    # 5. overlays for the first image and for any image holding an outlier
    if image_dir:
        picks = []
        if ok_images:
            picks.append(ok_images[0]["name"])
        picks += sorted({ln["name"] for ln in lines if ln["outlier"]})
        for name in picks:
            png = Path(image_dir) / f"{name}.png"
            if not png.exists():
                continue
            img = load_image(png)
            # edge positions are not stored, so re-measure; overlays only need
            # pixel coordinates, so the pixel size does not matter here
            res = measure_image(img, 1.0, name=name)
            out = draw_overlay(img, res)
            out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_dir / f"overlay_{name}.png"), out)
            charts.append((f"overlay_{name}.png",
                           f"{name}: detected edges at 10% (orange), 50% (green) and 90% (blue) height."))

    # summary tables
    status_counts = {}
    for r in images:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    cd_mid = stats.summarize([ln["cd_mid_nm"] for ln in lines])
    outliers = [ln for ln in lines if ln["outlier"]]
    bad = [r for r in images if r["status"] != "ok"]
    conn.close()

    def table(headers, rows_):
        head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows_)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    parts = [
        f"<h1>CD metrology run {run_id}</h1>",
        f"<p class='muted'>Source <code>{html.escape(run['source'])}</code> · {run['started_at']} · "
        f"{run['n_images']} images</p>",
        "<h2>Summary</h2>",
        table(["images", "status", "lines measured", "mean CD mid (nm)", "3σ (nm)", "range (nm)", "outlier lines"],
              [[len(images), ", ".join(f"{k}: {v}" for k, v in status_counts.items()), cd_mid.get("n", 0),
                f"{cd_mid.get('mean', 0):.2f}", f"{cd_mid.get('three_sigma', 0):.2f}",
                f"{cd_mid.get('range', 0):.2f}", len(outliers)]]),
    ]
    if acc:
        parts += ["<h2>Accuracy vs ground truth</h2>",
                  table(["lines", "bias (nm)", "σ (nm)", "RMSE (nm)", "max |error| (nm)"],
                        [[acc["n"], f"{acc['bias']:+.3f}", f"{acc['std']:.3f}", f"{acc['rmse']:.3f}",
                          f"{acc['max_abs']:.3f}"]])]
    if outliers:
        parts += ["<h2>Outlier lines (|robust z| > 3.5)</h2>",
                  table(["image", "line", "CD mid (nm)", "robust z"],
                        [[o["name"], o["idx"], f"{o['cd_mid_nm']:.2f}", f"{o['robust_z']:+.1f}"] for o in outliers])]
    if bad:
        parts += ["<h2>Images not fully measured</h2>",
                  table(["image", "status", "SNR", "reason"],
                        [[b["name"], b["status"], f"{b['snr']:.1f}", b["reason"]] for b in bad])]
    parts += ["<h2>Per-image results</h2>",
              table(["image", "wafer", "site", "status", "lines", "CD bottom", "CD mid", "CD top", "height", "SWA (°)"],
                    [[r["name"], r["wafer"], r["site"], r["status"], r["n_lines"],
                      *(f"{r[k]:.2f}" if r[k] is not None else "" for k in
                        ("mean_cd_bottom_nm", "mean_cd_mid_nm", "mean_cd_top_nm", "mean_height_nm", "mean_swa_deg"))]
                     for r in images])]
    parts += ["<h2>Charts</h2>"] + [
        f"<figure><img src='{c}' alt='{html.escape(cap)}'><figcaption>{html.escape(cap)}</figcaption></figure>"
        for c, cap in charts]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CD metrology run {run_id}</title>
<style>
body {{ background:{SURFACE}; color:{INK}; font:14px/1.5 system-ui, sans-serif; max-width:960px; margin:24px auto;
       padding:0 16px; }}
h1 {{ font-size:22px; margin-bottom:4px }} h2 {{ font-size:16px; margin-top:28px }}
.muted {{ color:{INK_2} }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums }}
th, td {{ text-align:left; padding:4px 8px; border-bottom:1px solid {GRID} }}
th {{ color:{INK_2}; font-weight:600 }}
figure {{ margin:16px 0 }} img {{ max-width:100%; border:1px solid {GRID} }}
figcaption {{ color:{INK_2}; font-size:12px }}
</style></head><body>{''.join(parts)}</body></html>"""
    out = out_dir / "report.html"
    out.write_text(page)
    return out
