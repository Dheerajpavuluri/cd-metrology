"""Small REST API over the results database, plus on-demand measurement.

    uvicorn cdmetro.api:app            # CDMETRO_DB=results/cd.sqlite by default

GET  /runs                         list runs
GET  /runs/{run_id}/images         per-image summary (filter: ?wafer=W01&status=ok)
GET  /runs/{run_id}/lines          per-line results (filter: ?outlier=true)
GET  /runs/{run_id}/summary        CD statistics, outlier count, status counts
POST /measure?pixel_nm=0.5         upload one image, get measurements back (not stored)
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile

from . import db, stats
from .measure import MeasurementError, measure_image, to_float

app = FastAPI(title="cd-metrology", version="0.1.0")


def get_conn() -> Iterator[sqlite3.Connection]:
    """One connection per request, closed afterwards."""
    conn = db.connect(os.environ.get("CDMETRO_DB", "results/cd.sqlite"))
    try:
        yield conn
    finally:
        conn.close()


def _check_run(conn: sqlite3.Connection, run_id: int) -> None:
    if not db.rows(conn, "SELECT id FROM runs WHERE id = ?", (run_id,)):
        raise HTTPException(404, f"run {run_id} not found")


@app.get("/runs")
def runs(conn: sqlite3.Connection = Depends(get_conn)):
    return db.rows(conn, "SELECT * FROM runs ORDER BY id DESC")


@app.get("/runs/{run_id}/images")
def images(run_id: int, wafer: str | None = None, status: str | None = None,
           conn: sqlite3.Connection = Depends(get_conn)):
    _check_run(conn, run_id)
    sql, params = "SELECT * FROM image_summary WHERE run_id = ?", [run_id]
    if wafer:
        sql += " AND wafer = ?"; params.append(wafer)
    if status:
        sql += " AND status = ?"; params.append(status)
    return db.rows(conn, sql + " ORDER BY wafer, site, name", params)


@app.get("/runs/{run_id}/lines")
def lines(run_id: int, outlier: bool | None = None, image: str | None = None,
          conn: sqlite3.Connection = Depends(get_conn)):
    _check_run(conn, run_id)
    sql = """SELECT i.name AS image, i.wafer, i.site, l.idx AS line, l.center_nm, l.cd_bottom_nm, l.cd_mid_nm,
                    l.cd_top_nm, l.height_nm, l.swa_deg, l.robust_z, l.outlier
             FROM lines l JOIN images i ON i.id = l.image_id WHERE i.run_id = ?"""
    params: list = [run_id]
    if outlier is not None:
        sql += " AND l.outlier = ?"; params.append(int(outlier))
    if image:
        sql += " AND i.name = ?"; params.append(image)
    return db.rows(conn, sql + " ORDER BY i.wafer, i.site, l.idx", params)


@app.get("/runs/{run_id}/summary")
def summary(run_id: int, conn: sqlite3.Connection = Depends(get_conn)):
    _check_run(conn, run_id)
    cds = [r["cd_mid_nm"] for r in db.rows(
        conn, "SELECT l.cd_mid_nm FROM lines l JOIN images i ON i.id = l.image_id WHERE i.run_id = ?", (run_id,))]
    status = db.rows(conn, "SELECT status, COUNT(*) AS n FROM images WHERE run_id = ? GROUP BY status", (run_id,))
    n_out = db.rows(conn, """SELECT COUNT(*) AS n FROM lines l JOIN images i ON i.id = l.image_id
                             WHERE i.run_id = ? AND l.outlier = 1""", (run_id,))[0]["n"]
    return {"run_id": run_id, "cd_mid_nm": stats.summarize(cds),
            "images_by_status": {r["status"]: r["n"] for r in status}, "outlier_lines": n_out}


@app.post("/measure")
async def measure(file: UploadFile = File(...), pixel_nm: float = Query(..., gt=0)):
    data = np.frombuffer(await file.read(), np.uint8)
    raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise HTTPException(400, "could not decode image")
    try:
        img = to_float(raw)
    except MeasurementError as e:
        raise HTTPException(400, str(e))
    return measure_image(img, pixel_nm, name=file.filename or "upload").to_dict()
