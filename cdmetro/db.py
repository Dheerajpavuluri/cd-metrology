"""SQLite storage for measurement runs.

One run = one batch of images measured together. Each image row keeps the
quality numbers and status; each line row keeps the CDs at three heights,
height, sidewall angle, outlier score and (for synthetic data) ground truth.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    source      TEXT NOT NULL,
    n_images    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    wafer       TEXT,
    site        INTEGER,
    pixel_nm    REAL NOT NULL,
    contrast    REAL,
    noise       REAL,
    snr         REAL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'partial', 'rejected', 'failed')),
    reason      TEXT,
    UNIQUE (run_id, name)
);
CREATE TABLE IF NOT EXISTS lines (
    id            INTEGER PRIMARY KEY,
    image_id      INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    center_nm     REAL,
    height_nm     REAL,
    cd_bottom_nm  REAL,
    cd_mid_nm     REAL,
    cd_top_nm     REAL,
    swa_deg       REAL,
    robust_z      REAL,
    outlier       INTEGER NOT NULL DEFAULT 0,
    true_cd_mid_nm REAL,
    UNIQUE (image_id, idx)
);
-- the UNIQUE constraints already index images by run_id and lines by image_id
CREATE INDEX IF NOT EXISTS ix_images_wafer ON images(run_id, wafer);

-- one row per measured image, the shape most reports want
CREATE VIEW IF NOT EXISTS image_summary AS
SELECT i.run_id, i.id AS image_id, i.name, i.wafer, i.site, i.status, i.reason, i.snr,
       COUNT(l.id)        AS n_lines,
       AVG(l.cd_mid_nm)   AS mean_cd_mid_nm,
       AVG(l.cd_top_nm)   AS mean_cd_top_nm,
       AVG(l.cd_bottom_nm) AS mean_cd_bottom_nm,
       AVG(l.height_nm)   AS mean_height_nm,
       AVG(l.swa_deg)     AS mean_swa_deg,
       SUM(l.outlier)     AS n_outliers
FROM images i LEFT JOIN lines l ON l.image_id = i.id
GROUP BY i.id;
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def save_run(conn: sqlite3.Connection, source: str, results: list[tuple[dict, dict]]) -> int:
    """Store one run. `results` holds (ImageResult dict, sidecar metadata) pairs."""
    with conn:
        cur = conn.execute(
            "INSERT INTO runs (started_at, source, n_images) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, len(results)),
        )
        run_id = cur.lastrowid
        for im, meta in results:
            cur = conn.execute(
                """INSERT INTO images (run_id, name, wafer, site, pixel_nm, contrast, noise, snr, status, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, im["name"], meta.get("wafer"), meta.get("site"), im["pixel_nm"],
                 im["contrast"], im["noise"], im["snr"], im["status"], im["reason"]),
            )
            image_id = cur.lastrowid
            conn.executemany(
                """INSERT INTO lines (image_id, idx, center_nm, height_nm, cd_bottom_nm, cd_mid_nm,
                                      cd_top_nm, swa_deg, robust_z, outlier, true_cd_mid_nm)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(image_id, ln["index"], ln["center_nm"], ln["height_nm"], ln["cd_nm"]["bottom"],
                  ln["cd_nm"]["mid"], ln["cd_nm"]["top"], ln["swa_deg"], ln["robust_z"],
                  int(ln["outlier"]), ln.get("true_cd_mid_nm"))
                 for ln in im["lines"]],
            )
    return run_id


def latest_run(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT MAX(id) AS id FROM runs").fetchone()
    return row["id"]           # NULL (None) when the table is empty


def rows(conn: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
