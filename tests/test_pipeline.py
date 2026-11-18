import csv
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cdmetro import db
from cdmetro.pipeline import run
from cdmetro.report import build
from cdmetro.synth import make_lot


@pytest.fixture(scope="module")
def lot(tmp_path_factory):
    root = tmp_path_factory.mktemp("lot")
    make_lot(root / "images", wafers=3, sites=9, seed=7)
    summary = run(root / "images", root / "cd.sqlite", root / "cd.csv")
    return root, summary


def test_run_summary(lot):
    root, s = lot
    assert s["images"] == 27
    assert s["status"] == {"ok": 26, "rejected": 1}      # W02_S08 is the low-contrast image
    assert s["lines"] == 26 * 5


def test_necked_line_is_the_only_outlier(lot):
    root, _ = lot
    conn = db.connect(root / "cd.sqlite")
    rows = db.rows(conn, """SELECT i.name, l.idx FROM lines l JOIN images i ON i.id = l.image_id
                            WHERE l.outlier = 1""")
    assert rows == [{"name": "W03_S05", "idx": 2}]


def test_database_matches_csv(lot):
    root, _ = lot
    conn = db.connect(root / "cd.sqlite")
    n_db = db.rows(conn, "SELECT COUNT(*) AS n FROM lines")[0]["n"]
    with open(root / "cd.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["line"]]
    assert len(rows) == n_db
    rejected = db.rows(conn, "SELECT name, reason FROM images WHERE status = 'rejected'")
    assert rejected[0]["name"] == "W02_S08" and "SNR" in rejected[0]["reason"]


def test_report_builds(lot, tmp_path):
    root, _ = lot
    page = build(root / "cd.sqlite", tmp_path, root / "images", baseline_wafer="W01")
    text = page.read_text()
    assert "W03_S05" in text and "Accuracy vs ground truth" in text
    for f in ("accuracy.png", "control_chart.png", "by_wafer.png", "site_map.png", "overlay_W03_S05.png"):
        assert (tmp_path / f).exists()


def test_api(lot, monkeypatch):
    root, _ = lot
    monkeypatch.setenv("CDMETRO_DB", str(root / "cd.sqlite"))
    from cdmetro.api import app
    client = TestClient(app)

    runs = client.get("/runs").json()
    run_id = runs[0]["id"]
    summary = client.get(f"/runs/{run_id}/summary").json()
    assert summary["outlier_lines"] == 1
    assert summary["images_by_status"]["rejected"] == 1

    w1 = client.get(f"/runs/{run_id}/images", params={"wafer": "W01"}).json()
    assert len(w1) == 9
    out = client.get(f"/runs/{run_id}/lines", params={"outlier": True}).json()
    assert out[0]["image"] == "W03_S05"
    assert client.get("/runs/999/summary").status_code == 404

    png = root / "images" / "W01_S01.png"
    pixel_nm = json.loads(png.with_suffix(".json").read_text())["pixel_nm"]
    r = client.post("/measure", params={"pixel_nm": pixel_nm},
                    files={"file": ("W01_S01.png", png.read_bytes(), "image/png")})
    assert r.status_code == 200
    assert len(r.json()["lines"]) == 5

    _, junk = cv2.imencode(".png", (cv2.imread(str(png)) * 0).astype("uint8"))
    r = client.post("/measure", params={"pixel_nm": 0.5}, files={"file": ("x.png", junk.tobytes(), "image/png")})
    assert r.status_code == 200 and r.json()["status"] in ("rejected", "failed")
    assert client.post("/measure", params={"pixel_nm": 0.5},
                       files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400


def test_duplicate_image_names_rejected(tmp_path):
    for ext in (".png", ".tif"):
        cv2.imwrite(str(tmp_path / f"a{ext}"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="unique"):
        run(tmp_path, tmp_path / "x.sqlite")


def test_report_unknown_run(lot, tmp_path):
    root, _ = lot
    with pytest.raises(ValueError, match="not found"):
        build(root / "cd.sqlite", tmp_path, run_id=99)


def test_api_without_database(tmp_path, monkeypatch):
    monkeypatch.setenv("CDMETRO_DB", str(tmp_path / "missing.sqlite"))
    from cdmetro.api import app
    assert TestClient(app).get("/runs").status_code == 503
