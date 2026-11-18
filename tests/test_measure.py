import numpy as np
import pytest

from cdmetro.measure import HEIGHTS, estimate_noise, measure_file, measure_image
from cdmetro.synth import Recipe, render


def _clean(**kw):
    r = Recipe(noise_sigma=0.0, wall_roughness_nm=0.0, cd_sigma_nm=0.0, **kw)
    img, truth = render(r, np.random.default_rng(1))
    # a whisper of noise so the noise estimate is not exactly zero
    img = img + np.random.default_rng(2).normal(0, 1e-4, img.shape).astype(np.float32)
    return img, truth, r


def test_vertical_walls_exact():
    img, truth, r = _clean(top_cd_nm=24.0)
    res = measure_image(img, r.pixel_nm)
    assert res.status == "ok"
    assert len(res.lines) == r.n_lines
    for ln, t in zip(res.lines, truth):
        for k in HEIGHTS:
            assert ln.cd_nm[k] == pytest.approx(t.cd_nm[k], abs=0.02)
        assert ln.swa_deg == pytest.approx(90.0, abs=0.2)


def test_tapered_lines_cd_height_and_angle():
    img, truth, r = _clean()
    res = measure_image(img, r.pixel_nm)
    expected_swa = 90 - np.degrees(np.arctan((r.bottom_cd_nm - r.top_cd_nm) / 2 / r.line_height_nm))
    for ln, t in zip(res.lines, truth):
        for k in HEIGHTS:
            assert ln.cd_nm[k] == pytest.approx(t.cd_nm[k], abs=0.05)
        assert ln.height_nm == pytest.approx(r.line_height_nm, abs=0.2)
        assert ln.swa_deg == pytest.approx(expected_swa, abs=0.3)


def test_noisy_images_are_unbiased():
    rng = np.random.default_rng(11)
    errors = []
    for _ in range(12):
        r = Recipe(cd_offset_nm=float(rng.normal(0, 1)))
        img, truth = render(r, rng)
        res = measure_image(img, r.pixel_nm)
        assert res.status == "ok"
        errors += [ln.cd_nm["mid"] - t.cd_nm["mid"] for ln, t in zip(res.lines, truth)]
    errors = np.array(errors)
    assert abs(errors.mean()) < 0.05          # nm
    assert errors.std() < 0.15
    assert np.abs(errors).max() < 0.4


def test_pixel_size_scales_results():
    img, truth, r = _clean()
    a = measure_image(img, 0.5)
    b = measure_image(img, 1.0)
    assert b.lines[0].cd_nm["mid"] == pytest.approx(2 * a.lines[0].cd_nm["mid"])


def test_noise_estimate_close_to_true_sigma():
    r = Recipe(noise_sigma=0.05)
    img, _ = render(r, np.random.default_rng(3))
    assert estimate_noise(img) == pytest.approx(0.05, rel=0.2)


def test_low_contrast_image_rejected():
    r = Recipe(line_level=0.31, noise_sigma=0.06)
    img, _ = render(r, np.random.default_rng(4))
    res = measure_image(img, r.pixel_nm)
    assert res.status == "rejected"
    assert res.lines == []
    assert "SNR" in res.reason


def test_blank_image_does_not_crash():
    img = np.full((128, 256), 0.4, np.float32) + np.random.default_rng(5).normal(0, 0.02, (128, 256)).astype(np.float32)
    res = measure_image(img, 0.5)
    assert res.status in ("rejected", "failed")


def test_bad_files_fail_without_raising(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    res, _ = measure_file(corrupt, pixel_nm=0.5)
    assert res.status == "failed" and "cannot read" in res.reason

    res, _ = measure_file(corrupt)            # no sidecar and no pixel size given
    assert res.status == "failed" and "pixel size" in res.reason
