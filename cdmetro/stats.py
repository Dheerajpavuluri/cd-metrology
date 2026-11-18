"""Statistics on measured CDs: outliers, control limits and accuracy."""

from __future__ import annotations

import numpy as np

ROBUST_Z_LIMIT = 3.5


def robust_z(values) -> np.ndarray:
    """Modified z-score (Iglewicz & Hoaglin): 0.6745 * (x - median) / MAD.

    Median and MAD are not dragged around by the outliers we are looking for,
    which a plain mean/standard deviation would be.
    """
    v = np.asarray(values, float)
    if v.size == 0:
        return v
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return np.zeros_like(v)
    return 0.6745 * (v - med) / mad


def flag_outliers(values, limit: float = ROBUST_Z_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    z = robust_z(values)
    return z, np.abs(z) > limit


def control_limits(values, baseline: int | None = None) -> dict:
    """Individuals chart limits: centre +/- 3 sigma, sigma from the average moving range.

    Sigma from the moving range (MR-bar / 1.128) reflects short-term variation,
    so a slow drift between wafers shows up against the limits instead of
    widening them. `baseline` limits the calculation to the first N points.
    """
    v = np.asarray(values, float)
    ref = v[:baseline] if baseline else v
    centre = float(ref.mean())
    mr = np.abs(np.diff(ref))
    sigma = float(mr.mean() / 1.128) if len(mr) else 0.0
    return {"centre": centre, "sigma": sigma, "ucl": centre + 3 * sigma, "lcl": centre - 3 * sigma}


def summarize(values) -> dict:
    v = np.asarray(values, float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "three_sigma": float(3 * v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
    }


def accuracy(measured, truth) -> dict:
    """Bias and spread of measured - truth."""
    err = np.asarray(measured, float) - np.asarray(truth, float)
    if err.size == 0:
        return {"n": 0}
    return {
        "n": int(err.size),
        "bias": float(err.mean()),
        "std": float(err.std(ddof=1)) if err.size > 1 else 0.0,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "max_abs": float(np.abs(err).max()),
    }
