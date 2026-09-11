"""Posterior predictive checks: does a fitted twin reproduce the recording it was fitted to?

Parameter sets drawn from the posterior are re-simulated and their features compared with the observed ones.
The same check with prior draws gives the uninformed baseline, so the gain from fitting is explicit.
"""
import numpy as np

from .features import FEATURE_NAMES, compute_features
from .sim import ChipConfig, simulate


def predictive(theta_samples, duration_s=300.0, warmup_s=10.0, cfg=ChipConfig(), seed=0):
    out = simulate(theta_samples, cfg, duration_s=duration_s, warmup_s=warmup_s, seed=seed)
    return np.array([compute_features(e, t, duration_s) for e, t in out])


def summary(x_obs, x_pred, level=0.9):
    """Per-feature: is the observation inside the central `level` predictive interval, and its z-score."""
    lo, hi = np.quantile(x_pred, [(1 - level) / 2, (1 + level) / 2], axis=0)
    sd = x_pred.std(0)
    z = np.where(sd > 0, (x_obs - x_pred.mean(0)) / np.where(sd > 0, sd, 1.0), np.nan)
    return {"feature": FEATURE_NAMES, "inside": (x_obs >= lo) & (x_obs <= hi), "z": z,
            "lo": lo, "hi": hi, "median": np.median(x_pred, 0)}


def energy_distance(x_obs, x_pred, scale):
    """Scaled mean distance between the observation and predictive draws minus the draws' own spread."""
    a = np.abs((x_pred - x_obs) / scale).sum(1).mean()
    idx = np.random.default_rng(0).permutation(len(x_pred))
    b = np.abs((x_pred - x_pred[idx]) / scale).sum(1).mean()
    return a - 0.5 * b
