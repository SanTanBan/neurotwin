"""Amortised Bayesian inference for the NeuroTwin with neural posterior estimation (sbi).

The posterior network is trained once on a bank of simulated cultures and then turns any recording's
features into a full posterior over the twin's biophysical parameters in milliseconds. Parameters are
handled in the space where the prior is uniform (log for scale parameters).

Sampling never uses rejection: for recordings unlike any simulation the flow can put mass outside the prior
box, which would stall rejection sampling. Samples are clipped to the box instead, and the fraction that
leaked outside is returned as a reliability signal.
"""
import numpy as np
import torch
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp
from sbi.inference import NPE
from sbi.utils import BoxUniform

from .sim import HI, LO, PARAM_NAMES, from_unbounded, to_unbounded

Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)


def make_prior():
    return BoxUniform(torch.tensor(Z_LO, dtype=torch.float32), torch.tensor(Z_HI, dtype=torch.float32))


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def train_npe(theta, x, density_estimator="nsf", seed=0, **train_kw):
    """theta: physical parameters [n, d]; x: features [n, f]. Returns (posterior, trainer)."""
    torch.manual_seed(seed)
    trainer = NPE(prior=make_prior(), density_estimator=density_estimator, show_progress_bars=False)
    trainer.append_simulations(_t(to_unbounded(theta)), _t(x))
    estimator = trainer.train(**train_kw)
    return trainer.build_posterior(estimator), trainer


def sample_z(posterior, X, n=500):
    """Samples in prior-uniform space for many observations: ([n, m, d] clipped, leakage [m])."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        s = posterior.sample_batched((n,), x=_t(X), show_progress_bars=False,
                                     reject_outside_prior=False).detach().numpy()
    leak = np.mean(np.any((s < Z_LO) | (s > Z_HI), axis=2), axis=0)
    return np.clip(s, Z_LO, Z_HI), leak


def sample(posterior, x_obs, n=2000):
    """Posterior samples in physical units for a single observation."""
    s, _ = sample_z(posterior, x_obs, n)
    return from_unbounded(s[:, 0])


def sample_many(posterior, X, n=500):
    """Posterior samples in physical units for many observations: ([n, m, d], leakage [m])."""
    s, leak = sample_z(posterior, X, n)
    return from_unbounded(s), leak


def recovery_metrics(posterior, theta_test, x_test, n=500, levels=(0.5, 0.9)):
    """Parameter recovery on held-out simulations, per parameter (in prior-uniform space)."""
    z_true = to_unbounded(theta_test)
    s, _ = sample_z(posterior, x_test, n)                                  # [n, m, d]
    width = Z_HI - Z_LO
    means = s.mean(0)
    ss_res = ((means - z_true) ** 2).sum(0)
    ss_tot = ((z_true - z_true.mean(0)) ** 2).sum(0)
    out = {"param": PARAM_NAMES, "r2": 1 - ss_res / ss_tot,
           "nrmse": np.sqrt(ss_res / len(z_true)) / width,
           "contraction": (1.0 - s.var(0) / (width ** 2 / 12.0)).mean(0), "post_mean_z": means}
    for lv in levels:
        lo, hi = np.quantile(s, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        out[f"coverage_{int(lv * 100)}"] = ((z_true >= lo) & (z_true <= hi)).mean(0)
    return out


def calibration(posterior, theta_test, x_test, n_post=500):
    """Simulation-based calibration (per-parameter ranks) and TARP joint expected coverage."""
    th, xs = _t(to_unbounded(theta_test)), _t(x_test)
    ranks, dap = run_sbc(th, xs, posterior, num_posterior_samples=n_post, show_progress_bar=False)
    sbc = check_sbc(ranks, th, dap, num_posterior_samples=n_post)
    ecp, alpha = run_tarp(th, xs, posterior, num_posterior_samples=n_post, show_progress_bar=False)
    atc, ks_p = check_tarp(ecp, alpha)
    return {"ranks": ranks.numpy(), "sbc_ks_pvals": sbc["ks_pvals"].numpy(),
            "c2st_ranks": sbc["c2st_ranks"].numpy(), "tarp_ecp": ecp.numpy(),
            "tarp_alpha": alpha.numpy(), "tarp_atc": float(atc), "tarp_ks_p": float(ks_p)}
