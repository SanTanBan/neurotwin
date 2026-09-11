"""Amortised Bayesian inference for the NeuroTwin with neural posterior estimation (sbi).

The posterior network is trained once on a bank of simulated cultures and then turns any recording's
features into a full posterior over the twin's biophysical parameters in milliseconds. Parameters are
handled in the space where the prior is uniform (log for scale parameters).
"""
import numpy as np
import torch
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp
from sbi.inference import NPE
from sbi.utils import BoxUniform

from .sim import HI, LO, PARAM_NAMES, from_unbounded, to_unbounded


def make_prior():
    lo = torch.tensor(to_unbounded(LO), dtype=torch.float32)
    hi = torch.tensor(to_unbounded(HI), dtype=torch.float32)
    return BoxUniform(lo, hi)


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def train_npe(theta, x, density_estimator="nsf", seed=0, **train_kw):
    """theta: physical parameters [n, d]; x: features [n, f]. Returns (posterior, trainer)."""
    torch.manual_seed(seed)
    trainer = NPE(prior=make_prior(), density_estimator=density_estimator, show_progress_bars=False)
    trainer.append_simulations(_t(to_unbounded(theta)), _t(x))
    estimator = trainer.train(**train_kw)
    return trainer.build_posterior(estimator), trainer


def sample(posterior, x_obs, n=2000):
    """Posterior samples in physical units for a single observation."""
    s = posterior.sample((n,), x=_t(x_obs), show_progress_bars=False)
    return from_unbounded(s.numpy())


def recovery_metrics(posterior, theta_test, x_test, n=500, levels=(0.5, 0.9)):
    """Parameter recovery on held-out simulations, per parameter (in prior-uniform space)."""
    z_true = to_unbounded(theta_test)
    width = to_unbounded(HI) - to_unbounded(LO)
    prior_var = width ** 2 / 12.0
    means, cover, contraction = [], {lv: [] for lv in levels}, []
    for zt, xo in zip(z_true, x_test):
        s = posterior.sample((n,), x=_t(xo), show_progress_bars=False).numpy()
        means.append(s.mean(0))
        contraction.append(1.0 - s.var(0) / prior_var)
        for lv in levels:
            lo, hi = np.quantile(s, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
            cover[lv].append((zt >= lo) & (zt <= hi))
    means = np.array(means)
    ss_res = ((means - z_true) ** 2).sum(0)
    ss_tot = ((z_true - z_true.mean(0)) ** 2).sum(0)
    out = {"param": PARAM_NAMES, "r2": 1 - ss_res / ss_tot,
           "nrmse": np.sqrt(ss_res / len(z_true)) / width,
           "contraction": np.array(contraction).mean(0)}
    for lv in levels:
        out[f"coverage_{int(lv * 100)}"] = np.array(cover[lv]).mean(0)
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
