"""NeuroTwin training & validation.

1. Train the amortized posterior (neural posterior estimation, neural spline flow) on the simulation bank.
2. Validate on held-out simulations: parameter recovery, simulation-based calibration (SBC), TARP coverage.
3. Apply to every 300-s window of the real recordings; aggregate per recording; trajectories vs age (DIV).
4. Reliability: out-of-support score for every real window; posterior predictive checks on a subset.
Outputs in /kaggle/working: posterior.pt, calibration.json, real_posteriors.csv, ppc.csv, figures/*.png
"""
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sbi==0.27.0"], check=False)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INPUT = "/kaggle/input" if os.path.isdir("/kaggle/input") else "."
FIG = os.path.join(WORK, "figures")
N_TEST = 1000
N_PPC_REC = 12
N_PPC_DRAWS = 16
WIN_S = 300.0

# ---- begin src/neurotwin/sim.py ----
"""Spiking-network simulator of a cultured neuronal network grown on a microelectrode-array (MEA) chip.

One simulation = one virtual culture: N leaky integrate-and-fire neurons (80% excitatory, 20% inhibitory)
scattered over the chip, distance-dependent random connectivity, Tsodyks-Markram short-term synaptic
depression (terminates network bursts), spike-frequency adaptation and a noisy background drive.
Virtual electrodes pick up spikes of neurons within `pickup_um` of their centre, giving multi-unit spike
trains in the same (time, electrode) format as real MEA recordings.

The inner loop is event-driven and compiled with numba; simulations run in parallel over CPU cores.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

# Parameters inferred by the digital twin: (low, high, scale). "log" = log-uniform prior.
PRIOR_BOUNDS = {
    "j_e": (0.5, 20.0, "log"),        # mV, EPSP of a fully recovered excitatory synapse
    "g_ie": (0.25, 4.0, "log"),       # inhibition/excitation ratio: IPSP = g_ie * j_e
    "u_std": (0.02, 0.8, "log"),      # release probability: fraction of synaptic resources used per spike
    "tau_rec": (0.2, 10.0, "log"),    # s, recovery time constant of synaptic resources
    "b_adapt": (0.0, 3.0, "lin"),     # mV, steady-state hyperpolarisation added per spike (adaptation)
    "tau_adapt": (0.5, 30.0, "log"),  # s, adaptation decay time constant
    "mu": (0.0, 14.0, "lin"),         # mV, mean background drive above rest (threshold is 15 mV above rest)
    "mu_sd": (0.0, 4.0, "lin"),       # mV, neuron-to-neuron spread of the drive (intrinsic excitability)
    "sigma": (1.0, 6.0, "lin"),       # mV, stationary std of membrane noise
    "lambda_um": (100.0, 3000.0, "log"),  # um, connection length scale
}
PARAM_NAMES = list(PRIOR_BOUNDS)
LOG = np.array([b[2] == "log" for b in PRIOR_BOUNDS.values()])
LO = np.array([b[0] for b in PRIOR_BOUNDS.values()])
HI = np.array([b[1] for b in PRIOR_BOUNDS.values()])


def to_unbounded(theta):
    """Physical parameters -> space where the prior is uniform (log for log-uniform parameters)."""
    theta = np.asarray(theta, float)
    return np.where(LOG, np.log(np.clip(theta, 1e-12, None)), theta)


def from_unbounded(z):
    z = np.asarray(z, float)
    return np.where(LOG, np.exp(z), z)


@dataclass
class ChipConfig:
    """Geometry and fixed biophysics shared by every simulation."""
    n_neurons: int = 400
    frac_exc: float = 0.8
    side_um: float = 1600.0     # square culture area covering the electrode grid
    grid: int = 8               # 8x8 layout with corners removed -> 60 electrodes (MCS-style 60MEA)
    pitch_um: float = 200.0
    drop_corners: bool = True
    pickup_um: float = 80.0     # ~3 neurons per electrode on average -> multi-unit activity
    k_out: float = 40.0         # mean out-degree
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    v_rest: float = -65.0
    v_thresh: float = -50.0
    v_reset: float = -60.0
    t_ref_ms: float = 2.0

    def electrodes(self):
        g = np.arange(self.grid) * self.pitch_um
        g = g - g.mean() + self.side_um / 2
        xy = np.array([(x, y) for y in g for x in g])
        if self.drop_corners:
            last = self.grid - 1
            keep = [i for i in range(len(xy)) if not (i // self.grid in (0, last) and i % self.grid in (0, last))]
            xy = xy[keep]
        return xy


def sample_prior(n, rng=None):
    rng = np.random.default_rng(rng)
    lo, hi = to_unbounded(LO), to_unbounded(HI)
    return from_unbounded(lo + (hi - lo) * rng.random((n, len(LO))))


def build_network(theta_row, cfg, rng):
    """Positions, CSR connectivity (by presynaptic neuron) and neuron->electrode map for one culture."""
    p = dict(zip(PARAM_NAMES, theta_row))
    N = cfg.n_neurons
    n_exc = int(round(cfg.frac_exc * N))
    pos = rng.random((N, 2)) * cfg.side_um
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))          # [post, pre]
    kern = np.exp(-d / p["lambda_um"])
    np.fill_diagonal(kern, 0.0)
    prob = np.minimum(1.0, kern * cfg.k_out / kern.sum(axis=0, keepdims=True))  # out-degree ~ k_out
    conn = rng.random((N, N)) < prob
    pre, post = np.nonzero(conn.T)                                           # sorted by presynaptic neuron
    w = np.where(pre < n_exc, p["j_e"], -p["g_ie"] * p["j_e"]).astype(np.float64)
    ptr = np.zeros(N + 1, np.int64)
    np.add.at(ptr, pre + 1, 1)
    ptr = np.cumsum(ptr)
    elec = cfg.electrodes()
    de = np.sqrt(((pos[:, None, :] - elec[None, :, :]) ** 2).sum(-1))       # [neuron, electrode]
    nearest = de.argmin(1)
    elec_of = np.where(de[np.arange(N), nearest] < cfg.pickup_um, nearest, -1).astype(np.int64)
    mu_vec = p["mu"] + p["mu_sd"] * rng.standard_normal(N)                 # heterogeneous excitability
    return ptr, post.astype(np.int64), w, elec_of, mu_vec


_CACHE = "__file__" in globals()   # numba's on-disk cache needs a source file (not available under exec)


@njit(cache=_CACHE)
def _run_one(n_steps, n_warm, dt, ptr, idx, w, elec_of, u, tau_rec, b_a, tau_a, mu_vec, sigma,
             tau_m, v_rest, v_th, v_reset, ref_steps, seed, out_e, out_t):
    np.random.seed(seed)
    N = elec_of.shape[0]
    decay_m = np.exp(-dt / tau_m)
    noise_step = sigma * np.sqrt(1.0 - decay_m * decay_m)
    decay_a = np.exp(-dt / (tau_a * 1000.0))
    rec_rate = 1.0 - np.exp(-dt / (tau_rec * 1000.0))
    v = v_rest + np.random.random(N) * (v_th - v_rest)
    a = np.zeros(N)
    x = np.ones(N)
    ref = np.zeros(N, np.int64)
    inp = np.zeros(N)
    nxt = np.zeros(N)
    max_ev = out_e.shape[0]
    n_ev = 0
    for t in range(n_steps):
        tmp = inp
        inp = nxt
        nxt = tmp
        nxt[:] = 0.0
        for i in range(N):
            if ref[i] > 0:
                ref[i] -= 1
                v[i] = v_reset
            else:
                vinf = v_rest + mu_vec[i] - a[i]
                v[i] = vinf + (v[i] - vinf) * decay_m + inp[i] + noise_step * np.random.standard_normal()
                if v[i] >= v_th:
                    v[i] = v_reset
                    ref[i] = ref_steps
                    a[i] += b_a
                    eff = u * x[i]
                    x[i] -= eff
                    for k in range(ptr[i], ptr[i + 1]):
                        nxt[idx[k]] += w[k] * eff
                    if t >= n_warm and elec_of[i] >= 0 and n_ev < max_ev:
                        out_e[n_ev] = elec_of[i]
                        out_t[n_ev] = (t - n_warm) * dt / 1000.0
                        n_ev += 1
            a[i] *= decay_a
            x[i] += rec_rate * (1.0 - x[i])
    return n_ev


@njit(parallel=True, cache=_CACHE)
def _run_batch(n_steps, n_warm, dt, ptrs, idx_all, w_all, off, elec_of, mu_vecs, pars, fixed, seeds,
               out_e, out_t, out_n):
    for b in prange(pars.shape[0]):
        lo, hi = off[b], off[b + 1]
        out_n[b] = _run_one(n_steps, n_warm, dt, ptrs[b], idx_all[lo:hi], w_all[lo:hi], elec_of[b],
                            pars[b, 0], pars[b, 1], pars[b, 2], pars[b, 3], mu_vecs[b], pars[b, 4],
                            fixed[0], fixed[1], fixed[2], fixed[3], np.int64(fixed[4]), seeds[b],
                            out_e[b], out_t[b])


def simulate(theta, cfg=ChipConfig(), duration_s=120.0, warmup_s=5.0, seed=0, max_rate_hz=60.0):
    """Simulate a batch of cultures (theta: [B, n_params]).

    Returns a list of (electrode_ids, spike_times_s) tuples, one per culture, sorted by time. Recording
    stops early for runaway cultures whose electrode event count exceeds `max_rate_hz` per electrode.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    B, N = theta.shape[0], cfg.n_neurons
    rng = np.random.default_rng(seed)
    nets = [build_network(th, cfg, rng) for th in theta]
    ptrs = np.stack([n[0] for n in nets])
    sizes = np.array([len(n[1]) for n in nets])
    off = np.concatenate([[0], np.cumsum(sizes)])
    idx_all = np.concatenate([n[1] for n in nets])
    w_all = np.concatenate([n[2] for n in nets])
    elec_of = np.stack([n[3] for n in nets])
    mu_vecs = np.stack([n[4] for n in nets])
    ix = [PARAM_NAMES.index(k) for k in ("u_std", "tau_rec", "b_adapt", "tau_adapt", "sigma")]
    pars = np.ascontiguousarray(theta[:, ix])
    fixed = np.array([cfg.tau_m_ms, cfg.v_rest, cfg.v_thresh, cfg.v_reset, round(cfg.t_ref_ms / cfg.dt_ms)])
    n_elec = len(cfg.electrodes())
    max_ev = int(max_rate_hz * n_elec * duration_s)
    out_e = np.empty((B, max_ev), np.int64)
    out_t = np.empty((B, max_ev), np.float64)
    out_n = np.zeros(B, np.int64)
    n_warm = int(warmup_s * 1000 / cfg.dt_ms)
    n_steps = n_warm + int(duration_s * 1000 / cfg.dt_ms)
    seeds = rng.integers(0, 2**31 - 1, B)
    _run_batch(n_steps, n_warm, cfg.dt_ms, ptrs, idx_all, w_all, off, elec_of, mu_vecs, pars, fixed, seeds,
               out_e, out_t, out_n)
    return [(out_e[b, :out_n[b]].copy(), out_t[b, :out_n[b]].copy()) for b in range(B)]

# ---- end src/neurotwin/sim.py ----

# ---- begin src/neurotwin/features.py ----
"""Summary statistics of MEA spike data.

The same function is applied to real recordings and to simulated cultures, so the digital twin is fitted on
exactly the quantities an experimenter would compute. Input is a flat list of spikes: electrode ids and
spike times (s). Rates are normalised by duration so that recordings of different lengths are comparable.
"""
import numpy as np

FEATURE_NAMES = [
    "log_rate",            # log10 mean firing rate per active electrode (Hz)
    "frac_active",         # fraction of electrodes with >= min_rate
    "log_eburst_rate",     # log10 single-electrode bursts per minute per active electrode
    "log_eburst_dur",      # log10 mean single-electrode burst duration (s)
    "frac_in_burst",       # fraction of spikes that fall in single-electrode bursts
    "log_nb_rate",         # log10 network bursts per minute
    "log_nb_dur",          # log10 mean network-burst duration (s)
    "nb_ibi_cv",           # coefficient of variation of inter-network-burst intervals
    "nb_frac_elec",        # mean fraction of active electrodes recruited per network burst
    "log_nb_spikes",       # log10 mean spikes per network burst per recruited electrode
    "log_nb_peak",         # log10 mean peak population rate inside network bursts (Hz, 10 ms bins)
    "nb_rise_frac",        # mean (peak time - start) / duration: burst shape asymmetry
    "frac_short_ibi",      # fraction of inter-network-burst intervals < 1 s (burst clustering)
    "log_tonic_rate",      # log10 firing rate outside network bursts (Hz per active electrode)
    "sync_corr",           # mean pairwise correlation of 100 ms spike counts between active electrodes
    "log_fano_pop",        # log10 Fano factor of population spike counts in 1 s bins
]
EPS = 1e-3


def _log(x):
    return float(np.log10(max(float(x), EPS)))


def electrode_bursts(ts, max_isi=0.1, min_spikes=4):
    """ISI-threshold burst detection on one electrode: runs of >= min_spikes spikes with ISI <= max_isi."""
    if len(ts) < min_spikes:
        return np.empty(0), np.empty(0), np.empty(0, int)
    close = np.diff(ts) <= max_isi
    d = np.diff(np.r_[0, close.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)      # spike indices: burst spans s..e
    n = e - s + 1
    keep = n >= min_spikes
    return ts[s[keep]], ts[e[keep]], n[keep]


def network_bursts(bursts, n_active, min_frac=0.2, min_elec=3, merge_gap=0.05):
    """Intervals where >= max(min_elec, min_frac*n_active) electrodes burst at the same time.

    bursts: list of (start, end, electrode). Each network burst is extended to the union of the electrode
    bursts overlapping its core; bursts closer than merge_gap are merged.
    """
    if not bursts or n_active == 0:
        return []
    k = max(min_elec, int(np.ceil(min_frac * n_active)))
    starts = np.array([b[0] for b in bursts]); ends = np.array([b[1] for b in bursts])
    elec = np.array([b[2] for b in bursts])
    t = np.r_[starts, ends]
    step = np.r_[np.ones(len(starts)), -np.ones(len(ends))]
    order = np.lexsort((-step, t))                    # at ties process starts before ends
    t, step = t[order], step[order]
    level = np.cumsum(step)
    cores, on = [], None
    for ti, lv in zip(t, level):
        if on is None and lv >= k:
            on = ti
        elif on is not None and lv < k:
            cores.append((on, ti)); on = None
    out = []
    for c0, c1 in cores:
        ov = (starts <= c1) & (ends >= c0)
        s, e = starts[ov].min(), ends[ov].max()
        if out and s - out[-1][1] <= merge_gap:
            out[-1] = (out[-1][0], max(e, out[-1][1]), out[-1][2] | set(elec[ov].tolist()))
        else:
            out.append((s, e, set(elec[ov].tolist())))
    return out


def compute_features(elec_ids, times, duration, n_electrodes=60, min_rate=0.1):
    """Return a feature vector (len(FEATURE_NAMES),) for one recording."""
    elec_ids = np.asarray(elec_ids); times = np.asarray(times, float)
    counts = np.bincount(elec_ids, minlength=n_electrodes)[:n_electrodes] if len(times) else np.zeros(n_electrodes)
    active = np.flatnonzero(counts / duration >= min_rate)
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    f["frac_active"] = len(active) / n_electrodes
    if len(active) == 0:
        for k in FEATURE_NAMES:
            if k.startswith("log_"):
                f[k] = np.log10(EPS)
        return np.array([f[k] for k in FEATURE_NAMES])
    trains = {e: np.sort(times[elec_ids == e]) for e in active}
    f["log_rate"] = _log(counts[active].mean() / duration)

    bursts, n_in = [], 0
    for e, ts in trains.items():
        s, en, n = electrode_bursts(ts)
        bursts += [(a, b, e) for a, b in zip(s, en)]
        n_in += n.sum()
    n_spk = counts[active].sum()
    f["log_eburst_rate"] = _log(len(bursts) / len(active) / (duration / 60))
    f["log_eburst_dur"] = _log(np.mean([b - a for a, b, _ in bursts]) if bursts else 0.0)
    f["frac_in_burst"] = n_in / max(n_spk, 1)

    nbs = network_bursts(bursts, len(active))
    all_t = np.sort(np.concatenate([trains[e] for e in active]))
    f["log_nb_rate"] = _log(len(nbs) / (duration / 60))
    in_nb = np.zeros(len(all_t), bool)
    if nbs:
        durs = np.array([b - a for a, b, _ in nbs])
        f["log_nb_dur"] = _log(durs.mean())
        f["nb_frac_elec"] = float(np.mean([len(p) for _, _, p in nbs])) / len(active)
        spikes_per, peaks, rise = [], [], []
        for a, b, p in nbs:
            i0, i1 = np.searchsorted(all_t, [a, b + 1e-9])
            in_nb[i0:i1] = True
            seg = all_t[i0:i1]
            spikes_per.append(len(seg) / len(p))
            if len(seg):
                h, edges = np.histogram(seg, bins=max(1, int(np.ceil((b - a) / 0.01))), range=(a, max(b, a + 0.01)))
                peaks.append(h.max() / 0.01)
                rise.append((edges[h.argmax()] - a) / max(b - a, 1e-3))
        f["log_nb_spikes"] = _log(np.mean(spikes_per))
        f["log_nb_peak"] = _log(np.mean(peaks) if peaks else 0.0)
        f["nb_rise_frac"] = float(np.mean(rise)) if rise else 0.0
        if len(nbs) >= 3:
            ibi = np.diff([a for a, _, _ in nbs])
            f["nb_ibi_cv"] = float(ibi.std() / max(ibi.mean(), 1e-9))
            f["frac_short_ibi"] = float((ibi < 1.0).mean())
    else:
        f["log_nb_dur"] = f["log_nb_spikes"] = f["log_nb_peak"] = np.log10(EPS)
    tonic_time = duration - sum(b - a for a, b, _ in nbs)
    f["log_tonic_rate"] = _log((~in_nb).sum() / len(active) / max(tonic_time, 1e-3))

    nb_bins = max(1, int(duration / 0.1))
    mat = np.stack([np.bincount(np.minimum((trains[e] / 0.1).astype(int), nb_bins - 1), minlength=nb_bins)
                    for e in active]).astype(float)
    if len(active) >= 2:
        sd = mat.std(1)
        ok = sd > 0
        if ok.sum() >= 2:
            c = np.corrcoef(mat[ok])
            f["sync_corr"] = float(c[np.triu_indices(ok.sum(), 1)].mean())
    pop = np.bincount(np.minimum(all_t.astype(int), int(duration) - 1), minlength=max(int(duration), 1))
    f["log_fano_pop"] = _log(pop.var() / max(pop.mean(), 1e-9))
    return np.array([f[k] for k in FEATURE_NAMES])

# ---- end src/neurotwin/features.py ----

# ---- begin src/neurotwin/infer.py ----
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

# ---- end src/neurotwin/infer.py ----

# ---- begin src/neurotwin/ppc.py ----
"""Posterior predictive checks: does a fitted twin reproduce the recording it was fitted to?

Parameter sets drawn from the posterior are re-simulated and their features compared with the observed ones.
The same check with prior draws gives the uninformed baseline, so the gain from fitting is explicit.
"""
import numpy as np





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

# ---- end src/neurotwin/ppc.py ----


def load_bank():
    files = sorted(glob.glob(os.path.join(INPUT, "**", "bank_shard*.npz"), recursive=True))
    th, xs = [], []
    for f in files:
        d = np.load(f)
        th.append(d["theta"]); xs.append(d["x"])
    theta, x = np.concatenate(th), np.concatenate(xs)
    inside = np.all((theta >= LO - 1e-9) & (theta <= HI + 1e-9), axis=1)   # support of the current prior
    ok = inside & np.isfinite(x).all(1)
    print(f"bank: {len(theta)} sims from {len(files)} shards, {ok.sum()} inside current prior", flush=True)
    return theta[ok], x[ok]


def out_of_support_scores(x_sim, x_query, k=10):
    """Mean distance to the k nearest simulations in z-scored feature space."""
    mu, sd = x_sim.mean(0), x_sim.std(0) + 1e-9
    a = torch.as_tensor((x_sim - mu) / sd, dtype=torch.float32)
    q = torch.as_tensor((x_query - mu) / sd, dtype=torch.float32)
    out = []
    for chunk in torch.split(q, 256):
        d = torch.cdist(chunk, a)
        out.append(d.topk(k, largest=False).values.mean(1))
    return torch.cat(out).numpy()


def fig_recovery(rec, theta_te, post_means):
    n = len(PARAM_NAMES)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.4))
    z = to_unbounded(theta_te)
    for i, a in enumerate(ax.flat[:n]):
        a.scatter(z[:, i], post_means[:, i], s=3, alpha=0.4)
        lim = [to_unbounded(LO)[i], to_unbounded(HI)[i]]
        a.plot(lim, lim, "k--", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  R2={rec['r2'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Parameter recovery on held-out simulations (posterior mean vs truth, prior-uniform space)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "recovery.png"), dpi=110); plt.close(fig)


def fig_sbc(cal):
    ranks = cal["ranks"]; n = ranks.shape[1]
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6))
    for i, a in enumerate(ax.flat[:n]):
        a.hist(ranks[:, i], bins=20, color="C0", alpha=0.8)
        a.axhline(len(ranks) / 20, color="k", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  KS p={cal['sbc_ks_pvals'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Simulation-based calibration: rank histograms (flat = calibrated)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sbc.png"), dpi=110); plt.close(fig)
    fig, a = plt.subplots(figsize=(4, 4))
    a.plot(cal["tarp_alpha"], cal["tarp_ecp"], lw=2); a.plot([0, 1], [0, 1], "k--")
    a.set_xlabel("credibility level"); a.set_ylabel("expected coverage")
    a.set_title(f"TARP coverage (ATC={cal['tarp_atc']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tarp.png"), dpi=110); plt.close(fig)


def fig_trajectories(agg):
    n = len(PARAM_NAMES)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.4 * ((n + 1) // 2), 6.4))
    for i, a in enumerate(ax.flat[:n]):
        p = PARAM_NAMES[i]
        for (pl, cu), g in agg.groupby(["plating", "culture"]):
            g = g.sort_values("div")
            a.plot(g["div"], g[f"{p}_med"], color=f"C{pl % 10}", alpha=0.35, lw=0.8)
        m = agg.groupby("div")[f"{p}_med"].median()
        a.plot(m.index, m.values, "k", lw=2)
        a.set_title(p, fontsize=10); a.set_xlabel("days in vitro")
        if LOG[i]:
            a.set_yscale("log")
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Inferred twin parameters across development (thin: cultures, coloured by plating; bold: median)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trajectories.png"), dpi=110); plt.close(fig)


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    theta, x = load_bank()
    perm = rng.permutation(len(theta))
    te, tr = perm[:N_TEST], perm[N_TEST:]

    posterior, trainer = train_npe(theta[tr], x[tr], density_estimator="nsf",
                                   training_batch_size=512, stop_after_epochs=30)
    torch.save(posterior, os.path.join(WORK, "posterior.pt"))
    print(f"trained on {len(tr)} sims in {(time.time() - t0) / 60:.1f} min", flush=True)

    rec = recovery_metrics(posterior, theta[te[:500]], x[te[:500]])
    cal = calibration(posterior, theta[te[:500]], x[te[:500]])
    fig_recovery(rec, theta[te[:500]], rec["post_mean_z"])
    fig_sbc(cal)
    summary_cal = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**rec, **cal}.items()
                   if k not in ("ranks", "tarp_ecp", "tarp_alpha", "post_mean_z")}
    with open(os.path.join(WORK, "calibration.json"), "w") as f:
        json.dump(summary_cal, f, indent=1)
    print(pd.DataFrame({k: rec[k] for k in ("param", "r2", "nrmse", "coverage_50", "coverage_90", "contraction")})
          .round(3).to_string(index=False), flush=True)
    print(f"TARP ATC={cal['tarp_atc']:.3f} KS p={cal['tarp_ks_p']:.3f}", flush=True)

    real_files = glob.glob(os.path.join(INPUT, "**", "wagenaar_dense_features.csv"), recursive=True)
    if not real_files:
        print("no real feature table attached; done"); return
    df = pd.read_csv(real_files[0])
    xr = df[FEATURE_NAMES].to_numpy(np.float32)
    thr = np.quantile(out_of_support_scores(x[tr], x[te]), 0.99)
    df["oos_score"] = out_of_support_scores(x[tr], xr)
    df["out_of_support"] = df["oos_score"] > thr
    samples, df["leakage"] = sample_many(posterior, xr, n=500)             # [500, n_windows, d]
    for i, p in enumerate(PARAM_NAMES):
        df[f"{p}_med"] = np.median(samples[:, :, i], 0)
        df[f"{p}_q05"] = np.quantile(samples[:, :, i], 0.05, 0)
        df[f"{p}_q95"] = np.quantile(samples[:, :, i], 0.95, 0)
    df.to_csv(os.path.join(WORK, "real_posteriors.csv"), index=False)
    print(f"real windows: {len(df)}, out-of-support (99% sim threshold): {df.out_of_support.mean():.2%}", flush=True)
    agg = df[~df.out_of_support].groupby(["plating", "culture", "div"])[[f"{p}_med" for p in PARAM_NAMES]] \
        .median().reset_index()
    fig_trajectories(agg)

    # posterior predictive checks on recordings spanning development
    picks = df[~df.out_of_support].drop_duplicates("file").sort_values("div")
    picks = picks.iloc[np.linspace(0, len(picks) - 1, min(N_PPC_REC, len(picks))).astype(int)]
    rows = []
    for _, r in picks.iterrows():
        xo = r[FEATURE_NAMES].to_numpy(np.float32)
        th = sample(posterior, xo, n=N_PPC_DRAWS)
        xp = predictive(th, duration_s=WIN_S, seed=int(rng.integers(2**31 - 1)))
        s_post = summary(xo, xp)
        s_prior = summary(xo, x[te[:N_PPC_DRAWS * 8]])
        rows.append({"file": r["file"], "div": r["div"], "post_inside90": s_post["inside"].mean(),
                     "prior_inside90": s_prior["inside"].mean(),
                     "post_abs_z": np.nanmedian(np.abs(s_post["z"])),
                     "prior_abs_z": np.nanmedian(np.abs(s_prior["z"]))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(WORK, "ppc.csv"), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
