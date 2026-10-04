"""NeuroTwin chip design: how long must a two-chamber chip be recorded to read its wiring direction?

A single 5-minute recording reads the direction of an axon-diode chip only weakly (nt-train2c). This notebook
asks the design question that follows. It simulates held-out two-chamber chips for 40 minutes each, cuts every
recording into eight 5-minute windows, fits each window with the two-chamber posterior trained in nt-train2c,
and pools the windows. With a uniform prior on the crossing probabilities, independent windows combine by
multiplying their posteriors; each window's marginal over (p_fwd, p_bwd) is a kernel density estimate of its
posterior draws. For 5, 10, 20 and 40 minutes of recording it reports how often the direction is read
correctly, how wide the uncertainty is, and whether the stated confidence can be trusted.

Outputs in /kaggle/working: design_length_sims.npz, design_length.csv, design_length_summary.csv,
figures/recording_length.png
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
from scipy.stats import binomtest, gaussian_kde  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ON_KAGGLE = os.name != "nt" and os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else "."
INPUT = "/kaggle/input" if ON_KAGGLE else "."
FIG = os.path.join(WORK, "figures")
N_CHIPS = int(os.environ.get("NT_N_CHIPS", 240))      # held-out chips (the environment overrides are for smoke tests)
N_WIN = int(os.environ.get("NT_N_WIN", 8))            # 5-minute windows per chip: 40 minutes of recording
BATCH = int(os.environ.get("NT_BATCH", 16))
N_POST = int(os.environ.get("NT_N_POST", 1000))       # posterior draws per window
WIN_S = 300.0
LENGTHS = [k for k in (1, 2, 4, 8) if k <= N_WIN]     # windows pooled: 5, 10, 20, 40 minutes
GRID = 61                                            # grid over (p_fwd, p_bwd) for the pooled posterior
SEED = 9100
TIME_BUDGET_S = 10.5 * 3600

# ---- begin src/neurotwin/sim.py ----
"""Spiking-network simulator of a cultured neuronal network grown on a microelectrode-array (MEA) chip.

One simulation = one virtual culture: N leaky integrate-and-fire neurons (80% excitatory, 20% inhibitory)
scattered over the chip, distance-dependent random connectivity, Tsodyks-Markram short-term synaptic
depression (terminates network bursts), spike-frequency adaptation, exponentially decaying synaptic currents
and a noisy background drive. Excitatory synapses can carry a slow component (NMDA-like) in addition to the
fast one: the fast part ignites a network burst, the slow part sustains its tail.
Virtual electrodes pick up spikes of neurons within `pickup_um` of their centre, giving multi-unit spike
trains in the same (time, electrode) format as real MEA recordings.

Optionally the chip has two chambers separated by a microchannel barrier (axon-diode design): axons cross
with separate forward/backward probabilities and a conduction delay.

The inner loop is event-driven and compiled with numba; simulations run in parallel over CPU cores.
Model versions: v2 = instantaneous synapses; v3 adds tau_syn; v4 adds f_slow/tau_slow. Each later version
reproduces the earlier one exactly at its neutral setting (tau_syn -> 0; f_slow = 0).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

# Parameters inferred by the digital twin: (low, high, scale). "log" = log-uniform prior.
PRIOR_BOUNDS = {
    "j_e": (0.5, 20.0, "log"),        # mV, total EPSP of a fully recovered excitatory synapse
    "g_ie": (0.25, 4.0, "log"),       # inhibition/excitation ratio: IPSP = g_ie * j_e
    "u_std": (0.02, 0.8, "log"),      # release probability: fraction of synaptic resources used per spike
    "tau_rec": (0.2, 10.0, "log"),    # s, recovery time constant of synaptic resources
    "b_adapt": (0.0, 3.0, "lin"),     # mV, steady-state hyperpolarisation added per spike (adaptation)
    "tau_adapt": (0.5, 30.0, "log"),  # s, adaptation decay time constant
    "mu": (0.0, 14.0, "lin"),         # mV, mean background drive above rest (threshold is 15 mV above rest)
    "mu_sd": (0.0, 4.0, "lin"),       # mV, neuron-to-neuron spread of the drive (intrinsic excitability)
    "sigma": (1.0, 6.0, "lin"),       # mV, stationary std of membrane noise
    "lambda_um": (100.0, 3000.0, "log"),  # um, connection length scale
    "tau_syn": (0.5, 150.0, "log"),   # ms, decay of the fast synaptic current
    "f_slow": (0.0, 0.8, "lin"),      # fraction of each excitatory weight carried by the slow (NMDA-like) current
    "tau_slow": (20.0, 400.0, "log"),  # ms, decay of the slow excitatory current
}
PARAM_NAMES = list(PRIOR_BOUNDS)
LOG = np.array([b[2] == "log" for b in PRIOR_BOUNDS.values()])
LO = np.array([b[0] for b in PRIOR_BOUNDS.values()])
HI = np.array([b[1] for b in PRIOR_BOUNDS.values()])
# Values that switch a later-version mechanism off, used to run parameter sets from older model versions.
NEUTRAL = {"tau_syn": 1e-3, "f_slow": 0.0, "tau_slow": 100.0}


def to_unbounded(theta):
    """Physical parameters -> space where the prior is uniform (log for log-uniform parameters)."""
    theta = np.asarray(theta, float)
    return np.where(LOG, np.log(np.clip(theta, 1e-12, None)), theta)


def from_unbounded(z):
    z = np.asarray(z, float)
    return np.where(LOG, np.exp(z), z)


def complete_theta(theta):
    """Pad parameter sets from older model versions (fewer columns) with the neutral values."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    k = theta.shape[1]
    if k == len(PARAM_NAMES):
        return theta
    pad = np.array([NEUTRAL[n] for n in PARAM_NAMES[k:]])
    return np.hstack([theta, np.tile(pad, (len(theta), 1))])


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
    compartments: int = 1       # 2 = two chambers joined by microchannels (axon-diode chip)
    barrier_um: float = 400.0   # width of the microchannel barrier strip (no somata)
    k_cross: float = 4.0        # mean cross-chamber targets per neuron when the crossing probability is 1
    exit_um: float = 300.0      # spread of crossing axons beyond the channel exit
    cross_delay_ms: float = 5.0  # conduction + synaptic delay of axons crossing the channels

    def chamber_of_electrodes(self):
        """0 = left chamber (A), 1 = right chamber (B), -1 = over the barrier."""
        x, half, gap = self.electrodes()[:, 0], self.side_um / 2, self.barrier_um / 2
        return np.where(x < half - gap, 0, np.where(x > half + gap, 1, -1))

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


def build_network(theta_row, cfg, rng, cross=None):
    """Positions, CSR connectivity (by presynaptic neuron) and neuron->electrode map for one culture.

    For two-chamber chips (cfg.compartments == 2) somata avoid the barrier strip, local wiring stays within a
    chamber, and axons cross through the channels with probability cross = (p_fwd: A->B, p_bwd: B->A),
    landing preferentially near the channel exit. Returns a per-synapse flag marking crossing synapses.
    """
    p = dict(zip(PARAM_NAMES, theta_row))
    N = cfg.n_neurons
    n_exc = int(round(cfg.frac_exc * N))
    pos = rng.random((N, 2)) * cfg.side_um
    two = cfg.compartments == 2
    if two:
        half, gap = cfg.side_um / 2, cfg.barrier_um / 2
        xs = pos[:, 0] * (cfg.side_um - cfg.barrier_um) / cfg.side_um
        pos[:, 0] = np.where(xs < half - gap, xs, xs + cfg.barrier_um)
        comp = (pos[:, 0] > half).astype(np.int64)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))          # [post, pre]
    kern = np.exp(-d / p["lambda_um"])
    np.fill_diagonal(kern, 0.0)
    if two:
        kern *= comp[:, None] == comp[None, :]                                # no direct wiring across the barrier
    prob = np.minimum(1.0, kern * cfg.k_out / kern.sum(axis=0, keepdims=True))  # out-degree ~ k_out
    if two and cross is not None:
        near = np.exp(-(np.abs(pos[:, 0] - half) - gap) / cfg.exit_um)       # closeness to the channel exit
        for src, dst, pc in ((0, 1, cross[0]), (1, 0, cross[1])):
            tgt = (comp == dst) * near
            tgt = tgt / tgt.sum()
            prob = np.minimum(1.0, prob + pc * cfg.k_cross * tgt[:, None] * (comp == src)[None, :])
    conn = rng.random((N, N)) < prob
    pre, post = np.nonzero(conn.T)                                           # sorted by presynaptic neuron
    w = np.where(pre < n_exc, p["j_e"], -p["g_ie"] * p["j_e"]).astype(np.float64)
    is_cross = (comp[pre] != comp[post]).astype(np.int64) if two else np.zeros(len(pre), np.int64)
    ptr = np.zeros(N + 1, np.int64)
    np.add.at(ptr, pre + 1, 1)
    ptr = np.cumsum(ptr)
    elec = cfg.electrodes()
    de = np.sqrt(((pos[:, None, :] - elec[None, :, :]) ** 2).sum(-1))       # [neuron, electrode]
    nearest = de.argmin(1)
    elec_of = np.where(de[np.arange(N), nearest] < cfg.pickup_um, nearest, -1).astype(np.int64)
    mu_vec = p["mu"] + p["mu_sd"] * rng.standard_normal(N)                 # heterogeneous excitability
    return ptr, post.astype(np.int64), w, elec_of, mu_vec, is_cross


_CACHE = "__file__" in globals()   # numba's on-disk cache needs a source file (not available under exec)


@njit(cache=_CACHE)
def _run_one(n_steps, n_warm, dt, ptr, idx, w, is_x, xdel, elec_of, u, tau_rec, b_a, tau_a, mu_vec, sigma,
             tau_syn, f_slow, tau_slow, tau_m, v_rest, v_th, v_reset, ref_steps, seed, out_e, out_t):
    np.random.seed(seed)
    N = elec_of.shape[0]
    decay_m = np.exp(-dt / tau_m)
    noise_step = sigma * np.sqrt(1.0 - decay_m * decay_m)
    decay_a = np.exp(-dt / (tau_a * 1000.0))
    rec_rate = 1.0 - np.exp(-dt / (tau_rec * 1000.0))
    a_syn = np.exp(-dt / tau_syn)      # fast synaptic current decay; ~0 reproduces instantaneous synapses
    amp = 1.0 - a_syn                  # keeps the total voltage delivered per spike equal to the weight
    a_slow = np.exp(-dt / tau_slow)    # slow excitatory current decay
    amp_slow = 1.0 - a_slow
    v = v_rest + np.random.random(N) * (v_th - v_rest)
    a = np.zeros(N)
    x = np.ones(N)
    s = np.zeros(N)                    # fast synaptic current state (mV delivered this step)
    s2 = np.zeros(N)                   # slow excitatory current state
    ref = np.zeros(N, np.int64)
    inp = np.zeros(N)
    nxt = np.zeros(N)
    inp2 = np.zeros(N)
    nxt2 = np.zeros(N)
    ring = np.zeros((xdel + 1, N))     # delayed fast input from channel-crossing synapses
    ring2 = np.zeros((xdel + 1, N))    # delayed slow input from channel-crossing synapses
    max_ev = out_e.shape[0]
    n_ev = 0
    for t in range(n_steps):
        tmp = inp
        inp = nxt
        nxt = tmp
        nxt[:] = 0.0
        tmp = inp2
        inp2 = nxt2
        nxt2 = tmp
        nxt2[:] = 0.0
        if xdel > 0:
            slot = t % (xdel + 1)
            for i in range(N):
                inp[i] += ring[slot, i]
                ring[slot, i] = 0.0
                inp2[i] += ring2[slot, i]
                ring2[slot, i] = 0.0
        for i in range(N):
            s[i] = s[i] * a_syn + inp[i]
            s2[i] = s2[i] * a_slow + inp2[i]
            if ref[i] > 0:
                ref[i] -= 1
                v[i] = v_reset
            else:
                vinf = v_rest + mu_vec[i] - a[i]
                v[i] = vinf + (v[i] - vinf) * decay_m + s[i] + s2[i] + noise_step * np.random.standard_normal()
                if v[i] >= v_th:
                    v[i] = v_reset
                    ref[i] = ref_steps
                    a[i] += b_a
                    e0 = u * x[i]
                    eff = e0 * amp                     # inhibitory synapses: fast only
                    eff_f = eff * (1.0 - f_slow)       # excitatory synapses: fast share
                    eff_s = e0 * f_slow * amp_slow     # excitatory synapses: slow share
                    x[i] -= e0
                    for k in range(ptr[i], ptr[i + 1]):
                        if w[k] > 0.0:
                            fast = w[k] * eff_f
                            slow = w[k] * eff_s
                        else:
                            fast = w[k] * eff
                            slow = 0.0
                        if is_x[k]:
                            ring[(t + xdel) % (xdel + 1), idx[k]] += fast
                            ring2[(t + xdel) % (xdel + 1), idx[k]] += slow
                        else:
                            nxt[idx[k]] += fast
                            nxt2[idx[k]] += slow
                    if t >= n_warm and elec_of[i] >= 0 and n_ev < max_ev:
                        out_e[n_ev] = elec_of[i]
                        out_t[n_ev] = (t - n_warm) * dt / 1000.0
                        n_ev += 1
            a[i] *= decay_a
            x[i] += rec_rate * (1.0 - x[i])
    return n_ev


@njit(parallel=True, cache=_CACHE)
def _run_batch(n_steps, n_warm, dt, ptrs, idx_all, w_all, isx_all, xdel, off, elec_of, mu_vecs, pars, fixed,
               seeds, out_e, out_t, out_n):
    for b in prange(pars.shape[0]):
        lo, hi = off[b], off[b + 1]
        out_n[b] = _run_one(n_steps, n_warm, dt, ptrs[b], idx_all[lo:hi], w_all[lo:hi], isx_all[lo:hi], xdel,
                            elec_of[b], pars[b, 0], pars[b, 1], pars[b, 2], pars[b, 3], mu_vecs[b], pars[b, 4],
                            pars[b, 5], pars[b, 6], pars[b, 7], fixed[0], fixed[1], fixed[2], fixed[3],
                            np.int64(fixed[4]), seeds[b], out_e[b], out_t[b])


def simulate(theta, cfg=ChipConfig(), duration_s=120.0, warmup_s=5.0, seed=0, max_rate_hz=60.0, cross=None):
    """Simulate a batch of cultures (theta: [B, n_params]; cross: optional [B, 2] crossing probabilities).

    Parameter sets from older model versions (fewer columns) are padded with neutral values, so they run
    exactly as in that version. Returns a list of (electrode_ids, spike_times_s) tuples, one per culture,
    sorted by time. Recording stops early for runaway cultures whose electrode event count exceeds
    `max_rate_hz` per electrode.
    """
    theta = complete_theta(theta)
    B, N = theta.shape[0], cfg.n_neurons
    rng = np.random.default_rng(seed)
    nets = [build_network(th, cfg, rng, None if cross is None else cross[b]) for b, th in enumerate(theta)]
    ptrs = np.stack([n[0] for n in nets])
    sizes = np.array([len(n[1]) for n in nets])
    off = np.concatenate([[0], np.cumsum(sizes)])
    idx_all = np.concatenate([n[1] for n in nets])
    w_all = np.concatenate([n[2] for n in nets])
    isx_all = np.concatenate([n[5] for n in nets])
    xdel = max(1, int(round(cfg.cross_delay_ms / cfg.dt_ms))) if cfg.compartments == 2 else 0
    elec_of = np.stack([n[3] for n in nets])
    mu_vecs = np.stack([n[4] for n in nets])
    ix = [PARAM_NAMES.index(k) for k in
          ("u_std", "tau_rec", "b_adapt", "tau_adapt", "sigma", "tau_syn", "f_slow", "tau_slow")]
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
    _run_batch(n_steps, n_warm, cfg.dt_ms, ptrs, idx_all, w_all, isx_all, xdel, off, elec_of, mu_vecs, pars,
               fixed, seeds, out_e, out_t, out_n)
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

# ---- begin src/neurotwin/directional.py ----
"""Directional signature of two-chamber (axon-diode) chips: which chamber leads the network activity.

Electrodes are labelled by chamber (A = left, B = right, -1 = over the barrier). Population events are
detected on the smoothed total population rate at 1-ms resolution. In each event every electrode's first
spike marks when it was recruited; the chamber of the earliest-recruited electrode is the event's leader
(the "burst leader" idea from the culture literature), and the lag between the chambers' robust onsets
(20th percentile of their electrodes' first spikes) quantifies propagation. A chamber counts as recruited
only if at least a quarter of its active electrodes fire twice in the event. Axon-diode chips are engineered
to make the leader consistently one chamber.
"""
import numpy as np
from scipy import ndimage



DIR_FEATURE_NAMES = [
    "log_event_rate",   # log10 population events per minute
    "frac_both",        # fraction of events that recruit both chambers
    "frac_A_only",      # fraction of events confined to chamber A
    "frac_B_only",
    "leader_A",         # fraction of two-chamber events whose earliest electrode is in A
    "dir_index",        # 2 * leader_A - 1: +1 = A always leads, -1 = B always leads
    "median_lag",       # median onset lag B - A (s) over two-chamber events, clipped to +-0.2
    "iqr_lag",          # spread (IQR) of that lag (s)
    "log_rate_ratio",   # log10 of B/A mean firing rate per active electrode
    "sync_AB",          # correlation of the chambers' 100-ms population counts
]


def _rate(t, duration, bin_s):
    n = max(1, int(duration / bin_s))
    return np.bincount(np.minimum((t / bin_s).astype(int), n - 1), minlength=n).astype(float)


def _events(times, duration, k_mad=8.0, bin_s=0.001, smooth_ms=5.0, merge_s=0.1):
    r = ndimage.gaussian_filter1d(_rate(times, duration, bin_s), smooth_ms)
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med)) + 1e-9
    above = r > med + k_mad * mad
    d = np.diff(np.r_[0, above.astype(np.int8), 0])
    ev = []
    for s, e in zip(np.flatnonzero(d == 1) * bin_s, np.flatnonzero(d == -1) * bin_s):
        if ev and s - ev[-1][1] < merge_s:
            ev[-1][1] = e
        else:
            ev.append([s, e])
    return ev


def directional_features(elec_ids, times, duration, elec_chamber, min_rate=0.1, pad_s=0.02, max_lag=0.2):
    elec_ids = np.asarray(elec_ids); times = np.asarray(times, float)
    f = dict.fromkeys(DIR_FEATURE_NAMES, 0.0)
    f["log_event_rate"] = np.log10(EPS)
    counts = np.bincount(elec_ids, minlength=len(elec_chamber))[:len(elec_chamber)] if len(times) else \
        np.zeros(len(elec_chamber))
    active = counts / duration >= min_rate
    actA, actB = np.flatnonzero(active & (elec_chamber == 0)), np.flatnonzero(active & (elec_chamber == 1))
    keep = np.isin(elec_ids, np.r_[actA, actB])
    elec_ids, times = elec_ids[keep], times[keep]
    order = np.argsort(times, kind="stable")
    elec_ids, times = elec_ids[order], times[order]
    side = elec_chamber[elec_ids] if len(elec_ids) else np.empty(0, int)
    tA, tB = times[side == 0], times[side == 1]
    f["log_rate_ratio"] = float(np.log10((len(tB) / max(len(actB), 1) + EPS) / (len(tA) / max(len(actA), 1) + EPS)))
    cA, cB = _rate(tA, duration, 0.1), _rate(tB, duration, 0.1)
    if cA.std() > 0 and cB.std() > 0:
        f["sync_AB"] = float(np.corrcoef(cA, cB)[0, 1])
    ev = _events(times, duration)
    if not ev:
        return np.array([f[k] for k in DIR_FEATURE_NAMES])
    f["log_event_rate"] = float(np.log10(max(len(ev) / (duration / 60), EPS)))
    n_both = n_a = n_b = lead_a = 0
    lags = []
    for s, e in ev:
        i0, i1 = np.searchsorted(times, [s - pad_s, e + pad_s])
        ei, ti = elec_ids[i0:i1], times[i0:i1]
        if len(ti) == 0:
            continue
        uniq, first_idx, cnt = np.unique(ei, return_index=True, return_counts=True)   # first spike per electrode
        rec = cnt >= 2
        first_t = ti[first_idx][rec]
        rec_e = uniq[rec]
        inA = np.isin(rec_e, actA); inB = np.isin(rec_e, actB)
        okA = inA.sum() >= max(1, 0.25 * len(actA))
        okB = inB.sum() >= max(1, 0.25 * len(actB))
        if okA and okB:
            n_both += 1
            lead_a += rec_e[np.argmin(first_t)] in actA
            lags.append(np.percentile(first_t[inB], 20) - np.percentile(first_t[inA], 20))
        elif okA:
            n_a += 1
        elif okB:
            n_b += 1
    n = max(n_both + n_a + n_b, 1)
    f["frac_both"], f["frac_A_only"], f["frac_B_only"] = n_both / n, n_a / n, n_b / n
    if n_both:
        f["leader_A"] = lead_a / n_both
        f["dir_index"] = 2 * f["leader_A"] - 1
        lags = np.clip(lags, -max_lag, max_lag)
        f["median_lag"] = float(np.median(lags))
        f["iqr_lag"] = float(np.subtract(*np.percentile(lags, [75, 25])))
    return np.array([f[k] for k in DIR_FEATURE_NAMES])

# ---- end src/neurotwin/directional.py ----

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


def find(pattern):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {INPUT}")
    return hits[0]


def posterior_space(box):
    """The two-chamber posterior lives on (base parameters in the bank's box, p_fwd, p_bwd)."""
    global LO, HI, LOG, Z_LO, Z_HI
    LO = np.array([box[k][0] for k in box] + [0.0, 0.0])
    HI = np.array([box[k][1] for k in box] + [1.0, 1.0])
    LOG = np.array([box[k][2] == "log" for k in box] + [False, False])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)


def sample_base(n, rng, box):
    cols = []
    for lo, hi, scale in box.values():
        u = rng.random(n)
        cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if scale == "log" else lo + u * (hi - lo))
    return np.stack(cols, 1)


def simulate_chips(box, rng, t0):
    """Held-out chips, each recorded for N_WIN windows; returns truth and per-window statistics."""
    cfg = ChipConfig(compartments=2)
    chamber = cfg.chamber_of_electrodes()
    dur = N_WIN * WIN_S
    cap = int(30.0 * len(cfg.electrodes()) * dur)
    thetas, crosses, xs, runaway, times = [], [], [], [], []
    while sum(len(t) for t in thetas) < N_CHIPS:
        tb = time.time()
        theta, cross = sample_base(BATCH, rng, box), rng.random((BATCH, 2))
        out = simulate(theta, cfg, duration_s=dur, warmup_s=10.0, seed=int(rng.integers(2**31 - 1)), cross=cross,
                       max_rate_hz=30.0)
        x = np.full((BATCH, N_WIN, len(FEATURE_NAMES) + len(DIR_FEATURE_NAMES)), np.nan)
        for b, (e, t) in enumerate(out):
            for w in range(N_WIN):
                m = (t >= w * WIN_S) & (t < (w + 1) * WIN_S)
                x[b, w] = np.r_[compute_features(e[m], t[m] - w * WIN_S, WIN_S),
                                directional_features(e[m], t[m] - w * WIN_S, WIN_S, chamber)]
        thetas.append(theta); crosses.append(cross); xs.append(x)
        runaway.append(np.array([len(e) >= cap for e, _ in out]))
        times.append(time.time() - tb)
        print(f"{sum(len(t) for t in thetas)} chips x {dur / 60:.0f} min, last batch {times[-1]:.0f} s, "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if time.time() - t0 + 1.5 * max(times) > TIME_BUDGET_S:
            break
    return (np.concatenate(thetas), np.concatenate(crosses), np.concatenate(xs), np.concatenate(runaway))


def pooled(samples, grid):
    """Pool windows' posteriors over (p_fwd, p_bwd): the product of their kernel density estimates on a grid.

    Returns mean asymmetry, P(p_fwd > p_bwd), and the 90% interval widths of p_fwd and of the asymmetry.
    """
    F, B = np.meshgrid(grid, grid, indexing="ij")
    pts = np.vstack([F.ravel(), B.ravel()])
    logp = np.zeros(F.size)
    for s in samples:                                    # s: [n, 2] draws of (p_fwd, p_bwd) for one window
        try:
            dens = gaussian_kde(s.T)(pts)
        except np.linalg.LinAlgError:                    # degenerate draws: this window adds no information
            dens = np.ones(F.size)
        logp += np.log(dens + 1e-6 * dens.max())          # a floor, so one window cannot veto a region
    p = np.exp(logp - logp.max()); p /= p.sum()
    asym = (F - B).ravel()

    def width(v):
        o = np.argsort(v); c = np.cumsum(p[o])
        return v[o][np.searchsorted(c, 0.95)] - v[o][np.searchsorted(c, 0.05)]

    return (p * asym).sum(), p[asym > 0].sum() + 0.5 * p[asym == 0].sum(), width(F.ravel()), width(asym)


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    box = {k: tuple(v) for k, v in json.load(open(find("bank2c_shard*.json")))["base_box"].items()}
    posterior = torch.load(find("posterior2c.pt"), weights_only=False)
    rng = np.random.default_rng(SEED)
    theta, cross, X, runaway = simulate_chips(box, rng, t0)
    np.savez_compressed(os.path.join(WORK, "design_length_sims.npz"), theta=theta, cross=cross, x=X,
                        runaway=runaway, base_names=np.array(list(box)),
                        feature_names=np.array(FEATURE_NAMES + DIR_FEATURE_NAMES))
    print(f"simulated {len(theta)} chips in {(time.time() - t0) / 60:.1f} min; {runaway.sum()} runaway", flush=True)

    posterior_space(box)
    ok_win = np.isfinite(X).all(2) & ~runaway[:, None]
    flat = X.reshape(-1, X.shape[2])
    idx = np.flatnonzero(ok_win.ravel())
    torch.manual_seed(0)
    s, leak = sample_many(posterior, flat[idx], n=N_POST)                 # [n, m, 12]
    draws = np.full((len(flat), N_POST, 2), np.nan)
    draws[idx] = np.transpose(s[:, :, -2:], (1, 0, 2))
    draws = draws.reshape(len(theta), N_WIN, N_POST, 2)
    print(f"posterior draws for {len(idx)} windows; median leakage {np.median(leak):.2f}", flush=True)

    grid = np.linspace(0.0, 1.0, GRID)
    rows = []
    for c in range(len(theta)):
        for k in LENGTHS:
            wins = [w for w in range(k) if ok_win[c, w]]
            if len(wins) < k:                    # a chip enters a length only if all its windows are usable
                continue
            m, p_fwd_leads, w_f, w_a = pooled([draws[c, w] for w in wins], grid)
            avg = np.mean([draws[c, w, :, 0].mean() - draws[c, w, :, 1].mean() for w in wins])
            rows.append({"chip": c, "minutes": k * WIN_S / 60, "p_fwd": cross[c, 0], "p_bwd": cross[c, 1],
                         "asym_true": cross[c, 0] - cross[c, 1], "asym_pooled": m, "p_forward_leads": p_fwd_leads,
                         "width90_p_fwd": w_f, "width90_asym": w_a, "asym_window_mean": avg})
    d = pd.DataFrame(rows)
    d["correct"] = np.sign(d.asym_pooled) == np.sign(d.asym_true)
    d["confidence"] = np.maximum(d.p_forward_leads, 1 - d.p_forward_leads)
    d["strong"] = d.asym_true.abs() > 0.3
    d.to_csv(os.path.join(WORK, "design_length.csv"), index=False)

    summ = []
    for mins, g in d.groupby("minutes"):
        st, dec = g[g.strong], g[g.confidence >= 0.9]
        r2 = 1 - ((g.asym_pooled - g.asym_true) ** 2).sum() / ((g.asym_true - g.asym_true.mean()) ** 2).sum()
        summ.append({"minutes": mins, "chips": len(g), "direction_correct": g.correct.mean(),
                     "direction_correct_strong": st.correct.mean(), "strong_chips": len(st),
                     "direction_correct_window_mean": (np.sign(g.asym_window_mean) == np.sign(g.asym_true)).mean(),
                     "asym_r2": r2, "median_width90_p_fwd": g.width90_p_fwd.median(),
                     "median_width90_asym": g.width90_asym.median(), "decisive_share": len(dec) / len(g),
                     "decisive_correct": dec.correct.mean() if len(dec) else np.nan, "decisive_chips": len(dec),
                     "mean_confidence": g.confidence.mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(WORK, "design_length_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n" + summ.round(3).to_string(index=False), flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for col, lab, n_col in (("direction_correct", "all chips", "chips"),
                            ("direction_correct_strong", "engineered asymmetry > 0.3", "strong_chips")):
        ok = summ[n_col] > 0
        ci = [binomtest(int(round(a * n)), int(n)).proportion_ci() for a, n in zip(summ[col][ok], summ[n_col][ok])]
        ax[0].errorbar(summ.minutes[ok], summ[col][ok], yerr=[summ[col][ok] - [c.low for c in ci],
                                                               [c.high for c in ci] - summ[col][ok]],
                       marker="o", capsize=3, label=lab)
    ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(summ.minutes); ax[0].set_xticklabels([f"{m:g}" for m in summ.minutes])
    ax[0].set_xlabel("recording length (min)"); ax[0].set_ylabel("direction read correctly"); ax[0].legend(frameon=False)
    ax[0].set_title("Reading the wiring direction")
    ax[1].plot(summ.minutes, summ.median_width90_asym, "o-", label="asymmetry p_fwd - p_bwd")
    ax[1].plot(summ.minutes, summ.median_width90_p_fwd, "s-", label="p_fwd")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(summ.minutes); ax[1].set_xticklabels([f"{m:g}" for m in summ.minutes])
    ax[1].set_xlabel("recording length (min)"); ax[1].set_ylabel("median width of the 90% interval")
    ax[1].legend(frameon=False); ax[1].set_title("Uncertainty shrinks with recording length")
    bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    for mins, g in d.groupby("minutes"):
        k = np.digitize(g.confidence, bins) - 1
        acc = g.groupby(k).correct.mean()
        conf = g.groupby(k).confidence.mean()
        ax[2].plot(conf.values, acc.values, "o-", label=f"{mins:g} min")
    ax[2].plot([0.5, 1], [0.5, 1], "k--", lw=0.8)
    ax[2].set_xlabel("stated probability of the direction"); ax[2].set_ylabel("observed share correct")
    ax[2].set_title("Is the stated confidence honest?"); ax[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "recording_length.png"), dpi=110); plt.close(fig)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
