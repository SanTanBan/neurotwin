"""NeuroTwin round-3 training & validation (model v4, focused prior).

1. Train the amortised posterior on the round-2 bank. The prior used for inference is read from the bank's
   own metadata, so it always matches how the bank was sampled.
2. Validate on held-out simulations: parameter recovery, simulation-based calibration, TARP coverage.
3. Apply to every 300-s window of the real recordings; report out-of-support and posterior leakage.
4. Posterior predictive checks against a *mismatched* twin (parameters from another recording's posterior),
   which is the baseline a useful twin has to beat, with per-feature detail.
5. Developmental trajectories from the windows that pass the reliability checks.
Outputs in /kaggle/working: posterior.pt, calibration.json, real_posteriors.csv, ppc.csv, ppc_features.csv,
trajectories.csv, figures/*.png
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
N_CAL = 500
N_PPC_REC = 16
N_PPC_DRAWS = 24
WIN_S = 300.0
# Statistics the model family cannot reproduce (documented misspecification) are excluded from inference
# but still checked in the posterior predictive checks.
EXCLUDE = ["nb_rise_frac"]

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


def apply_box(box):
    """Use the bank's sampling box as the inference prior (round 2 samples a focused region)."""
    global LO, HI, LOG, Z_LO, Z_HI
    LO = np.array([box[k][0] for k in PARAM_NAMES], float)
    HI = np.array([box[k][1] for k in PARAM_NAMES], float)
    LOG = np.array([box[k][2] == "log" for k in PARAM_NAMES])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)
    print("inference prior:", {k: (round(LO[i], 3), round(HI[i], 3), "log" if LOG[i] else "lin")
                               for i, k in enumerate(PARAM_NAMES)}, flush=True)


def load_bank():
    files = sorted(glob.glob(os.path.join(INPUT, "**", "bank3_shard*.npz"), recursive=True))
    meta = sorted(glob.glob(os.path.join(INPUT, "**", "bank3_shard*.json"), recursive=True))
    box = json.load(open(meta[0]))["focus_box"]
    apply_box({k: tuple(v) for k, v in box.items()})
    json.dump(box, open(os.path.join(WORK, "focus_box.json"), "w"), indent=1)   # the demo needs it too
    theta = np.concatenate([np.load(f)["theta"] for f in files])
    x = np.concatenate([np.load(f)["x"] for f in files])
    ok = np.isfinite(x).all(1) & np.all((theta >= LO - 1e-9) & (theta <= HI + 1e-9), axis=1)
    print(f"bank: {len(theta)} simulations from {len(files)} shards, {ok.sum()} usable", flush=True)
    return theta[ok], x[ok]


def out_of_support_scores(x_sim, x_query, k=10):
    """Mean distance to the k nearest simulations in z-scored feature space."""
    mu, sd = x_sim.mean(0), x_sim.std(0) + 1e-9
    a = torch.as_tensor((x_sim - mu) / sd, dtype=torch.float32)
    q = torch.as_tensor((x_query - mu) / sd, dtype=torch.float32)
    return torch.cat([torch.cdist(c, a).topk(k, largest=False).values.mean(1) for c in torch.split(q, 256)]).numpy()


def fig_recovery(rec, theta_te, post_means):
    n = len(PARAM_NAMES)
    cols = (n + 1) // 2
    fig, ax = plt.subplots(2, cols, figsize=(3.2 * cols, 6.6))
    z = to_unbounded(theta_te)
    for i, a in enumerate(ax.flat[:n]):
        a.scatter(z[:, i], post_means[:, i], s=3, alpha=0.4)
        a.plot([Z_LO[i], Z_HI[i]], [Z_LO[i], Z_HI[i]], "k--", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  R2={rec['r2'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Round 2: parameter recovery on held-out simulations")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "recovery.png"), dpi=110); plt.close(fig)


def fig_sbc(cal):
    ranks = cal["ranks"]; n = ranks.shape[1]; cols = (n + 1) // 2
    fig, ax = plt.subplots(2, cols, figsize=(3.2 * cols, 6.2))
    for i, a in enumerate(ax.flat[:n]):
        a.hist(ranks[:, i], bins=20, color="C0", alpha=0.8)
        a.axhline(len(ranks) / 20, color="k", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  KS p={cal['sbc_ks_pvals'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Round 2: simulation-based calibration (flat = calibrated)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sbc.png"), dpi=110); plt.close(fig)
    fig, a = plt.subplots(figsize=(4, 4))
    a.plot(cal["tarp_alpha"], cal["tarp_ecp"], lw=2); a.plot([0, 1], [0, 1], "k--")
    a.set_xlabel("credibility level"); a.set_ylabel("expected coverage")
    a.set_title(f"TARP coverage (ATC={cal['tarp_atc']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tarp.png"), dpi=110); plt.close(fig)


def fig_trajectories(agg):
    n = len(PARAM_NAMES); cols = (n + 1) // 2
    fig, ax = plt.subplots(2, cols, figsize=(3.4 * cols, 6.8))
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
    fig.suptitle("Inferred twin parameters across development (thin: cultures; bold: median)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trajectories.png"), dpi=110); plt.close(fig)


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    theta, x_all = load_bank()
    keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in EXCLUDE]
    x = x_all[:, keep]
    print(f"inference uses {len(keep)} of {len(FEATURE_NAMES)} statistics; excluded: {EXCLUDE}", flush=True)
    perm = rng.permutation(len(theta))
    te, tr = perm[:N_TEST], perm[N_TEST:]

    posterior, trainer = train_npe(theta[tr], x[tr], density_estimator="nsf",
                                   training_batch_size=512, stop_after_epochs=30)
    torch.save(posterior, os.path.join(WORK, "posterior.pt"))
    print(f"trained on {len(tr)} simulations in {(time.time() - t0) / 60:.1f} min", flush=True)

    rec = recovery_metrics(posterior, theta[te[:N_CAL]], x[te[:N_CAL]])
    cal = calibration(posterior, theta[te[:N_CAL]], x[te[:N_CAL]])
    fig_recovery(rec, theta[te[:N_CAL]], rec["post_mean_z"])
    fig_sbc(cal)
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**rec, **cal}.items()
               if k not in ("ranks", "tarp_ecp", "tarp_alpha", "post_mean_z")},
              open(os.path.join(WORK, "calibration.json"), "w"), indent=1)
    print(pd.DataFrame({k: rec[k] for k in ("param", "r2", "nrmse", "coverage_50", "coverage_90", "contraction")})
          .round(3).to_string(index=False), flush=True)
    print(f"TARP ATC={cal['tarp_atc']:.3f} KS p={cal['tarp_ks_p']:.3f}", flush=True)

    real_files = glob.glob(os.path.join(INPUT, "**", "wagenaar_dense_features.csv"), recursive=True)
    if not real_files:
        print("no real feature table attached; done"); return
    df = pd.read_csv(real_files[0])
    if "file" not in df:
        df["file"] = df[["plating", "culture", "div"]].astype(str).agg("-".join, axis=1)
    xr_all = df[FEATURE_NAMES].to_numpy(np.float32)
    xr = xr_all[:, keep]
    thr = np.quantile(out_of_support_scores(x[tr], x[te]), 0.99)
    sub = rng.choice(len(tr), size=min(5000, len(tr)), replace=False)      # assets for the interactive app
    np.savez(os.path.join(WORK, "twin_assets.npz"), mu=x[tr].mean(0), sd=x[tr].std(0), threshold=thr,
             x=((x[tr][sub] - x[tr].mean(0)) / (x[tr].std(0) + 1e-9)).astype(np.float32),
             feature_names=np.array(FEATURE_NAMES), param_names=np.array(PARAM_NAMES), keep=np.array(keep))
    df["oos_score"] = out_of_support_scores(x[tr], xr)
    thr16 = np.quantile(out_of_support_scores(x_all[tr], x_all[te]), 0.99)
    df["out_of_support_all16"] = out_of_support_scores(x_all[tr], xr_all) > thr16
    df["out_of_support"] = df["oos_score"] > thr
    samples, df["leakage"] = sample_many(posterior, xr, n=500)
    for i, p in enumerate(PARAM_NAMES):
        df[f"{p}_med"] = np.median(samples[:, :, i], 0)
        df[f"{p}_q05"] = np.quantile(samples[:, :, i], 0.05, 0)
        df[f"{p}_q95"] = np.quantile(samples[:, :, i], 0.95, 0)
    df.to_csv(os.path.join(WORK, "real_posteriors.csv"), index=False)
    good = df[~df.out_of_support & (df.leakage < 0.2)]
    print(f"real windows {len(df)}: out of support {df.out_of_support.mean():.1%}, "
          f"mean leakage {df.leakage.mean():.2f}, reliable windows {len(good)} ({len(good) / len(df):.1%}); "
          f"out of support using all 16 statistics {df.out_of_support_all16.mean():.1%}",
          flush=True)
    fig, a = plt.subplots(1, 2, figsize=(11, 4))
    a[0].hist(df.leakage, bins=40); a[0].set_xlabel("posterior mass outside the prior box"); a[0].set_ylabel("windows")
    a[1].scatter(df["div"], df.oos_score, s=4, alpha=0.3); a[1].axhline(thr, color="r", ls="--")
    a[1].set_xlabel("days in vitro"); a[1].set_ylabel("distance to nearest simulations")
    fig.suptitle("Reliability of the fit on real recordings")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability.png"), dpi=110); plt.close(fig)

    # posterior predictive checks: own twin vs a twin fitted to a different recording
    pool = good if len(good) >= N_PPC_REC else df
    picks = pool.drop_duplicates("file").sort_values("div")
    picks = picks.iloc[np.linspace(0, len(picks) - 1, min(N_PPC_REC, len(picks))).astype(int)]
    rows, feat_rows = [], []
    for n_i, (_, r) in enumerate(picks.iterrows()):
        xo = r[FEATURE_NAMES].to_numpy(np.float32)
        other = picks.iloc[(n_i + len(picks) // 2) % len(picks)]
        th_own = sample(posterior, xo[keep], n=N_PPC_DRAWS)
        th_other = sample(posterior, other[FEATURE_NAMES].to_numpy(np.float32)[keep], n=N_PPC_DRAWS)
        xp_own = predictive(th_own, duration_s=WIN_S, seed=int(rng.integers(2**31 - 1)))
        xp_other = predictive(th_other, duration_s=WIN_S, seed=int(rng.integers(2**31 - 1)))
        s_own, s_other = summary(xo, xp_own), summary(xo, xp_other)
        rows.append({"file": r["file"], "div": r["div"], "own_inside90": s_own["inside"].mean(),
                     "mismatched_inside90": s_other["inside"].mean(),
                     "own_abs_z": np.nanmedian(np.abs(s_own["z"])),
                     "mismatched_abs_z": np.nanmedian(np.abs(s_other["z"]))})
        for j, f in enumerate(FEATURE_NAMES):
            feat_rows.append({"file": r["file"], "div": r["div"], "feature": f, "observed": xo[j],
                              "own_median": s_own["median"][j], "own_z": s_own["z"][j],
                              "mismatched_z": s_other["z"][j], "inside90": bool(s_own["inside"][j])})
        print(rows[-1], flush=True)
    ppc = pd.DataFrame(rows); ppc.to_csv(os.path.join(WORK, "ppc.csv"), index=False)
    pd.DataFrame(feat_rows).to_csv(os.path.join(WORK, "ppc_features.csv"), index=False)
    print(f"\nPPC: own twin covers {ppc.own_inside90.mean():.2f} of statistics vs {ppc.mismatched_inside90.mean():.2f} "
          f"for a mismatched twin; median |z| {ppc.own_abs_z.median():.2f} vs {ppc.mismatched_abs_z.median():.2f}",
          flush=True)

    if len(good) > 50:
        agg = good.groupby(["plating", "culture", "div"])[[f"{p}_med" for p in PARAM_NAMES]].median().reset_index()
        agg.to_csv(os.path.join(WORK, "trajectories.csv"), index=False)
        fig_trajectories(agg)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
