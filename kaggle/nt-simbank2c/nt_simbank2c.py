"""NeuroTwin two-chamber (axon-diode) simulation bank (one shard).

Virtual two-chamber chips: somata in two chambers separated by a microchannel barrier, axons crossing with
forward/backward probabilities (p_fwd: A->B, p_bwd: B->A) and a 5-ms delay. Base parameters are drawn from a
box covering realistic bursting cultures (found by prior-predictive exploration against real recordings);
crossing probabilities are uniform on [0, 1]. For every culture both the global MEA signature and the
directional signature are stored, so a posterior over (base parameters, p_fwd, p_bwd) can be trained.
"""
import json
import os
import subprocess
import sys
import time

try:
    import numba  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"], check=True)

import numpy as np

SHARD = 0
DURATION_S = 300.0
WARMUP_S = 10.0
BATCH = 64
TIME_BUDGET_S = 11.0 * 3600
MAX_SIMS = 100_000
OUT = f"/kaggle/working/bank2c_shard{SHARD}.npz" if os.path.isdir("/kaggle/working") else f"bank2c_shard{SHARD}.npz"

BASE_BOX = {
    "j_e": (6.0, 20.0, "log"), "g_ie": (0.3, 1.5, "log"), "u_std": (0.05, 0.4, "log"),
    "tau_rec": (0.5, 5.0, "log"), "b_adapt": (0.0, 2.0, "lin"), "tau_adapt": (1.0, 20.0, "log"),
    "mu": (3.0, 10.0, "lin"), "mu_sd": (0.0, 4.0, "lin"), "sigma": (2.0, 5.0, "lin"),
    "lambda_um": (300.0, 2000.0, "log"),
}

# ---- begin src/neurotwin/sim.py ----
"""Spiking-network simulator of a cultured neuronal network grown on a microelectrode-array (MEA) chip.

One simulation = one virtual culture: N leaky integrate-and-fire neurons (80% excitatory, 20% inhibitory)
scattered over the chip, distance-dependent random connectivity, Tsodyks-Markram short-term synaptic
depression (terminates network bursts), spike-frequency adaptation and a noisy background drive.
Virtual electrodes pick up spikes of neurons within `pickup_um` of their centre, giving multi-unit spike
trains in the same (time, electrode) format as real MEA recordings.

Optionally the chip has two chambers separated by a microchannel barrier (axon-diode design): axons cross
with separate forward/backward probabilities and a conduction delay.

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
    ring = np.zeros((xdel + 1, N))      # delayed input from channel-crossing synapses
    max_ev = out_e.shape[0]
    n_ev = 0
    for t in range(n_steps):
        tmp = inp
        inp = nxt
        nxt = tmp
        nxt[:] = 0.0
        if xdel > 0:
            slot = t % (xdel + 1)
            for i in range(N):
                inp[i] += ring[slot, i]
                ring[slot, i] = 0.0
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
                        if is_x[k]:
                            ring[(t + xdel) % (xdel + 1), idx[k]] += w[k] * eff
                        else:
                            nxt[idx[k]] += w[k] * eff
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
                            fixed[0], fixed[1], fixed[2], fixed[3], np.int64(fixed[4]), seeds[b],
                            out_e[b], out_t[b])


def simulate(theta, cfg=ChipConfig(), duration_s=120.0, warmup_s=5.0, seed=0, max_rate_hz=60.0, cross=None):
    """Simulate a batch of cultures (theta: [B, n_params]; cross: optional [B, 2] crossing probabilities).

    Returns a list of (electrode_ids, spike_times_s) tuples, one per culture, sorted by time. Recording
    stops early for runaway cultures whose electrode event count exceeds `max_rate_hz` per electrode.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
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


def sample_base(n, rng):
    cols = []
    for name in PARAM_NAMES:
        lo, hi, scale = BASE_BOX[name]
        u = rng.random(n)
        cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if scale == "log" else lo + u * (hi - lo))
    return np.stack(cols, 1)


def main():
    t0 = time.time()
    rng = np.random.default_rng(5000 + SHARD)
    cfg = ChipConfig(compartments=2)
    chamber = cfg.chamber_of_electrodes()
    thetas, crosses, xg, xd, times = [], [], [], [], []
    while len(thetas) * BATCH < MAX_SIMS:
        tb = time.time()
        theta, cross = sample_base(BATCH, rng), rng.random((BATCH, 2))
        out = simulate(theta, cfg, duration_s=DURATION_S, warmup_s=WARMUP_S, seed=int(rng.integers(2**31 - 1)),
                       cross=cross)
        xg.append(np.array([compute_features(e, t, DURATION_S) for e, t in out]))
        xd.append(np.array([directional_features(e, t, DURATION_S, chamber) for e, t in out]))
        thetas.append(theta); crosses.append(cross)
        times.append(time.time() - tb)
        np.savez_compressed(OUT, theta=np.concatenate(thetas), cross=np.concatenate(crosses),
                            x_global=np.concatenate(xg), x_dir=np.concatenate(xd),
                            param_names=np.array(PARAM_NAMES), feature_names=np.array(FEATURE_NAMES),
                            dir_feature_names=np.array(DIR_FEATURE_NAMES))
        print(f"[2c shard {SHARD}] {len(thetas) * BATCH} sims, last batch {times[-1]:.0f}s, "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if time.time() - t0 + 1.5 * max(times) > TIME_BUDGET_S:
            break
    meta = {"shard": SHARD, "n_sims": len(thetas) * BATCH, "duration_s": DURATION_S, "base_box": BASE_BOX,
            "chip": vars(cfg), "elapsed_min": (time.time() - t0) / 60}
    with open(OUT.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("done", meta)


if __name__ == "__main__":
    main()
