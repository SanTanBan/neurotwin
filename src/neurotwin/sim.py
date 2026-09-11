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
