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
