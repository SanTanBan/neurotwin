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
