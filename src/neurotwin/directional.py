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

from .features import EPS

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
