"""NeuroTwin real-data signatures: developing rat cortical cultures on 60-electrode MEAs.

Downloads the daily spontaneous-activity recordings of dense cultures from Wagenaar, Pine & Potter (2006),
"An extremely rich repertoire of bursting patterns during the development of cortical cultures",
BMC Neuroscience 7:11 (data free for research use with citation of that article). Each recording is cut into
300-s windows and the NeuroTwin electrophysiological signature is computed for every window.
Outputs: wagenaar_dense_features.csv and the raw files of one example culture for demos.
"""
import os
import re
import shutil
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

LIST_URL = "https://potterlab.bme.gatech.edu/development-data/html/wget/daily.spont.dense.text.0.0.0.list"
WIN_S = 300.0
N_WIN = 6
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RAW = os.path.join(WORK, "raw_tmp")
KEEP_CULTURE = "2-2-"   # raw files kept in the output for demos

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

# ---- begin src/neurotwin/data.py ----
"""Real MEA recordings: the Wagenaar, Pine & Potter (2006) developing cortical culture dataset.

Files are named <plating>-<culture>-<div>.spk.txt.bz2 and hold two columns (time in s, channel 0..59).
Channel 60 marks stimuli and is dropped. Each recording is cut into consecutive fixed-length windows so its
features match the simulated window length used to train the twin.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd



NAME = re.compile(r"(\d+)-(\d+)-(\d+)\.spk\.txt\.bz2$")


def parse_name(path):
    m = NAME.search(str(path))
    return {"plating": int(m.group(1)), "culture": int(m.group(2)), "div": int(m.group(3))}


def load_spk(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["t", "ch"], compression="bz2")
    df = df[df.ch < 60]
    return df.ch.to_numpy(np.int64), df.t.to_numpy(np.float64)


def windowed_features(ch, t, win_s=300.0, n_windows=None):
    """Features of consecutive windows [k*win_s, (k+1)*win_s); returns array [n_windows, n_features]."""
    n_avail = int(t.max() // win_s) if len(t) else 0
    n = n_avail if n_windows is None else min(n_windows, n_avail)
    rows = []
    for k in range(n):
        m = (t >= k * win_s) & (t < (k + 1) * win_s)
        rows.append(compute_features(ch[m], t[m] - k * win_s, win_s))
    return np.array(rows).reshape(-1, len(FEATURE_NAMES))


def build_table(folder, win_s=300.0, n_windows=6, pattern="*.spk.txt.bz2"):
    """One row per (recording, window): metadata columns + features."""
    rows = []
    for p in sorted(Path(folder).glob(pattern)):
        meta = parse_name(p)
        ch, t = load_spk(p)
        X = windowed_features(ch, t, win_s, n_windows)
        for k, x in enumerate(X):
            rows.append({**meta, "file": p.name, "window": k, "duration_s": float(t.max()) if len(t) else 0.0,
                         **dict(zip(FEATURE_NAMES, x))})
    return pd.DataFrame(rows)

# ---- end src/neurotwin/data.py ----


def download_all():
    os.makedirs(RAW, exist_ok=True)
    txt = urllib.request.urlopen(LIST_URL, timeout=120).read().decode("utf-8", "replace")
    urls = [urllib.parse.urljoin(LIST_URL, m) for m in re.findall(r"(\S+\.spk\.txt\.bz2)", txt)]
    t0, fails = time.time(), []
    for i, u in enumerate(urls):
        fn = os.path.join(RAW, os.path.basename(u))
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            continue
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(u.replace("http://", "https://"), timeout=180).read()
                with open(fn, "wb") as f:
                    f.write(data)
                break
            except Exception as e:  # retry transient network errors
                if attempt == 2:
                    fails.append((os.path.basename(u), repr(e)[:100]))
                time.sleep(5)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(urls)} downloaded, {time.time() - t0:.0f}s", flush=True)
    print(f"download done: {len(urls)} listed, {len(fails)} failed", fails[:5], flush=True)


def main():
    download_all()
    t0 = time.time()
    df = build_table(RAW, win_s=WIN_S, n_windows=N_WIN)
    df.to_csv(os.path.join(WORK, "wagenaar_dense_features.csv"), index=False)
    print(f"features: {len(df)} windows from {df.file.nunique()} recordings in {time.time() - t0:.0f}s")
    keep = os.path.join(WORK, "raw_example")
    os.makedirs(keep, exist_ok=True)
    for fn in os.listdir(RAW):
        if fn.startswith(KEEP_CULTURE):
            shutil.copy(os.path.join(RAW, fn), keep)
    shutil.rmtree(RAW)
    print(df.groupby("div")[["log_rate", "log_nb_rate", "sync_corr", "log_fano_pop"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
