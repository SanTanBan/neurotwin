"""Real MEA recordings: the Wagenaar, Pine & Potter (2006) developing cortical culture dataset.

Files are named <plating>-<culture>-<div>.spk.txt.bz2 and hold two columns (time in s, channel 0..59).
Channel 60 marks stimuli and is dropped. Each recording is cut into consecutive fixed-length windows so its
features match the simulated window length used to train the twin.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, compute_features

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
