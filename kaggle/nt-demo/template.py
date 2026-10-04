"""NeuroTwin demo: fit a digital twin to one real MEA recording and run an in-silico experiment.

This is the short, reviewer-facing notebook. It takes a real recording of a developing rat cortical culture
(Wagenaar, Pine & Potter 2006), computes its electrophysiological signature, fits the twin with the
pre-trained amortised posterior in about a second, states plainly whether the fit can be trusted, checks that
the fitted twin reproduces the recording, and then asks the twin a question no experiment answered:
what would this culture do with half the inhibition?
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
WIN_S = 300.0
RECORDING = "2-2-20"          # culture 2-2 at 20 days in vitro
N_POST = 500

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py

# @@include src/neurotwin/data.py

# @@include src/neurotwin/infer.py

# @@include src/neurotwin/ppc.py


def find(pattern):
    hits = glob.glob(os.path.join(INPUT, "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {INPUT}")
    return hits[0]


def apply_box(box):
    global LO, HI, LOG, Z_LO, Z_HI
    LO = np.array([box[k][0] for k in PARAM_NAMES], float)
    HI = np.array([box[k][1] for k in PARAM_NAMES], float)
    LOG = np.array([box[k][2] == "log" for k in PARAM_NAMES])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)


def raster(ax, e, t, t0, t1, title, color="k"):
    m = (t >= t0) & (t < t1)
    ax.scatter(t[m] - t0, e[m], s=0.4, c=color)
    ax.set_title(title, fontsize=10); ax.set_xlabel("time (s)"); ax.set_ylabel("electrode")


def main():
    t0 = time.time()
    box = json.load(open(find("focus_box.json")))
    apply_box({k: tuple(v) for k, v in box.items()})
    posterior = torch.load(find("posterior.pt"), weights_only=False)
    assets = np.load(find("twin_assets.npz"))

    ch, t = load_spk(find(f"{RECORDING}.spk.txt.bz2"))
    x = compute_features(ch[t < WIN_S], t[t < WIN_S], WIN_S)
    print(f"recording {RECORDING}: {len(t):,} spikes, {len(np.unique(ch))} active electrodes, {t.max():.0f} s")
    print(pd.DataFrame({"statistic": FEATURE_NAMES, "value": np.round(x, 3)}).to_string(index=False))

    t1 = time.time()
    keep = assets["keep"] if "keep" in assets.files else np.arange(len(x))   # statistics used for inference
    samples, leak = sample_many(posterior, x[keep][None], n=N_POST)
    s = samples[:, 0]
    print(f"\ntwin fitted in {time.time() - t1:.2f} s")
    q = np.percentile(s, [5, 50, 95], axis=0)
    print(pd.DataFrame({"parameter": PARAM_NAMES, "median": np.round(q[1], 3),
                        "90% interval": [f"{a:.3g} - {b:.3g}" for a, b in zip(q[0], q[2])]}).to_string(index=False))

    z = (x[keep] - assets["mu"]) / (assets["sd"] + 1e-9)
    d = np.sort(np.linalg.norm(assets["x"] - z, axis=1))[:10].mean()
    trusted = d <= float(assets["threshold"]) and leak[0] < 0.2
    print(f"\nreliability: distance to simulated data {d:.2f} (threshold {float(assets['threshold']):.2f}), "
          f"posterior mass outside the parameter ranges {leak[0]:.0%} -> "
          f"{'fit is inside the model’s range' if trusted else 'FIT IS EXTRAPOLATION, do not trust it'}")

    theta = np.median(s, 0)
    modified = theta.copy()
    modified[PARAM_NAMES.index("g_ie")] *= 0.5                    # the in-silico experiment
    sims = simulate(np.stack([theta, modified]), ChipConfig(), duration_s=WIN_S, warmup_s=10.0, seed=7)
    x_twin = compute_features(*sims[0], WIN_S)
    x_half = compute_features(*sims[1], WIN_S)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    raster(ax[0], ch, t, 600, 660, f"real culture {RECORDING} (60 s)")
    raster(ax[1], *sims[0], 0, 60, "its digital twin", "C3")
    raster(ax[2], *sims[1], 0, 60, "twin with inhibition halved", "C0")
    fig.tight_layout(); fig.savefig(os.path.join(WORK, "demo_rasters.png"), dpi=110); plt.close(fig)

    cmp = pd.DataFrame({"statistic": FEATURE_NAMES, "real": np.round(x, 2), "twin": np.round(x_twin, 2),
                        "twin, half inhibition": np.round(x_half, 2)})
    cmp.to_csv(os.path.join(WORK, "demo_comparison.csv"), index=False)
    print("\n" + cmp.to_string(index=False))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
