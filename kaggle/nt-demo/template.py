"""NeuroTwin demo: fit a digital twin to one real MEA recording and run an in-silico experiment.

This is the short, reviewer-facing notebook. It takes a real recording of a developing rat cortical culture
(Wagenaar, Pine & Potter 2006), computes its electrophysiological signature, fits the twin with the
pre-trained amortised posterior in about a second, states plainly whether the fit can be trusted, checks that
the fitted twin reproduces the recording, and then asks the twin a question no experiment answered:
what would this culture do with half the inhibition? The question is put to 16 posterior draws and to the posterior
median on 8 random networks, which separates what the recording leaves uncertain from what is
intrinsically random.
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

ON_KAGGLE = os.name != "nt" and os.path.isdir("/kaggle/working")   # a local D:\kaggle folder must not count
WORK = "/kaggle/working" if ON_KAGGLE else "."
INPUT = "/kaggle/input" if ON_KAGGLE else "."
WIN_S = 300.0
RECORDING = "2-2-20"          # culture 2-2 at 20 days in vitro
N_POST = 500
N_DRAWS = 16                  # posterior draws re-simulated for the predictive check and the experiment
N_NETS = 8                    # random networks for the posterior-median twin

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
    """The posterior's parameters are those in its box (an older model version may have fewer)."""
    global LO, HI, LOG, Z_LO, Z_HI, POST_NAMES
    POST_NAMES = [k for k in PARAM_NAMES if k in box]
    LO = np.array([box[k][0] for k in POST_NAMES], float)
    HI = np.array([box[k][1] for k in POST_NAMES], float)
    LOG = np.array([box[k][2] == "log" for k in POST_NAMES])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)


def raster(ax, ax_rate, e, t, t0, t1, title, color="k"):
    """Raster plus population rate (50-ms bins), so network bursts stand out."""
    m = (t >= t0) & (t < t1)
    ax.scatter(t[m] - t0, e[m], s=0.3, c=color)
    ax.set_title(title, fontsize=11); ax.set_ylabel("electrode"); ax.tick_params(labelbottom=False)
    h, edges = np.histogram(t[m] - t0, bins=np.arange(0, t1 - t0 + 1e-9, 0.05))
    ax_rate.plot(edges[:-1], h / 0.05, color=color, lw=0.7)
    ax_rate.set_xlabel("time (s)"); ax_rate.set_ylabel("spikes/s")


def paired(ax, x0, before, after):
    """One line per twin: as fitted (red, at x0) and after the in-silico change (blue, at x0 + 1)."""
    for b, a in zip(before, after):
        ax.plot([x0, x0 + 1], [b, a], color="0.75", lw=0.8)
    ax.scatter(np.full(len(before), x0), before, color="C3", s=14, zorder=3)
    ax.scatter(np.full(len(after), x0 + 1.0), after, color="C0", s=14, zorder=3)


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

    torch.manual_seed(0)                                         # reproducible posterior draws
    t1 = time.time()
    keep = assets["keep"] if "keep" in assets.files else np.arange(len(x))   # statistics used for inference
    samples, leak = sample_many(posterior, x[keep][None], n=N_POST)
    s = samples[:, 0]
    print(f"\ntwin fitted in {time.time() - t1:.2f} s")
    q = np.percentile(s, [5, 50, 95], axis=0)
    print(pd.DataFrame({"parameter": POST_NAMES, "median": np.round(q[1], 3),
                        "90% interval": [f"{a:.3g} - {b:.3g}" for a, b in zip(q[0], q[2])]}).to_string(index=False))

    z = (x[keep] - assets["mu"]) / (assets["sd"] + 1e-9)
    d = np.sort(np.linalg.norm(assets["x"] - z, axis=1))[:10].mean()
    trusted = d <= float(assets["threshold"]) and leak[0] < 0.2
    print(f"\nreliability: distance to simulated data {d:.2f} (threshold {float(assets['threshold']):.2f}), "
          f"posterior mass outside the parameter ranges {leak[0]:.0%} -> "
          f"{'fit is inside the model’s range' if trusted else 'FIT IS EXTRAPOLATION, do not trust it'}")

    # In-silico experiment: halve inhibition. Every modified run is paired with an unmodified run on the same
    # network and noise (same seed), so a pair differs only by the intervention. Two sources of spread are kept
    # apart: parameter uncertainty (N_DRAWS posterior draws, each on its own network) and network randomness
    # (the posterior median on N_NETS networks).
    gi = POST_NAMES.index("g_ie")
    pick = np.random.default_rng(0).choice(len(s), N_DRAWS, replace=False)
    base = np.vstack([s[pick], np.repeat(np.median(s, 0)[None], N_NETS, axis=0)])
    half = base.copy()
    half[:, gi] *= 0.5
    t2 = time.time()
    sims_b = simulate(base, ChipConfig(), duration_s=WIN_S, warmup_s=10.0, seed=7)
    sims_h = simulate(half, ChipConfig(), duration_s=WIN_S, warmup_s=10.0, seed=7)
    print(f"\n{2 * len(base)} simulations of {WIN_S:.0f} s in {time.time() - t2:.0f} s")
    xb = np.array([compute_features(*o, WIN_S) for o in sims_b])
    xh = np.array([compute_features(*o, WIN_S) for o in sims_h])
    capped = sum(len(e) >= int(60.0 * 60 * WIN_S) for e, _ in sims_b + sims_h)
    if capped:
        print(f"warning: {capped} runaway simulations hit the event cap")
    dr, nt = slice(0, N_DRAWS), slice(N_DRAWS, None)       # posterior draws | median twin on N_NETS networks
    eff = xh - xb

    lo, hi = np.percentile(xb[dr], [5, 95], axis=0)
    inside = (x >= lo) & (x <= hi)
    print(f"predictive check: {inside.sum()} of {len(x)} statistics of the recording lie inside the 90% range "
          f"of {N_DRAWS} re-simulated posterior draws")
    k = FEATURE_NAMES.index("sync_corr")
    print(f"halving inhibition, synchrony of the median twin: {np.median(xb[nt, k]):.2f} -> {np.median(xh[nt, k]):.2f}, "
          f"up by more than 0.1 on {(eff[nt, k] > 0.1).sum()}/{N_NETS} networks; posterior draws: up in "
          f"{(eff[dr, k] > 0).sum()}/{N_DRAWS}, by more than 0.1 in {(eff[dr, k] > 0.1).sum()}/{N_DRAWS}")

    j = N_DRAWS                                                 # the median twin on its first network
    fig, ax = plt.subplots(2, 4, figsize=(20, 5.6), height_ratios=[3, 1], sharex="col")
    raster(ax[0, 0], ax[1, 0], ch, t, 0, 60, f"real culture {RECORDING}: first 60 s of the fitted window")
    raster(ax[0, 1], ax[1, 1], *sims_b[j], 0, 60, "its digital twin (posterior median)", "C3")
    raster(ax[0, 2], ax[1, 2], *sims_h[j], 0, 60, "the same twin with inhibition halved (in silico)", "C0")
    for a, name, label in ((ax[0, 3], "sync_corr", "synchrony"), (ax[1, 3], "log_fano_pop", "log10 Fano")):
        c = FEATURE_NAMES.index(name)
        paired(a, 0, xb[dr, c], xh[dr, c])
        paired(a, 2, xb[nt, c], xh[nt, c])
        a.axhline(x[c], color="k", ls="--", lw=0.8)
        a.set_ylabel(label); a.set_xlim(-0.4, 3.4)
        a.set_xticks([0, 1, 2, 3]); a.set_xticklabels(["fitted", "halved", "fitted", "halved"])
    ax[0, 3].set_title(f"left: {N_DRAWS} posterior draws; right: the median twin\n"
                       f"on {N_NETS} random networks (dashed: the recording)", fontsize=11)
    ax[0, 3].tick_params(labelbottom=False)
    fig.tight_layout(); fig.savefig(os.path.join(WORK, "demo_rasters.png"), dpi=110); plt.close(fig)

    q = lambda v: f"{np.percentile(v, 5):.2f} to {np.percentile(v, 95):.2f}"  # noqa: E731
    cmp = pd.DataFrame({"statistic": FEATURE_NAMES, "real": np.round(x, 2),
                        "draws: median": np.round(np.median(xb[dr], 0), 2), "draws: 90% range": [q(v) for v in xb[dr].T],
                        "real inside": inside, "median twin": np.round(np.median(xb[nt], 0), 2),
                        "halved, draws: median change": np.round(np.median(eff[dr], 0), 2),
                        "halved, draws: 90% range": [q(v) for v in eff[dr].T],
                        "halved, median twin: median change": np.round(np.median(eff[nt], 0), 2),
                        "halved, median twin: range": [f"{v.min():.2f} to {v.max():.2f}" for v in eff[nt].T]})
    cmp.to_csv(os.path.join(WORK, "demo_comparison.csv"), index=False)
    rows = [f"draw {i}" for i in range(N_DRAWS)] + [f"median twin, network {i}" for i in range(N_NETS)]
    pd.concat([pd.DataFrame({"row": rows}), pd.DataFrame(base, columns=POST_NAMES),
               pd.DataFrame(xb, columns=FEATURE_NAMES), pd.DataFrame(xh, columns=[f + "_half" for f in FEATURE_NAMES])],
              axis=1).to_csv(os.path.join(WORK, "demo_draws.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n" + cmp.to_string(index=False))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
