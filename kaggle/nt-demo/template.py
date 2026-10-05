"""NeuroTwin demo: fit a digital twin to one real MEA recording and run in-silico experiments.

This is the short, reviewer-facing notebook. It takes a real recording of a developing rat cortical culture
(Wagenaar, Pine & Potter 2006), computes its electrophysiological signature, fits the twin with the
pre-trained amortised posterior in well under a second, states plainly whether the fit can be trusted, checks
that re-simulated twins reproduce the recording, and then asks two questions no experiment answered: what
would this culture do with half its inhibition, or with half its excitation? Each question is put to 16
posterior draws, and to the draw closest to the recording on 8 further random networks, which separates what
the recording leaves uncertain from what is intrinsically random. Every modified run shares its network and
noise with an unmodified run, so a pair differs only by the intervention.
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
START_S = 1200.0              # its fifth 300-s window: the culture bursts, and the window passes every check
N_POST = 500
N_DRAWS = 16                  # posterior draws re-simulated for the predictive check and the experiments
N_NETS = 8                    # further random networks for the twin shown in the figure
EXPERIMENTS = [("inhibition halved", "g_ie", 0.5, "C0"),      # name, parameter, factor, colour
               ("excitation halved", "j_e", 0.5, "C2")]

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


def paired(ax, x0, before, after, color):
    """One line per twin: as fitted (red, at x0) and after the in-silico change (at x0 + 1)."""
    for b, a in zip(before, after):
        ax.plot([x0, x0 + 1], [b, a], color="0.75", lw=0.8)
    ax.scatter(np.full(len(before), x0), before, color="C3", s=14, zorder=3)
    ax.scatter(np.full(len(after), x0 + 1.0), after, color=color, s=14, zorder=3)


def main():
    t0 = time.time()
    box = json.load(open(find("focus_box.json")))
    apply_box({k: tuple(v) for k, v in box.items()})
    posterior = torch.load(find("posterior.pt"), weights_only=False)
    assets = np.load(find("twin_assets.npz"))

    ch, t = load_spk(find(f"{RECORDING}.spk.txt.bz2"))
    m = (t >= START_S) & (t < START_S + WIN_S)
    e_obs, t_obs = ch[m], t[m] - START_S
    x = compute_features(e_obs, t_obs, WIN_S)
    print(f"recording {RECORDING}: {len(t):,} spikes, {len(np.unique(ch))} active electrodes, {t.max():.0f} s; "
          f"fitting the window {START_S:.0f}-{START_S + WIN_S:.0f} s")
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
    dist, thr, exact = support_check(assets, z)
    if not exact:
        print("note: these assets hold a subsample of the training bank; the support check is conservative")
    trusted = dist <= thr and leak[0] < 0.2
    print(f"\nreliability: distance to simulated data {dist:.2f} (threshold {thr:.2f}), "
          f"posterior mass outside the parameter ranges {leak[0]:.0%} -> "
          f"{'fit is inside the model’s range' if trusted else 'FIT IS EXTRAPOLATION, do not trust it'}")

    # Twins: N_DRAWS posterior draws, each simulated on its own random network, as fitted and under each
    # experiment. The same seed gives every arm the same networks and noise, so a pair differs only by the change.
    pick = np.random.default_rng(0).choice(len(s), N_DRAWS, replace=False)
    arms = {"as fitted": s[pick]}
    for name, par, factor, _ in EXPERIMENTS:
        arms[name] = s[pick].copy()
        arms[name][:, POST_NAMES.index(par)] *= factor
    t2 = time.time()
    sims = {a: simulate(th, ChipConfig(), duration_s=WIN_S, warmup_s=10.0, seed=7) for a, th in arms.items()}
    feats = {a: np.array([compute_features(*o, WIN_S) for o in v]) for a, v in sims.items()}
    xb = feats["as fitted"]
    lo, hi = np.percentile(xb, [5, 95], axis=0)
    inside = (x >= lo) & (x <= hi)
    print(f"\npredictive check: {inside.sum()} of {len(x)} statistics of the recording lie inside the 90% range "
          f"of {N_DRAWS} re-simulated posterior draws")

    # The twin shown in the figure is the draw whose simulation is closest to the recording (z-scored distance
    # over the inference statistics). It is then re-run on N_NETS further random networks, to see how much of an
    # experiment's effect is network randomness rather than parameter uncertainty.
    zdist = np.linalg.norm((xb[:, keep] - x[keep]) / (assets["sd"] + 1e-9), axis=1)
    j = int(np.argmin(zdist))
    print(f"twin shown: posterior draw {j}, the closest of {N_DRAWS} to the recording (distance {zdist[j]:.2f}; "
          f"median over draws {np.median(zdist):.2f})")
    nets = {a: simulate(np.repeat(th[j][None], N_NETS, axis=0), ChipConfig(), duration_s=WIN_S, warmup_s=10.0, seed=8)
            for a, th in arms.items()}
    fnets = {a: np.array([compute_features(*o, WIN_S) for o in v]) for a, v in nets.items()}
    n_sims = sum(len(v) for v in sims.values()) + sum(len(v) for v in nets.values())
    print(f"{n_sims} simulations of {WIN_S:.0f} s in {time.time() - t2:.0f} s")
    capped = sum(len(e) >= int(60.0 * 60 * WIN_S) for v in list(sims.values()) + list(nets.values()) for e, _ in v)
    if capped:
        print(f"warning: {capped} runaway simulations hit the event cap")

    k = FEATURE_NAMES.index("sync_corr")
    for name, *_ in EXPERIMENTS:
        d = feats[name][:, k] - xb[:, k]
        dn = fnets[name][:, k] - fnets["as fitted"][:, k]
        print(f"{name}: twin shown, synchrony {xb[j, k]:.2f} -> {feats[name][j, k]:.2f} (on {N_NETS} more networks the "
              f"change is {dn.min():+.2f} to {dn.max():+.2f}); posterior draws: median change {np.median(d):+.2f}, "
              f"90% range {np.percentile(d, 5):+.2f} to {np.percentile(d, 95):+.2f}, up by more than 0.05 in "
              f"{(d > 0.05).sum()}/{N_DRAWS}, down by more than 0.05 in {(d < -0.05).sum()}/{N_DRAWS}")

    fig = plt.figure(figsize=(20, 10))
    outer = fig.add_gridspec(2, 1, height_ratios=[4, 2.6], hspace=0.28)
    top = outer[0].subgridspec(2, 4, height_ratios=[3, 1], hspace=0.08, wspace=0.25)
    bottom = outer[1].subgridspec(1, 2, wspace=0.15)
    panels = [(e_obs, t_obs, f"real culture {RECORDING}, minutes {START_S / 60:.0f}-{START_S / 60 + 1:.0f}", "k"),
              (*sims["as fitted"][j], "its digital twin (the closest posterior draw)", "C3")]
    panels += [(*sims[name][j], f"the same twin, {name} (in silico)", c) for name, _, _, c in EXPERIMENTS]
    for i, (e, tt, title, color) in enumerate(panels):
        a = fig.add_subplot(top[0, i])
        raster(a, fig.add_subplot(top[1, i], sharex=a), e, tt, 0, 60, title, color)
    for i, (name, _, _, c) in enumerate(EXPERIMENTS):
        a = fig.add_subplot(bottom[0, i])
        paired(a, 0, xb[:, k], feats[name][:, k], c)
        a.plot([0, 1], [xb[j, k], feats[name][j, k]], color="k", lw=1.6, zorder=4)
        paired(a, 2, fnets["as fitted"][:, k], fnets[name][:, k], c)
        a.axhline(x[k], color="k", ls="--", lw=0.8)
        a.set_xlim(-0.4, 3.4); a.set_xticks([0, 1, 2, 3]); a.set_ylabel("synchrony")
        a.set_xticklabels(["as fitted", name, "as fitted", name])
        a.set_title(f"{name}\nleft: {N_DRAWS} posterior draws (black: the twin shown); right: the twin shown on "
                    f"{N_NETS} more networks; dashed: the recording", fontsize=10)
    fig.savefig(os.path.join(WORK, "demo_rasters.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    qr = lambda v: f"{np.percentile(v, 5):.2f} to {np.percentile(v, 95):.2f}"  # noqa: E731
    cols = {"statistic": FEATURE_NAMES, "real": np.round(x, 2),
            "draws: median": np.round(np.median(xb, 0), 2), "draws: 90% range": [qr(v) for v in xb.T],
            "real inside": inside, "twin shown": np.round(xb[j], 2)}
    for name, *_ in EXPERIMENTS:
        d, dn = feats[name] - xb, fnets[name] - fnets["as fitted"]
        cols[f"{name}, draws: median change"] = np.round(np.median(d, 0), 2)
        cols[f"{name}, draws: 90% range"] = [qr(v) for v in d.T]
        cols[f"{name}, twin shown on {N_NETS} networks: range"] = [f"{v.min():.2f} to {v.max():.2f}" for v in dn.T]
    cmp = pd.DataFrame(cols)
    cmp.to_csv(os.path.join(WORK, "demo_comparison.csv"), index=False)
    rows = [f"draw {i}" for i in range(N_DRAWS)] + [f"twin shown (draw {j}), network {i}" for i in range(N_NETS)]
    parts = [pd.DataFrame({"row": rows, "distance": np.r_[zdist, np.full(N_NETS, np.nan)]}),
             pd.DataFrame(np.vstack([s[pick], np.repeat(s[pick][j][None], N_NETS, axis=0)]), columns=POST_NAMES)]
    parts += [pd.DataFrame(np.vstack([feats[a], fnets[a]]),
                           columns=[n if a == "as fitted" else f"{n} | {a}" for n in FEATURE_NAMES]) for a in arms]
    pd.concat(parts, axis=1).to_csv(os.path.join(WORK, "demo_draws.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + cmp.iloc[:, :6].to_string(index=False))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
