"""NeuroTwin chip design: how long must a two-chamber chip be recorded to read its wiring direction?

A single 5-minute recording reads the direction of an axon-diode chip only weakly (nt-train2c). This notebook
asks the design question that follows. It simulates held-out two-chamber chips for 40 minutes each, cuts every
recording into eight 5-minute windows, fits each window with the two-chamber posterior trained in nt-train2c,
and pools the windows. With a uniform prior on the crossing probabilities, independent windows combine by
multiplying their posteriors; each window's marginal over (p_fwd, p_bwd) is a kernel density estimate of its
posterior draws. For 5, 10, 20 and 40 minutes of recording it reports how often the direction is read
correctly, how wide the uncertainty is, and whether the stated confidence can be trusted.

Outputs in /kaggle/working: design_length_sims.npz, design_length.csv, design_length_summary.csv,
figures/recording_length.png
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
from scipy.stats import binomtest, gaussian_kde  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ON_KAGGLE = os.name != "nt" and os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else "."
INPUT = "/kaggle/input" if ON_KAGGLE else "."
FIG = os.path.join(WORK, "figures")
N_CHIPS = int(os.environ.get("NT_N_CHIPS", 240))      # held-out chips (the environment overrides are for smoke tests)
N_WIN = int(os.environ.get("NT_N_WIN", 8))            # 5-minute windows per chip: 40 minutes of recording
BATCH = int(os.environ.get("NT_BATCH", 16))
N_POST = int(os.environ.get("NT_N_POST", 1000))       # posterior draws per window
WIN_S = 300.0
LENGTHS = [k for k in (1, 2, 4, 8) if k <= N_WIN]     # windows pooled: 5, 10, 20, 40 minutes
GRID = 61                                            # grid over (p_fwd, p_bwd) for the pooled posterior
SEED = 9100
TIME_BUDGET_S = 10.5 * 3600

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py

# @@include src/neurotwin/directional.py

# @@include src/neurotwin/infer.py


def find(pattern):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {INPUT}")
    return hits[0]


def posterior_space(box):
    """The two-chamber posterior lives on (base parameters in the bank's box, p_fwd, p_bwd)."""
    global LO, HI, LOG, Z_LO, Z_HI
    LO = np.array([box[k][0] for k in box] + [0.0, 0.0])
    HI = np.array([box[k][1] for k in box] + [1.0, 1.0])
    LOG = np.array([box[k][2] == "log" for k in box] + [False, False])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)


def sample_base(n, rng, box):
    cols = []
    for lo, hi, scale in box.values():
        u = rng.random(n)
        cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if scale == "log" else lo + u * (hi - lo))
    return np.stack(cols, 1)


def simulate_chips(box, rng, t0):
    """Held-out chips, each recorded for N_WIN windows; returns truth and per-window statistics."""
    cfg = ChipConfig(compartments=2)
    chamber = cfg.chamber_of_electrodes()
    dur = N_WIN * WIN_S
    cap = int(30.0 * len(cfg.electrodes()) * dur)
    thetas, crosses, xs, runaway, times = [], [], [], [], []
    while sum(len(t) for t in thetas) < N_CHIPS:
        tb = time.time()
        theta, cross = sample_base(BATCH, rng, box), rng.random((BATCH, 2))
        out = simulate(theta, cfg, duration_s=dur, warmup_s=10.0, seed=int(rng.integers(2**31 - 1)), cross=cross,
                       max_rate_hz=30.0)
        x = np.full((BATCH, N_WIN, len(FEATURE_NAMES) + len(DIR_FEATURE_NAMES)), np.nan)
        for b, (e, t) in enumerate(out):
            for w in range(N_WIN):
                m = (t >= w * WIN_S) & (t < (w + 1) * WIN_S)
                x[b, w] = np.r_[compute_features(e[m], t[m] - w * WIN_S, WIN_S),
                                directional_features(e[m], t[m] - w * WIN_S, WIN_S, chamber)]
        thetas.append(theta); crosses.append(cross); xs.append(x)
        runaway.append(np.array([len(e) >= cap for e, _ in out]))
        times.append(time.time() - tb)
        print(f"{sum(len(t) for t in thetas)} chips x {dur / 60:.0f} min, last batch {times[-1]:.0f} s, "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if time.time() - t0 + 1.5 * max(times) > TIME_BUDGET_S:
            break
    return (np.concatenate(thetas), np.concatenate(crosses), np.concatenate(xs), np.concatenate(runaway))


def pooled(samples, grid):
    """Pool windows' posteriors over (p_fwd, p_bwd): the product of their kernel density estimates on a grid.

    Returns mean asymmetry, P(p_fwd > p_bwd), and the 90% interval widths of p_fwd and of the asymmetry.
    """
    F, B = np.meshgrid(grid, grid, indexing="ij")
    pts = np.vstack([F.ravel(), B.ravel()])
    logp = np.zeros(F.size)
    for s in samples:                                    # s: [n, 2] draws of (p_fwd, p_bwd) for one window
        try:
            dens = gaussian_kde(s.T)(pts)
        except np.linalg.LinAlgError:                    # degenerate draws: this window adds no information
            dens = np.ones(F.size)
        logp += np.log(dens + 1e-6 * dens.max())          # a floor, so one window cannot veto a region
    p = np.exp(logp - logp.max()); p /= p.sum()
    asym = (F - B).ravel()

    def width(v):
        o = np.argsort(v); c = np.cumsum(p[o])
        return v[o][np.searchsorted(c, 0.95)] - v[o][np.searchsorted(c, 0.05)]

    return (p * asym).sum(), p[asym > 0].sum() + 0.5 * p[asym == 0].sum(), width(F.ravel()), width(asym)


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    box = {k: tuple(v) for k, v in json.load(open(find("bank2c_shard*.json")))["base_box"].items()}
    posterior = torch.load(find("posterior2c.pt"), weights_only=False)
    rng = np.random.default_rng(SEED)
    theta, cross, X, runaway = simulate_chips(box, rng, t0)
    np.savez_compressed(os.path.join(WORK, "design_length_sims.npz"), theta=theta, cross=cross, x=X,
                        runaway=runaway, base_names=np.array(list(box)),
                        feature_names=np.array(FEATURE_NAMES + DIR_FEATURE_NAMES))
    print(f"simulated {len(theta)} chips in {(time.time() - t0) / 60:.1f} min; {runaway.sum()} runaway", flush=True)

    posterior_space(box)
    ok_win = np.isfinite(X).all(2) & ~runaway[:, None]
    flat = X.reshape(-1, X.shape[2])
    idx = np.flatnonzero(ok_win.ravel())
    torch.manual_seed(0)
    s, leak = sample_many(posterior, flat[idx], n=N_POST)                 # [n, m, 12]
    draws = np.full((len(flat), N_POST, 2), np.nan)
    draws[idx] = np.transpose(s[:, :, -2:], (1, 0, 2))
    draws = draws.reshape(len(theta), N_WIN, N_POST, 2)
    print(f"posterior draws for {len(idx)} windows; median leakage {np.median(leak):.2f}", flush=True)

    grid = np.linspace(0.0, 1.0, GRID)
    rows = []
    for c in range(len(theta)):
        for k in LENGTHS:
            wins = [w for w in range(k) if ok_win[c, w]]
            if len(wins) < k:                    # a chip enters a length only if all its windows are usable
                continue
            m, p_fwd_leads, w_f, w_a = pooled([draws[c, w] for w in wins], grid)
            avg = np.mean([draws[c, w, :, 0].mean() - draws[c, w, :, 1].mean() for w in wins])
            rows.append({"chip": c, "minutes": k * WIN_S / 60, "p_fwd": cross[c, 0], "p_bwd": cross[c, 1],
                         "asym_true": cross[c, 0] - cross[c, 1], "asym_pooled": m, "p_forward_leads": p_fwd_leads,
                         "width90_p_fwd": w_f, "width90_asym": w_a, "asym_window_mean": avg})
    d = pd.DataFrame(rows)
    d["correct"] = np.sign(d.asym_pooled) == np.sign(d.asym_true)
    d["confidence"] = np.maximum(d.p_forward_leads, 1 - d.p_forward_leads)
    d["strong"] = d.asym_true.abs() > 0.3
    d.to_csv(os.path.join(WORK, "design_length.csv"), index=False)

    summ = []
    for mins, g in d.groupby("minutes"):
        st, dec = g[g.strong], g[g.confidence >= 0.9]
        r2 = 1 - ((g.asym_pooled - g.asym_true) ** 2).sum() / ((g.asym_true - g.asym_true.mean()) ** 2).sum()
        summ.append({"minutes": mins, "chips": len(g), "direction_correct": g.correct.mean(),
                     "direction_correct_strong": st.correct.mean(), "strong_chips": len(st),
                     "direction_correct_window_mean": (np.sign(g.asym_window_mean) == np.sign(g.asym_true)).mean(),
                     "asym_r2": r2, "median_width90_p_fwd": g.width90_p_fwd.median(),
                     "median_width90_asym": g.width90_asym.median(), "decisive_share": len(dec) / len(g),
                     "decisive_correct": dec.correct.mean() if len(dec) else np.nan, "decisive_chips": len(dec),
                     "mean_confidence": g.confidence.mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(WORK, "design_length_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n" + summ.round(3).to_string(index=False), flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for col, lab, n_col in (("direction_correct", "all chips", "chips"),
                            ("direction_correct_strong", "engineered asymmetry > 0.3", "strong_chips")):
        ok = summ[n_col] > 0
        ci = [binomtest(int(round(a * n)), int(n)).proportion_ci() for a, n in zip(summ[col][ok], summ[n_col][ok])]
        ax[0].errorbar(summ.minutes[ok], summ[col][ok], yerr=[summ[col][ok] - [c.low for c in ci],
                                                               [c.high for c in ci] - summ[col][ok]],
                       marker="o", capsize=3, label=lab)
    ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(summ.minutes); ax[0].set_xticklabels([f"{m:g}" for m in summ.minutes])
    ax[0].set_xlabel("recording length (min)"); ax[0].set_ylabel("direction read correctly"); ax[0].legend(frameon=False)
    ax[0].set_title("Reading the wiring direction")
    ax[1].plot(summ.minutes, summ.median_width90_asym, "o-", label="asymmetry p_fwd - p_bwd")
    ax[1].plot(summ.minutes, summ.median_width90_p_fwd, "s-", label="p_fwd")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(summ.minutes); ax[1].set_xticklabels([f"{m:g}" for m in summ.minutes])
    ax[1].set_xlabel("recording length (min)"); ax[1].set_ylabel("median width of the 90% interval")
    ax[1].legend(frameon=False); ax[1].set_title("Uncertainty shrinks with recording length")
    bins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    for mins, g in d.groupby("minutes"):
        k = np.digitize(g.confidence, bins) - 1
        acc = g.groupby(k).correct.mean()
        conf = g.groupby(k).confidence.mean()
        ax[2].plot(conf.values, acc.values, "o-", label=f"{mins:g} min")
    ax[2].plot([0.5, 1], [0.5, 1], "k--", lw=0.8)
    ax[2].set_xlabel("stated probability of the direction"); ax[2].set_ylabel("observed share correct")
    ax[2].set_title("Is the stated confidence honest?"); ax[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "recording_length.png"), dpi=110); plt.close(fig)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
