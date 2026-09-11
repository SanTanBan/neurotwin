"""NeuroTwin training & validation.

1. Train the amortized posterior (neural posterior estimation, neural spline flow) on the simulation bank.
2. Validate on held-out simulations: parameter recovery, simulation-based calibration (SBC), TARP coverage.
3. Apply to every 300-s window of the real recordings; aggregate per recording; trajectories vs age (DIV).
4. Reliability: out-of-support score for every real window; posterior predictive checks on a subset.
Outputs in /kaggle/working: posterior.pt, calibration.json, real_posteriors.csv, ppc.csv, figures/*.png
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
FIG = os.path.join(WORK, "figures")
N_TEST = 1000
N_PPC_REC = 12
N_PPC_DRAWS = 16
WIN_S = 300.0

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py

# @@include src/neurotwin/infer.py

# @@include src/neurotwin/ppc.py


def load_bank():
    files = sorted(glob.glob(os.path.join(INPUT, "**", "bank_shard*.npz"), recursive=True))
    th, xs = [], []
    for f in files:
        d = np.load(f)
        th.append(d["theta"]); xs.append(d["x"])
    theta, x = np.concatenate(th), np.concatenate(xs)
    inside = np.all((theta >= LO - 1e-9) & (theta <= HI + 1e-9), axis=1)   # support of the current prior
    ok = inside & np.isfinite(x).all(1)
    print(f"bank: {len(theta)} sims from {len(files)} shards, {ok.sum()} inside current prior", flush=True)
    return theta[ok], x[ok]


def out_of_support_scores(x_sim, x_query, k=10):
    """Mean distance to the k nearest simulations in z-scored feature space."""
    mu, sd = x_sim.mean(0), x_sim.std(0) + 1e-9
    a = torch.as_tensor((x_sim - mu) / sd, dtype=torch.float32)
    q = torch.as_tensor((x_query - mu) / sd, dtype=torch.float32)
    out = []
    for chunk in torch.split(q, 256):
        d = torch.cdist(chunk, a)
        out.append(d.topk(k, largest=False).values.mean(1))
    return torch.cat(out).numpy()


def fig_recovery(rec, theta_te, post_means):
    n = len(PARAM_NAMES)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.4))
    z = to_unbounded(theta_te)
    for i, a in enumerate(ax.flat[:n]):
        a.scatter(z[:, i], post_means[:, i], s=3, alpha=0.4)
        lim = [to_unbounded(LO)[i], to_unbounded(HI)[i]]
        a.plot(lim, lim, "k--", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  R2={rec['r2'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Parameter recovery on held-out simulations (posterior mean vs truth, prior-uniform space)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "recovery.png"), dpi=110); plt.close(fig)


def fig_sbc(cal):
    ranks = cal["ranks"]; n = ranks.shape[1]
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6))
    for i, a in enumerate(ax.flat[:n]):
        a.hist(ranks[:, i], bins=20, color="C0", alpha=0.8)
        a.axhline(len(ranks) / 20, color="k", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  KS p={cal['sbc_ks_pvals'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Simulation-based calibration: rank histograms (flat = calibrated)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sbc.png"), dpi=110); plt.close(fig)
    fig, a = plt.subplots(figsize=(4, 4))
    a.plot(cal["tarp_alpha"], cal["tarp_ecp"], lw=2); a.plot([0, 1], [0, 1], "k--")
    a.set_xlabel("credibility level"); a.set_ylabel("expected coverage")
    a.set_title(f"TARP coverage (ATC={cal['tarp_atc']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tarp.png"), dpi=110); plt.close(fig)


def fig_trajectories(agg):
    n = len(PARAM_NAMES)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.4 * ((n + 1) // 2), 6.4))
    for i, a in enumerate(ax.flat[:n]):
        p = PARAM_NAMES[i]
        for (pl, cu), g in agg.groupby(["plating", "culture"]):
            g = g.sort_values("div")
            a.plot(g["div"], g[f"{p}_med"], color=f"C{pl % 10}", alpha=0.35, lw=0.8)
        m = agg.groupby("div")[f"{p}_med"].median()
        a.plot(m.index, m.values, "k", lw=2)
        a.set_title(p, fontsize=10); a.set_xlabel("days in vitro")
        if LOG[i]:
            a.set_yscale("log")
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Inferred twin parameters across development (thin: cultures, coloured by plating; bold: median)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trajectories.png"), dpi=110); plt.close(fig)


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    theta, x = load_bank()
    perm = rng.permutation(len(theta))
    te, tr = perm[:N_TEST], perm[N_TEST:]

    posterior, trainer = train_npe(theta[tr], x[tr], density_estimator="nsf",
                                   training_batch_size=512, stop_after_epochs=30)
    torch.save(posterior, os.path.join(WORK, "posterior.pt"))
    print(f"trained on {len(tr)} sims in {(time.time() - t0) / 60:.1f} min", flush=True)

    rec = recovery_metrics(posterior, theta[te[:500]], x[te[:500]])
    cal = calibration(posterior, theta[te[:500]], x[te[:500]])
    fig_recovery(rec, theta[te[:500]], rec["post_mean_z"])
    fig_sbc(cal)
    summary_cal = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**rec, **cal}.items()
                   if k not in ("ranks", "tarp_ecp", "tarp_alpha", "post_mean_z")}
    with open(os.path.join(WORK, "calibration.json"), "w") as f:
        json.dump(summary_cal, f, indent=1)
    print(pd.DataFrame({k: rec[k] for k in ("param", "r2", "nrmse", "coverage_50", "coverage_90", "contraction")})
          .round(3).to_string(index=False), flush=True)
    print(f"TARP ATC={cal['tarp_atc']:.3f} KS p={cal['tarp_ks_p']:.3f}", flush=True)

    real_files = glob.glob(os.path.join(INPUT, "**", "wagenaar_dense_features.csv"), recursive=True)
    if not real_files:
        print("no real feature table attached; done"); return
    df = pd.read_csv(real_files[0])
    if "file" not in df:
        df["file"] = df[["plating", "culture", "div"]].astype(str).agg("-".join, axis=1)
    xr = df[FEATURE_NAMES].to_numpy(np.float32)
    thr = np.quantile(out_of_support_scores(x[tr], x[te]), 0.99)
    df["oos_score"] = out_of_support_scores(x[tr], xr)
    df["out_of_support"] = df["oos_score"] > thr
    samples, df["leakage"] = sample_many(posterior, xr, n=500)             # [500, n_windows, d]
    for i, p in enumerate(PARAM_NAMES):
        df[f"{p}_med"] = np.median(samples[:, :, i], 0)
        df[f"{p}_q05"] = np.quantile(samples[:, :, i], 0.05, 0)
        df[f"{p}_q95"] = np.quantile(samples[:, :, i], 0.95, 0)
    df.to_csv(os.path.join(WORK, "real_posteriors.csv"), index=False)
    print(f"real windows: {len(df)}, out-of-support (99% sim threshold): {df.out_of_support.mean():.2%}", flush=True)
    agg = df[~df.out_of_support].groupby(["plating", "culture", "div"])[[f"{p}_med" for p in PARAM_NAMES]] \
        .median().reset_index()
    fig_trajectories(agg)

    # posterior predictive checks on recordings spanning development
    picks = df[~df.out_of_support].drop_duplicates("file").sort_values("div")
    picks = picks.iloc[np.linspace(0, len(picks) - 1, min(N_PPC_REC, len(picks))).astype(int)]
    rows = []
    for _, r in picks.iterrows():
        xo = r[FEATURE_NAMES].to_numpy(np.float32)
        th = sample(posterior, xo, n=N_PPC_DRAWS)
        xp = predictive(th, duration_s=WIN_S, seed=int(rng.integers(2**31 - 1)))
        s_post = summary(xo, xp)
        s_prior = summary(xo, x[te[:N_PPC_DRAWS * 8]])
        rows.append({"file": r["file"], "div": r["div"], "post_inside90": s_post["inside"].mean(),
                     "prior_inside90": s_prior["inside"].mean(),
                     "post_abs_z": np.nanmedian(np.abs(s_post["z"])),
                     "prior_abs_z": np.nanmedian(np.abs(s_prior["z"]))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(WORK, "ppc.csv"), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
