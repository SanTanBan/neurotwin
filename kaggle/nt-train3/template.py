"""NeuroTwin round-3 training & validation (model v4, focused prior).

1. Train the amortised posterior on the round-2 bank. The prior used for inference is read from the bank's
   own metadata, so it always matches how the bank was sampled.
2. Validate on held-out simulations: parameter recovery, simulation-based calibration, TARP coverage.
3. Apply to every 300-s window of the real recordings; report out-of-support and posterior leakage.
4. Posterior predictive checks against a *mismatched* twin (parameters from another recording's posterior),
   which is the baseline a useful twin has to beat, with per-feature detail.
5. Developmental trajectories from the windows that pass the reliability checks.
Outputs in /kaggle/working: posterior.pt, calibration.json, real_posteriors.csv, ppc.csv, ppc_features.csv,
trajectories.csv, figures/*.png
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

WORK = "/kaggle/working" if (os.name != "nt" and os.path.isdir("/kaggle/working")) else "."
INPUT = "/kaggle/input" if (os.name != "nt" and os.path.isdir("/kaggle/input")) else "."
FIG = os.path.join(WORK, "figures")
N_TEST = 1000
N_CAL = 500
N_PPC_REC = 16
N_PPC_DRAWS = 24
WIN_S = 300.0
# Statistics the model family cannot reproduce (documented misspecification) are excluded from inference
# but still checked in the posterior predictive checks.
EXCLUDE = ["nb_rise_frac"]

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py

# @@include src/neurotwin/infer.py

# @@include src/neurotwin/ppc.py


def apply_box(box):
    """Use the bank's sampling box as the inference prior (round 2 samples a focused region)."""
    global LO, HI, LOG, Z_LO, Z_HI
    LO = np.array([box[k][0] for k in PARAM_NAMES], float)
    HI = np.array([box[k][1] for k in PARAM_NAMES], float)
    LOG = np.array([box[k][2] == "log" for k in PARAM_NAMES])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)
    print("inference prior:", {k: (round(LO[i], 3), round(HI[i], 3), "log" if LOG[i] else "lin")
                               for i, k in enumerate(PARAM_NAMES)}, flush=True)


def load_bank():
    files = sorted(glob.glob(os.path.join(INPUT, "**", "bank3_shard*.npz"), recursive=True))
    meta = sorted(glob.glob(os.path.join(INPUT, "**", "bank3_shard*.json"), recursive=True))
    box = json.load(open(meta[0]))["focus_box"]
    apply_box({k: tuple(v) for k, v in box.items()})
    json.dump(box, open(os.path.join(WORK, "focus_box.json"), "w"), indent=1)   # the demo needs it too
    theta = np.concatenate([np.load(f)["theta"] for f in files])
    x = np.concatenate([np.load(f)["x"] for f in files])
    ok = np.isfinite(x).all(1) & np.all((theta >= LO - 1e-9) & (theta <= HI + 1e-9), axis=1)
    print(f"bank: {len(theta)} simulations from {len(files)} shards, {ok.sum()} usable", flush=True)
    return theta[ok], x[ok]


def out_of_support_scores(x_sim, x_query, k=10):
    """Mean distance to the k nearest simulations in z-scored feature space."""
    mu, sd = x_sim.mean(0), x_sim.std(0) + 1e-9
    a = torch.as_tensor((x_sim - mu) / sd, dtype=torch.float32)
    q = torch.as_tensor((x_query - mu) / sd, dtype=torch.float32)
    return torch.cat([torch.cdist(c, a).topk(k, largest=False).values.mean(1) for c in torch.split(q, 256)]).numpy()


def fig_recovery(rec, theta_te, post_means):
    n = len(PARAM_NAMES)
    cols = (n + 1) // 2
    fig, ax = plt.subplots(2, cols, figsize=(3.2 * cols, 6.6))
    z = to_unbounded(theta_te)
    for i, a in enumerate(ax.flat[:n]):
        a.scatter(z[:, i], post_means[:, i], s=3, alpha=0.4)
        a.plot([Z_LO[i], Z_HI[i]], [Z_LO[i], Z_HI[i]], "k--", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  R2={rec['r2'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Round 2: parameter recovery on held-out simulations")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "recovery.png"), dpi=110); plt.close(fig)


def fig_sbc(cal):
    ranks = cal["ranks"]; n = ranks.shape[1]; cols = (n + 1) // 2
    fig, ax = plt.subplots(2, cols, figsize=(3.2 * cols, 6.2))
    for i, a in enumerate(ax.flat[:n]):
        a.hist(ranks[:, i], bins=20, color="C0", alpha=0.8)
        a.axhline(len(ranks) / 20, color="k", lw=0.8)
        a.set_title(f"{PARAM_NAMES[i]}  KS p={cal['sbc_ks_pvals'][i]:.2f}", fontsize=9)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Round 2: simulation-based calibration (flat = calibrated)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sbc.png"), dpi=110); plt.close(fig)
    fig, a = plt.subplots(figsize=(4, 4))
    a.plot(cal["tarp_alpha"], cal["tarp_ecp"], lw=2); a.plot([0, 1], [0, 1], "k--")
    a.set_xlabel("credibility level"); a.set_ylabel("expected coverage")
    a.set_title(f"TARP coverage (ATC={cal['tarp_atc']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tarp.png"), dpi=110); plt.close(fig)


def fig_trajectories(agg):
    n = len(PARAM_NAMES); cols = (n + 1) // 2
    fig, ax = plt.subplots(2, cols, figsize=(3.4 * cols, 6.8))
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
    fig.suptitle("Inferred twin parameters across development (thin: cultures; bold: median)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trajectories.png"), dpi=110); plt.close(fig)


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    theta, x_all = load_bank()
    keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in EXCLUDE]
    x = x_all[:, keep]
    print(f"inference uses {len(keep)} of {len(FEATURE_NAMES)} statistics; excluded: {EXCLUDE}", flush=True)
    perm = rng.permutation(len(theta))
    te, tr = perm[:N_TEST], perm[N_TEST:]

    posterior, trainer = train_npe(theta[tr], x[tr], density_estimator="nsf",
                                   training_batch_size=512, stop_after_epochs=30)
    torch.save(posterior, os.path.join(WORK, "posterior.pt"))
    print(f"trained on {len(tr)} simulations in {(time.time() - t0) / 60:.1f} min", flush=True)

    rec = recovery_metrics(posterior, theta[te[:N_CAL]], x[te[:N_CAL]])
    cal = calibration(posterior, theta[te[:N_CAL]], x[te[:N_CAL]])
    fig_recovery(rec, theta[te[:N_CAL]], rec["post_mean_z"])
    fig_sbc(cal)
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**rec, **cal}.items()
               if k not in ("ranks", "tarp_ecp", "tarp_alpha", "post_mean_z")},
              open(os.path.join(WORK, "calibration.json"), "w"), indent=1)
    print(pd.DataFrame({k: rec[k] for k in ("param", "r2", "nrmse", "coverage_50", "coverage_90", "contraction")})
          .round(3).to_string(index=False), flush=True)
    print(f"TARP ATC={cal['tarp_atc']:.3f} KS p={cal['tarp_ks_p']:.3f}", flush=True)

    real_files = glob.glob(os.path.join(INPUT, "**", "wagenaar_dense_features.csv"), recursive=True)
    if not real_files:
        print("no real feature table attached; done"); return
    df = pd.read_csv(real_files[0])
    if "file" not in df:
        df["file"] = df[["plating", "culture", "div"]].astype(str).agg("-".join, axis=1)
    xr_all = df[FEATURE_NAMES].to_numpy(np.float32)
    xr = xr_all[:, keep]
    thr = np.quantile(out_of_support_scores(x[tr], x[te]), 0.99)
    # assets for the demo and the app: the whole training bank, so their support check equals this one
    np.savez_compressed(os.path.join(WORK, "twin_assets.npz"), mu=x[tr].mean(0), sd=x[tr].std(0), threshold=thr,
                        threshold_ref=thr, x=((x[tr] - x[tr].mean(0)) / (x[tr].std(0) + 1e-9)).astype(np.float32),
                        feature_names=np.array(FEATURE_NAMES), param_names=np.array(PARAM_NAMES), keep=np.array(keep))
    df["oos_score"] = out_of_support_scores(x[tr], xr)
    thr16 = np.quantile(out_of_support_scores(x_all[tr], x_all[te]), 0.99)
    df["out_of_support_all16"] = out_of_support_scores(x_all[tr], xr_all) > thr16
    df["out_of_support"] = df["oos_score"] > thr
    samples, df["leakage"] = sample_many(posterior, xr, n=500)
    for i, p in enumerate(PARAM_NAMES):
        df[f"{p}_med"] = np.median(samples[:, :, i], 0)
        df[f"{p}_q05"] = np.quantile(samples[:, :, i], 0.05, 0)
        df[f"{p}_q95"] = np.quantile(samples[:, :, i], 0.95, 0)
    df.to_csv(os.path.join(WORK, "real_posteriors.csv"), index=False)
    good = df[~df.out_of_support & (df.leakage < 0.2)]
    print(f"real windows {len(df)}: out of support {df.out_of_support.mean():.1%}, "
          f"mean leakage {df.leakage.mean():.2f}, reliable windows {len(good)} ({len(good) / len(df):.1%}); "
          f"out of support using all 16 statistics {df.out_of_support_all16.mean():.1%}",
          flush=True)
    fig, a = plt.subplots(1, 2, figsize=(11, 4))
    a[0].hist(df.leakage, bins=40); a[0].set_xlabel("posterior mass outside the prior box"); a[0].set_ylabel("windows")
    a[1].scatter(df["div"], df.oos_score, s=4, alpha=0.3); a[1].axhline(thr, color="r", ls="--")
    a[1].set_xlabel("days in vitro"); a[1].set_ylabel("distance to nearest simulations")
    fig.suptitle("Reliability of the fit on real recordings")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability.png"), dpi=110); plt.close(fig)

    # posterior predictive checks: own twin vs a twin fitted to a different recording
    pool = good if len(good) >= N_PPC_REC else df
    picks = pool.drop_duplicates("file").sort_values("div")
    picks = picks.iloc[np.linspace(0, len(picks) - 1, min(N_PPC_REC, len(picks))).astype(int)]
    rows, feat_rows = [], []
    for n_i, (_, r) in enumerate(picks.iterrows()):
        xo = r[FEATURE_NAMES].to_numpy(np.float32)
        other = picks.iloc[(n_i + len(picks) // 2) % len(picks)]
        th_own = sample(posterior, xo[keep], n=N_PPC_DRAWS)
        th_other = sample(posterior, other[FEATURE_NAMES].to_numpy(np.float32)[keep], n=N_PPC_DRAWS)
        xp_own = predictive(th_own, duration_s=WIN_S, seed=int(rng.integers(2**31 - 1)))
        xp_other = predictive(th_other, duration_s=WIN_S, seed=int(rng.integers(2**31 - 1)))
        s_own, s_other = summary(xo, xp_own), summary(xo, xp_other)
        rows.append({"file": r["file"], "div": r["div"], "own_inside90": s_own["inside"].mean(),
                     "mismatched_inside90": s_other["inside"].mean(),
                     "own_abs_z": np.nanmedian(np.abs(s_own["z"])),
                     "mismatched_abs_z": np.nanmedian(np.abs(s_other["z"]))})
        for j, f in enumerate(FEATURE_NAMES):
            feat_rows.append({"file": r["file"], "div": r["div"], "feature": f, "observed": xo[j],
                              "own_median": s_own["median"][j], "own_z": s_own["z"][j],
                              "mismatched_z": s_other["z"][j], "inside90": bool(s_own["inside"][j])})
        print(rows[-1], flush=True)
    ppc = pd.DataFrame(rows); ppc.to_csv(os.path.join(WORK, "ppc.csv"), index=False)
    pd.DataFrame(feat_rows).to_csv(os.path.join(WORK, "ppc_features.csv"), index=False)
    print(f"\nPPC: own twin covers {ppc.own_inside90.mean():.2f} of statistics vs {ppc.mismatched_inside90.mean():.2f} "
          f"for a mismatched twin; median |z| {ppc.own_abs_z.median():.2f} vs {ppc.mismatched_abs_z.median():.2f}",
          flush=True)

    if len(good) > 50:
        agg = good.groupby(["plating", "culture", "div"])[[f"{p}_med" for p in PARAM_NAMES]].median().reset_index()
        agg.to_csv(os.path.join(WORK, "trajectories.csv"), index=False)
        fig_trajectories(agg)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
