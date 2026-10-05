"""NeuroTwin two-chamber (axon-diode) inference.

Trains an amortised posterior over the culture's biophysical parameters *and* the forward/backward axon
crossing probabilities (p_fwd, p_bwd) of a two-chamber chip, from the global MEA signature plus the
directional signature. It then answers the questions a chip designer actually asks:

- How precisely can the wiring direction be read off a single recording, and with what uncertainty?
- How does engineered channel asymmetry translate into functional directionality (the design map)?
- How reliable is the answer when the two chambers are only weakly coupled?

Outputs in /kaggle/working: posterior2c.pt, calibration2c.json, design_map.csv, figures/*.png
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
N_TEST = 800
N_CAL = 400

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py

# @@include src/neurotwin/directional.py

# @@include src/neurotwin/infer.py


def load_bank2c():
    """Returns (theta+cross, features, names). The inference prior is the bank's own sampling box."""
    files = sorted(glob.glob(os.path.join(INPUT, "**", "bank2c_shard*.npz"), recursive=True))
    meta = sorted(glob.glob(os.path.join(INPUT, "**", "bank2c_shard*.json"), recursive=True))
    box = {k: tuple(v) for k, v in json.load(open(meta[0]))["base_box"].items()}
    d0 = np.load(files[0])
    base_names = [str(s) for s in d0["param_names"]]
    theta = np.concatenate([np.load(f)["theta"] for f in files])
    cross = np.concatenate([np.load(f)["cross"] for f in files])
    xg = np.concatenate([np.load(f)["x_global"] for f in files])
    xd = np.concatenate([np.load(f)["x_dir"] for f in files])
    names = base_names + ["p_fwd", "p_bwd"]
    full = np.c_[theta, cross]
    x = np.c_[xg, xd]
    global PARAM_NAMES, LO, HI, LOG, Z_LO, Z_HI
    PARAM_NAMES = names
    LO = np.array([box[k][0] for k in base_names] + [0.0, 0.0])
    HI = np.array([box[k][1] for k in base_names] + [1.0, 1.0])
    LOG = np.array([box[k][2] == "log" for k in base_names] + [False, False])
    Z_LO, Z_HI = to_unbounded(LO), to_unbounded(HI)
    ok = np.isfinite(x).all(1)
    feat_names = [str(s) for s in d0["feature_names"]] + [str(s) for s in d0["dir_feature_names"]]
    print(f"two-chamber bank: {len(full)} simulations, {ok.sum()} usable, {x.shape[1]} statistics", flush=True)
    return full[ok], x[ok], feat_names


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    theta, x, feat_names = load_bank2c()
    perm = rng.permutation(len(theta))
    te, tr = perm[:N_TEST], perm[N_TEST:]
    posterior, _ = train_npe(theta[tr], x[tr], density_estimator="nsf", training_batch_size=512,
                             stop_after_epochs=30)
    torch.save(posterior, os.path.join(WORK, "posterior2c.pt"))
    print(f"trained on {len(tr)} simulations in {(time.time() - t0) / 60:.1f} min", flush=True)

    rec = recovery_metrics(posterior, theta[te[:N_CAL]], x[te[:N_CAL]])
    cal = calibration(posterior, theta[te[:N_CAL]], x[te[:N_CAL]])
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**rec, **cal}.items()
               if k not in ("ranks", "tarp_ecp", "tarp_alpha", "post_mean_z")},
              open(os.path.join(WORK, "calibration2c.json"), "w"), indent=1)
    tbl = pd.DataFrame({k: rec[k] for k in ("param", "r2", "nrmse", "coverage_50", "coverage_90", "contraction")})
    print(tbl.round(3).to_string(index=False), flush=True)
    print(f"TARP ATC={cal['tarp_atc']:.3f} KS p={cal['tarp_ks_p']:.3f}", flush=True)

    # How well is the wiring direction recovered, and does the posterior know when it cannot tell?
    i_f, i_b = PARAM_NAMES.index("p_fwd"), PARAM_NAMES.index("p_bwd")
    s, leak = sample_many(posterior, x[te], n=400)
    med = np.median(s, 0)
    width = np.percentile(s, 95, axis=0) - np.percentile(s, 5, axis=0)
    truth = theta[te]
    asym_true = truth[:, i_f] - truth[:, i_b]
    asym_est = med[:, i_f] - med[:, i_b]
    rows = pd.DataFrame({"p_fwd_true": truth[:, i_f], "p_bwd_true": truth[:, i_b],
                         "p_fwd_med": med[:, i_f], "p_bwd_med": med[:, i_b],
                         "p_fwd_width90": width[:, i_f], "p_bwd_width90": width[:, i_b],
                         "asym_true": asym_true, "asym_est": asym_est, "leakage": leak,
                         "coupling": np.maximum(truth[:, i_f], truth[:, i_b])})
    rows.to_csv(os.path.join(WORK, "design_map.csv"), index=False)
    sign_ok = np.sign(asym_est) == np.sign(asym_true)
    strong = np.abs(asym_true) > 0.3
    print(f"\ndirection (sign of p_fwd - p_bwd) correct: {sign_ok.mean():.1%} overall, "
          f"{sign_ok[strong].mean():.1%} when the engineered asymmetry exceeds 0.3", flush=True)
    print(f"asymmetry estimate: R2={1 - ((asym_est - asym_true) ** 2).sum() / ((asym_true - asym_true.mean()) ** 2).sum():.2f}, "
          f"median 90% width for p_fwd {np.median(width[:, i_f]):.2f}", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].scatter(asym_true, asym_est, s=5, alpha=0.4); ax[0].plot([-1, 1], [-1, 1], "k--")
    ax[0].set_xlabel("engineered asymmetry  p_fwd - p_bwd"); ax[0].set_ylabel("inferred asymmetry")
    ax[0].set_title("Reading the wiring direction off one recording")
    sc = ax[1].scatter(truth[:, i_f], truth[:, i_b], c=asym_est, s=6, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set_xlabel("p_fwd (A->B)"); ax[1].set_ylabel("p_bwd (B->A)"); ax[1].set_title("Design map: inferred asymmetry")
    plt.colorbar(sc, ax=ax[1])
    b = np.linspace(0, 1, 11)
    k = np.digitize(rows.coupling, b) - 1
    m = pd.DataFrame({"k": k, "w": width[:, i_f]}).groupby("k").w.median()
    ax[2].plot(b[m.index] + 0.05, m.values, "o-")
    ax[2].set_xlabel("strength of the stronger direction"); ax[2].set_ylabel("median 90% posterior width, p_fwd")
    ax[2].set_title("Uncertainty vs coupling strength")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "two_chamber.png"), dpi=110); plt.close(fig)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
