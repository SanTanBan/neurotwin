"""NeuroTwin label-free imaging, second attempt: GPU ensemble on all 13 focal planes, calibrated readouts.

The first CPU run (one model, 5 focal planes) reached pixel correlations of 0.81 (nuclei) and 0.71 (dead
cells) against 0.92 / 0.85 for the published in-silico-labeling model, and, more seriously, its nuclei counts
were 89% too high because blurrier predictions create spurious peaks that per-pixel uncertainty does not
capture. This run changes three things:

1. all 13 phase-contrast focal planes as input, a 3-seed ensemble and longer training (GPU);
2. the readout itself is calibrated on validation wells: the nuclei detector's smoothing, threshold and
   spacing, and the dead-cell threshold, are chosen so that counts on *predicted* images match counts on the
   real stains of validation wells;
3. readout intervals come from the validation residuals (split-conformal style) instead of pixel noise.

Evaluation on the 8 official test wells against the real stains and the published model's predictions.
Data: Google in-silico-labeling dataset (Christiansen et al. 2018), CC BY 4.0.
"""
import glob
import itertools
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
INPUT = "/kaggle/input" if os.path.isdir("/kaggle/input") else "."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MODELS = 3 if DEVICE == "cuda" else 1
ITERS = 8000 if DEVICE == "cuda" else 2500
BATCH = 16 if DEVICE == "cuda" else 8
CROP = 256
LR = 1e-3
N_VAL = 8
TARGETS = ("dapi", "dead")
DATA_DIR = "isl_alicia13"

# @@include src/neurotwin/vstain.py


def load_well(path):
    d = np.load(path)
    w = {"key": os.path.basename(path)[:-4],
         "phase": np.stack([percentile_norm(p.astype(np.float32)) for p in d["phase"]])}
    for k in ("dapi", "dead", "dapi_isl", "dead_isl"):
        if k in d.files:
            w[k] = percentile_norm(d[k].astype(np.float32))
    return w


def random_batch(wells, rng):
    xs, ys = [], []
    for _ in range(BATCH):
        w = wells[rng.integers(len(wells))]
        H, W = w["phase"].shape[1:]
        y0, x0 = rng.integers(0, H - CROP), rng.integers(0, W - CROP)
        x = w["phase"][:, y0:y0 + CROP, x0:x0 + CROP]
        y = np.stack([w[t][y0:y0 + CROP, x0:x0 + CROP] for t in TARGETS])
        k = rng.integers(4)
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        xs.append(x.copy()); ys.append(y.copy())
    return torch.tensor(np.stack(xs), device=DEVICE), torch.tensor(np.stack(ys), device=DEVICE)


def train_one(train, val, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet(train[0]["phase"].shape[0], len(TARGETS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=ITERS, pct_start=0.1)
    vx, vy = random_batch(val, np.random.default_rng(999))
    best, best_state, t0 = np.inf, None, time.time()
    for it in range(1, ITERS + 1):
        model.train()
        x, y = random_batch(train, rng)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            m, lv = model(x)
        loss = nll_loss(m.float(), lv.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == ITERS:
            model.eval()
            with torch.no_grad():
                vm, vlv = model(vx)
                vloss = nll_loss(vm.float(), vlv.float(), vy).item()
            if vloss < best:
                best, best_state = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[seed {seed}] it {it} train {loss.item():.3f} val {vloss:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model


def ensemble_predict(models, phase):
    means, vars_ = [], []
    for m in models:
        mu, sd = predict_tiled(m, phase, tile=512, overlap=64, device=DEVICE)
        means.append(mu); vars_.append(sd ** 2)
    means = np.stack(means)
    return means.mean(0), np.sqrt(np.stack(vars_).mean(0) + means.var(0))


def readout(dapi, dead, p):
    peaks = count_nuclei(dapi, sigma=p["sigma"], min_dist=p["min_dist"],
                         thr=nuclei_threshold(dapi, sigma=p["sigma"], k_mad=p["k_mad"]))
    return len(peaks), dead_fraction(peaks, dead, rel_thresh=p["dead_rel"])


DEFAULT = {"sigma": 2.0, "k_mad": 5.0, "min_dist": 5, "dead_rel": 0.3}


def calibrate_readout(val_pred, val_true):
    """Choose detector settings so readouts on predicted images match readouts on the real stains."""
    truth = [readout(w["dapi"], w["dead"], DEFAULT) for w in val_true]
    grid = [dict(zip(("sigma", "k_mad", "min_dist", "dead_rel"), v))
            for v in itertools.product([1.5, 2.0, 3.0, 4.0], [3.0, 5.0, 8.0, 12.0], [4, 5, 7, 9], [0.2, 0.3, 0.4, 0.5])]
    best, best_err = DEFAULT, np.inf
    for p in grid:
        est = [readout(mu[0], mu[1], p) for mu in val_pred]
        err = np.mean([abs(e[0] - t[0]) / max(t[0], 1) for e, t in zip(est, truth)]) + \
            np.mean([abs(e[1] - t[1]) for e, t in zip(est, truth)])
        if err < best_err:
            best, best_err = p, err
    est = [readout(mu[0], mu[1], best) for mu in val_pred]
    rel_res = np.array([(e[0] - t[0]) / max(t[0], 1) for e, t in zip(est, truth)])
    dead_res = np.array([e[1] - t[1] for e, t in zip(est, truth)])
    print(f"calibrated readout {best} (validation error {best_err:.3f}); "
          f"count residuals {np.round(rel_res, 3)}", flush=True)
    return best, rel_res, dead_res


def main():
    t0 = time.time()
    root = glob.glob(os.path.join(INPUT, "**", DATA_DIR), recursive=True)[0]
    wells = [load_well(p) for p in sorted(glob.glob(os.path.join(root, "train", "*.npz")))]
    test = [load_well(p) for p in sorted(glob.glob(os.path.join(root, "test", "*.npz")))]
    wells = [w for w in wells if all(t in w for t in TARGETS)]
    perm = np.random.default_rng(0).permutation(len(wells))
    val, train = [wells[i] for i in perm[:N_VAL]], [wells[i] for i in perm[N_VAL:]]
    print(f"{len(train)} train / {len(val)} val / {len(test)} test wells, {train[0]['phase'].shape[0]} focal "
          f"planes, {N_MODELS} models x {ITERS} iterations on {DEVICE}", flush=True)
    models = [train_one(train, val, seed) for seed in range(N_MODELS)]
    torch.save([m.state_dict() for m in models], os.path.join(WORK, "vstain2_ensemble.pt"))

    val_pred, errs, sds = [], [], []
    for w in val:
        mu, sd = ensemble_predict(models, w["phase"])
        val_pred.append(mu)
        y = np.stack([w[t] for t in TARGETS])
        errs.append(np.abs(y - mu).ravel()); sds.append(sd.ravel())
    scale = float(np.quantile(np.concatenate(errs) / (np.concatenate(sds) + 1e-9), 0.90) / 1.645)
    params, rel_res, dead_res = calibrate_readout(val_pred, val)
    q_count = float(np.quantile(np.abs(rel_res), 0.9))
    q_dead = float(np.quantile(np.abs(dead_res), 0.9))
    json.dump({"readout": params, "std_scale": scale, "q_count_rel": q_count, "q_dead_abs": q_dead},
              open(os.path.join(WORK, "readout_calibration.json"), "w"), indent=1)

    rows = []
    os.makedirs(os.path.join(WORK, "figures"), exist_ok=True)
    for i, w in enumerate(test):
        mu, sd = ensemble_predict(models, w["phase"])
        sd = sd * scale
        r = {"key": w["key"]}
        for c, t in enumerate(TARGETS):
            r[f"{t}_r_ours"] = pearson(mu[c], w[t])
            r[f"{t}_cov90"] = interval_coverage(w[t], mu[c], sd[c])
            if f"{t}_isl" in w:
                r[f"{t}_r_isl"] = pearson(w[f"{t}_isl"], w[t])
        r["count_true"], r["dead_true"] = readout(w["dapi"], w["dead"], DEFAULT)
        r["count_ours_default"], r["dead_ours_default"] = readout(mu[0], mu[1], DEFAULT)
        r["count_ours"], r["dead_ours"] = readout(mu[0], mu[1], params)
        r["count_lo"], r["count_hi"] = r["count_ours"] * (1 - q_count), r["count_ours"] * (1 + q_count)
        r["dead_lo"], r["dead_hi"] = r["dead_ours"] - q_dead, r["dead_ours"] + q_dead
        if "dapi_isl" in w:
            r["count_isl"], r["dead_isl"] = readout(w["dapi_isl"], w["dead_isl"], DEFAULT)
        rows.append(r)
        print(r, flush=True)
        if i < 3:
            fig, ax = plt.subplots(2, 5, figsize=(20, 8))
            mid = w["phase"][w["phase"].shape[0] // 2]
            for c, t in enumerate(TARGETS):
                panels = [(mid, "phase (mid focus)", "gray"), (w[t], f"real {t}", "magma"),
                          (mu[c], f"NeuroTwin {t} (r={r[f'{t}_r_ours']:.2f})", "magma"),
                          (w.get(f"{t}_isl", np.zeros_like(mid)), f"ISL-2018 {t} (r={r.get(f'{t}_r_isl', np.nan):.2f})", "magma"),
                          (sd[c], f"predictive std {t}", "viridis")]
                for a, (img, title, cmap) in zip(ax[c], panels):
                    a.imshow(img[:600, :600], cmap=cmap); a.set_title(title, fontsize=10); a.axis("off")
            fig.tight_layout(); fig.savefig(os.path.join(WORK, "figures", f"test_{w['key']}.png"), dpi=80); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK, "vstain2_test_metrics.csv"), index=False)
    mape = lambda col: float((np.abs(df[col] - df.count_true) / df.count_true.clip(lower=1)).mean())
    summ = {"n_test": len(df), "device": DEVICE, "n_models": N_MODELS, "iters": ITERS,
            "planes": int(train[0]["phase"].shape[0]), "minutes": (time.time() - t0) / 60,
            "dapi_r_ours": float(df.dapi_r_ours.mean()), "dapi_r_isl": float(df.dapi_r_isl.mean()),
            "dead_r_ours": float(df.dead_r_ours.mean()), "dead_r_isl": float(df.dead_r_isl.mean()),
            "dapi_cov90": float(df.dapi_cov90.mean()), "dead_cov90": float(df.dead_cov90.mean()),
            "count_mape_ours_default": mape("count_ours_default"), "count_mape_ours": mape("count_ours"),
            "count_mape_isl": mape("count_isl"),
            "dead_mae_ours": float((df.dead_ours - df.dead_true).abs().mean()),
            "dead_mae_isl": float((df.dead_isl - df.dead_true).abs().mean()),
            "count_interval_coverage": float(((df.count_true >= df.count_lo) & (df.count_true <= df.count_hi)).mean()),
            "dead_interval_coverage": float(((df.dead_true >= df.dead_lo) & (df.dead_true <= df.dead_hi)).mean())}
    json.dump(summ, open(os.path.join(WORK, "vstain2_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
