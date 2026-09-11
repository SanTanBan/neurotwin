"""NeuroTwin label-free imaging: uncertainty-aware virtual staining of rat cortical cultures.

Trains U-Nets that predict nuclei (DAPI) and dead-cell (propidium iodide) stains from a phase-contrast
z-stack, each pixel with a predictive variance (heteroscedastic Gaussian NLL); several seeds form a deep
ensemble when a GPU is available. The predictive std is recalibrated on validation wells with one scale
factor. Evaluation uses the held-out test wells of Christiansen et al. (2018) and compares against the real
stains and against the published in-silico-labeling predictions for the same wells, at pixel level
(Pearson r) and on experimenter-level readouts (nuclei count, dead-cell fraction) with interval coverage.
Data: Google in-silico-labeling dataset, CC BY 4.0.
"""
import glob
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
ITERS = 6000 if DEVICE == "cuda" else 2500
BATCH = 16 if DEVICE == "cuda" else 8
CROP = 256
LR = 1e-3
N_VAL = 6
TARGETS = ("dapi", "dead")

# @@include src/neurotwin/vstain.py


def load_well(path):
    d = np.load(path)
    w = {"key": os.path.basename(path)[:-4],
         "phase": np.stack([percentile_norm(p.astype(np.float32)) for p in d["phase"]])}
    for k in ("dapi", "dead", "dapi_isl", "dead_isl"):
        if k in d.files:
            w[k] = percentile_norm(d[k].astype(np.float32))       # Pearson r is scale-free; norm for training
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
        if it % 250 == 0 or it == ITERS:
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
    return means.mean(0), np.sqrt(np.stack(vars_).mean(0) + means.var(0))   # aleatoric + epistemic


def main():
    t0 = time.time()
    root = glob.glob(os.path.join(INPUT, "**", "isl_alicia"), recursive=True)[0]
    wells = [load_well(p) for p in sorted(glob.glob(os.path.join(root, "train", "*.npz")))]
    test = [load_well(p) for p in sorted(glob.glob(os.path.join(root, "test", "*.npz")))]
    wells = [w for w in wells if all(t in w for t in TARGETS)]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(wells))
    val, train = [wells[i] for i in perm[:N_VAL]], [wells[i] for i in perm[N_VAL:]]
    print(f"{len(train)} train / {len(val)} val / {len(test)} test wells on {DEVICE}", flush=True)
    models = [train_one(train, val, seed) for seed in range(N_MODELS)]
    torch.save([m.state_dict() for m in models], os.path.join(WORK, "vstain_ensemble.pt"))

    # one-parameter recalibration of the predictive std on validation wells (target: 90% coverage)
    errs, sds = [], []
    for w in val:
        mu, sd = ensemble_predict(models, w["phase"])
        y = np.stack([w[t] for t in TARGETS])
        errs.append(np.abs(y - mu).ravel()); sds.append(sd.ravel())
    ratio = np.concatenate(errs) / (np.concatenate(sds) + 1e-9)
    scale = float(np.quantile(ratio, 0.90) / 1.645)
    print(f"std recalibration factor: {scale:.3f}", flush=True)

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
        true_p = count_nuclei(w["dapi"])
        r["count_true"], r["dead_true"] = len(true_p), dead_fraction(true_p, w["dead"])
        ro = readouts_with_intervals(mu, sd)
        r.update({f"{k}_ours": v for k, v in ro.items()})
        if "dapi_isl" in w:
            isl_p = count_nuclei(w["dapi_isl"])
            r["count_isl"], r["dead_isl"] = len(isl_p), dead_fraction(isl_p, w["dead_isl"])
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
    df.to_csv(os.path.join(WORK, "vstain_test_metrics.csv"), index=False)
    summ = {"n_test": len(df), "std_scale": scale, "device": DEVICE, "n_models": N_MODELS, "iters": ITERS,
            "minutes": (time.time() - t0) / 60}
    for col in df.columns:
        if col != "key":
            summ[col + "_mean"] = float(df[col].mean())
    if "count_isl" in df:
        for who in ("ours", "isl"):
            col = "count_ours" if who == "ours" else "count_isl"
            summ[f"count_mape_{who}"] = float((np.abs(df[col] - df.count_true) / df.count_true.clip(lower=1)).mean())
    summ["count_interval_coverage"] = float(((df.count_true >= df.count_lo_ours) & (df.count_true <= df.count_hi_ours)).mean())
    json.dump(summ, open(os.path.join(WORK, "vstain_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
