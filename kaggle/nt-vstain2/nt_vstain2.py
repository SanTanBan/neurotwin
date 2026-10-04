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
LR = 5e-4
N_VAL = 8
TARGETS = ("dapi", "dead")
DATA_DIR = "isl_alicia13"

# ---- begin src/neurotwin/vstain.py ----
"""Label-free virtual staining with calibrated uncertainty.

A U-Net maps a stack of transmitted-light focal planes (phase contrast) of a neural culture to fluorescence
channels (nuclei = DAPI, dead cells = propidium iodide). Every output pixel is a Gaussian: the network
predicts a mean and a log-variance and is trained with the heteroscedastic negative log-likelihood, so it
reports where it is unsure. Readouts that matter to an experimenter - nuclei count and dead-cell fraction -
are computed from the prediction, with intervals obtained by sampling the per-pixel predictive distribution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class UNet(nn.Module):
    """Plain U-Net; outputs 2 maps per target channel: mean and log-variance."""

    def __init__(self, in_ch, out_ch, base=32, depth=4, dropout=0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([conv_block(in_ch, chs[0])] +
                                  [conv_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(chs[0], 2 * out_ch, 1)
        self.out_ch = out_ch

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.drop(skips.pop())
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        out = self.head(x)
        mean, logvar = out[:, :self.out_ch], out[:, self.out_ch:].clamp(-10, 6)
        return mean, logvar


def nll_loss(mean, logvar, target):
    """Heteroscedastic Gaussian negative log-likelihood (per pixel, averaged)."""
    return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar)).mean()


def percentile_norm(img, lo=1.0, hi=99.9, ref=None):
    """Scale an image to ~[0, 1] using percentiles (of `ref` if given, e.g. of a whole well)."""
    p1, p2 = np.percentile(img if ref is None else ref, [lo, hi])
    return ((img - p1) / max(p2 - p1, 1e-6)).astype(np.float32)


@torch.no_grad()
def predict_tiled(model, stack, tile=512, overlap=64, device="cpu"):
    """Predict mean and std for a whole image [C, H, W] with overlapping tiles (cosine blending)."""
    model.eval()
    C, H0, W0 = stack.shape
    H, W = max(tile, H0), max(tile, W0)             # images smaller than a tile are reflect-padded
    if (H, W) != (H0, W0):
        stack = np.pad(stack, ((0, 0), (0, H - H0), (0, W - W0)), mode="reflect")
    step = tile - overlap
    w1 = np.hanning(tile + 2)[1:-1]
    win = torch.tensor(np.outer(w1, w1), dtype=torch.float32, device=device) + 1e-3
    acc_m = torch.zeros((model.out_ch, H, W), device=device)
    acc_v = torch.zeros_like(acc_m)
    acc_w = torch.zeros((H, W), device=device)
    x = torch.as_tensor(stack, dtype=torch.float32, device=device)
    ys = list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if H > tile and (H - tile) % step else [])
    xs = list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if W > tile and (W - tile) % step else [])
    for y0 in ys:
        for x0 in xs:
            m, lv = model(x[None, :, y0:y0 + tile, x0:x0 + tile])
            acc_m[:, y0:y0 + tile, x0:x0 + tile] += m[0] * win
            acc_v[:, y0:y0 + tile, x0:x0 + tile] += torch.exp(lv[0]) * win
            acc_w[y0:y0 + tile, x0:x0 + tile] += win
    mean, std = (acc_m / acc_w).cpu().numpy(), np.sqrt((acc_v / acc_w).cpu().numpy())
    return mean[:, :H0, :W0], std[:, :H0, :W0]


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def interval_coverage(target, mean, std, z=1.645):
    """Fraction of pixels whose true value falls inside mean +- z*std (0.90 expected for z=1.645)."""
    return float((np.abs(target - mean) <= z * std).mean())


def nuclei_threshold(dapi, sigma=2.0, k_mad=5.0):
    """Detection threshold from the image's own background: max(Otsu, median + k * robust noise)."""
    from skimage.filters import threshold_otsu
    s = ndimage.gaussian_filter(dapi, sigma)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return float(max(threshold_otsu(s), med + k_mad * mad))


def count_nuclei(dapi, sigma=2.0, min_dist=5, thr=None):
    """Nuclei = local maxima of the smoothed nuclear stain above a background-derived threshold.

    Pass `thr` to reuse one threshold across images (e.g. across draws of a predictive distribution).
    Returns the (y, x) peak coordinates.
    """
    s = ndimage.gaussian_filter(dapi, sigma)
    if thr is None:
        thr = nuclei_threshold(dapi, sigma)
    mx = ndimage.maximum_filter(s, size=2 * min_dist + 1)
    return np.argwhere((s == mx) & (s > thr))


def dead_fraction(peaks, dead, rel_thresh=0.3, radius=3):
    """Fraction of nuclei whose dead-cell stain (propidium iodide) is bright around the nucleus centre."""
    if len(peaks) == 0:
        return 0.0
    s = ndimage.uniform_filter(dead, 2 * radius + 1)
    thr = rel_thresh * np.percentile(s, 99.5)
    return float((s[peaks[:, 0], peaks[:, 1]] > thr).mean())


def readouts_with_intervals(mean, std, n_draws=20, rng=0):
    """Nuclei count and dead fraction from the predicted stains, with 90% intervals from pixel sampling.

    Draws add spatially smooth noise (the per-pixel std modulates a smoothed white-noise field), so the
    interval reflects uncertainty at the scale of a nucleus rather than independent pixel noise. The
    detection threshold is fixed from the mean prediction so every draw is scored the same way; the reported
    estimate is the median over draws, which keeps it inside its own interval.
    """
    g = np.random.default_rng(rng)
    thr = nuclei_threshold(mean[0])
    counts, fracs = [], []
    for _ in range(n_draws):
        eps = ndimage.gaussian_filter(g.standard_normal(mean.shape[1:]), 2.0)
        eps /= eps.std() + 1e-9
        d = mean + std * eps[None]
        p = count_nuclei(d[0], thr=thr)
        counts.append(len(p)); fracs.append(dead_fraction(p, d[1]))
    return {"count": float(np.median(counts)), "count_lo": float(np.percentile(counts, 5)),
            "count_hi": float(np.percentile(counts, 95)), "dead_frac": float(np.median(fracs)),
            "dead_lo": float(np.percentile(fracs, 5)), "dead_hi": float(np.percentile(fracs, 95))}

# ---- end src/neurotwin/vstain.py ----


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
        m, lv = model(x)                       # full precision: fp16 overflowed in the variance term
        loss = nll_loss(m, lv, y)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if it % 500 == 0 or it == ITERS:
            model.eval()
            with torch.no_grad():
                vm, vlv = model(vx)
                vloss = nll_loss(vm.float(), vlv.float(), vy).item()
            if vloss < best:
                best, best_state = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[seed {seed}] it {it} train {loss.item():.3f} val {vloss:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, best


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
    # counts and dead fraction are calibrated separately, so a poor dead-cell channel cannot drag the
    # nuclei detector away from its own optimum
    best, best_err = dict(DEFAULT), np.inf
    for p in [g for g in grid if g["dead_rel"] == DEFAULT["dead_rel"]]:
        est = [readout(mu[0], mu[1], p) for mu in val_pred]
        err = np.mean([abs(e[0] - t[0]) / max(t[0], 1) for e, t in zip(est, truth)])
        if err < best_err:
            best, best_err = dict(p), err
    dead_err = np.inf
    for rel in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        q = dict(best, dead_rel=rel)
        e = np.mean([abs(readout(mu[0], mu[1], q)[1] - t[1]) for mu, t in zip(val_pred, truth)])
        if e < dead_err:
            dead_err, best = e, q
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
    trained = [train_one(train, val, seed) for seed in range(N_MODELS)]
    best_val = min(b for _, b in trained)
    models = [m for m, b in trained if np.isfinite(b) and b <= best_val + 0.5]   # drop diverged members
    print(f"ensemble: kept {len(models)} of {len(trained)} models (best validation NLLs "
          f"{[round(float(b), 3) for _, b in trained]})", flush=True)
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
