"""Rebuild a round-2 training run's twin assets with the whole training bank as the support reference.

nt-train2 stored a 5,000-simulation subsample for the app, but its out-of-support threshold was calibrated on
the whole training bank, and distances to a sparser reference set are larger. This replays nt-train2's split
(same filter, numpy seed 0), checks the result against the stored feature means and standard deviations, and
writes assets whose reference set and threshold belong together (round 3 writes such assets itself).

    python tools/rebuild_twin_assets.py --bank runs/bank2/s0 --train runs/nt-train2 --out runs/nt-train2-full
"""
import argparse
import glob
import json
import os

import numpy as np

N_TEST = 1000          # as in kaggle/nt-train2/template.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/bank2/s0")
    ap.add_argument("--train", default="runs/nt-train2")
    ap.add_argument("--out", default="runs/nt-train2-full")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.bank, "**", "bank2_shard*.npz"), recursive=True))
    box = json.load(open(glob.glob(os.path.join(a.train, "**", "focus_box.json"), recursive=True)[0]))
    lo = np.array([v[0] for v in box.values()]); hi = np.array([v[1] for v in box.values()])
    theta = np.concatenate([np.load(f)["theta"] for f in files])
    x = np.concatenate([np.load(f)["x"] for f in files])
    ok = np.isfinite(x).all(1) & np.all((theta >= lo - 1e-9) & (theta <= hi + 1e-9), axis=1)
    theta, x = theta[ok], x[ok]
    tr = np.random.default_rng(0).permutation(len(theta))[N_TEST:]
    old = np.load(glob.glob(os.path.join(a.train, "**", "twin_assets.npz"), recursive=True)[0])
    mu, sd = x[tr].mean(0), x[tr].std(0)
    if not (np.allclose(mu, old["mu"], rtol=1e-5, atol=1e-6) and np.allclose(sd, old["sd"], rtol=1e-5, atol=1e-6)):
        raise SystemExit("the replayed split does not match the stored assets; not writing anything")
    os.makedirs(a.out, exist_ok=True)
    out = {k: old[k] for k in old.files}
    out.update(x=((x[tr] - mu) / (sd + 1e-9)).astype(np.float32), threshold_ref=old["threshold"])
    np.savez_compressed(os.path.join(a.out, "twin_assets.npz"), **out)
    print(f"{len(tr)} training simulations (stored subsample: {len(old['x'])}); threshold {float(old['threshold']):.3f}"
          f" -> {os.path.join(a.out, 'twin_assets.npz')}")


if __name__ == "__main__":
    main()
