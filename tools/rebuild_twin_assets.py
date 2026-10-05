"""Rebuild a round-2 training run's twin assets with the whole training bank as the support reference.

nt-train2 stored a 5,000-simulation subsample for the app, but its out-of-support threshold was calibrated on
the whole training bank, and distances to a sparser reference set are larger. This replays nt-train2's split
(neurotwin.infer.rebuild_assets), checks it against the stored feature means and standard deviations, and
writes assets whose reference set and threshold belong together (round 3 writes such assets itself).

    python tools/rebuild_twin_assets.py --bank runs/bank2/s0 --train runs/nt-train2 --out runs/nt-train2-full
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from neurotwin.infer import rebuild_assets  # noqa: E402


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
    old = np.load(glob.glob(os.path.join(a.train, "**", "twin_assets.npz"), recursive=True)[0])
    new = rebuild_assets(old, theta, x, lo, hi)
    if new is None:
        raise SystemExit("the replayed split does not match the stored assets; not writing anything")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "twin_assets.npz"), **new)
    print(f"{len(new['x'])} training simulations (stored subsample: {len(old['x'])}); "
          f"threshold {float(old['threshold']):.3f} -> {os.path.join(a.out, 'twin_assets.npz')}")


if __name__ == "__main__":
    main()
