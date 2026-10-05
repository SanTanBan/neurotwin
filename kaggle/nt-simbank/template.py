"""NeuroTwin simulation bank (one shard).

Samples twin parameters from the prior, simulates virtual MEA cultures and stores (theta, features).
Several shards run as separate Kaggle CPU sessions; SHARD selects a disjoint random stream.
Results are saved after every batch, and the run stops cleanly before the session time limit.
"""
import json
import os
import subprocess
import sys
import time

try:
    import numba  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"], check=True)

import numpy as np

SHARD = 0
DURATION_S = 300.0
WARMUP_S = 10.0
BATCH = 64
TIME_BUDGET_S = 11.0 * 3600
MAX_SIMS = 100_000
OUT = f"/kaggle/working/bank_shard{SHARD}.npz" if (os.name != "nt" and os.path.isdir("/kaggle/working")) else f"bank_shard{SHARD}.npz"

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py


def main():
    t0 = time.time()
    rng = np.random.default_rng(1000 + SHARD)
    cfg = ChipConfig()
    thetas, feats = [], []
    batch_times = []
    while len(thetas) * BATCH < MAX_SIMS:
        tb = time.time()
        theta = sample_prior(BATCH, rng)
        out = simulate(theta, cfg, duration_s=DURATION_S, warmup_s=WARMUP_S, seed=int(rng.integers(2**31 - 1)))
        X = np.array([compute_features(e, t, DURATION_S) for e, t in out])
        thetas.append(theta); feats.append(X)
        batch_times.append(time.time() - tb)
        np.savez_compressed(OUT, theta=np.concatenate(thetas), x=np.concatenate(feats),
                            param_names=np.array(PARAM_NAMES), feature_names=np.array(FEATURE_NAMES))
        n = len(thetas) * BATCH
        print(f"[shard {SHARD}] {n} sims, last batch {batch_times[-1]:.0f}s, elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 + 1.5 * max(batch_times) > TIME_BUDGET_S:
            break
    meta = {"shard": SHARD, "n_sims": len(thetas) * BATCH, "duration_s": DURATION_S, "warmup_s": WARMUP_S,
            "prior": {k: list(v) for k, v in PRIOR_BOUNDS.items()}, "chip": vars(cfg),
            "elapsed_min": (time.time() - t0) / 60}
    with open(OUT.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("done", meta)


if __name__ == "__main__":
    main()
