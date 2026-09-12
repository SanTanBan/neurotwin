"""NeuroTwin round-2 simulation bank (one shard): model v3, prior focused on real cultures.

Round 1 showed that 76% of real recording windows lie outside the support of the broad prior, because
strongly bursting cultures occupy only a small part of it, and that instantaneous synapses cannot produce
the long, fast-rising bursts of mature cultures. Round 2 therefore uses model v3 (exponentially decaying
synaptic currents) and samples from a box centred on the region that best matched real mature recordings.
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
OUT = f"/kaggle/working/bank2_shard{SHARD}.npz" if os.path.isdir("/kaggle/working") else f"bank2_shard{SHARD}.npz"

# Focused prior: 10-90% range of the simulations closest to real mature windows, widened for safety.
FOCUS_BOX = {
    "j_e": (4.0, 20.0, "log"), "g_ie": (0.3, 2.5, "log"), "u_std": (0.04, 0.5, "log"),
    "tau_rec": (0.2, 6.0, "log"), "b_adapt": (0.0, 1.2, "lin"), "tau_adapt": (0.7, 25.0, "log"),
    "mu": (1.0, 11.0, "lin"), "mu_sd": (0.5, 4.0, "lin"), "sigma": (1.5, 5.5, "lin"),
    "lambda_um": (120.0, 2000.0, "log"), "tau_syn": (2.0, 150.0, "log"),
}

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py


def sample_focus(n, rng):
    cols = []
    for name in PARAM_NAMES:
        lo, hi, scale = FOCUS_BOX[name]
        u = rng.random(n)
        cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if scale == "log" else lo + u * (hi - lo))
    return np.stack(cols, 1)


def main():
    t0 = time.time()
    rng = np.random.default_rng(9000 + SHARD)
    cfg = ChipConfig()
    thetas, feats, batch_times = [], [], []
    while len(thetas) * BATCH < MAX_SIMS:
        tb = time.time()
        theta = sample_focus(BATCH, rng)
        out = simulate(theta, cfg, duration_s=DURATION_S, warmup_s=WARMUP_S, seed=int(rng.integers(2**31 - 1)))
        feats.append(np.array([compute_features(e, t, DURATION_S) for e, t in out]))
        thetas.append(theta)
        batch_times.append(time.time() - tb)
        np.savez_compressed(OUT, theta=np.concatenate(thetas), x=np.concatenate(feats),
                            param_names=np.array(PARAM_NAMES), feature_names=np.array(FEATURE_NAMES))
        print(f"[round2 shard {SHARD}] {len(thetas) * BATCH} sims, last batch {batch_times[-1]:.0f}s, "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if time.time() - t0 + 1.5 * max(batch_times) > TIME_BUDGET_S:
            break
    meta = {"shard": SHARD, "n_sims": len(thetas) * BATCH, "duration_s": DURATION_S, "warmup_s": WARMUP_S,
            "focus_box": FOCUS_BOX, "chip": vars(cfg), "elapsed_min": (time.time() - t0) / 60}
    with open(OUT.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("done", meta)


if __name__ == "__main__":
    main()
