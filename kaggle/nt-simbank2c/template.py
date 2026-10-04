"""NeuroTwin two-chamber (axon-diode) simulation bank (one shard).

Virtual two-chamber chips: somata in two chambers separated by a microchannel barrier, axons crossing with
forward/backward probabilities (p_fwd: A->B, p_bwd: B->A) and a 5-ms delay. Base parameters are drawn from a
box covering realistic bursting cultures (found by prior-predictive exploration against real recordings);
crossing probabilities are uniform on [0, 1]. For every culture both the global MEA signature and the
directional signature are stored, so a posterior over (base parameters, p_fwd, p_bwd) can be trained.
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
OUT = f"/kaggle/working/bank2c_shard{SHARD}.npz" if os.path.isdir("/kaggle/working") else f"bank2c_shard{SHARD}.npz"

BASE_BOX = {
    "j_e": (6.0, 20.0, "log"), "g_ie": (0.3, 1.5, "log"), "u_std": (0.05, 0.4, "log"),
    "tau_rec": (0.5, 5.0, "log"), "b_adapt": (0.0, 2.0, "lin"), "tau_adapt": (1.0, 20.0, "log"),
    "mu": (3.0, 10.0, "lin"), "mu_sd": (0.0, 4.0, "lin"), "sigma": (2.0, 5.0, "lin"),
    "lambda_um": (300.0, 2000.0, "log"),
}

# @@include src/neurotwin/sim.py

# @@include src/neurotwin/features.py

# @@include src/neurotwin/directional.py


def sample_base(n, rng):
    """The ten model-v2 parameters; simulate() runs later mechanisms at their neutral (v2-exact) setting."""
    cols = []
    for name in BASE_BOX:
        lo, hi, scale = BASE_BOX[name]
        u = rng.random(n)
        cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if scale == "log" else lo + u * (hi - lo))
    return np.stack(cols, 1)


def main():
    t0 = time.time()
    rng = np.random.default_rng(5000 + SHARD)
    cfg = ChipConfig(compartments=2)
    chamber = cfg.chamber_of_electrodes()
    thetas, crosses, xg, xd, times = [], [], [], [], []
    while len(thetas) * BATCH < MAX_SIMS:
        tb = time.time()
        theta, cross = sample_base(BATCH, rng), rng.random((BATCH, 2))
        out = simulate(theta, cfg, duration_s=DURATION_S, warmup_s=WARMUP_S, seed=int(rng.integers(2**31 - 1)),
                       cross=cross)
        xg.append(np.array([compute_features(e, t, DURATION_S) for e, t in out]))
        xd.append(np.array([directional_features(e, t, DURATION_S, chamber) for e, t in out]))
        thetas.append(theta); crosses.append(cross)
        times.append(time.time() - tb)
        np.savez_compressed(OUT, theta=np.concatenate(thetas), cross=np.concatenate(crosses),
                            x_global=np.concatenate(xg), x_dir=np.concatenate(xd),
                            param_names=np.array(list(BASE_BOX)), feature_names=np.array(FEATURE_NAMES),
                            dir_feature_names=np.array(DIR_FEATURE_NAMES))
        print(f"[2c shard {SHARD}] {len(thetas) * BATCH} sims, last batch {times[-1]:.0f}s, "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if time.time() - t0 + 1.5 * max(times) > TIME_BUDGET_S:
            break
    meta = {"shard": SHARD, "n_sims": len(thetas) * BATCH, "duration_s": DURATION_S, "base_box": BASE_BOX,
            "chip": vars(cfg), "elapsed_min": (time.time() - t0) / 60}
    with open(OUT.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("done", meta)


if __name__ == "__main__":
    main()
