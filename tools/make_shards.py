"""Create per-shard Kaggle kernel folders from a built script (the SHARD constant is substituted).

Usage:
    python tools/make_shards.py kaggle/nt-simbank/nt_simbank.py kaggle/nt-simbank/shards 3
Each shard folder can then be pushed with `python tools/kgl.py push <folder>`.
"""
import json
import re
import sys
from pathlib import Path

OWNER = "santanubanerjee9"


def make_shards(script, out_root, n, name="nt-simbank"):
    src = Path(script).read_text(encoding="utf-8")
    code_file = Path(script).name
    folders = []
    for k in range(n):
        d = Path(out_root) / f"s{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / code_file).write_text(re.sub(r"^SHARD = \d+", f"SHARD = {k}", src, flags=re.M), encoding="utf-8")
        meta = {
            "id": f"{OWNER}/{name}-s{k}", "title": f"{name.replace('-', ' ')} s{k}",
            "code_file": code_file, "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
            "machine_shape": "", "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        folders.append(str(d))
    return folders


if __name__ == "__main__":
    print("\n".join(make_shards(sys.argv[1], sys.argv[2], int(sys.argv[3]))))
