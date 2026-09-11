"""Inline project modules into a single self-contained Kaggle script.

A kernel template marks inclusion points with lines like
    # @@include src/neurotwin/sim.py
Each marker is replaced by the module source with package-relative imports removed (the included modules
then share one namespace). Usage:
    python tools/build_kernel.py kaggle/<kernel_dir>/template.py kaggle/<kernel_dir>/<script>.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL_IMPORT = re.compile(r"^from \.[\w.]* import .*$|^from \.[\w.]* import \($[\s\S]*?^\)$", re.M)


def build(template, out):
    text = Path(template).read_text(encoding="utf-8")

    def inline(m):
        src = (ROOT / m.group(1).strip()).read_text(encoding="utf-8")
        return f"# ---- begin {m.group(1).strip()} ----\n{REL_IMPORT.sub('', src)}\n# ---- end {m.group(1).strip()} ----"

    Path(out).write_text(re.sub(r"^# @@include (.+)$", inline, text, flags=re.M), encoding="utf-8")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
    print("built", sys.argv[2])
