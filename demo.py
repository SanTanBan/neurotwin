"""NeuroTwin entry script: fit a digital twin to one MEA recording and run two in-silico experiments.

    python demo.py                                  # culture 2-2 at DIV 20, minutes 20-25 (the report's example)
    python demo.py --recording 2-2-14 --start 0     # any recording of the public Wagenaar et al. (2006) dataset
    python demo.py --file my_chip.txt --start 0     # your own recording: two columns, time (s) and electrode 0-59

It runs kaggle/nt-demo/nt_demo.py, the exact script of the Kaggle notebook nt-demo, in a working folder with
the round-3 posterior shipped in assets/round3/. Dataset recordings are downloaded from the authors' archive on
first use. Outputs in --out (default runs/demo): demo_rasters.png, demo_comparison.csv and demo_draws.csv.
About 10 minutes on a 4-core laptop; only CPU is needed.
"""
import argparse
import bz2
import os
import re
import runpy
import shutil
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_URL = "https://potterlab.bme.gatech.edu/development-data/html/wget/daily.spont.dense.text.0.0.0.list"
ASSETS = ("posterior.pt", "twin_assets.npz", "focus_box.json")


def fetch_recording(name, dest):
    """Download one recording (plating-culture-DIV, e.g. 2-2-20) of the Wagenaar et al. (2006) dataset."""
    listing = urllib.request.urlopen(LIST_URL, timeout=120).read().decode("utf-8", "replace")
    urls = [urllib.parse.urljoin(LIST_URL, m) for m in re.findall(r"(\S+\.spk\.txt\.bz2)", listing)]
    hits = [u for u in urls if os.path.basename(u) == f"{name}.spk.txt.bz2"]
    if not hits:
        raise SystemExit(f"recording {name} is not in the dataset's file list")
    data = urllib.request.urlopen(hits[0].replace("http://", "https://"), timeout=300).read()
    with open(dest, "wb") as f:
        f.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", default="2-2-20", help="dataset recording as plating-culture-DIV")
    ap.add_argument("--file", help="your own spike file (time in s, electrode 0-59), used instead of --recording")
    ap.add_argument("--start", type=float, default=1200.0, help="start (s) of the 300-s window to fit")
    ap.add_argument("--assets", default=os.path.join(HERE, "assets", "round3"), help="trained posterior and assets")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "demo"), help="working and output folder")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for f in ASSETS:
        shutil.copy(os.path.join(a.assets, f), out)
    name = re.sub(r"\.(spk\.)?(txt|csv)(\.bz2)?$", "", os.path.basename(a.file)) if a.file else a.recording
    target = os.path.join(out, f"{name}.spk.txt.bz2")
    if a.file:                                           # the demo reads bz2-compressed text
        raw = open(a.file, "rb").read()
        with open(target, "wb") as f:
            f.write(raw if a.file.endswith(".bz2") else bz2.compress(raw))
    elif not os.path.exists(target):
        print(f"downloading {name} ...", flush=True)
        fetch_recording(name, target)

    os.environ["NT_RECORDING"], os.environ["NT_START_S"] = name, str(a.start)
    os.chdir(out)                                        # the notebook reads inputs from and writes to "."
    runpy.run_path(os.path.join(HERE, "kaggle", "nt-demo", "nt_demo.py"), run_name="__main__")
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
