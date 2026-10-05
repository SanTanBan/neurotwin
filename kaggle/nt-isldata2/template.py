"""NeuroTwin imaging data: rat cortical cultures from the Google in-silico-labeling dataset (CC BY 4.0).

Christiansen et al. (2018), "In silico labeling: predicting fluorescent labels in unlabeled images",
Cell 173:792-803. Condition alicia_2_0 (Finkbeiner lab, rat cortical cultures): a phase-contrast z-stack
with DAPI (nuclei) and propidium iodide (dead cells), plus the published model's own predictions for the
test wells, which serve as the baseline. All 13 focal planes and both stains are block-averaged and stored as
one compact uint16 file per well.
"""
import concurrent.futures as cf
import io
import os
import re
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

BASE = "https://storage.googleapis.com/in-silico-labeling/paper_data/"
COND = "alicia_2_0"
Z_KEEP = list(range(13))
DS = 2
WORK = "/kaggle/working" if (os.name != "nt" and os.path.isdir("/kaggle/working")) else "."
OUT = os.path.join(WORK, "isl_alicia13")


def list_files(split):
    url = BASE + f"{split}_single_channel_images_data_index.html"
    html = urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "replace")
    rows = []
    for u in re.findall(r'href="([^"]+\.png)"', html):
        name = u.rsplit("/", 1)[-1]
        if f"condition-{COND}" not in name:
            continue
        date = "".join(f"{int(v):02d}" for v in re.search(r"year-(\d+),month-(\d+),day-(\d+)", name).groups())
        z = re.search(r"z_depth-(\d+)", name)
        rows.append({"split": split, "key": f"{date}_{re.search(r'well-([A-Z]+\d+)', name).group(1)}",
                     "channel": re.search(r"channel,value-([A-Z_]+)", name).group(1),
                     "z": int(z.group(1)) if z else -1,
                     "kind": "PREDICTED" if "value-PREDICTED" in name else "ORIGINAL",
                     "url": u if u.startswith("http") else BASE + u})
    return pd.DataFrame(rows)


def fetch(url, tries=3):
    for i in range(tries):
        try:
            a = np.array(Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=180).read())))
            h, w = (a.shape[0] // DS) * DS, (a.shape[1] // DS) * DS
            return a[:h, :w].reshape(h // DS, DS, w // DS, DS).mean((1, 3)).round().astype(np.uint16)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5)


def prepare_well(split, key, g):
    ph = g[(g.channel == "PHASE_CONTRAST") & (g.kind == "ORIGINAL") & g.z.isin(Z_KEEP)].sort_values("z")
    arrays = {"phase": np.stack([fetch(u) for u in ph.url]), "z": ph.z.to_numpy()}
    for ch, name in (("DAPI_CONFOCAL", "dapi"), ("DEAD_CONFOCAL", "dead")):
        for kind, suffix in (("ORIGINAL", ""), ("PREDICTED", "_isl")):
            row = g[(g.channel == ch) & (g.kind == kind)]
            if len(row):
                arrays[name + suffix] = fetch(row.url.iloc[0])
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, split, f"{key}.npz"), **arrays)
    return {"split": split, "key": key, "n_z": len(ph), "shape": arrays["phase"].shape[1:],
            "has_isl_pred": "dapi_isl" in arrays, "has_dead": "dead" in arrays}


def main():
    t0 = time.time()
    df = pd.concat([list_files("train"), list_files("test")], ignore_index=True)
    print(df.groupby(["split", "channel", "kind"]).size().to_string(), flush=True)
    jobs = [(s, k, g) for (s, k), g in df.groupby(["split", "key"])]
    rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(lambda j: prepare_well(*j), jobs)):
            rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} wells, {time.time() - t0:.0f}s", flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(OUT, "index.csv"), index=False)
    print(idx.groupby("split").agg(n=("key", "size"), with_isl_pred=("has_isl_pred", "sum")).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
