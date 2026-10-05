"""NeuroTwin real-data signatures: developing rat cortical cultures on 60-electrode MEAs.

Downloads the daily spontaneous-activity recordings of dense cultures from Wagenaar, Pine & Potter (2006),
"An extremely rich repertoire of bursting patterns during the development of cortical cultures",
BMC Neuroscience 7:11 (data free for research use with citation of that article). Each recording is cut into
300-s windows and the NeuroTwin electrophysiological signature is computed for every window.
Outputs: wagenaar_dense_features.csv and the raw files of one example culture for demos.
"""
import os
import re
import shutil
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

LIST_URL = "https://potterlab.bme.gatech.edu/development-data/html/wget/daily.spont.dense.text.0.0.0.list"
WIN_S = 300.0
N_WIN = 6
WORK = "/kaggle/working" if (os.name != "nt" and os.path.isdir("/kaggle/working")) else "."
RAW = os.path.join(WORK, "raw_tmp")
KEEP_CULTURE = "2-2-"   # raw files kept in the output for demos

# @@include src/neurotwin/features.py

# @@include src/neurotwin/data.py


def download_all():
    os.makedirs(RAW, exist_ok=True)
    txt = urllib.request.urlopen(LIST_URL, timeout=120).read().decode("utf-8", "replace")
    urls = [urllib.parse.urljoin(LIST_URL, m) for m in re.findall(r"(\S+\.spk\.txt\.bz2)", txt)]
    t0, fails = time.time(), []
    for i, u in enumerate(urls):
        fn = os.path.join(RAW, os.path.basename(u))
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            continue
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(u.replace("http://", "https://"), timeout=180).read()
                with open(fn, "wb") as f:
                    f.write(data)
                break
            except Exception as e:  # retry transient network errors
                if attempt == 2:
                    fails.append((os.path.basename(u), repr(e)[:100]))
                time.sleep(5)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(urls)} downloaded, {time.time() - t0:.0f}s", flush=True)
    print(f"download done: {len(urls)} listed, {len(fails)} failed", fails[:5], flush=True)


def main():
    download_all()
    t0 = time.time()
    df = build_table(RAW, win_s=WIN_S, n_windows=N_WIN)
    df.to_csv(os.path.join(WORK, "wagenaar_dense_features.csv"), index=False)
    print(f"features: {len(df)} windows from {df.file.nunique()} recordings in {time.time() - t0:.0f}s")
    keep = os.path.join(WORK, "raw_example")
    os.makedirs(keep, exist_ok=True)
    for fn in os.listdir(RAW):
        if fn.startswith(KEEP_CULTURE):
            shutil.copy(os.path.join(RAW, fn), keep)
    shutil.rmtree(RAW)
    print(df.groupby("div")[["log_rate", "log_nb_rate", "sync_corr", "log_fano_pop"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
