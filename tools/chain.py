"""Wait for Kaggle notebooks to finish, then push the next stages in order and fetch their outputs.

    python tools/chain.py --wait owner/a,owner/b --grace-hours 2.5 \
                          --push kaggle/nt-train2:runs/nt-train2 kaggle/nt-demo:runs/nt-demo

The first --wait notebook is the anchor: once it reaches a terminal state, the others get `grace-hours`
to finish too. With --drop-missing-sources, any kernel_source that is not COMPLETE is removed from a
notebook's metadata before pushing, so one failed bank cannot block the whole chain.
"""
import argparse
import json
import time
from pathlib import Path

from kgl import fetch, kaggle, status, wait

TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def is_terminal(slug):
    s = status(slug)
    return any(k in s for k in TERMINAL), s


def is_complete(slug):
    return "COMPLETE" in status(slug)


def wait_for(slugs, grace_hours, poll=300):
    anchor = slugs[0]
    while True:
        done, s = is_terminal(anchor)
        print(time.strftime("%H:%M"), anchor, s, flush=True)
        if done:
            break
        time.sleep(poll)
    deadline = time.time() + grace_hours * 3600
    for other in slugs[1:]:
        while time.time() < deadline:
            done, s = is_terminal(other)
            print(time.strftime("%H:%M"), other, s, flush=True)
            if done:
                break
            time.sleep(poll)


def push_stage(folder, out, drop_missing, retries=12, retry_wait=600):
    meta_path = Path(folder) / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if drop_missing and meta.get("kernel_sources"):
        keep = [k for k in meta["kernel_sources"] if is_complete(k)]
        if keep != meta["kernel_sources"]:
            print(f"dropping unavailable sources: {set(meta['kernel_sources']) - set(keep)}", flush=True)
            meta["kernel_sources"] = keep
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for _ in range(retries):
        msg = kaggle("kernels", "push", "-p", folder).splitlines()[-1]
        print(time.strftime("%H:%M"), msg, flush=True)
        if "successfully pushed" in msg:
            break
        time.sleep(retry_wait)
    else:
        print(f"could not push {folder}", flush=True)
        return False
    print("FINAL:", wait(meta["id"]), flush=True)
    print(fetch(meta["id"], out)[-5000:], flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", default="", help="comma-separated slugs to wait for (first is the anchor)")
    ap.add_argument("--grace-hours", type=float, default=2.0)
    ap.add_argument("--push", nargs="+", help="folder:output_dir stages, pushed in order")
    ap.add_argument("--drop-missing-sources", action="store_true")
    a = ap.parse_args()
    if a.wait:
        wait_for(a.wait.split(","), a.grace_hours)
    for stage in a.push:
        folder, out = stage.split(":")
        if not push_stage(folder, out, a.drop_missing_sources):
            break


if __name__ == "__main__":
    main()
