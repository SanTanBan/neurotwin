"""Minimal Kaggle kernel runner: push a kernel folder, wait for it, fetch outputs and a clean log.

Usage:
    python tools/kgl.py run   kaggle/<kernel_dir>      # push + wait + fetch
    python tools/kgl.py push  kaggle/<kernel_dir>
    python tools/kgl.py wait  owner/slug               # wait + fetch
    python tools/kgl.py fetch owner/slug [--out DIR]
"""
import argparse
import json
import os
import subprocess
import time

ENV = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=ENV)
    return (r.stdout + r.stderr).strip()


def status(slug):
    return kaggle("kernels", "status", slug).splitlines()[-1]


def wait(slug, poll=20, timeout=12 * 3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = status(slug)
        if any(k in s for k in TERMINAL):
            return s
        time.sleep(poll)
    return "TIMEOUT"


def fetch(slug, out, pattern=None):
    os.makedirs(out, exist_ok=True)
    kaggle("kernels", "output", slug, "-p", out, *(["--file-pattern", pattern] if pattern else []))
    log = os.path.join(out, slug.split("/")[1] + ".log")
    if not os.path.exists(log):
        return "(no log found)"
    raw = open(log, encoding="utf-8", errors="replace").read()
    try:                                        # Kaggle logs are JSON, but a partial fetch is plain text
        entries = json.loads(raw)
        text = "".join(e.get("data", "") for e in entries if "SyntaxWarning" not in e.get("data", ""))
    except (json.JSONDecodeError, AttributeError, TypeError):
        text = raw
    open(os.path.join(out, "log.txt"), "w", encoding="utf-8").write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "wait", "fetch", "run"])
    ap.add_argument("target", help="kernel folder (push/run) or owner/slug (wait/fetch)")
    ap.add_argument("--out", default=None, help="output dir (default runs/<slug>)")
    ap.add_argument("--tail", type=int, default=4000, help="chars of log to print")
    a = ap.parse_args()

    if a.cmd in ("push", "run"):
        slug = json.load(open(os.path.join(a.target, "kernel-metadata.json")))["id"]
        print(kaggle("kernels", "push", "-p", a.target).splitlines()[-1])
        if a.cmd == "push":
            return
    else:
        slug = a.target
    if a.cmd in ("wait", "run"):
        print("FINAL:", wait(slug))
    out = a.out or os.path.join("runs", slug.split("/")[1])
    print(fetch(slug, out)[-a.tail:])


if __name__ == "__main__":
    main()
