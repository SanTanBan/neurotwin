"""Interactive NeuroTwin demo.

Load a microelectrode-array recording of a neural chip culture, fit its digital twin in about a second, see
what the fit is and is not sure about, check that the twin reproduces the recording, and then run in-silico
experiments on that twin ("what if inhibition were halved?").

Run locally:   python app/app.py
The assets (a trained posterior and bank summary) are produced by the Kaggle training notebook.
"""
import glob
import json
import os
import sys

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from neurotwin.features import FEATURE_NAMES, compute_features  # noqa: E402
from neurotwin.infer import sample_many  # noqa: E402
from neurotwin.sim import ChipConfig, PARAM_NAMES, simulate  # noqa: E402

ASSETS = os.environ.get("NEUROTWIN_ASSETS", "runs/nt-train2")
WIN_S = 300.0
CFG = ChipConfig()


def _load_assets():
    post = glob.glob(os.path.join(ASSETS, "**", "posterior.pt"), recursive=True)
    bank = glob.glob(os.path.join(ASSETS, "**", "twin_assets.npz"), recursive=True)
    box = glob.glob(os.path.join(ASSETS, "**", "focus_box.json"), recursive=True)
    posterior = torch.load(post[0], weights_only=False) if post else None
    stats = np.load(bank[0]) if bank else None
    names = PARAM_NAMES
    if box:                       # the posterior's parameters and prior box (older models have fewer)
        b = json.load(open(box[0]))
        names = [k for k in PARAM_NAMES if k in b]
        import neurotwin.infer as inf
        import neurotwin.sim as sim
        sim.LO = np.array([b[k][0] for k in names], float)
        sim.HI = np.array([b[k][1] for k in names], float)
        sim.LOG = np.array([b[k][2] == "log" for k in names])
        inf.Z_LO, inf.Z_HI = sim.to_unbounded(sim.LO), sim.to_unbounded(sim.HI)
    return posterior, stats, names


POSTERIOR, BANK, POST_NAMES = _load_assets()


def read_spikes(path):
    """Read (time, electrode) spikes from a .spk.txt[.bz2] or a two-column CSV."""
    comp = "bz2" if str(path).endswith(".bz2") else None
    sep = "," if str(path).endswith(".csv") else r"\s+"
    df = pd.read_csv(path, sep=sep, header=None, usecols=[0, 1], names=["t", "ch"], compression=comp,
                     comment="#")
    df = df[pd.to_numeric(df.ch, errors="coerce").notna()]
    return df.ch.to_numpy(int), df.t.to_numpy(float)


def raster_fig(ch, t, t0, t1, title, color="k"):
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.2), height_ratios=[3, 1], sharex=True)
    m = (t >= t0) & (t < t1)
    ax[0].scatter(t[m] - t0, ch[m], s=0.4, c=color)
    ax[0].set_ylabel("electrode"); ax[0].set_title(title, fontsize=10)
    h, e = np.histogram(t[m] - t0, bins=np.arange(0, t1 - t0, 0.05))
    ax[1].plot(e[:-1], h / 0.05, lw=0.7, c=color)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("pop. rate (Hz)")
    fig.tight_layout()
    return fig


def analyse(file, start_s):
    if file is None:
        return None, None, "Upload a recording (two columns: time in seconds, electrode 0-59)."
    ch, t = read_spikes(file.name if hasattr(file, "name") else file)
    start = float(start_s)
    m = (t >= start) & (t < start + WIN_S)
    x = compute_features(ch[m], t[m] - start, WIN_S)
    tbl = pd.DataFrame({"statistic": FEATURE_NAMES, "value": np.round(x, 3)})
    msg = (f"{len(t):,} spikes on {len(np.unique(ch))} electrodes over {t.max():.0f} s. "
           f"Analysing the {WIN_S:.0f}-s window starting at {start:.0f} s.")
    return raster_fig(ch, t, start, min(start + 60, t.max()), "recording (first 60 s of the window)"), tbl, msg


def _keep(x):
    """Statistics the posterior was trained on (a training run may exclude misspecified ones)."""
    return x[BANK["keep"]] if BANK is not None and "keep" in BANK.files else x


def fit_twin(file, start_s):
    if POSTERIOR is None:
        return None, None, "No trained posterior found: set NEUROTWIN_ASSETS to the training output folder."
    ch, t = read_spikes(file.name if hasattr(file, "name") else file)
    start = float(start_s)
    m = (t >= start) & (t < start + WIN_S)
    x = compute_features(ch[m], t[m] - start, WIN_S)
    samples, leak = sample_many(POSTERIOR, _keep(x)[None], n=500)
    s = samples[:, 0]
    rows = []
    for i, p in enumerate(POST_NAMES):
        q = np.percentile(s[:, i], [5, 50, 95])
        rows.append({"parameter": p, "median": round(q[1], 3), "90% interval": f"{q[0]:.3g} – {q[2]:.3g}"})
    warn = ""
    if BANK is not None:
        z = (_keep(x) - BANK["mu"]) / (BANK["sd"] + 1e-9)
        d = np.linalg.norm(BANK["x"] - z, axis=1)
        oos = np.mean(np.sort(d)[:10]) > float(BANK["threshold"])
        warn = ("\n\n**Reliability: this recording is unlike anything the simulator produces, so the fit is "
                "extrapolation and should not be trusted.**" if oos else
                "\n\nReliability: the recording lies inside the simulated range.")
    if leak[0] > 0.2:
        warn += f"\n\n**{leak[0]:.0%} of the posterior falls outside the parameter ranges — treat with care.**"
    fig, ax = plt.subplots(1, len(POST_NAMES), figsize=(2.0 * len(POST_NAMES), 3))
    for i, a in enumerate(ax):
        a.violinplot(s[:, i], showextrema=False)
        a.set_title(POST_NAMES[i], fontsize=8); a.set_xticks([])
    fig.suptitle("posterior over the twin's biophysical parameters")
    fig.tight_layout()
    return fig, pd.DataFrame(rows), "Twin fitted." + warn


def in_silico(file, start_s, inhibition, drive, synaptic_decay, seed):
    """Run the fitted twin forward, with chosen parameters scaled: an in-silico experiment."""
    if POSTERIOR is None:
        return None, "No trained posterior found."
    ch, t = read_spikes(file.name if hasattr(file, "name") else file)
    start = float(start_s)
    m = (t >= start) & (t < start + WIN_S)
    x = compute_features(ch[m], t[m] - start, WIN_S)
    samples, _ = sample_many(POSTERIOR, _keep(x)[None], n=200)
    theta = np.median(samples[:, 0], axis=0)
    mod = theta.copy()
    for name, factor in (("g_ie", inhibition), ("mu", drive), ("tau_syn", synaptic_decay)):
        if name in POST_NAMES:
            mod[POST_NAMES.index(name)] *= float(factor)
    out = simulate(np.stack([theta, mod]), CFG, duration_s=120.0, warmup_s=10.0, seed=int(seed))
    figs = []
    for (e, tt), name in zip(out, ["fitted twin", "after the in-silico change"]):
        figs.append(raster_fig(e, tt, 0, 60, name, color="C3"))
    xa, xb = compute_features(*out[0], 120.0), compute_features(*out[1], 120.0)
    diff = pd.DataFrame({"statistic": FEATURE_NAMES, "fitted twin": np.round(xa, 2),
                         "after change": np.round(xb, 2), "difference": np.round(xb - xa, 2)})
    return figs[1], diff


with gr.Blocks(title="NeuroTwin") as demo:
    gr.Markdown("# NeuroTwin\nA digital twin of a neural organ-on-a-chip culture, fitted to its own recording.")
    with gr.Row():
        rec = gr.File(label="MEA recording (time, electrode)")
        start = gr.Number(value=0, label="window start (s)")
    with gr.Tab("1. Recording"):
        b1 = gr.Button("Analyse")
        r_fig = gr.Plot(); r_tbl = gr.Dataframe(); r_msg = gr.Markdown()
        b1.click(analyse, [rec, start], [r_fig, r_tbl, r_msg])
    with gr.Tab("2. Fit the twin"):
        b2 = gr.Button("Fit")
        f_fig = gr.Plot(); f_tbl = gr.Dataframe(); f_msg = gr.Markdown()
        b2.click(fit_twin, [rec, start], [f_fig, f_tbl, f_msg])
    with gr.Tab("3. In-silico experiment"):
        with gr.Row():
            inh = gr.Slider(0.25, 4.0, 1.0, label="inhibition x")
            drv = gr.Slider(0.5, 2.0, 1.0, label="background drive x")
            syn = gr.Slider(0.25, 4.0, 1.0, label="synaptic decay x")
            sd = gr.Number(value=0, label="seed")
        b3 = gr.Button("Simulate")
        w_fig = gr.Plot(); w_tbl = gr.Dataframe()
        b3.click(in_silico, [rec, start, inh, drv, syn, sd], [w_fig, w_tbl])

if __name__ == "__main__":
    demo.launch()
