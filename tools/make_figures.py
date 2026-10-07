"""Figures for the report and the video, built from finished runs.

    python tools/make_figures.py [--train runs/nt-train2]

Writes to runs/figs/: ppc_own_vs_mismatched.png, maturation.png, architecture.png, title.png, closing.png,
and prints the statistics quoted in the report (including a per-culture sign test for maturation trends).
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = "runs/figs"
TRENDS = [("j_e_med", "excitatory synaptic\nstrength (mV)"), ("tau_syn_med", "synaptic current\ndecay (ms)"),
          ("mu_med", "background\ndrive (mV)"), ("mu_sd_med", "excitability\nspread (mV)"),
          ("b_adapt_med", "adaptation\nincrement (mV)"), ("tau_adapt_med", "adaptation\ntime constant (s)")]


def ppc_figure(train):
    p = pd.read_csv(os.path.join(train, "ppc.csv")).sort_values("div")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(p))
    ax[0].vlines(x, p.own_abs_z, p.mismatched_abs_z, color="0.75", lw=1.5)
    ax[0].scatter(x, p.mismatched_abs_z, color="0.45", s=40, label="twin of a different recording")
    ax[0].scatter(x, p.own_abs_z, color="C3", s=40, label="its own twin", zorder=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(p["div"], fontsize=8)
    ax[0].set_xlabel("recording (days in vitro)"); ax[0].set_ylabel("median |z| of the 16 statistics")
    ax[0].set_title("Each twin reproduces its own recording"); ax[0].legend(frameon=False)
    ax[1].bar(["own twin", "different\nrecording's twin"], [p.own_inside90.mean(), p.mismatched_inside90.mean()],
              color=["C3", "0.55"])
    ax[1].axhline(0.9, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("statistics inside the 90% predictive interval"); ax[1].set_ylim(0, 1)
    ax[1].set_title(f"Coverage {p.own_inside90.mean():.2f} vs {p.mismatched_inside90.mean():.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ppc_own_vs_mismatched.png"), dpi=110); plt.close(fig)
    print(f"PPC: own coverage {p.own_inside90.mean():.2f} vs mismatched {p.mismatched_inside90.mean():.2f}; "
          f"median |z| {p.own_abs_z.median():.2f} vs {p.mismatched_abs_z.median():.2f}; "
          f"own better in {(p.own_abs_z < p.mismatched_abs_z).mean():.0%} of {len(p)} recordings")


def maturation_figure(train, trends=TRENDS, rows=1):
    t = pd.read_csv(os.path.join(train, "trajectories.csv"))
    cols = -(-len(trends) // rows)
    fig, ax = plt.subplots(rows, cols, figsize=(17 if rows == 1 else 4.2 * cols, 3.6 * rows), squeeze=False)
    ax = ax.ravel()
    print(f"maturation: {t.groupby(['plating', 'culture']).ngroups} cultures, {len(t)} culture-days")
    for a, (col, label) in zip(ax, trends):
        for _, g in t.groupby(["plating", "culture"]):
            g = g.sort_values("div")
            a.plot(g["div"], g[col], color="0.8", lw=0.8)
        b = t.groupby(pd.cut(t["div"], [0, 8, 14, 21, 28, 40]), observed=True)[col]
        mid = [i.mid for i in b.median().index]
        a.plot(mid, b.median().values, "o-", color="C3", lw=2)
        a.fill_between(mid, b.quantile(0.25).values, b.quantile(0.75).values, color="C3", alpha=0.15)
        a.set_title(label, fontsize=10); a.set_xlabel("days in vitro")
        # per-culture test: within each culture with >= 3 reliable days, does the parameter rise with age?
        signs = []
        for _, g in t.groupby(["plating", "culture"]):
            if len(g) >= 3 and g["div"].nunique() >= 3:
                signs.append(np.sign(np.corrcoef(g["div"], g[col])[0, 1]))
        signs = [s for s in signs if s != 0]
        up = sum(s > 0 for s in signs)
        pv = binomtest(up, len(signs), 0.5).pvalue if signs else np.nan
        print(f"  {col:15s} cultures rising {up}/{len(signs)}  (sign test p={pv:.3g})")
    fig.suptitle("How the twins change as cultures mature (windows that pass every reliability check)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "maturation.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


def box(ax, xy, w, h, text, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="0.3", lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=11)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color="0.3", lw=1.4))


def architecture_figure():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis("off")
    box(ax, (0.3, 4.6), 3.0, 1.6, "MEA recording\n60 electrodes,\nspike times", "#dbe9f6")
    box(ax, (4.2, 4.6), 3.0, 1.6, "Electrophysiological\nsignature\n16 statistics", "#dbe9f6")
    box(ax, (0.3, 1.0), 3.0, 1.9, "Chip simulator\nLIF network, depression,\nadaptation, fast + slow\nsynapses", "#f6e3d3")
    box(ax, (4.2, 1.0), 3.0, 1.9, "Bank of tens of\nthousands of\nvirtual cultures", "#f6e3d3")
    box(ax, (8.2, 2.8), 3.0, 1.9, "Amortised posterior\n(neural spline flow)\nfit in < 1 s", "#e4f1dc")
    box(ax, (12.3, 4.7), 3.4, 1.6, "Reliability flags\ncalibration, support,\nleakage, predictive checks", "#fbe7ea")
    box(ax, (12.3, 2.6), 3.4, 1.6, "Mechanistic parameters\nwith 90% intervals", "#fbe7ea")
    box(ax, (12.3, 0.5), 3.4, 1.6, "In-silico experiments\n(e.g. halve inhibition)", "#fbe7ea")
    arrow(ax, (3.3, 5.4), (4.2, 5.4)); arrow(ax, (3.3, 1.95), (4.2, 1.95))
    arrow(ax, (7.2, 5.4), (8.2, 4.1)); arrow(ax, (7.2, 1.95), (8.2, 3.3))
    arrow(ax, (11.2, 4.1), (12.3, 5.4)); arrow(ax, (11.2, 3.75), (12.3, 3.4)); arrow(ax, (11.2, 3.2), (12.3, 1.3))
    ax.text(8, 6.7, "NeuroTwin: from a chip recording to a calibrated, mechanistic digital twin",
            ha="center", fontsize=15, weight="bold")
    fig.savefig(os.path.join(FIG, "architecture.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


def slide(name, title, lines):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off"); fig.patch.set_facecolor("white")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=40, weight="bold", transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(0.5, 0.45 - 0.07 * i, line, ha="center", va="center", fontsize=20, transform=ax.transAxes,
                color="0.25")
    fig.savefig(os.path.join(FIG, name), dpi=120); plt.close(fig)


def imaging_figure(run="runs/nt-vstain2-v2"):
    """Readouts on the 8 test wells: nuclei count and dead fraction with our 90% intervals, against the truth and
    the published in-silico-labeling model, plus per-well pixel correlation."""
    path = os.path.join(run, "vstain2_test_metrics.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, (true, ours, lo, hi, isl, lab) in zip(ax[:2], [
            ("count_true", "count_ours", "count_lo", "count_hi", "count_isl", "nuclei per well"),
            ("dead_true", "dead_ours", "dead_lo", "dead_hi", "dead_isl", "dead-cell fraction")]):
        a.errorbar(x - 0.15, d[ours], yerr=[d[ours] - d[lo], d[hi] - d[ours]], fmt="o", color="C3", capsize=3,
                   label="NeuroTwin, with 90% interval")
        a.scatter(x + 0.15, d[isl], color="0.55", marker="s", label="published model (2018)", zorder=3)
        a.scatter(x, d[true], color="k", marker="_", s=300, label="real stain", zorder=4)
        a.set_xticks(x); a.set_xticklabels([k.split("_")[-1] for k in d["key"]]); a.set_xlabel("test well")
        a.set_ylabel(lab); a.legend(frameon=False, fontsize=9)
    err = lambda col: (d[col] - d.count_true).abs().div(d.count_true).mean()  # noqa: E731
    ax[0].set_title(f"Nuclei count: mean error {err('count_ours'):.1%} (published {err('count_isl'):.1%})")
    ax[1].set_title(f"Dead-cell fraction: mean error {(d.dead_ours - d.dead_true).abs().mean():.3f} "
                    f"(published {(d.dead_isl - d.dead_true).abs().mean():.3f})")
    for col, lab, c in (("dapi", "nuclei (DAPI)", "C0"), ("dead", "dead cells (PI)", "C1")):
        ax[2].scatter(d[f"{col}_r_isl"], d[f"{col}_r_ours"], color=c, label=lab)
    ax[2].plot([0.6, 1], [0.6, 1], "k--", lw=0.8)
    ax[2].set_xlim(0.6, 1); ax[2].set_ylim(0.6, 1)
    ax[2].set_xlabel("pixel correlation, published model"); ax[2].set_ylabel("pixel correlation, NeuroTwin")
    ax[2].set_title("Pixel fidelity: below the published model"); ax[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "imaging_readouts.png"), dpi=110); plt.close(fig)


def card_image(demo="docs/figures/demo_in_silico.png", size=(16, 9), name="card.png"):
    """Card and thumbnail for the Kaggle writeup (1600 x 900): title on the left, a real culture and its twin
    (the first two raster columns of the worked example) on the right."""
    if not os.path.exists(demo):
        return
    from PIL import Image
    im = Image.open(demo)
    rasters = im.crop((0, 0, int(im.width * 0.485), int(im.height * 0.6)))
    fig = plt.figure(figsize=size, dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.70, "NeuroTwin", fontsize=64, weight="bold", color="#1a1a1a")
    fig.text(0.04, 0.58, "Calibrated digital twins of\nneural organ-on-a-chip cultures", fontsize=27,
             color="#333333", va="top", linespacing=1.3)
    fig.text(0.04, 0.33, "recording in  →  mechanistic twin out\nwith checks that say when to trust it",
             fontsize=19, color="#555555", va="top", linespacing=1.4)
    fig.text(0.04, 0.10, "End-to-End System  ·  AI4S Open Innovation: AI for Life Science", fontsize=15,
             color="#777777")
    ax = fig.add_axes([0.43, 0.08, 0.55, 0.84])
    ax.imshow(rasters); ax.axis("off")
    fig.savefig(os.path.join(FIG, name), dpi=100); plt.close(fig)


def video_slides():
    """Text slides for the video, and the top (raster) half of the worked-example figure."""
    slide("simulator.png", "The chip simulator", ["400 model neurons on the real 60-electrode layout",
                                                  "synaptic depression  ·  spike-frequency adaptation",
                                                  "fast and slow synaptic currents  ·  noisy, uneven drive",
                                                  "about 10 CPU-seconds per 5-minute culture"])
    slide("diagnosis.png", "Diagnosis, then a fix", ["Round 1: 3 in 4 real windows outside the simulator's range",
                                                     "real network bursts last about 2 s, simulated ones 0.4 s",
                                                     "fix 1: synaptic currents with realistic time courses",
                                                     "fix 2: simulate where real cultures live"])
    slide("public.png", "Open and reproducible", ["code: github.com/SanTanBan/neurotwin  (Apache-2.0)",
                                                  "notebooks: kaggle.com/santanubanerjee9  (nt-*)",
                                                  "data: Wagenaar et al. 2006; Christiansen et al. 2018",
                                                  "app: fit a twin to your own recording"])
    demo = "docs/figures/demo_in_silico.png"
    if os.path.exists(demo):
        from PIL import Image
        im = Image.open(demo)
        im.crop((0, 0, im.width, int(im.height * 0.6))).save(os.path.join(FIG, "demo_top.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/nt-train2")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    ppc_figure(a.train)
    maturation_figure(a.train)
    architecture_figure()
    slide("title.png", "NeuroTwin", ["Calibrated digital twins of neural organ-on-a-chip cultures",
                                     "AI4S Open Innovation: AI for Life Science  |  End-to-End System"])
    slide("closing.png", "From description to prediction", ["for neural organs-on-chips, with uncertainty you can check",
                                                             "code and notebooks: github.com/SanTanBan/neurotwin"])
    imaging_figure()
    video_slides()
    card_image()
    card_image(size=(16, 8), name="card_2x1.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
