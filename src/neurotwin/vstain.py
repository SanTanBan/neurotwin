"""Label-free virtual staining with calibrated uncertainty.

A U-Net maps a stack of transmitted-light focal planes (phase contrast) of a neural culture to fluorescence
channels (nuclei = DAPI, dead cells = propidium iodide). Every output pixel is a Gaussian: the network
predicts a mean and a log-variance and is trained with the heteroscedastic negative log-likelihood, so it
reports where it is unsure. Readouts that matter to an experimenter - nuclei count and dead-cell fraction -
are computed from the prediction, with intervals obtained by sampling the per-pixel predictive distribution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class UNet(nn.Module):
    """Plain U-Net; outputs 2 maps per target channel: mean and log-variance."""

    def __init__(self, in_ch, out_ch, base=32, depth=4, dropout=0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([conv_block(in_ch, chs[0])] +
                                  [conv_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(chs[0], 2 * out_ch, 1)
        self.out_ch = out_ch

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.drop(skips.pop())
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        out = self.head(x)
        mean, logvar = out[:, :self.out_ch], out[:, self.out_ch:].clamp(-10, 6)
        return mean, logvar


def nll_loss(mean, logvar, target):
    """Heteroscedastic Gaussian negative log-likelihood (per pixel, averaged)."""
    return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar)).mean()


def percentile_norm(img, lo=1.0, hi=99.9, ref=None):
    """Scale an image to ~[0, 1] using percentiles (of `ref` if given, e.g. of a whole well)."""
    p1, p2 = np.percentile(img if ref is None else ref, [lo, hi])
    return ((img - p1) / max(p2 - p1, 1e-6)).astype(np.float32)


@torch.no_grad()
def predict_tiled(model, stack, tile=512, overlap=64, device="cpu"):
    """Predict mean and std for a whole image [C, H, W] with overlapping tiles (cosine blending)."""
    model.eval()
    C, H0, W0 = stack.shape
    H, W = max(tile, H0), max(tile, W0)             # images smaller than a tile are reflect-padded
    if (H, W) != (H0, W0):
        stack = np.pad(stack, ((0, 0), (0, H - H0), (0, W - W0)), mode="reflect")
    step = tile - overlap
    w1 = np.hanning(tile + 2)[1:-1]
    win = torch.tensor(np.outer(w1, w1), dtype=torch.float32, device=device) + 1e-3
    acc_m = torch.zeros((model.out_ch, H, W), device=device)
    acc_v = torch.zeros_like(acc_m)
    acc_w = torch.zeros((H, W), device=device)
    x = torch.as_tensor(stack, dtype=torch.float32, device=device)
    ys = list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if H > tile and (H - tile) % step else [])
    xs = list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if W > tile and (W - tile) % step else [])
    for y0 in ys:
        for x0 in xs:
            m, lv = model(x[None, :, y0:y0 + tile, x0:x0 + tile])
            acc_m[:, y0:y0 + tile, x0:x0 + tile] += m[0] * win
            acc_v[:, y0:y0 + tile, x0:x0 + tile] += torch.exp(lv[0]) * win
            acc_w[y0:y0 + tile, x0:x0 + tile] += win
    mean, std = (acc_m / acc_w).cpu().numpy(), np.sqrt((acc_v / acc_w).cpu().numpy())
    return mean[:, :H0, :W0], std[:, :H0, :W0]


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def interval_coverage(target, mean, std, z=1.645):
    """Fraction of pixels whose true value falls inside mean +- z*std (0.90 expected for z=1.645)."""
    return float((np.abs(target - mean) <= z * std).mean())


def nuclei_threshold(dapi, sigma=2.0, k_mad=5.0):
    """Detection threshold from the image's own background: max(Otsu, median + k * robust noise)."""
    from skimage.filters import threshold_otsu
    s = ndimage.gaussian_filter(dapi, sigma)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return float(max(threshold_otsu(s), med + k_mad * mad))


def count_nuclei(dapi, sigma=2.0, min_dist=5, thr=None):
    """Nuclei = local maxima of the smoothed nuclear stain above a background-derived threshold.

    Pass `thr` to reuse one threshold across images (e.g. across draws of a predictive distribution).
    Returns the (y, x) peak coordinates.
    """
    s = ndimage.gaussian_filter(dapi, sigma)
    if thr is None:
        thr = nuclei_threshold(dapi, sigma)
    mx = ndimage.maximum_filter(s, size=2 * min_dist + 1)
    return np.argwhere((s == mx) & (s > thr))


def dead_fraction(peaks, dead, rel_thresh=0.3, radius=3):
    """Fraction of nuclei whose dead-cell stain (propidium iodide) is bright around the nucleus centre."""
    if len(peaks) == 0:
        return 0.0
    s = ndimage.uniform_filter(dead, 2 * radius + 1)
    thr = rel_thresh * np.percentile(s, 99.5)
    return float((s[peaks[:, 0], peaks[:, 1]] > thr).mean())


def readouts_with_intervals(mean, std, n_draws=20, rng=0):
    """Nuclei count and dead fraction from the predicted stains, with 90% intervals from pixel sampling.

    Draws add spatially smooth noise (the per-pixel std modulates a smoothed white-noise field), so the
    interval reflects uncertainty at the scale of a nucleus rather than independent pixel noise. The
    detection threshold is fixed from the mean prediction so every draw is scored the same way; the reported
    estimate is the median over draws, which keeps it inside its own interval.
    """
    g = np.random.default_rng(rng)
    thr = nuclei_threshold(mean[0])
    counts, fracs = [], []
    for _ in range(n_draws):
        eps = ndimage.gaussian_filter(g.standard_normal(mean.shape[1:]), 2.0)
        eps /= eps.std() + 1e-9
        d = mean + std * eps[None]
        p = count_nuclei(d[0], thr=thr)
        counts.append(len(p)); fracs.append(dead_fraction(p, d[1]))
    return {"count": float(np.median(counts)), "count_lo": float(np.percentile(counts, 5)),
            "count_hi": float(np.percentile(counts, 95)), "dead_frac": float(np.median(fracs)),
            "dead_lo": float(np.percentile(fracs, 5)), "dead_hi": float(np.percentile(fracs, 95))}
