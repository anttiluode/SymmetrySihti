"""The Sihti sieve: purify an image step by step and keep everything that bleeds out.

x_0 = r_0 + r_1 + ... + r_{K-1} + core     (exact, for any purifier)

Two purifiers:
  'lattice' : heat flow on the plain pixel grid (Gaussian blur, sigma = sqrt(2 t)).
              Persistence = scale. Fast.
  'owneq'   : heat flow on the image's OWN colour-affinity graph (neighbours coupled
              by Lab similarity, weights fixed from the first frame of the call).
              Persistence = grouping: object layout survives in the core, texture /
              shading inside objects bleeds into the residues.
Both flows are symmetric and conserve mass, so every residue band has zero mean
per channel (up to boundary handling for 'lattice').
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def octave_depths(dmax):
    d, out = 1, [0]
    while d <= dmax:
        out.append(d); d *= 2
    return out


def _srgb_to_lab(rgb):
    c = np.clip(rgb, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1]); b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def owneq_weights(x, sigma=12.0):
    """Edge weights of the image's own colour graph. wx couples (i,j)-(i,j+1); wy couples (i,j)-(i+1,j)."""
    lab = _srgb_to_lab(x if x.ndim == 3 else np.repeat(x[..., None], 3, -1))
    wx = np.exp(-((lab[:, 1:] - lab[:, :-1]) ** 2).sum(-1) / sigma ** 2)
    wy = np.exp(-((lab[1:] - lab[:-1]) ** 2).sum(-1) / sigma ** 2)
    return wx, wy


def _owneq_step(x, wx, wy, tau=0.24):
    """One explicit heat step on the weighted 4-neighbour graph (zero flux at the border).
    Weights are symmetric, so total mass is conserved exactly; tau<=0.25 keeps it stable."""
    fx = (x[:, 1:] - x[:, :-1]) * (wx[..., None] if x.ndim == 3 else wx)
    fy = (x[1:] - x[:-1]) * (wy[..., None] if x.ndim == 3 else wy)
    d = np.zeros_like(x)
    d[:, :-1] += fx; d[:, 1:] -= fx
    d[:-1] += fy; d[1:] -= fy
    return x + tau * d


def sieve(x, purifier='lattice', dmax=64, sigma=12.0):
    """Return (bands, depths). bands = [r_0, ..., r_{K-1}, core], fine to coarse."""
    x = np.asarray(x, dtype=np.float64)
    depths = octave_depths(dmax)
    snaps = []
    if purifier == 'lattice':
        for d in depths:
            if d == 0: snaps.append(x.copy()); continue
            s = np.sqrt(2.0 * d)
            snaps.append(gaussian_filter(x, sigma=(s, s, 0) if x.ndim == 3 else s, mode='reflect'))
    elif purifier == 'owneq':
        wx, wy = owneq_weights(x, sigma)
        cur, t = x.copy(), 0
        for d in depths:
            while t < d:
                cur = _owneq_step(cur, wx, wy); t += 1
            snaps.append(cur.copy())
    else:
        raise ValueError(purifier)
    bands = [snaps[k] - snaps[k + 1] for k in range(len(snaps) - 1)] + [snaps[-1]]
    return bands, depths


def reconstruct(bands):
    out = np.zeros_like(bands[0])
    for b in bands: out = out + b
    return out
