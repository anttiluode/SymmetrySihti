"""Noether Sihti: the purifier flow written as a conservation law.

The purifier is heat flow on a weighted 4-neighbour graph (plain lattice: all weights 1;
own-EQ: weights from the image's own Lab colour differences). Its symmetry is "add a constant
to a connected component", so the conserved charge is the colour mass of each component.
Locally that is a continuity equation:

    x_{t+1} = x_t - div J_t,       J_t(edge i->j) = tau * w_ij * (x_i - x_j)

and therefore, for the Sihti band between depths d_k and d_{k+1},

    r_k = x_{d_k} - x_{d_{k+1}} = div( sum_{t in window} J_t )                (exact)

Every residue is the divergence of the current that flowed during its time window: the
residue stack is a record of where the charge went. Energy has its own exact ledger:

    ||f_0||^2 = sum_k E_k + ||f_core||^2,   E_k = ||f_{d_k}||^2 - ||f_{d_{k+1}}||^2 >= 0,   f = x - mean

(E_k >= 0 because the step operator S = I - tau L has spectrum in [-1, 1]).
"""
import numpy as np
from .sieve import octave_depths, owneq_weights


def graph_weights(x, purifier, sigma=12.0):
    H, W = x.shape[:2]
    if purifier == 'lattice': return np.ones((H, W - 1)), np.ones((H - 1, W))
    if purifier == 'owneq': return owneq_weights(x, sigma)
    raise ValueError(purifier)


def divergence(jx, jy):
    """Net outflow at each pixel for currents on horizontal (jx) and vertical (jy) edges."""
    H, W = jy.shape[0] + 1, jx.shape[1] + 1
    d = np.zeros((H, W) + jx.shape[2:])
    d[:, :-1] += jx; d[:, 1:] -= jx
    d[:-1] += jy; d[1:] -= jy
    return d


def noether_sieve(x, purifier='owneq', dmax=64, sigma=12.0, tau=0.24):
    """Returns dict with bands (Sihti residues + core), per-band integrated currents (jx, jy),
    energy ledger, weights and depths."""
    x = np.asarray(x, float); wx, wy = graph_weights(x, purifier, sigma)
    wxe, wye = (wx[..., None], wy[..., None]) if x.ndim == 3 else (wx, wy)
    depths = octave_depths(dmax); cur = x.copy(); t = 0
    snaps, JX, JY = [x.copy()], [], []
    for d in depths[1:]:
        jx_acc = np.zeros(x.shape[:1] + (x.shape[1] - 1,) + x.shape[2:])
        jy_acc = np.zeros((x.shape[0] - 1,) + x.shape[1:])
        while t < d:
            jx = tau * wxe * (cur[:, :-1] - cur[:, 1:])          # flow left -> right
            jy = tau * wye * (cur[:-1] - cur[1:])                # flow top -> bottom
            cur = cur - divergence(jx, jy); jx_acc += jx; jy_acc += jy; t += 1
        snaps.append(cur.copy()); JX.append(jx_acc); JY.append(jy_acc)
    bands = [snaps[k] - snaps[k + 1] for k in range(len(snaps) - 1)] + [snaps[-1]]
    # The total colour (the global conserved charge) never dissipates, and since S is symmetric
    # with S 1 = 1 it decouples exactly: ||x_t||^2 = ||mean||^2 + ||x_t - mean||^2.
    # So the ledger is reported for the fluctuation x - mean, which is what actually flows.
    mean = x.mean(axis=(0, 1))
    norms = np.array([float(((s - mean) ** 2).sum()) for s in snaps])
    charge_share = float((mean ** 2).sum() * x.shape[0] * x.shape[1] / (x ** 2).sum())
    return dict(bands=bands, jx=JX, jy=JY, depths=depths, wx=wx, wy=wy, charge_share=charge_share,
                energy=np.append(norms[:-1] - norms[1:], norms[-1]) / norms[0])


def boundary_current_share(res, x, sigma=12.0, w_edge=0.2):
    """Per band: share of the total |current| that crosses strong colour edges
    (edges whose OWN-EQ weight < w_edge), whichever purifier was used.
    Low share = the band's charge stays inside regions (near-conserved per object)."""
    owx, owy = owneq_weights(np.asarray(x, float), sigma)
    out = []
    for jx, jy in zip(res['jx'], res['jy']):
        ax, ay = np.abs(jx).sum(-1) if jx.ndim == 3 else np.abs(jx), np.abs(jy).sum(-1) if jy.ndim == 3 else np.abs(jy)
        tot = ax.sum() + ay.sum()
        out.append(float((ax[owx < w_edge].sum() + ay[owy < w_edge].sum()) / tot) if tot > 0 else 0.0)
    return np.array(out)


def current_image(jx, jy):
    """Visualise an integrated current as colour: hue = direction, brightness = magnitude."""
    from matplotlib.colors import hsv_to_rgb
    if jx.ndim == 3: jx, jy = jx.sum(-1), jy.sum(-1)
    H, W = jy.shape[0] + 1, jx.shape[1] + 1
    vx = np.zeros((H, W)); vy = np.zeros((H, W))
    vx[:, :-1] += jx / 2; vx[:, 1:] += jx / 2; vy[:-1] += jy / 2; vy[1:] += jy / 2
    mag = np.hypot(vx, vy); m = np.percentile(mag, 99.5) + 1e-12
    hsv = np.stack([(np.arctan2(vy, vx) / (2 * np.pi)) % 1, np.ones_like(mag), np.clip(mag / m, 0, 1)], -1)
    return hsv_to_rgb(hsv)
