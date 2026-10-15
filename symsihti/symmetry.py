"""Exact symmetry projectors on the pixel grid.

For a finite group G acting on pixel positions about a centre (cx, cy),
    P_G x [p] = mean_{g in G} x[g p]       if the whole orbit of p lies inside the image,
    P_G x [p] = x[p]                       otherwise (orbit left untouched).
Because whole orbits are either averaged or left alone, P_G is an exact orthogonal
projector: P^2 = P, P^T = P. Only the square-lattice symmetries are exact on pixels
(mirrors, 180 deg, 90 deg), so those are the groups offered.

Centres may be integers or half-integers (stored as twice the value). For 90-degree
groups, 2cx and 2cy must have the same parity so rotated pixels land on pixels.
"""
import numpy as np

# each group element maps offsets (u, v) = (i - cy, j - cx) to new offsets
_E = lambda u, v: (u, v)
_MLR = lambda u, v: (u, -v)       # mirror across a vertical axis (left <-> right)
_MUD = lambda u, v: (-u, v)       # mirror across a horizontal axis (top <-> bottom)
_R180 = lambda u, v: (-u, -v)
_R90 = lambda u, v: (v, -u)
_R270 = lambda u, v: (-v, u)
_T1 = lambda u, v: (v, u)         # diagonal mirrors
_T2 = lambda u, v: (-v, -u)

GROUPS = {
    'none': [_E],
    'mirror L|R': [_E, _MLR],
    'mirror T|B': [_E, _MUD],
    'rotate 180': [_E, _R180],
    'mirror both (D2)': [_E, _MLR, _MUD, _R180],
    'rotate 90 (C4)': [_E, _R90, _R180, _R270],
    'kaleido (D4)': [_E, _MLR, _MUD, _R180, _R90, _R270, _T1, _T2],
}
NEEDS_SQUARE_LATTICE = {'rotate 90 (C4)', 'kaleido (D4)'}

_cache = {}


def _maps(name, H, W, c2y, c2x):
    """Integer gather maps for every group element + mask of pixels whose orbit is fully inside."""
    key = (name, H, W, c2y, c2x)
    if key in _cache: return _cache[key]
    if name in NEEDS_SQUARE_LATTICE and (c2y - c2x) % 2:
        c2x += 1  # nudge so 90-degree rotations map pixels to pixels
    I, J = np.mgrid[0:H, 0:W]
    U2, V2 = 2 * I - c2y, 2 * J - c2x           # doubled offsets, exact integers
    gi, gj, ok = [], [], np.ones((H, W), bool)
    for g in GROUPS[name]:
        u2, v2 = g(U2, V2)
        i2, j2 = u2 + c2y, v2 + c2x             # doubled target coordinates
        ii, jj = i2 // 2, j2 // 2
        ok &= (i2 % 2 == 0) & (j2 % 2 == 0) & (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        gi.append(np.clip(ii, 0, H - 1)); gj.append(np.clip(jj, 0, W - 1))
    _cache[key] = (gi, gj, ok)
    if len(_cache) > 64: _cache.pop(next(iter(_cache)))
    return _cache[key]


def projector(x, name, center=None):
    """P_G x. center = (cy, cx) in pixels (half-integers allowed); default = image centre."""
    if name == 'none': return x
    H, W = x.shape[:2]
    cy, cx = center if center is not None else ((H - 1) / 2, (W - 1) / 2)
    gi, gj, ok = _maps(name, H, W, int(round(2 * cy)), int(round(2 * cx)))
    avg = sum(x[a, b] for a, b in zip(gi, gj)) / len(gi)
    m = ok[..., None] if x.ndim == 3 else ok
    return np.where(m, avg, x)


def apply_strength(band, name, s, center=None):
    """out = P r + (1 - s)(r - P r).
    s = 0: unchanged;  s = 1: fully symmetric;  s = 2: symmetry-breaking part flipped in sign
    (for a mirror this IS the mirror image);  s = -1: symmetry-breaking part doubled."""
    if name == 'none' or s == 0: return band
    p = projector(band, name, center)
    return p + (1.0 - s) * (band - p)


def symmetric_fraction(band, name, center=None):
    """||P r||^2 / ||r||^2 in [0, 1]  (exact energy split, since P is an orthogonal projector)."""
    if name == 'none': return 1.0
    e = float((band ** 2).sum())
    if e == 0: return 1.0
    p = projector(band, name, center)
    return float((p ** 2).sum()) / e
