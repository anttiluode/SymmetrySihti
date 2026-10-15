"""Symmetry Sieve: a Sihti whose purifiers are a chain of nested symmetry groups.

For G_0 ⊃ G_1 ⊃ ... ⊃ G_n = {e}, the group averages P_k are orthogonal projectors with
P_j P_k = P_j for j <= k (averaging over the big group swallows the small one), so

    c_0 = P_0 x,   c_k = P_k x - P_{k-1} x,   x = sum_k c_k,   ||x||^2 = sum_k ||c_k||^2   (exact).

c_k is "the part of the image that has symmetry G_k but breaks G_{k-1}" (an isotypic split).
Nesting only holds if no orbit leaves the image, so the sieve works on the largest window
centred on the chosen centre (square for chains containing 90-degree rotations).
"""
import math, numpy as np
from .symmetry import projector, NEEDS_SQUARE_LATTICE

CHAINS = {
    'D4 > C4 > C2 > 1': ['kaleido (D4)', 'rotate 90 (C4)', 'rotate 180', 'none'],
    'D4 > D2 > mirror > 1': ['kaleido (D4)', 'mirror both (D2)', 'mirror L|R', 'none'],
    'D2 > mirror > 1': ['mirror both (D2)', 'mirror L|R', 'none'],
    'mirror > 1': ['mirror L|R', 'none'],
}
LEVEL_LABELS = {
    'kaleido (D4)': 'D4-symmetric', 'rotate 90 (C4)': 'C4 not D4', 'rotate 180': 'C2 not C4',
    'mirror both (D2)': 'D2 not D4', 'mirror L|R': 'mirror not D2', 'none': 'breaks all',
}


def window(shape, center, chain):
    """Rows/cols slices of the largest window centred on `center` (half-integers allowed)."""
    H, W = shape[:2]
    cy, cx = center if center is not None else ((H - 1) / 2, (W - 1) / 2)
    cy, cx = round(2 * cy) / 2, round(2 * cx) / 2
    square = any(g in NEEDS_SQUARE_LATTICE for g in CHAINS[chain])
    if square and (2 * cy - 2 * cx) % 2: cx += 0.5 if cx + 0.5 <= W - 1 else -0.5
    hy, hx = min(cy, H - 1 - cy), min(cx, W - 1 - cx)        # half-extents, same fractional part as centre
    if square:
        h = min(hy, hx); hy = hx = h
    i0, i1 = int(round(cy - hy)), int(round(cy + hy)); j0, j1 = int(round(cx - hx)), int(round(cx + hx))
    return slice(i0, i1 + 1), slice(j0, j1 + 1)


def symmetry_sieve(x, chain='D4 > C4 > C2 > 1', center=None):
    """Returns (components, labels, (rows, cols)). Components live on the window."""
    rs, cs = window(x.shape, center, chain)
    w = np.asarray(x, float)[rs, cs]
    projs = [projector(w, g) for g in CHAINS[chain]]          # window centre = symmetry centre
    comps = [projs[0]] + [projs[k] - projs[k - 1] for k in range(1, len(projs))]
    return comps, [LEVEL_LABELS[g] for g in CHAINS[chain]], (rs, cs)


def energy_fractions(comps):
    e = np.array([float((c ** 2).sum()) for c in comps]); tot = e.sum()
    return e / tot if tot > 0 else e


def scale_symmetry_table(bands, chain='D4 > C4 > C2 > 1', center=None):
    """Energy of every Sihti band split exactly over the symmetry levels (rows sum to 1)."""
    return np.array([energy_fractions(symmetry_sieve(b, chain, center)[0]) for b in bands])
