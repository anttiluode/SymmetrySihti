import numpy as np
from .sieve import sieve, reconstruct
from .symmetry import apply_strength, symmetric_fraction


def render(x, purifier='lattice', dmax=64, settings=None, center=None, bands=None):
    """settings: list of (group_name, strength) per band (fine..core). Missing -> unchanged.
    Returns (output_image, bands, per-band symmetric fraction)."""
    if bands is None: bands, _ = sieve(x, purifier, dmax)
    settings = settings or []
    out, fracs = np.zeros_like(bands[0]), []
    for k, b in enumerate(bands):
        name, s = settings[k] if k < len(settings) else ('none', 0.0)
        out += apply_strength(b, name, s, center)
        fracs.append(symmetric_fraction(b, name, center) if name != 'none' else None)
    return out, bands, fracs
