import numpy as np, pytest
from symsihti import sieve, reconstruct, GROUPS, projector, apply_strength, symmetric_fraction, render

rng = np.random.default_rng(0)
IMG = rng.random((37, 52, 3))          # odd/even sizes on purpose


@pytest.mark.parametrize('pur', ['lattice', 'owneq'])
def test_lossless(pur):
    bands, _ = sieve(IMG, pur, 32)
    assert np.abs(reconstruct(bands) - IMG).max() < 1e-12


def test_owneq_residues_are_mean_free():
    bands, _ = sieve(IMG, 'owneq', 32)
    for r in bands[:-1]:
        assert np.abs(r.mean((0, 1))).max() < 1e-12     # mass-conserving flow
    assert np.allclose(bands[-1].mean((0, 1)), IMG.mean((0, 1)))


@pytest.mark.parametrize('g', list(GROUPS))
@pytest.mark.parametrize('center', [None, (10, 20), (10.5, 30.5), (17, 9.5)])
def test_projector_is_orthogonal_projector(g, center):
    a, b = rng.random(IMG.shape), rng.random(IMG.shape)
    Pa = projector(a, g, center)
    assert np.abs(projector(Pa, g, center) - Pa).max() < 1e-12            # idempotent
    assert abs((Pa * b).sum() - (a * projector(b, g, center)).sum()) < 1e-9  # self-adjoint


@pytest.mark.parametrize('g', list(GROUPS))
def test_strength_semantics(g):
    r = rng.random(IMG.shape)
    assert np.array_equal(apply_strength(r, g, 0.0), r)
    P = projector(r, g)
    assert np.abs(apply_strength(r, g, 1.0) - P).max() < 1e-12
    f = symmetric_fraction(r, g)
    assert abs(f * (r ** 2).sum() - (P ** 2).sum()) < 1e-9 and 0 <= f <= 1 + 1e-12


def test_mirror_strength_two_is_the_mirror_image():
    x = rng.random((20, 31, 3))                     # odd width: centre column is a pixel
    assert np.abs(apply_strength(x, 'mirror L|R', 2.0) - x[:, ::-1]).max() < 1e-12


@pytest.mark.parametrize('pur', ['lattice', 'owneq'])
def test_all_bands_full_symmetry_equals_projecting_the_image(pur):
    # P is linear, so symmetrizing every band equals symmetrizing the whole image
    bands, _ = sieve(IMG, pur, 16)
    out, _, _ = render(IMG, settings=[('mirror L|R', 1.0)] * len(bands), bands=bands)
    assert np.abs(out - projector(IMG, 'mirror L|R')).max() < 1e-12


def test_square_kaleido_is_d4_invariant():
    x = rng.random((40, 40, 3)); y = projector(x, 'kaleido (D4)')
    for t in [y[:, ::-1], y[::-1], y.transpose(1, 0, 2), np.rot90(y)]:
        assert np.abs(t - y).max() < 1e-12
