import numpy as np, pytest
from symsihti.groupsieve import CHAINS, symmetry_sieve, energy_fractions, scale_symmetry_table
from symsihti.noether import noether_sieve, divergence, boundary_current_share
from symsihti.symmetry import projector
from symsihti import sieve

rng = np.random.default_rng(1)
IMG = rng.random((41, 58, 3))


@pytest.mark.parametrize('chain', list(CHAINS))
@pytest.mark.parametrize('center', [None, (20, 30), (15.5, 22.5), (12, 40.5), (30, 9)])
def test_symmetry_sieve_is_exact_and_parseval(chain, center):
    comps, labels, (rs, cs) = symmetry_sieve(IMG, chain, center)
    w = IMG[rs, cs]
    assert np.abs(sum(comps) - w).max() < 1e-12                               # lossless
    assert abs(sum((c ** 2).sum() for c in comps) - (w ** 2).sum()) < 1e-9   # Parseval
    for a in range(len(comps)):                                               # mutually orthogonal
        for b in range(a + 1, len(comps)):
            assert abs((comps[a] * comps[b]).sum()) < 1e-9
    # component k is invariant under its own group (it has the symmetry it claims)
    for g, c in zip(CHAINS[chain], comps):
        assert np.abs(projector(c, g) - c).max() < 1e-12


def test_scale_symmetry_rows_sum_to_one():
    bands, _ = sieve(IMG, 'lattice', 16)
    T = scale_symmetry_table(bands, 'D4 > C4 > C2 > 1')
    assert np.allclose(T.sum(1), 1)


@pytest.mark.parametrize('pur', ['lattice', 'owneq'])
def test_noether_continuity_charge_energy(pur):
    res = noether_sieve(IMG, pur, 32)
    assert np.abs(sum(res['bands']) - IMG).max() < 1e-12                       # Sihti identity
    for r, jx, jy in zip(res['bands'][:-1], res['jx'], res['jy']):
        assert np.abs(r - divergence(jx, jy)).max() < 1e-12                    # residue = div of current
        assert np.abs(r.sum((0, 1))).max() < 1e-10                             # each residue carries no net charge
    assert np.allclose(res['bands'][-1].sum((0, 1)), IMG.sum((0, 1)))          # charge conserved into core
    assert np.all(res['energy'] >= -1e-15) and abs(res['energy'].sum() - 1) < 1e-12   # energy ledger


def test_owneq_keeps_current_inside_regions():
    # two noisy regions with a strong colour edge between columns 19|20:
    # under lattice flow colour charge leaks across it, under own-EQ flow it does not
    x = np.zeros((40, 40, 3)); x[:, :20] = [0.9, 0.2, 0.2]; x[:, 20:] = [0.2, 0.3, 0.9]
    x += 0.05 * rng.standard_normal(x.shape)
    for pur, lo, hi in [('lattice', 0.5, 1e9), ('owneq', -1, 1e-20)]:
        res = noether_sieve(x, pur, 64)
        crossed = sum(np.abs(jx[:, 19]).sum() for jx in res['jx'])     # total charge moved across the edge
        assert lo < crossed < hi, (pur, crossed)
        left_charge = res['bands'][-1][:, :20].sum()                   # region charge in the core
        assert (abs(left_charge - x[:, :20].sum()) < 1e-12) == (pur == 'owneq')
