"""Figures for part B: the Symmetry Sieve (group chain) and Noether currents (purifier flow).
usage: python demo_b.py [image] [--center CY CX]"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from demo import load
from symsihti.groupsieve import symmetry_sieve, energy_fractions, scale_symmetry_table
from symsihti.noether import noether_sieve, boundary_current_share, current_image


def gear(n=241, teeth=8, defect=True, seed=0):
    """A part with 8-fold symmetry (contains the lattice's D4) + lighting gradient + a broken tooth."""
    r = np.random.default_rng(seed); c = (n - 1) / 2; y, x = np.mgrid[0:n, 0:n] - c
    rad, th = np.hypot(x, y) / c, np.arctan2(y, x)
    edge = 0.78 + 0.08 * (np.cos(teeth * th) > 0.2)
    body = (rad < edge) & (rad > 0.18)
    if defect: body &= ~((np.abs(th - 0.42) < 0.17) & (rad > 0.72))      # chipped tooth
    img = np.full((n, n, 3), [0.12, 0.13, 0.16]); img[body] = [0.72, 0.70, 0.64]
    holes = np.zeros_like(rad, bool)
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 4; holes |= np.hypot(x - 0.45 * c * np.cos(a), y - 0.45 * c * np.sin(a)) < 0.08 * c
    img[holes] = [0.12, 0.13, 0.16]
    img *= (0.85 + 0.25 * (x + 0.6 * y) / n)[..., None]                    # lighting gradient
    return np.clip(img + 0.015 * r.standard_normal(img.shape), 0, 1)


def show(a): return np.clip(a, 0, 1)
def signed(a): return np.clip(0.5 + a / (6 * a.std() + 1e-12), 0, 1)

if __name__ == '__main__':
    args = sys.argv[1:]; path = args[0] if args and not args[0].startswith('--') else None
    center = None
    if '--center' in args: i = args.index('--center'); center = (float(args[i + 1]), float(args[i + 2]))

    # ---------- 1. Symmetry Sieve on a part ----------
    g = gear(); comps, labels, _ = symmetry_sieve(g, 'D4 > C4 > C2 > 1'); fr = energy_fractions(comps)
    fig, ax = plt.subplots(1, 6, figsize=(21, 4))
    ax[0].imshow(show(g)); ax[0].set_title('input: 8-fold part, chipped tooth,\nlighting gradient')
    for k, (c, l) in enumerate(zip(comps, labels)):
        ax[k + 1].imshow(show(c) if k == 0 else signed(c)); ax[k + 1].set_title(f'{l}\nenergy {100 * fr[k]:.3f}%')
    ax[5].imshow(np.abs(g - comps[0]).sum(-1), cmap='magma'); ax[5].set_title('|x - P_D4 x|: the defect is 7x brighter\nthan its 7 orbit copies (+ lighting)')
    for a in ax: a.axis('off')
    plt.suptitle('Symmetry Sieve: x = sum of orthogonal parts, ||x||^2 = sum of energies (exact)'); plt.tight_layout(); plt.subplots_adjust(top=0.78)
    plt.savefig('figures/b_symmetry_sieve_part.png', dpi=80); plt.close()
    gd = gear(defect=False); fr0 = energy_fractions(symmetry_sieve(gd, 'D4 > C4 > C2 > 1')[0])
    print('part energies  with defect:', np.round(100 * fr, 4), ' without:', np.round(100 * fr0, 4))

    # ---------- 2. Noether currents on a photo ----------
    x = load(path, 240); K = None
    fig, ax = plt.subplots(2, 6, figsize=(21, 6.6))
    for row, pur in enumerate(['lattice', 'owneq']):
        res = noether_sieve(x, pur, 64); share = boundary_current_share(res, x); E = res['energy']; d = res['depths']
        K = len(res['bands'])
        ax[row, 0].imshow(show(res['bands'][-1])); ax[row, 0].set_title(f'{pur}: core (depth {d[-1]})\nkeeps {100 * E[-1]:.1f}% of energy')
        for j, k in enumerate([0, 2, 4, 5]):
            ax[row, j + 1].imshow(current_image(res['jx'][k], res['jy'][k]))
            ax[row, j + 1].set_title(f'current, window {d[k]}->{d[k + 1]}\nE {100 * E[k]:.1f}%  edge-crossing {100 * share[k]:.0f}%')
        ax[row, 5].imshow(signed(res['bands'][4])); ax[row, 5].set_title('r4 = div(current 8->16)\n(exact identity)')
        print(f'{pur:8s} energy ledger %:', np.round(100 * E, 2), ' edge-crossing current %:', np.round(100 * share))
    for a in ax.ravel(): a.axis('off')
    plt.suptitle('Noether currents: every Sihti residue is the divergence of the current that flowed in its time window.  '
                 'hue = direction, brightness = amount moved', fontsize=11)
    plt.tight_layout(); plt.savefig('figures/b_noether_currents.png', dpi=80); plt.close()

    # ---------- 3. scale x symmetry table ----------
    from symsihti import sieve
    bands, d = sieve(x, 'lattice', 64)
    T = scale_symmetry_table(bands, 'mirror > 1', center)
    print('mirror-symmetric share of each lattice band (fine -> core):', np.round(T[:, 0], 3))
