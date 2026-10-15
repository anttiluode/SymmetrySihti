"""Render a gallery of scale-selective symmetry effects to figures/ (no webcam needed).
usage: python demo.py [image_path] [--center CY CX] [--purifier owneq|lattice]"""
import sys, time, numpy as np
from PIL import Image
from symsihti import sieve, render, reconstruct

def load(path=None, width=360):
    if path: im = Image.open(path).convert('RGB')
    else:
        try:
            from skimage import data; im = Image.fromarray(data.chelsea())
        except Exception:
            im = Image.fromarray(synthetic())
    h = int(round(im.height * width / im.width)); return np.asarray(im.resize((width, h), Image.LANCZOS)) / 255.0

def synthetic(h=240, w=360):
    y, x = np.mgrid[0:h, 0:w] / h; img = np.zeros((h, w, 3))
    img[:] = [0.25, 0.35, 0.55]; img[y > 0.7] = [0.3, 0.5, 0.25]
    disk = (x - 0.6) ** 2 + (y - 0.45) ** 2 < 0.06; img[disk] = [0.85, 0.6, 0.3]
    img[disk] += 0.15 * np.sin(40 * x + 25 * y)[disk][:, None]
    sq = (abs(x - 1.15) < 0.18) & (abs(y - 0.55) < 0.25); img[sq] = [0.7, 0.2, 0.25]
    return (np.clip(img + 0.03 * np.random.default_rng(0).standard_normal(img.shape), 0, 1) * 255).astype(np.uint8)

if __name__ == '__main__':
    args = sys.argv[1:]; path = args[0] if args and not args[0].startswith('--') else None
    center = None; pur = 'owneq'
    if '--center' in args: i = args.index('--center'); center = (float(args[i + 1]), float(args[i + 2]))
    if '--purifier' in args: pur = args[args.index('--purifier') + 1]
    x = load(path); t = time.time(); bands, depths = sieve(x, pur, 64); dt = time.time() - t
    K = len(bands); M = 'mirror L|R'
    print(f'image {x.shape}, purifier {pur}, {K} bands (depths {depths}), sieve {dt:.2f}s, '
          f'reconstruction error {np.abs(reconstruct(bands) - x).max():.1e}')
    fine = range(0, K - 3)          # detail bands
    recipes = {
        'original (all sliders 0)': [],
        'core symmetric, details untouched': [('none', 0)] * (K - 1) + [(M, 1)],
        'details symmetric, core untouched': [(M, 1) if k in fine else ('none', 0) for k in range(K)],
        'everything symmetric (= P x)': [(M, 1)] * K,
        'asymmetry doubled (s = -1)': [(M, -1)] * K,
        'details mirror-flipped (s = 2)': [(M, 2) if k in fine else ('none', 0) for k in range(K)],
    }
    tiles = []
    for name, st in recipes.items():
        out, _, fr = render(x, settings=st, center=center, bands=bands)
        tiles.append((name, np.clip(out, 0, 1)))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.2))
    for a, (n, im) in zip(ax.ravel(), tiles): a.imshow(im); a.set_title(n, fontsize=11); a.axis('off')
    if center is not None:
        for a in ax.ravel(): a.axvline(center[1], color='w', lw=0.5, alpha=0.4)
    plt.suptitle(f'SymmetrySihti: one mirror, applied per scale band ({pur} purifier)', fontsize=12)
    plt.tight_layout(); plt.savefig(f'figures/gallery_{pur}.png', dpi=90)
    # band strip with symmetric-energy fraction
    _, _, fr = render(x, settings=[(M, 0.0)] * K, center=center, bands=bands)
    from symsihti import symmetric_fraction
    fig, ax = plt.subplots(1, K, figsize=(2.4 * K, 2.4))
    for k, b in enumerate(bands):
        v = b if k == K - 1 else 0.5 + b / (6 * b.std() + 1e-9)
        ax[k].imshow(np.clip(v, 0, 1)); ax[k].axis('off')
        lab = 'core' if k == K - 1 else f'r{k} ({depths[k]}-{depths[k+1]})'
        ax[k].set_title(f'{lab}\nsym {symmetric_fraction(b, M, center):.2f}', fontsize=9)
    plt.tight_layout(); plt.savefig(f'figures/bands_{pur}.png', dpi=90); print('saved figures')
