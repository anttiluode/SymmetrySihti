# SymmetrySihti

**A symmetry slider for every scale band of an image.** It uses the Sihti sieve: purify the image step by step, keep everything that bleeds out, and sum it back exactly. Each band gets its own symmetry operator and its own strength.

![app](figures/app_screenshot.png)

```
pip install -r requirements.txt
python symsihti_live.py            # webcam (falls back to a test image)
python symsihti_live.py photo.jpg
python demo.py photo.jpg --center CY CX --purifier lattice   # gallery figure, no webcam
python -m pytest tests
```

Left-click the output to put the symmetry centre there (e.g. between the eyes). Right-click resets it.

## What it does

1. **Sieve.** Run a purifier and keep the residues at octave depths 0, 1, 2, 4 … d_max. With r_k = x_{d_k} − x_{d_{k+1}}, we have `x = r_0 + r_1 + … + core`, exactly (reconstruction error ~1e-16).
   - `lattice`: heat flow on the plain pixel grid. Persistence means scale. Fast (~10 fps at 320 px).
   - `owneq`: heat flow on the image's own colour-affinity graph. Object layout survives in the core, and the texture inside objects bleeds into the residues (Sihti's G4 result). ~7 fps.
2. **Symmetry per band.** Each band r gets `P r + (1 − s)(r − P r)`, where P averages over a symmetry group about the chosen centre.

| s | effect |
|---|---|
| −1 | the band's asymmetry is doubled |
| 0 | unchanged (all zeros = the exact original) |
| 1 | the band becomes fully symmetric |
| 2 | the symmetry-breaking part is flipped; for a mirror, this is exactly the band's mirror image |

Groups: mirror L|R, mirror T|B, rotate 180, both mirrors (D2), rotate 90 (C4), kaleido (D4).

3. **"sym" readout.** For each band, it shows the share of the band's energy that is symmetric, ‖P r‖² / ‖r‖². Because P is an exact orthogonal projector, this energy split is exact, not an estimate.

![gallery](figures/gallery_lattice.png)

The two purifiers treat the same slider very differently, because they put different things into the detail bands:

![gallery owneq](figures/gallery_owneq.png)

With `lattice`, the eyes live in the detail bands, so "details mirror-flipped" paints one eye's sharp detail onto the other eye's blurred base. With `owneq`, the eyes are objects and stay in the core, so the same setting barely touches them.

## What is exact (tests/test_core.py, 42 tests)

- Lossless sieve for both purifiers.
- The own-EQ flow conserves mass, so every residue band has zero mean, and the core carries the image's mean colour.
- Every projector satisfies P² = P and Pᵀ = P to 1e-12, for any centre (integer or half-integer). Pixels whose symmetry orbit leaves the image are left untouched as whole orbits. That is what keeps P an exact projector, and it is why a visible seam appears where the mirror runs out of image.
- Strength 2 on a mirror is exactly the mirror image.
- Symmetrizing *every* band equals symmetrizing the whole image, since P is linear. The effects only come from treating the bands *differently*.
- Only square-lattice symmetries are offered (mirrors, 90° and 180° rotations), because only these map pixels exactly onto pixels. 5- or 7-fold symmetry would need resampling and would no longer be exact.

## Honest scope

Mirror and kaleidoscope filters and Laplacian/diffusion pyramids are all old. What this repo adds is the combination: a lossless, object-aware residue stack (Sihti) with an exact symmetry projector applied per scale band, plus an exact energy readout per band. It is a toy and an instrument, not a discovery.

The planned next step (B) is the *Symmetry Sieve*: use a chain of nested groups (D4 ⊃ C4 ⊃ C2 ⊃ {e}) as the purifier itself. The residues are then exactly orthogonal (‖x‖² = Σ‖r_k‖² + ‖core‖²), and each residue is the part of the image that has one symmetry but breaks the next.

Built from [Sihti](https://github.com/anttiluode/Sihti) and [Noether-Ledger](https://github.com/anttiluode/Noether-Ledger).
