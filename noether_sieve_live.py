"""Part B live: the Symmetry Sieve and Noether currents, on webcam or an image.

    python noether_sieve_live.py            (webcam if available, otherwise the test image)
    python noether_sieve_live.py photo.jpg

Tab "Symmetry Sieve": a chain of nested groups splits the image into exactly orthogonal parts
    (fully symmetric / symmetric under the smaller group only / ... / breaks everything),
    with exact energy shares, plus a map of where the image breaks the full symmetry.
    Left-click on the input to set the symmetry centre, right-click to reset it.
Tab "Noether currents": the Sihti purifier flow as a conservation law. For each time window,
    the current that flowed (hue = direction, brightness = amount), the energy dissipated in it,
    and how much of the current crossed strong colour edges. lattice = plain heat flow;
    owneq = the image's own colour graph, where objects' colour charge is nearly conserved.
"""
import sys, time, numpy as np, tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from symsihti.groupsieve import CHAINS, symmetry_sieve, energy_fractions, window
from symsihti.noether import noether_sieve, boundary_current_share, current_image, divergence

try:
    import cv2
except Exception:
    cv2 = None

IN_W = 420


def photo(a, w):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return ImageTk.PhotoImage(im.resize((w, max(1, int(round(im.height * w / im.width)))), Image.BILINEAR))


def signed(a): return np.clip(0.5 + a / (6 * a.std() + 1e-12), 0, 1)


def heat(a):
    from matplotlib import cm
    m = np.percentile(a, 99.7) + 1e-12
    return cm.magma(np.clip(a / m, 0, 1))[..., :3]


def test_image():
    try:
        from skimage import data
        return np.asarray(data.chelsea()) / 255.0
    except Exception:
        from demo import synthetic
        return synthetic() / 255.0


class App:
    def __init__(self, root, path=None):
        self.root = root; root.title('SymmetrySihti B — Symmetry Sieve & Noether currents')
        self.cap = None; self.still = None; self.x = None; self.center = None
        self.last = time.time(); self.fps = 0.0; self.dirty = True; self.cache = None

        top = ttk.Frame(root); top.pack(fill='x', padx=6, pady=4)
        ttk.Button(top, text='Webcam', command=self.use_webcam).pack(side='left')
        ttk.Button(top, text='Open image…', command=self.open_image).pack(side='left')
        ttk.Button(top, text='Test image', command=lambda: self.use_still(test_image())).pack(side='left')
        ttk.Button(top, text='Test part', command=self.use_part).pack(side='left')
        ttk.Label(top, text='  width').pack(side='left')
        self.width = tk.StringVar(value='240')
        cb = ttk.Combobox(top, textvariable=self.width, values=['160', '200', '240', '320'], width=5, state='readonly')
        cb.pack(side='left'); cb.bind('<<ComboboxSelected>>', lambda e: self.reload_still())
        self.freeze = tk.BooleanVar(value=False); self.selfie = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='freeze', variable=self.freeze).pack(side='left', padx=(10, 0))
        ttk.Checkbutton(top, text='selfie view', variable=self.selfie).pack(side='left')

        body = ttk.Frame(root); body.pack(fill='both', expand=True, padx=6)
        left = ttk.Frame(body); left.pack(side='left', anchor='n')
        ttk.Label(left, text='input (click = symmetry centre, right-click = reset)').pack(anchor='w')
        self.in_canvas = tk.Canvas(left, width=IN_W, height=300, bg='black', highlightthickness=0); self.in_canvas.pack()
        self.in_canvas.bind('<Button-1>', self.set_center); self.in_canvas.bind('<Button-3>', lambda e: setattr(self, 'center', None))

        self.nb = ttk.Notebook(body); self.nb.pack(side='left', fill='both', expand=True, padx=(8, 0))
        t1 = ttk.Frame(self.nb); t2 = ttk.Frame(self.nb)
        self.nb.add(t1, text='Symmetry Sieve'); self.nb.add(t2, text='Noether currents')
        self.nb.bind('<<NotebookTabChanged>>', lambda e: self.mark_dirty())

        row = ttk.Frame(t1); row.pack(fill='x', pady=4)
        ttk.Label(row, text='group chain').pack(side='left')
        self.chain = tk.StringVar(value='D2 > mirror > 1')
        cb = ttk.Combobox(row, textvariable=self.chain, values=list(CHAINS), width=22, state='readonly'); cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self.mark_dirty())
        self.sieve_canvas = tk.Canvas(t1, width=980, height=560, bg='#1b1b1b', highlightthickness=0); self.sieve_canvas.pack()

        row = ttk.Frame(t2); row.pack(fill='x', pady=4)
        ttk.Label(row, text='purifier').pack(side='left')
        self.purifier = tk.StringVar(value='owneq')
        cb = ttk.Combobox(row, textvariable=self.purifier, values=['owneq', 'lattice'], width=8, state='readonly'); cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self.mark_dirty())
        ttk.Label(row, text='  depth').pack(side='left')
        self.dmax = tk.StringVar(value='64')
        cb = ttk.Combobox(row, textvariable=self.dmax, values=['16', '32', '64'], width=4, state='readonly'); cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self.mark_dirty())
        self.show = tk.StringVar(value='current')
        for v in ['current', 'residue = div(current)']:
            ttk.Radiobutton(row, text=v, value=v, variable=self.show, command=self.mark_dirty).pack(side='left', padx=4)
        self.noether_canvas = tk.Canvas(t2, width=980, height=560, bg='#1b1b1b', highlightthickness=0); self.noether_canvas.pack()

        self.status = ttk.Label(root, text=''); self.status.pack(anchor='w', padx=6, pady=4)
        self.imgs = []
        if path: self.use_still(np.asarray(Image.open(path).convert('RGB')) / 255.0)
        elif not self.use_webcam(quiet=True): self.use_still(test_image())
        root.after(10, self.loop)

    # ---------- sources ----------
    def use_webcam(self, quiet=False):
        if cv2 is None:
            self.status.config(text='opencv not installed: pip install opencv-python'); return False
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            if not quiet: self.status.config(text='no webcam found')
            return False
        if self.cap: self.cap.release()
        self.cap, self.still = cap, None; self.mark_dirty(); return True

    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[('images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('all', '*.*')])
        if p: self.use_still(np.asarray(Image.open(p).convert('RGB')) / 255.0)

    def use_part(self):
        from demo_b import gear
        self.chain.set('D4 > C4 > C2 > 1'); self.use_still(gear())

    def use_still(self, img):
        if self.cap: self.cap.release(); self.cap = None
        self.still_full = img; self.reload_still()

    def reload_still(self):
        if self.cap is None and getattr(self, 'still_full', None) is not None:
            self.still = self.resize(self.still_full); self.center = None; self.mark_dirty()

    def resize(self, img):
        w = int(self.width.get()); im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        return np.asarray(im.resize((w, max(8, int(round(im.height * w / im.width)))), Image.LANCZOS)) / 255.0

    def mark_dirty(self):
        self.dirty = True
        if self.still is not None: self.x = None

    def grab(self):
        if self.cap is not None and not self.freeze.get():
            ok, f = self.cap.read()
            if ok:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) / 255.0
                if self.selfie.get(): f = f[:, ::-1]
                new = self.resize(f)
                if self.x is None or new.shape != self.x.shape: self.center = None
                self.x = new; self.dirty = True
        elif self.still is not None and self.x is None:
            self.x = self.still; self.dirty = True

    def set_center(self, ev):
        if self.x is None: return
        H, W = self.x.shape[:2]; sc = IN_W / W
        self.center = (round(2 * min(max(ev.y / sc, 0), H - 1)) / 2, round(2 * min(max(ev.x / sc, 0), W - 1)) / 2)
        self.dirty = True

    # ---------- loop ----------
    def loop(self):
        try:
            self.grab()
            if self.x is not None and self.dirty:
                self.dirty = False; self.imgs = []
                self.draw_input()
                if self.nb.index('current') == 0: self.draw_sieve()
                else: self.draw_noether()
                now = time.time(); self.fps = 0.8 * self.fps + 0.2 / max(now - self.last, 1e-3); self.last = now
        except Exception as e:
            self.status.config(text=f'error: {e!r}')
        self.root.after(15, self.loop)

    def draw_input(self):
        H, W = self.x.shape[:2]; sc = IN_W / W; ph = photo(self.x, IN_W); self.imgs.append(ph)
        c = self.in_canvas; c.config(height=ph.height()); c.delete('all'); c.create_image(0, 0, anchor='nw', image=ph)
        if self.nb.index('current') == 0:
            rs, cs = window(self.x.shape, self.center, self.chain.get())
            c.create_rectangle(cs.start * sc, rs.start * sc, cs.stop * sc, rs.stop * sc, outline='#7fd', dash=(3, 3))

    def tile(self, canvas, img, x, y, w, title):
        ph = photo(img, w); self.imgs.append(ph)
        canvas.create_image(x, y + 16, anchor='nw', image=ph)
        canvas.create_text(x, y, anchor='nw', text=title, fill='#ddd', font=('TkDefaultFont', 9))
        return ph.height() + 22

    def draw_sieve(self):
        comps, labels, (rs, cs) = symmetry_sieve(self.x, self.chain.get(), self.center)
        fr = energy_fractions(comps); cv = self.sieve_canvas; cv.delete('all')
        n = len(comps); w = min(230, (980 - 10 * n) // n)
        h = 0
        for k, (c, l) in enumerate(zip(comps, labels)):
            h = max(h, self.tile(cv, c if k == 0 else signed(c), 10 + k * (w + 10), 6, w, f'{l}   {100 * fr[k]:.2f}%'))
        win = self.x[rs, cs]; brk = np.abs(win - comps[0]).sum(-1)
        hw = int(min(2 * w, (560 - h - 40) * brk.shape[1] / brk.shape[0]))
        self.tile(cv, heat(brk), 10, h + 16, hw, 'where it breaks the full symmetry  |x - P x|')
        txt = '\n'.join(f'{l:>16s}  {100 * f:7.3f} %' for l, f in zip(labels, fr))
        err = abs(sum(float((c ** 2).sum()) for c in comps) - float((win ** 2).sum())) / float((win ** 2).sum())
        cv.create_text(30 + hw, h + 40, anchor='nw', fill='#ddd', font=('TkFixedFont', 11),
                       text=f'energy split (exact, orthogonal parts)\n\n{txt}\n\nParseval error {err:.1e}')
        self.status.config(text=f'{self.fps:4.1f} fps   window {cs.stop - cs.start}x{rs.stop - rs.start}   chain {self.chain.get()}   '
                                f'| each part has the symmetry it is named for and none of the bigger one; the parts are orthogonal')

    def draw_noether(self):
        res = noether_sieve(self.x, self.purifier.get(), int(self.dmax.get()))
        share = boundary_current_share(res, self.x); E, d = res['energy'], res['depths']
        cv = self.noether_canvas; cv.delete('all'); K = len(res['bands']) - 1
        cols = 4; w = (980 - 10 * (cols + 1)) // cols; y = 6; hmax = 0
        cont = max(float(np.abs(r - divergence(jx, jy)).max()) for r, jx, jy in zip(res['bands'], res['jx'], res['jy']))
        for k in range(K + 1):
            col, row = k % cols, k // cols
            if k == K:
                img, title = res['bands'][-1], f'core (≥{d[-1]})  keeps {100 * E[-1]:.1f}% of fluctuation energy'
            else:
                img = current_image(res['jx'][k], res['jy'][k]) if self.show.get() == 'current' else signed(res['bands'][k])
                title = f'{d[k]}→{d[k + 1]}   E {100 * E[k]:.1f}%   edge-crossing {100 * share[k]:.0f}%'
            hh = self.tile(cv, img, 10 + col * (w + 10), 6 + row * (hmax + 8), w, title); hmax = max(hmax, hh)
        self.status.config(text=f'{self.fps:4.1f} fps   purifier {self.purifier.get()}   continuity error max|r - div J| = {cont:.1e}   '
                                f'global colour charge = {100 * res["charge_share"]:.1f}% of total energy (never dissipates)')


if __name__ == '__main__':
    root = tk.Tk(); app = App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.cap.release() if app.cap else None, root.destroy()))
    root.mainloop()
