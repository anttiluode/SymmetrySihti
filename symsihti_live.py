"""SymmetrySihti live — webcam / image, Sihti residue sieve, one symmetry slider per scale band.

    python symsihti_live.py            (webcam if available, otherwise the test image)
    python symsihti_live.py photo.jpg

Left-click the output to put the symmetry centre there; right-click resets it to the middle.
Each band (fine detail r0 ... coarse core) has its own group and strength:
    s = -1  asymmetry doubled      s = 0  unchanged (exact original)
    s =  1  fully symmetric        s = 2  symmetry-breaking part flipped (for a mirror: the mirror image)
"""
import sys, time, numpy as np, tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from symsihti.sieve import sieve, octave_depths
from symsihti.symmetry import GROUPS, apply_strength, symmetric_fraction

try:
    import cv2
except Exception:
    cv2 = None

GROUP_NAMES = list(GROUPS)
DISPLAY_W = 640


def to_photo(a, w):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    h = max(1, int(round(im.height * w / im.width)))
    return ImageTk.PhotoImage(im.resize((w, h), Image.BILINEAR))


def test_image():
    try:
        from skimage import data
        return np.asarray(data.chelsea()) / 255.0
    except Exception:
        from demo import synthetic
        return synthetic() / 255.0


class App:
    def __init__(self, root, path=None):
        self.root = root; root.title('SymmetrySihti — symmetry per scale band')
        self.cap = None; self.still = None; self.x = None; self.bands = None; self.depths = None
        self.dirty = True; self.center = None; self.last = time.time(); self.fps = 0.0

        top = ttk.Frame(root); top.pack(fill='x', padx=6, pady=4)
        ttk.Button(top, text='Webcam', command=self.use_webcam).pack(side='left')
        ttk.Button(top, text='Open image…', command=self.open_image).pack(side='left')
        ttk.Button(top, text='Test image', command=lambda: self.use_still(test_image())).pack(side='left')
        ttk.Label(top, text='  purifier').pack(side='left')
        self.purifier = tk.StringVar(value='lattice')
        cb = ttk.Combobox(top, textvariable=self.purifier, values=['lattice', 'owneq'], width=8, state='readonly')
        cb.pack(side='left'); cb.bind('<<ComboboxSelected>>', lambda e: self.mark_dirty())
        ttk.Label(top, text='  depth').pack(side='left')
        self.dmax = tk.StringVar(value='64')
        cb = ttk.Combobox(top, textvariable=self.dmax, values=['16', '32', '64', '128'], width=5, state='readonly')
        cb.pack(side='left'); cb.bind('<<ComboboxSelected>>', lambda e: (self.build_band_rows(), self.mark_dirty()))
        ttk.Label(top, text='  width').pack(side='left')
        self.width = tk.StringVar(value='320')
        cb = ttk.Combobox(top, textvariable=self.width, values=['200', '240', '320', '400', '480'], width=5, state='readonly')
        cb.pack(side='left'); cb.bind('<<ComboboxSelected>>', lambda e: self.reload_still())
        self.freeze = tk.BooleanVar(value=False); self.selfie = tk.BooleanVar(value=True); self.show_axis = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='freeze', variable=self.freeze).pack(side='left', padx=(10, 0))
        ttk.Checkbutton(top, text='selfie view', variable=self.selfie).pack(side='left')
        ttk.Checkbutton(top, text='show centre', variable=self.show_axis).pack(side='left')
        ttk.Button(top, text='Save', command=self.save).pack(side='left', padx=8)

        mid = ttk.Frame(root); mid.pack(fill='both', expand=True, padx=6)
        self.out_canvas = tk.Canvas(mid, width=DISPLAY_W, height=480, bg='black', highlightthickness=0)
        self.out_canvas.pack(side='left')
        self.out_canvas.bind('<Button-1>', self.set_center); self.out_canvas.bind('<Button-3>', self.reset_center)
        right = ttk.Frame(mid); right.pack(side='left', fill='y', padx=6)
        ttk.Label(right, text='input').pack(anchor='w')
        self.in_canvas = tk.Canvas(right, width=240, height=180, bg='black', highlightthickness=0); self.in_canvas.pack()
        ttk.Label(right, text='presets').pack(anchor='w', pady=(8, 0))
        for name, fn in [('all zero (original)', self.preset_zero), ('uncanny: core symmetric', self.preset_core),
                         ('twin details', self.preset_details), ('exaggerate asymmetry', self.preset_exaggerate),
                         ('flip details', self.preset_flip), ('kaleido core', self.preset_kaleido)]:
            ttk.Button(right, text=name, command=fn).pack(fill='x')
        row = ttk.Frame(right); row.pack(fill='x', pady=(8, 0))
        self.all_group = tk.StringVar(value='mirror L|R')
        ttk.Combobox(row, textvariable=self.all_group, values=GROUP_NAMES, width=16, state='readonly').pack(side='left')
        ttk.Button(row, text='set all', command=self.set_all_groups).pack(side='left')

        self.strip_canvas = tk.Canvas(root, width=DISPLAY_W + 250, height=96, bg='#222', highlightthickness=0)
        self.strip_canvas.pack(padx=6, pady=(4, 0), anchor='w')
        self.bands_frame = ttk.Frame(root); self.bands_frame.pack(fill='x', padx=6, pady=4)
        self.status = ttk.Label(root, text=''); self.status.pack(anchor='w', padx=6, pady=(0, 4))
        self.rows = []; self.build_band_rows()

        if path: self.use_still(np.asarray(Image.open(path).convert('RGB')) / 255.0)
        elif not self.use_webcam(quiet=True): self.use_still(test_image())
        self.root.after(10, self.loop)

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

    def use_still(self, img):
        if self.cap: self.cap.release(); self.cap = None
        self.still_full = img; self.reload_still()

    def reload_still(self):
        if self.cap is None and getattr(self, 'still_full', None) is not None:
            self.still = self.resize(self.still_full); self.center = None; self.mark_dirty()

    def resize(self, img):
        w = int(self.width.get()); im = Image.fromarray((img * 255).astype(np.uint8))
        h = max(8, int(round(im.height * w / im.width)))
        return np.asarray(im.resize((w, h), Image.LANCZOS)) / 255.0

    def grab(self):
        if self.cap is not None and not self.freeze.get():
            ok, f = self.cap.read()
            if ok:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) / 255.0
                if self.selfie.get(): f = f[:, ::-1]
                new = self.resize(f)
                if self.x is None or new.shape != self.x.shape: self.center = None
                self.x = new; self.dirty = True
        elif self.still is not None and (self.x is None or self.x is not self.still):
            self.x = self.still; self.dirty = True

    def mark_dirty(self):
        self.dirty = True
        if self.still is not None: self.x = None

    # ---------- band rows ----------
    def build_band_rows(self):
        old = [(r['group'].get(), r['s'].get()) for r in self.rows]
        for w in self.bands_frame.winfo_children(): w.destroy()
        depths = octave_depths(int(self.dmax.get())); K = len(depths)
        self.rows = []
        for k in range(K):
            lab = f'core  (≥{depths[-1]})' if k == K - 1 else f'r{k}  ({depths[k]}→{depths[k + 1]})'
            g = tk.StringVar(value=old[k][0] if k < len(old) else 'mirror L|R')
            s = tk.DoubleVar(value=old[k][1] if k < len(old) else 0.0)
            col, r0 = (k % 2) * 6, k // 2
            ttk.Label(self.bands_frame, text=lab, width=14).grid(row=r0, column=col, sticky='w')
            ttk.Combobox(self.bands_frame, textvariable=g, values=GROUP_NAMES, width=15, state='readonly').grid(row=r0, column=col + 1)
            tk.Scale(self.bands_frame, variable=s, from_=-1, to=2, resolution=0.05, orient='horizontal', length=170,
                     showvalue=True, highlightthickness=0).grid(row=r0, column=col + 2)
            frac = ttk.Label(self.bands_frame, text='', width=9); frac.grid(row=r0, column=col + 3, sticky='w')
            ttk.Label(self.bands_frame, text='   ').grid(row=r0, column=col + 4)
            self.rows.append(dict(group=g, s=s, frac=frac))

    def settings(self):
        return [(r['group'].get(), float(r['s'].get())) for r in self.rows]

    def set_strengths(self, fn):
        K = len(self.rows)
        for k, r in enumerate(self.rows): r['s'].set(fn(k, K))

    def preset_zero(self): self.set_strengths(lambda k, K: 0.0)
    def preset_core(self): self.set_all_groups('mirror L|R'); self.set_strengths(lambda k, K: 1.0 if k >= K - 2 else 0.0)
    def preset_details(self): self.set_all_groups('mirror L|R'); self.set_strengths(lambda k, K: 1.0 if k < K - 3 else 0.0)
    def preset_exaggerate(self): self.set_all_groups('mirror L|R'); self.set_strengths(lambda k, K: -1.0)
    def preset_flip(self): self.set_all_groups('mirror L|R'); self.set_strengths(lambda k, K: 2.0 if k < K - 3 else 0.0)
    def preset_kaleido(self):
        for k, r in enumerate(self.rows): r['group'].set('kaleido (D4)' if k >= len(self.rows) - 3 else 'mirror L|R')
        self.set_strengths(lambda k, K: 1.0 if k >= K - 3 else 0.0)

    def set_all_groups(self, name=None):
        for r in self.rows: r['group'].set(name or self.all_group.get())

    # ---------- centre ----------
    def set_center(self, ev):
        if self.bands is None: return
        sc = DISPLAY_W / self.bands[0].shape[1]
        H, W = self.bands[0].shape[:2]
        self.center = (min(max(ev.y / sc, 0), H - 1), min(max(ev.x / sc, 0), W - 1))
        self.center = (round(self.center[0] * 2) / 2, round(self.center[1] * 2) / 2)

    def reset_center(self, ev=None): self.center = None

    # ---------- main loop ----------
    def loop(self):
        try:
            self.grab()
            if self.x is not None and (self.dirty or self.bands is None):
                self.bands, self.depths = sieve(self.x, self.purifier.get(), int(self.dmax.get()))
                self.dirty = False
                if len(self.bands) != len(self.rows): self.build_band_rows()
            if self.bands is not None: self.draw()
        except Exception as e:
            self.status.config(text=f'error: {e!r}')
        self.root.after(15, self.loop)

    def draw(self):
        st, out = self.settings(), np.zeros_like(self.bands[0])
        for k, b in enumerate(self.bands):
            g, s = st[k] if k < len(st) else ('none', 0.0)
            out += apply_strength(b, g, s, self.center)
        self.out_img = to_photo(out, DISPLAY_W); self.last_out = out
        self.out_canvas.config(height=self.out_img.height())
        self.out_canvas.delete('all'); self.out_canvas.create_image(0, 0, anchor='nw', image=self.out_img)
        H, W = self.bands[0].shape[:2]; sc = DISPLAY_W / W
        cy, cx = self.center if self.center is not None else ((H - 1) / 2, (W - 1) / 2)
        if self.show_axis.get():
            self.out_canvas.create_line((cx + .5) * sc, 0, (cx + .5) * sc, H * sc, fill='#7fd', dash=(3, 4))
            self.out_canvas.create_line(0, (cy + .5) * sc, W * sc, (cy + .5) * sc, fill='#7fd', dash=(3, 4))
        self.in_img = to_photo(sum(self.bands), 240)
        self.in_canvas.config(height=self.in_img.height())
        self.in_canvas.delete('all'); self.in_canvas.create_image(0, 0, anchor='nw', image=self.in_img)
        # band strip: residues shown as 0.5 + r (contrast-normalised), core as is
        K, tw = len(self.bands), (DISPLAY_W + 250) // len(self.bands)
        self.strip_imgs = []; self.strip_canvas.delete('all')
        for k, b in enumerate(self.bands):
            v = b if k == K - 1 else 0.5 + b / (6 * b.std() + 1e-9)
            ph = to_photo(v[::2, ::2], tw - 4); self.strip_imgs.append(ph)
            self.strip_canvas.create_image(k * tw + 2, 2, anchor='nw', image=ph)
            if k == 0: self.strip_canvas.config(height=ph.height() + 4)
            if k < len(self.rows):
                g = st[k][0]
                self.rows[k]['frac'].config(text='' if g == 'none' else f'sym {symmetric_fraction(b, g, self.center):.2f}')
        now = time.time(); self.fps = 0.9 * self.fps + 0.1 / max(now - self.last, 1e-3); self.last = now
        self.status.config(text=f'{self.fps:4.1f} fps   {W}x{H}   purifier {self.purifier.get()}   centre ({cy:.1f}, {cx:.1f})'
                                f'   |   s: -1 exaggerate · 0 original · 1 symmetric · 2 flipped   |   "sym" = symmetric share of band energy')

    def save(self):
        if getattr(self, 'last_out', None) is None: return
        name = time.strftime('symsihti_%Y%m%d_%H%M%S.png')
        Image.fromarray((np.clip(self.last_out, 0, 1) * 255).astype(np.uint8)).save(name)
        self.status.config(text=f'saved {name}')


if __name__ == '__main__':
    root = tk.Tk(); app = App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.cap.release() if app.cap else None, root.destroy()))
    root.mainloop()
