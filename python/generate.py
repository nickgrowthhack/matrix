"""Generate a new binary-code image from the extracted mold (shape + layout).

Per cell: sample occupancy from a SPATIALLY-CORRELATED field (Gaussian copula,
anisotropic -> matches the real vertical/horizontal clustering), then type from
the spatial fields, then a faithful outline from the EFA/PCA shape model
(bootstrap), scaled by the local size field, jittered. Rendered into an
anti-aliased coverage mask -> RGBA with TRANSPARENT knock-out glyphs (like the
source), soft edges.
"""
import json
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.special import ndtri
from shape_model import ShapeModel
import common as C

AA = cv2.LINE_AA


def load_ids(cls):
    arr = np.loadtxt(C.MODEL / f"efa_{cls}_ids.csv", skiprows=1, dtype=int)
    return {int(i): r for r, i in enumerate(np.atleast_1d(arr))}


class Generator:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        d = np.load(C.MODEL / "layout.npz")
        self.occ = d["occ"]; self.type_p = d["type_p"]; self.merge = d["merge"]
        self.size = {"0": d["size_0"], "1": d["size_1"], "blob": d["size_blob"]}
        self.s = json.loads((C.MODEL / "layout_scalars.json").read_text())
        self.grid = json.loads((C.INTERIM / "grid.json").read_text())
        self._colc = np.array(self.grid["col_centers"]); self._rowc = np.array(self.grid["row_centers"])
        self.sm = {k: ShapeModel(k) for k in ("1", "blob", "0_outer", "0_inner")}
        self.o_ids = load_ids("0_outer"); self.i_ids = load_ids("0_inner")
        self.common0 = sorted(set(self.o_ids) & set(self.i_ids))
        # id -> score-row, per class, so reconstruction can use each glyph's OWN shape
        self.id2row = {"1": load_ids("1"), "blob": load_ids("blob"),
                       "0_outer": self.o_ids, "0_inner": self.i_ids}
        self._props = None  # lazy regionprops for exact-contour fallback
        # --- tunables (auto-calibrated; overridable via data/model/gen_params.json) ---
        self.occ_gain = 1.11     # #3 density: lift occupancy to hit white-fraction
        self.sigma_h = 0.42      # #1 horizontal correlation length (cells)
        self.sigma_v = 0.78      # #1 vertical correlation length (cells) > horiz
        self.blob_cap = 0.52     # blob WIDTH cap as fraction of pitch_x (no side clumps)
        self.merge_gain = 1.5    # #4 boost merge probability (orig merges undercounted)
        self.var_clip = 0.45     # clip on per-glyph size lognormal: lower = more uniform
        self.gap = 2             # collision: min clear pixels kept between glyphs
        self.collision = True    # shrink a glyph locally if it would touch a placed one
        self.overlap_tol = 0     # px of contact allowed before shrinking
        self.fit_steps = 4
        self.gain = {"0": 1.0, "1": 1.0, "blob": 1.0}  # per-class size gain (blob ↑ w/o 0/1 ↑)
        self.size_var = {"0": 0.0, "1": 0.0, "blob": 0.0}  # extra size log-std (collision compresses spread)
        # per-class: True = one fixed scale (strict "quadradinho" look); False = use
        # the position field. 0/1 are uniform in the source; blobs grow into the dissolve.
        self.uniform_size = {"0": True, "1": True, "blob": False}
        # RECONSTRUCTION "esvaecimento" (glyphs lose internal ink toward the right):
        # (1) per-glyph in _place_fit a '0' resizes its HOLE to match its measured
        #     fill (thick walls left, thin right) — the right tool for 0s;
        # (2) a gentle rightward EROSION field thins blobs/1s as they dissolve.
        self.fade = True
        self.fade_x0 = 0.50     # fraction of width where thinning starts
        self.fade_k = 3.4       # slope
        self.fade_max = 2.2     # max erosion (px-ish)
        pj = C.MODEL / "gen_params.json"
        if pj.exists():
            for k, v in json.loads(pj.read_text()).items():
                if k in ("gain", "size_var", "uniform_size"):
                    getattr(self, k).update(v)
                else:
                    setattr(self, k, v)
        # fixed per-class scale = typical glyph size (median of the field), so every
        # element of a class is the SAME size on the grid (user's "quadradinho" model)
        self.size_const = {t: float(np.median(self.size[t])) for t in ("0", "1", "blob")}

    def _boot(self, sm, jit=0.25, min_sol=0.0):
        # min_sol>0: reject concave/self-intersecting shapes (blobs should be convex
        # -> a 'U' has solidity ~0.6; a clean blob ~0.95). Resample until convex.
        out = None
        for _ in range(8):
            i = self.rng.integers(len(sm.scores))
            s = sm.scores[i] + self.rng.standard_normal(len(sm.sdev)) * sm.sdev * jit
            out = sm.coe_to_outline(sm.mean + s @ sm.rotation.T)
            if min_sol <= 0:
                return out
            pts = out.astype(np.float32)
            a = abs(cv2.contourArea(pts))
            hull = abs(cv2.contourArea(cv2.convexHull(pts)))
            if hull > 0 and a / hull >= min_sol:
                return out
        return out

    def _boot0(self, jit=0.25):
        gid = self.common0[self.rng.integers(len(self.common0))]
        out = []
        for sm, idmap in ((self.sm["0_outer"], self.o_ids), (self.sm["0_inner"], self.i_ids)):
            row = idmap[gid]
            sc = sm.scores[row] + self.rng.standard_normal(len(sm.sdev)) * sm.sdev * jit
            out.append(sm.coe_to_outline(sm.mean + sc @ sm.rotation.T))
        return out  # outer, inner

    def _sample_occupancy(self):
        """Correlated-Bernoulli occupancy via a Gaussian copula: threshold an
        anisotropically-smoothed Gaussian field at the per-cell probability.
        Reproduces the real spatial clustering (esp. vertical), not salt & pepper.
        """
        nR, nC = self.s["n_rows"], self.s["n_cols"]
        p = np.clip(self.occ * self.occ_gain, 1e-4, 1 - 1e-4)
        z = gaussian_filter(self.rng.standard_normal((nR, nC)),
                            sigma=(self.sigma_v, self.sigma_h), mode="nearest")
        z = (z - z.mean()) / (z.std() + 1e-9)
        return z <= ndtri(p)   # P(z<=ndtri(p)) = p  -> correct marginal

    def _fit(self, out, cx, cy, size, cov, kernel):
        """Shrink `size` until the glyph keeps a `gap`-px clearance from already
        placed glyphs. Lets glyphs be full-size (more white) yet never touch
        (no chains) — only crowded spots shrink. Returns the fitted size."""
        H, W = cov.shape
        g = self.gap
        for _ in range(self.fit_steps):
            pts = out * size + [cx, cy]
            x, y, wd, ht = cv2.boundingRect(pts.astype(np.int32))
            x0, y0 = max(0, x - g - 1), max(0, y - g - 1)
            x1, y1 = min(W, x + wd + g + 1), min(H, y + ht + g + 1)
            if x1 <= x0 or y1 <= y0:
                return size
            ex = (cov[y0:y1, x0:x1] > 0).astype(np.uint8)
            if not ex.any():
                return size
            cand = np.zeros_like(ex)
            cv2.fillPoly(cand, [(pts - [x0, y0]).astype(np.int32)], 1)
            # allow a little contact (overlap_tol px) so glyphs keep their size
            # spread and the authentic ~few touches survive; shrink only past it
            if int((cand & cv2.dilate(ex, kernel)).sum()) > self.overlap_tol:
                size *= 0.88
            else:
                return size
        return size

    def _place(self, cov, t, r, c, kernel):
        """GENERATE one glyph of type t FROM THE MOLD (new shape+size sampled), at
        cell (r,c) with jitter + collision. Shared by variation and reconstruction;
        nothing is copied from the original glyph — only its type/cell are honoured."""
        rng = self.rng
        px, py = self.s["pitch_x"], self.s["pitch_y"]
        jdx, jdy = self.s["jitter_dx_std"], self.s["jitter_dy_std"]
        logstd = self.s["size_logstd"]; SAFE = 1.10
        base = self.size_const[t] if self.uniform_size.get(t) else self.size[t][r, c]
        lt = logstd[t] + self.size_var[t]
        size = base * self.gain[t] * np.exp(np.clip(rng.normal(0, lt), -self.var_clip, self.var_clip))
        cx = self._colc[c] + np.clip(rng.normal(0, jdx), -1.3 * jdx, 1.3 * jdx)
        cy = self._rowc[r] + np.clip(rng.normal(0, jdy), -1.3 * jdy, 1.3 * jdy)
        if t == "0":
            outer, inner = self._boot0()
            size = min(size, SAFE * py / np.ptp(outer[:, 1]))
            if self.collision:
                size = self._fit(outer, cx, cy, size, cov, kernel)
            cv2.fillPoly(cov, [(outer * size + [cx, cy]).astype(np.int32)], 255, lineType=AA)
            cv2.fillPoly(cov, [(inner * size + [cx, cy]).astype(np.int32)], 0, lineType=AA)
        else:
            out = self._boot(self.sm[t], min_sol=(0.88 if t == "blob" else 0.0))
            if t == "blob":
                size = min(size, self.blob_cap * px)
            size = min(size, SAFE * py / np.ptp(out[:, 1]))
            if self.collision:
                size = self._fit(out, cx, cy, size, cov, kernel)
            cv2.fillPoly(cov, [(out * size + [cx, cy]).astype(np.int32)], 255, lineType=AA)
            if t == "blob" and c + 1 < self.s["n_cols"] and rng.random() < self.merge[r, c] * self.merge_gain:
                out2 = self._boot(self.sm["blob"], min_sol=0.88)
                cv2.fillPoly(cov, [(out2 * size + [cx + 0.62 * px, cy]).astype(np.int32)], 255, lineType=AA)

    def _place_fit(self, cov, t, cx, cy, tw, th, fill=None):
        """RECONSTRUCTION placement: GENERATE a glyph of type t from the mold (new
        outline sampled), scale its bbox to the target size (tw x th) at (cx,cy abs),
        AND match the original glyph's FILL ratio (internal ink density, which fades
        toward the right). Shape = mold; size/position/fill = the original's layout.
        A '0' matches fill by resizing its HOLE (thicker walls = fuller); a blob/1
        matches by eroding the solid shape. No collision (dense blobs touch)."""
        area_box = max(1e-6, tw * th)

        def tf(o, sx, sy, bx, by):
            return np.column_stack([(o[:, 0] - bx) * sx + cx, (o[:, 1] - by) * sy + cy])

        if t == "0":
            outer, inner = self._boot0()
            x, y = outer[:, 0], outer[:, 1]
            bx = (x.min() + x.max()) / 2.0; by = (y.min() + y.max()) / 2.0
            sx = tw / max(1e-6, x.max() - x.min()); sy = th / max(1e-6, y.max() - y.min())
            fo = tf(outer, sx, sy, bx, by); fi = tf(inner, sx, sy, bx, by)
            if fill is not None:                       # nudge the hole toward target fill:
                oa = abs(cv2.contourArea(fo.astype(np.float32)))   # smaller hole = thicker
                ia = abs(cv2.contourArea(fi.astype(np.float32)))   # walls = fuller (left),
                des_i = oa - fill * area_box                       # bigger = thinner (right)
                if ia > 60:
                    # gentle, capped: thicken/thin walls modestly but NEVER close the
                    # hole (>=60px floor + s>=0.85) so a '0' never turns into a blob
                    s = 0.85 if des_i <= 0 else min(1.15, max(0.85, (des_i / ia) ** 0.5))
                    s = max(s, (60.0 / ia) ** 0.5)
                    fi = (fi - [cx, cy]) * s + [cx, cy]
            cv2.fillPoly(cov, [fo.astype(np.int32)], 255, lineType=AA)
            cv2.fillPoly(cov, [fi.astype(np.int32)], 0, lineType=AA)
            return fi.astype(np.int32)                  # hole, to re-punch after all glyphs
        else:
            out = self._boot(self.sm[t], min_sol=(0.88 if t == "blob" else 0.0))
            x, y = out[:, 0], out[:, 1]
            bx = (x.min() + x.max()) / 2.0; by = (y.min() + y.max()) / 2.0
            sx = tw / max(1e-6, x.max() - x.min()); sy = th / max(1e-6, y.max() - y.min())
            cv2.fillPoly(cov, [tf(out, sx, sy, bx, by).astype(np.int32)], 255, lineType=AA)

    def render(self):
        """Variation: sample occupancy AND type per cell, then GENERATE each glyph."""
        cov = np.zeros((self.s["H"], self.s["W"]), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.gap + 1, 2 * self.gap + 1))
        placed = {"0": 0, "1": 0, "blob": 0}
        occupied = self._sample_occupancy()
        rows, cols = np.where(occupied)
        for r, c in zip(rows.tolist(), cols.tolist()):
            p = self.type_p[:, r, c]; p = p / p.sum()
            t = self.rng.choice(("0", "1", "blob"), p=p)
            self._place(cov, t, r, c, kernel)
            placed[t] += 1
        return cov, placed

    def render_reconstruct(self):
        """Reconstruction: GENERATE each glyph FROM THE MOLD (new shape+size sampled,
        from scratch — nothing copied), but honour the ORIGINAL's content: each cell
        gets the SAME type it has in the original (0/1/blob/empty), never a sampled
        type. So 'cell has a 0' -> generate a brand-new 0 from the mold in that cell.

        The content map is built per-cell from the validated pixels (s6_content_map),
        NOT from connected-component->grid assignment — which dropped cells wherever
        dense blobs merged (leaving holes) and mis-typed a few. Every cell of the grid
        is honoured here, with no exception."""
        cm = json.loads((C.INTERIM / "content_map.json").read_text())
        cov = np.zeros((self.s["H"], self.s["W"]), np.uint8)
        placed = {"0": 0, "1": 0, "blob": 0}
        # content map is a flat list of glyph placements (a cell with two narrow
        # glyphs the grid merged contributes two entries); each is generated from
        # the mold, fitted to its measured size+position.
        holes = []
        for gph in cm["glyphs"]:
            h = self._place_fit(cov, gph["t"], gph["cx"], gph["cy"], gph["bw"], gph["bh"],
                                 gph.get("fill"))
            if h is not None:
                holes.append(h)
            placed[gph["t"]] += 1
        if self.fade:
            cov = self._apply_fade(cov)              # gentle rightward thinning (blobs/1s)
        # re-punch the '0' counters LAST so a neighbour glyph drawn over them in the
        # dense region can't fill them in (else a '0' reads as a solid blob)
        for h in holes:
            cv2.fillPoly(cov, [h], 0, lineType=AA)
        placed["dot"] = self._draw_small(cov)        # faithful small-dot detail layer
        return cov, placed

    def _apply_fade(self, cov):
        """Gentle rightward erosion: blends cov -> eroded per column by a weight that
        rises toward the right, thinning blobs/1s as they dissolve (0s are matched
        per-glyph via their hole, so this is a light top-up for the rest)."""
        W = cov.shape[1]
        w = np.clip((np.arange(W) / W - self.fade_x0) * self.fade_k, 0, self.fade_max)
        er1 = cv2.erode(cov, np.ones((3, 3), np.uint8))
        er2 = cv2.erode(cov, np.ones((5, 5), np.uint8))
        w1 = np.clip(w, 0, 1)[None, :]; w2 = np.clip(w - 1, 0, 1)[None, :]
        covf = cov * (1 - w1) + er1 * w1
        covf = covf * (1 - w2) + er2 * w2
        return np.clip(covf, 0, 255).astype(np.uint8)

    def _draw_region(self, cov, region, off):
        """Copy the component's EXACT mask into cov (hard edges, full fill) -> a
        faithful copy with no AA thinning. `region` is True on the glyph (holes are
        False, so they stay background)."""
        x0, y0 = off
        h, w = region.shape
        cov[y0:y0 + h, x0:x0 + w][region] = 255

    def _draw_small(self, cov, lo=4, hi=60):
        """Faithful detail layer: redraw the small specks/dots that the model
        pipeline drops (area < MIN_AREA). Keeps them out of the shape model but
        present in the image."""
        from skimage import measure
        labels = measure.label(C.binarize(C.load_gray()), connectivity=2)
        n = 0
        for p in measure.regionprops(labels):
            if lo <= p.area < hi:
                self._draw_region(cov, p.image, [p.bbox[1], p.bbox[0]])
                n += 1
        return n

    def render_extraction(self):
        """Extraction test (≈ copy): re-segment the original and redraw EVERY
        component (incl. tiny dots) from its exact contour. Maximally faithful;
        deterministic (seed only names the file)."""
        from skimage import measure
        labels = measure.label(C.binarize(C.load_gray()), connectivity=2)
        W, H = self.s["W"], self.s["H"]
        cov = np.zeros((H, W), np.uint8)
        placed = {"0": 0, "1": 0, "blob": 0, "dot": 0}
        for p in measure.regionprops(labels):
            if p.area < 4:
                continue
            self._draw_region(cov, p.image, [p.bbox[1], p.bbox[0]])
            minr, minc, maxr, maxc = p.bbox
            h, w = maxr - minr, maxc - minc
            if p.area < 60:
                placed["dot"] += 1
            elif int(1 - p.euler_number) >= 1:
                placed["0"] += 1
            elif w / h < 0.5:
                placed["1"] += 1
            else:
                placed["blob"] += 1
        # RGB is strictly 2-tone, so the redrawn bgr already equals the original's
        # RGB; reuse the ORIGINAL's alpha channel (the soft anti-aliased edges we
        # extract too) -> brightness/halo matches the original exactly.
        bgr = cov_to_bgr(cov)
        alpha = cv2.imread(str(C.SRC), cv2.IMREAD_UNCHANGED)[:, :, 3]
        return np.dstack([bgr, alpha]), placed


def cov_to_bgr(cov):
    """Coverage -> BGR, white glyphs on the dark layer (for side-by-side display)."""
    c = cov.astype(np.float32) / 255.0
    bg = np.array(C.BG_RGB[::-1], np.float32)
    return (bg[None, None, :] * (1 - c)[..., None] + 255.0 * c[..., None]).astype(np.uint8)


def cov_to_bgra(cov):
    """Coverage -> BGRA knock-out: glyphs transparent (soft alpha), dark opaque."""
    return np.dstack([cov_to_bgr(cov), (255 - cov).astype(np.uint8)])


def render_rgba(gen, seed, mode="variation"):
    gen.rng = np.random.default_rng(seed)
    if mode == "reconstruct":
        cov, placed = gen.render_reconstruct()
    elif mode == "extract":
        return gen.render_extraction()  # returns final BGRA (custom soft alpha)
    else:
        cov, placed = gen.render()
    return cov_to_bgra(cov), placed


# output filename per mode (all seed-named so each run is a distinct file)
def out_name(mode, seed):
    if mode == "reconstruct":
        return f"reconstruction_seed{seed}.png"
    if mode == "extract":
        return f"extraction_seed{seed}.png"
    return f"generated_seed{seed}.png" if seed != 7 else "generated.png"


def main():
    import sys
    args = sys.argv[1:]
    MODE = {"reconstruct": "reconstruct", "recon": "reconstruct",
            "extract": "extract", "extraction": "extract"}
    if args and args[0] in MODE:
        mode = MODE[args[0]]
        seed = int(args[1]) if len(args) > 1 else 0
        g = Generator(seed=seed)
        rgba, placed = render_rgba(g, seed, mode=mode)
        out = C.OUTPUT / out_name(mode, seed)
        cv2.imwrite(str(out), rgba)
        print(f"wrote {out}  placed: {placed}")
        return
    seed = int(args[0]) if args else 7
    g = Generator(seed=seed)
    cov, placed = g.render()
    out = C.OUTPUT / (f"generated_seed{seed}.png" if seed != 7 else "generated.png")
    cv2.imwrite(str(out), cov_to_bgra(cov))
    print(f"wrote {out}  placed glyphs: {placed} total={sum(placed.values())}")
    img = cov_to_bgr(cov)
    orig = cv2.imread(str(C.SRC))
    thumb = cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10))
    ot = cv2.resize(orig, (orig.shape[1] // 10, orig.shape[0] // 10))
    sep = np.full((thumb.shape[0], 8, 3), (0, 180, 0), np.uint8)
    cv2.imwrite(str(C.REPORTS / "gen_vs_orig_thumb.png"), np.hstack([ot, sep, thumb]))
    cw, ch = 1000, 700
    cv2.imwrite(str(C.REPORTS / "gen_vs_orig_crop.png"),
                np.hstack([orig[:ch, :cw], np.full((ch, 8, 3), (0, 180, 0), np.uint8), img[:ch, :cw]]))
    print("wrote reports/gen_vs_orig_thumb.png and gen_vs_orig_crop.png")


if __name__ == "__main__":
    main()
