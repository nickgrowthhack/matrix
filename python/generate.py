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
        self.sm = {k: ShapeModel(k) for k in ("1", "blob", "0_outer", "0_inner")}
        self.o_ids = load_ids("0_outer"); self.i_ids = load_ids("0_inner")
        self.common0 = sorted(set(self.o_ids) & set(self.i_ids))
        # --- tunables (auto-calibrated; overridable via data/model/gen_params.json) ---
        self.occ_gain = 1.11     # #3 density: lift occupancy to hit white-fraction
        self.sigma_h = 0.42      # #1 horizontal correlation length (cells)
        self.sigma_v = 0.78      # #1 vertical correlation length (cells) > horiz
        self.blob_cap = 0.52     # blob WIDTH cap as fraction of pitch_x (no side clumps)
        self.merge_gain = 1.5    # #4 boost merge probability (orig merges undercounted)
        self.var_clip = 0.45     # clip on per-glyph size lognormal: lower = more uniform
        self.gain = {"0": 1.0, "1": 1.0, "blob": 1.0}  # per-class size gain (blob ↑ w/o 0/1 ↑)
        pj = C.MODEL / "gen_params.json"
        if pj.exists():
            for k, v in json.loads(pj.read_text()).items():
                if k == "gain":
                    self.gain.update(v)
                else:
                    setattr(self, k, v)

    def _boot(self, sm, jit=0.25):
        i = self.rng.integers(len(sm.scores))
        s = sm.scores[i] + self.rng.standard_normal(len(sm.sdev)) * sm.sdev * jit
        return sm.coe_to_outline(sm.mean + s @ sm.rotation.T)

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

    def render(self):
        W, H = self.s["W"], self.s["H"]
        nC, nR = self.s["n_cols"], self.s["n_rows"]
        cov = np.zeros((H, W), np.uint8)            # glyph coverage (255 = glyph)
        col_c = np.array(self.grid["col_centers"]); row_c = np.array(self.grid["row_centers"])
        jdx, jdy = self.s["jitter_dx_std"], self.s["jitter_dy_std"]
        logstd = self.s["size_logstd"]
        px = self.s["pitch_x"]
        rng = self.rng
        placed = {"0": 0, "1": 0, "blob": 0}
        occupied = self._sample_occupancy()
        rows, cols = np.where(occupied)

        py = self.s["pitch_y"]
        for r, c in zip(rows.tolist(), cols.tolist()):
            p = self.type_p[:, r, c]; p = p / p.sum()
            t = rng.choice(("0", "1", "blob"), p=p)
            size = self.size[t][r, c] * self.gain[t] * np.exp(np.clip(rng.normal(0, logstd[t]), -self.var_clip, self.var_clip))
            # bounded jitter: real glyphs stay within the cell; an unbounded
            # Gaussian tail would pull neighbours together into false merges
            cx = col_c[c] + np.clip(rng.normal(0, jdx), -1.3 * jdx, 1.3 * jdx)
            cy = row_c[r] + np.clip(rng.normal(0, jdy), -1.3 * jdy, 1.3 * jdy)
            if t == "0":
                outer, inner = self._boot0()
                size = min(size, 0.85 * py / np.ptp(outer[:, 1]))   # height clamp (orig p99=64<pitch)
                op = (outer * size + [cx, cy]).astype(np.int32)
                ip = (inner * size + [cx, cy]).astype(np.int32)
                cv2.fillPoly(cov, [op], 255, lineType=AA)
                cv2.fillPoly(cov, [ip], 0, lineType=AA)       # carve the hole back
            else:
                out = self._boot(self.sm["1" if t == "1" else "blob"])
                if t == "blob":
                    size = min(size, self.blob_cap * px)         # width cap (no side clumps)
                size = min(size, 0.85 * py / np.ptp(out[:, 1]))  # height clamp (no vertical chains)
                op = (out * size + [cx, cy]).astype(np.int32)
                cv2.fillPoly(cov, [op], 255, lineType=AA)
                # #4 merge: a second blob overlapping the right side -> peanut/'W'
                if t == "blob" and c + 1 < nC and rng.random() < self.merge[r, c] * self.merge_gain:
                    out2 = self._boot(self.sm["blob"])
                    op2 = (out2 * size + [cx + 0.62 * px, cy]).astype(np.int32)
                    cv2.fillPoly(cov, [op2], 255, lineType=AA)
            placed[t] += 1
        return cov, placed


def cov_to_bgr(cov):
    """Coverage -> BGR, white glyphs on the dark layer (for side-by-side display)."""
    c = cov.astype(np.float32) / 255.0
    bg = np.array(C.BG_RGB[::-1], np.float32)
    return (bg[None, None, :] * (1 - c)[..., None] + 255.0 * c[..., None]).astype(np.uint8)


def cov_to_bgra(cov):
    """Coverage -> BGRA knock-out: glyphs transparent (soft alpha), dark opaque."""
    return np.dstack([cov_to_bgr(cov), (255 - cov).astype(np.uint8)])


def render_rgba(gen, seed):
    gen.rng = np.random.default_rng(seed)
    cov, placed = gen.render()
    return cov_to_bgra(cov), placed


def main():
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
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
