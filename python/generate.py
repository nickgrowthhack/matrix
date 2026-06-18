"""Generate a new binary-code image from the extracted mold (shape + layout).

For each grid cell: sample occupancy, then type (0/1/blob) from the spatial
fields; sample a faithful outline from the EFA/PCA shape model (bootstrap);
scale by the local size field; jitter position; rasterize with hard edges to
match the pristine 2-color source.
"""
import json
import numpy as np
import cv2
from shape_model import ShapeModel
from pathlib import Path
import common as C


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

    def render(self):
        W, H = self.s["W"], self.s["H"]
        nC, nR = self.s["n_cols"], self.s["n_rows"]
        canvas = np.full((H, W, 3), C.BG_RGB[::-1], np.uint8)  # BGR
        col_c = np.array(self.grid["col_centers"]); row_c = np.array(self.grid["row_centers"])
        jdx, jdy = self.s["jitter_dx_std"], self.s["jitter_dy_std"]
        logstd = self.s["size_logstd"]
        rng = self.rng
        placed = {"0": 0, "1": 0, "blob": 0}

        for r in range(nR):
            for c in range(nC):
                if rng.random() > self.occ[r, c]:
                    continue
                p = self.type_p[:, r, c]
                p = p / p.sum()
                t = rng.choice(("0", "1", "blob"), p=p)
                mult = np.exp(np.clip(rng.normal(0, logstd[t]), -0.45, 0.45))
                size = self.size[t][r, c] * mult
                if t == "blob":
                    # safety cap; the robust size field already keeps blobs to
                    # single-cell width, so packed blob fields keep clean gaps.
                    relax = max(0.0, (0.20 - self.occ[r, c]) / 0.20)
                    size = min(size, (0.55 + 0.30 * relax) * self.s["pitch_x"])
                cx = col_c[c] + rng.normal(0, jdx)
                cy = row_c[r] + rng.normal(0, jdy)
                if t == "0":
                    outer, inner = self._boot0()
                    op = (outer * size + [cx, cy]).astype(np.int32)
                    ip = (inner * size + [cx, cy]).astype(np.int32)
                    cv2.fillPoly(canvas, [op], (255, 255, 255))
                    cv2.fillPoly(canvas, [ip], C.BG_RGB[::-1])
                else:
                    key = "1" if t == "1" else "blob"
                    out = self._boot(self.sm[key])
                    op = (out * size + [cx, cy]).astype(np.int32)
                    cv2.fillPoly(canvas, [op], (255, 255, 255))
                    # simple merge: extend a blob to the right neighbour
                    if t == "blob" and c + 1 < nC and rng.random() < self.merge[r, c]:
                        out2 = self._boot(self.sm["blob"])
                        cx2 = col_c[c + 1] + rng.normal(0, jdx)
                        op2 = (out2 * size + [cx2, cy]).astype(np.int32)
                        cv2.fillPoly(canvas, [op2], (255, 255, 255))
                placed[t] += 1
        return canvas, placed


def to_rgba(img):
    """BGR canvas -> BGRA with glyphs (white) as transparent knock-outs."""
    white = np.all(img == 255, axis=2)
    alpha = np.where(white, 0, 255).astype(np.uint8)
    return np.dstack([img, alpha])


def render_rgba(gen, seed):
    """Re-render with a given seed reusing an already-loaded Generator."""
    gen.rng = np.random.default_rng(seed)
    img, placed = gen.render()
    return to_rgba(img), placed


def main():
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    g = Generator(seed=seed)
    img, placed = g.render()
    out = C.OUTPUT / (f"generated_seed{seed}.png" if seed != 7 else "generated.png")
    # faithful to the source: glyphs are TRANSPARENT knock-outs in the dark layer
    # (the source is RGBA with alpha=0 on the glyphs), not opaque white.
    white = np.all(img == 255, axis=2)
    alpha = np.where(white, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out), np.dstack([img, alpha]))
    print(f"wrote {out}  placed glyphs: {placed} total={sum(placed.values())}")
    # thumbnail + 1:1 crop comparison vs original
    thumb = cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10))
    cv2.imwrite(str(C.REPORTS / "gen_thumb.png"), thumb)
    orig = cv2.imread(str(C.SRC))
    ot = cv2.resize(orig, (orig.shape[1] // 10, orig.shape[0] // 10))
    sep = np.full((thumb.shape[0], 8, 3), (0, 180, 0), np.uint8)
    cv2.imwrite(str(C.REPORTS / "gen_vs_orig_thumb.png"), np.hstack([ot, sep, thumb]))
    # 1:1 crops (top-left dense + a mid band)
    cw, ch = 1000, 700
    crop_g = img[:ch, :cw]; crop_o = orig[:ch, :cw]
    cv2.imwrite(str(C.REPORTS / "gen_vs_orig_crop.png"),
                np.hstack([crop_o, np.full((ch, 8, 3), (0, 180, 0), np.uint8), crop_g]))
    print("wrote reports/gen_vs_orig_thumb.png and gen_vs_orig_crop.png")


if __name__ == "__main__":
    main()
