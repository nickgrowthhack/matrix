"""Validate the shape model: render reconstructed REAL glyphs vs SAMPLED glyphs.
Top half of each class block = real (K-harmonic reconstruction),
bottom half = newly sampled from the PCA model."""
import numpy as np
import cv2
from shape_model import ShapeModel
import common as C

CELL = 90
PXSCALE = 26  # normalized outline (~[-1,1]) -> pixels


def render_outline(outer, inner=None, cell=CELL):
    img = np.full((cell, cell, 3), 30, np.uint8)
    c = cell // 2
    op = (outer * PXSCALE + c).astype(np.int32)
    cv2.fillPoly(img, [op], (255, 255, 255))
    if inner is not None:
        ip = (inner * PXSCALE + c).astype(np.int32)
        cv2.fillPoly(img, [ip], (50, 51, 50))
    return img


def grid(tiles, ncol):
    nrow = int(np.ceil(len(tiles) / ncol))
    canvas = np.full((nrow * CELL, ncol * CELL, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, ncol)
        canvas[r * CELL:(r + 1) * CELL, cc * CELL:(cc + 1) * CELL] = t
    return canvas


def main():
    rng = np.random.default_rng(3)
    ncol = 10
    blocks = []
    sm1 = ShapeModel("1"); smb = ShapeModel("blob")
    smo = ShapeModel("0_outer"); smi = ShapeModel("0_inner")

    def block(title_real_sampled):
        pass

    MODE = "bootstrap"
    # class 1
    real = [render_outline(sm1.real_outline(i)) for i in rng.choice(len(sm1.scores), ncol, replace=False)]
    samp = [render_outline(sm1.sample_outline(rng, mode=MODE)) for _ in range(ncol)]
    b1 = np.vstack([grid(real, ncol), np.full((4, ncol * CELL, 3), 90, np.uint8), grid(samp, ncol)])

    # blob
    real = [render_outline(smb.real_outline(i)) for i in rng.choice(len(smb.scores), ncol, replace=False)]
    samp = [render_outline(smb.sample_outline(rng, mode=MODE)) for _ in range(ncol)]
    bb = np.vstack([grid(real, ncol), np.full((4, ncol * CELL, 3), 90, np.uint8), grid(samp, ncol)])

    # 0 (outer+inner). Sampled: share a real index so the ring stays correlated.
    idx = rng.choice(len(smo.scores), ncol, replace=False)
    real = [render_outline(smo.real_outline(i), smi.real_outline(i)) for i in idx]
    sidx = rng.choice(len(smo.scores), ncol, replace=False)
    samp = []
    for i in sidx:
        co = smo.mean + (smo.scores[i] + rng.standard_normal(len(smo.sdev)) * smo.sdev * 0.25) @ smo.rotation.T
        ci = smi.mean + (smi.scores[i] + rng.standard_normal(len(smi.sdev)) * smi.sdev * 0.25) @ smi.rotation.T
        samp.append(render_outline(smo.coe_to_outline(co), smi.coe_to_outline(ci)))
    b0 = np.vstack([grid(real, ncol), np.full((4, ncol * CELL, 3), 90, np.uint8), grid(samp, ncol)])

    sep = np.full((10, ncol * CELL, 3), (0, 120, 0), np.uint8)
    full = np.vstack([b1, sep, bb, sep, b0])
    cv2.imwrite(str(C.REPORTS / "s4b_shapes.png"), full)
    print("wrote reports/s4b_shapes.png")
    print("layout: [1 real / 1 sampled] [blob real / sampled] [0 real / sampled]")


if __name__ == "__main__":
    main()
