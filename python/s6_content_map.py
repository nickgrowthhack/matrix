"""Stage 6: per-cell CONTENT MAP, sampled 100% from the validated pixels.

For EVERY grid cell (no exception) decide its content directly from the same
validated binarization the extractor uses — NOT from connected-component->grid
assignment (which drops cells where dense blobs merge). Each cell becomes one of
{empty, 0, 1, blob}:

  coverage < TAU_EMPTY                         -> empty
  enclosed hole present & coverage moderate    -> '0'   (the counter)
  filled (high coverage), no hole              -> 'blob'
  thin / low coverage, no hole                 -> '1'

Thresholds were calibrated from the per-cell metric distributions of the
confidently-labelled glyphs (see _recon/cell_metrics). Output:
  data/interim/content_map.json  (n_rows x n_cols grid of types + coverage)
  reports/s6_content_map*.png     (colour overlay for visual validation)
"""
import json
import numpy as np
import cv2
from scipy.ndimage import label
import common as C

# --- calibrated thresholds (per-cell pixel metrics; measured distributions) ---
# A '1' is a thin TALL stroke -> LOW coverage by nature, so emptiness is decided
# by white AREA, never by coverage (coverage<X wrongly ate the thinnest 1s).
# With an un-padded window the width gap is clean: 1 width<=24, blob width>=33.
MIN_AREA = 100      # white px below -> empty, UNLESS a tall thin '1' stroke (below)
TAU_HOLE = 60       # enclosed-background px to count as a '0' counter
ZERO_COV_MAX = 0.40  # a real '0' never fills this much (guards dense masses w/ pockets)
ONE_WBB = 29        # white-bbox width at/below -> '1' (clean gap to blob's >=33)
ONE_COV_MAX = 0.25  # border-only: above this a clipped-narrow cell is a blob, not '1'
TALL_MIN = 30       # a thin '1' stroke is this tall even when its area is tiny
THIN1_AREA = 40     # floor so 1-2px noise isn't promoted to a '1'
HOLE_PAD = 8        # extra px for the hole window so a jittered '0' ring still closes


def _edges(centers, pitch):
    """Cell boundaries = midpoints to the neighbouring grid centers (Voronoi in 1-D).
    Adapts to NON-UNIFORM spacing: a squeezed row/col gets a narrow cell instead of
    the fixed pitch, so its window stops at the neighbour and never bleeds into it.
    Ends fall back to half-pitch."""
    c = np.asarray(centers, float); n = len(c)
    lo = np.empty(n); hi = np.empty(n)
    for i in range(n):
        lo[i] = (c[i - 1] + c[i]) / 2 if i > 0 else c[i] - pitch / 2
        hi[i] = (c[i] + c[i + 1]) / 2 if i < n - 1 else c[i] + pitch / 2
    return lo, hi


def _crop(M, xl, xr, yt, yb):
    H, W = M.shape
    x0 = max(0, int(round(xl))); x1 = min(W, int(round(xr)))
    y0 = max(0, int(round(yt))); y1 = min(H, int(round(yb)))
    return M[y0:y1, x0:x1], x0, y0


def _band(M, xl, xr, yt, yb, gy):
    """Crop the ADAPTIVE cell window, then keep only the contiguous white ROW-BAND
    around the cell center (bounded by whitespace gaps). The center-midpoint boundary
    can fall inside a TALL neighbour row's glyph tail (when this row's glyphs are much
    smaller); clipping to the band that holds the cell's own glyph drops that tail.
    Returns (band_mask, x0, y0) with neighbour rows zeroed, or (None, x0, y0)."""
    w, x0, y0 = _crop(M, xl, xr, yt, yb)
    if w.size == 0:
        return None, x0, y0
    rp = w.sum(1)
    runs = []
    i = 0
    while i < len(rp):
        if rp[i] > 0:
            j = i
            while j + 1 < len(rp) and rp[j + 1] > 0:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    if not runs:
        return None, x0, y0
    ci = gy - y0
    a, b = min(runs, key=lambda rn: -1 if rn[0] - 1 <= ci <= rn[1] + 1
               else min(abs(rn[0] - ci), abs(rn[1] - ci)))
    if not (a - 1 <= ci <= b + 1) and min(abs(a - ci), abs(b - ci)) > 0.30 * len(rp):
        return None, x0, y0
    band = np.zeros_like(w); band[a:b + 1] = w[a:b + 1]
    return band, x0, y0


def _hole(mask):
    """Enclosed-background area inside a glyph mask (its '0' counter, if any)."""
    wp = np.pad(mask, HOLE_PAD, constant_values=False)
    lab, n = label(~wp)
    border = set(lab[0, :]).union(lab[-1, :]).union(lab[:, 0]).union(lab[:, -1])
    return sum(int((lab == i).sum()) for i in range(1, n + 1) if i not in border)


def cell_metrics(M, xl, xr, yt, yb, gx, gy):
    """Whole-band metrics (single glyph): area, white-bbox w/h, coverage, hole,
    centroid offset from the grid center. Used for diagnostics / the overview type."""
    band, x0, y0 = _band(M, xl, xr, yt, yb, gy)
    if band is None:
        return 0, 0, 0, 0.0, 0, 0.0, 0.0
    area = int(band.sum()); cov = float(band.mean())
    hw = (xr - xl) / 2; hh = (yb - yt) / 2
    ys_i, xs_i = np.nonzero(band)
    if xs_i.size:
        wbb = int(xs_i.max() - xs_i.min() + 1); hbb = int(ys_i.max() - ys_i.min() + 1)
        offx = max(-hw, min(hw, float(xs_i.mean() + x0 - gx)))
        offy = max(-hh, min(hh, float(ys_i.mean() + y0 - gy)))
        r0, r1 = ys_i.min(), ys_i.max() + 1
        hole = _hole(band[r0:r1])
    else:
        wbb = hbb = 0; offx = offy = 0.0; hole = 0
    return area, wbb, hbb, cov, int(hole), offx, offy


def cell_glyphs(M, xl, xr, yt, yb, gx, gy, border=False):
    """Per-cell glyph LIST. Most cells hold one glyph; where the validated pixels
    show 2+ SEPARATED ink components (a real gap between them — e.g. two narrow
    columns the grid merged into one cell), each becomes its own glyph. A single '0'
    is ONE connected component, so its hole never causes a false split. Returns a
    list of {t, bw, bh, cx, cy} (cx,cy absolute px). 8-connectivity keeps a glyph
    whole; edge slivers from a neighbour column are dropped by area + centrality."""
    band, x0, y0 = _band(M, xl, xr, yt, yb, gy)
    if band is None:
        return []
    lab, n = label(band, structure=np.ones((3, 3), int))   # 8-connected
    win_area = band.size; full_w = xr - xl
    out = []
    for cid in range(1, n + 1):
        ys_i, xs_i = np.nonzero(lab == cid)
        if xs_i.size == 0:
            continue
        cxw = float(xs_i.mean() + x0); cyw = float(ys_i.mean() + y0)
        if abs(cxw - gx) > 0.42 * full_w:        # neighbour sliver, not this cell's glyph
            continue
        area = int(xs_i.size)
        wbb = int(xs_i.max() - xs_i.min() + 1); hbb = int(ys_i.max() - ys_i.min() + 1)
        x0c, x1c = xs_i.min(), xs_i.max() + 1; y0c, y1c = ys_i.min(), ys_i.max() + 1
        hole = _hole((lab[y0c:y1c, x0c:x1c] == cid))
        t = classify(area, wbb, hbb, area / win_area, hole, border)
        if t == "empty":
            continue
        # fill = ink / bbox area: the original glyph's INTERNAL density (it drops
        # toward the right as glyphs dissolve). Reconstruction matches it per glyph.
        out.append({"t": t, "bw": wbb, "bh": hbb, "cx": round(cxw, 1), "cy": round(cyw, 1),
                    "fill": round(area / max(1, wbb * hbb), 3)})
    return out


def classify(area, wbb, hbb, cov, hole, border=False):
    thin1 = wbb <= ONE_WBB and hbb >= TALL_MIN and area >= THIN1_AREA
    if area < MIN_AREA and not thin1:
        return "empty"
    if hole >= TAU_HOLE and cov < ZERO_COV_MAX:
        return "0"
    # WIDTH is the primary 1-vs-blob signal (clean gap: 1<=24px, blob>=33px).
    # At the image border the window is clipped so a wide blob can look narrow;
    # there only, a high coverage (which a thin '1' never has) overrides to blob.
    if wbb <= ONE_WBB and not (border and cov >= ONE_COV_MAX):
        return "1"
    return "blob"


def build():
    g = json.loads((C.INTERIM / "grid.json").read_text())
    cc = np.array(g["col_centers"]); rc = np.array(g["row_centers"])
    px, py = g["pitch_x"], g["pitch_y"]; nR, nC = g["n_rows"], g["n_cols"]
    M = C.binarize(C.load_gray()).astype(bool)
    xlo, xhi = _edges(cc, px); ylo, yhi = _edges(rc, py)  # adaptive cell bounds

    cells = [["empty"] * nC for _ in range(nR)]    # main (cell-center) glyph type, for the overview
    glyphs = []                                    # flat placement list the reconstruction renders
    split_cells = 0
    counts = {"0": 0, "1": 0, "blob": 0}
    for r in range(nR):
        for col in range(nC):
            gl = cell_glyphs(M, xlo[col], xhi[col], ylo[r], yhi[r], cc[col], rc[r],
                             border=(col == 0 or col == nC - 1))
            if not gl:
                continue
            if len(gl) >= 2:
                split_cells += 1
            for gph in gl:
                glyphs.append(gph)
                counts[gph["t"]] += 1
            cells[r][col] = min(gl, key=lambda gp: abs(gp["cx"] - cc[col]))["t"]

    out = {"n_rows": nR, "n_cols": nC, "cells": cells, "glyphs": glyphs}
    (C.INTERIM / "content_map.json").write_text(json.dumps(out))
    print(f"content map: {len(glyphs)} glyphs {counts}; {split_cells} cells split into 2+ glyphs")

    # spot-check the dense all-blob region the user flagged
    from collections import Counter
    reg = [cells[r][col] for r in range(22, 26) for col in range(0, 18)]
    print("rows22-25 x cols0-17 (should be all blob):", dict(Counter(reg)))

    _overlay(M, cells, xlo, xhi, ylo, yhi, nR, nC)
    return out


def _overlay(M, cells, xlo, xhi, ylo, yhi, nR, nC):
    """Tint every cell by its class over the original (white glyphs on dark),
    using the adaptive cell bounds so squeezed rows show their true extent."""
    base = np.where(M[..., None], 255, np.array(C.BG_RGB[::-1], np.uint8)[None, None, :]).astype(np.uint8)
    color = {"0": (255, 60, 60), "1": (40, 220, 40), "blob": (40, 40, 255), "empty": None}
    vis = base.copy()
    for r in range(nR):
        for col in range(nC):
            t = cells[r][col]
            if color.get(t) is None:
                continue
            x0, y0 = int(round(xlo[col])), int(round(ylo[r]))
            x1, y1 = int(round(xhi[col])), int(round(yhi[r]))
            vis[y0:y1, x0:x1] = (0.62 * vis[y0:y1, x0:x1] + 0.38 * np.array(color[t])).astype(np.uint8)
    # full thumbnail + native crops for validation
    H, W = base.shape[:2]
    cv2.imwrite(str(C.REPORTS / "s6_content_map_overview.png"),
                cv2.resize(vis, (W // 6, H // 6), interpolation=cv2.INTER_AREA))
    crops = {"dense_blobs": (0, 1540, 1010, 1845), "mix": (1700, 600, 2700, 1000)}
    for name, (x0, y0, x1, y1) in crops.items():
        cv2.imwrite(str(C.REPORTS / f"s6_content_map_{name}.png"), vis[y0:y1, x0:x1])
    print("wrote reports/s6_content_map*.png  (blue=0, green=1, red=blob, untinted=empty)")


if __name__ == "__main__":
    build()
