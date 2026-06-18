"""Stage 1: binarize, detect the glyph grid, segment connected components.

Outputs:
  data/interim/mask.png          binary glyph mask (0/255)
  data/interim/grid.json         pitch + phase + column/row centers
  data/interim/components.json   per-component: bbox, area, centroid, holes, cell
  reports/s1_overlay.png         visual check on a crop (grid + boxes by hole-count)
"""
import json
import numpy as np
import cv2
from skimage import measure
import common as C


def estimate_axis(proj, p_lo, p_hi):
    """Given a 1-D projection (peaks at cell centers), return (pitch, centers)."""
    c = proj - proj.mean()
    best_p, best = p_lo, -np.inf
    for p in range(p_lo, p_hi + 1):
        s = float(np.dot(c[:-p], c[p:]))
        if s > best:
            best, best_p = s, p
    p = best_p
    # phase: offset whose comb of period p maximizes sampled projection (cell centers)
    best_o, best_v = 0, -np.inf
    for o in range(p):
        v = float(proj[o::p].sum())
        if v > best_v:
            best_v, best_o = v, o
    centers = np.arange(best_o, len(proj), p)
    return p, centers.tolist()


def main():
    gray = C.load_gray()
    H, W = gray.shape
    mask = C.binarize(gray)  # 1 = glyph
    cv2.imwrite(str(C.INTERIM / "mask.png"), (mask * 255).astype(np.uint8))

    # --- grid via projections over the densest band ---
    col_proj = mask[: int(H * 0.45), :].sum(axis=0).astype(np.float64)
    row_proj = mask[:, : int(W * 0.5)].sum(axis=1).astype(np.float64)
    pitch_x0, _ = estimate_axis(col_proj, 40, 80)
    pitch_y0, _ = estimate_axis(row_proj, 55, 95)

    # --- connected components + properties (before grid, to fit grid from data) ---
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)
    comps = []
    MIN_AREA = 60  # px; the source is clean so this only drops stray specks
    for p in props:
        if p.area < MIN_AREA:
            continue
        minr, minc, maxr, maxc = p.bbox
        cy, cx = p.centroid
        comps.append({
            "id": int(p.label),
            "x": int(minc), "y": int(minr), "w": int(maxc - minc), "h": int(maxr - minr),
            "area": int(p.area), "cx": float(cx), "cy": float(cy),
            "holes": int(1 - p.euler_number), "extent": float(p.extent),
        })

    # --- data-driven grid centers (rows/cols are uniform-ish but not rigid) ---
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    def detect_centers(coords, pitch0, extent):
        h, _ = np.histogram(coords, bins=np.arange(0, extent + 2, 2.0))
        sm = gaussian_filter1d(h.astype(float), 1.6)
        pk, _ = find_peaks(sm, distance=max(1, int(pitch0 * 0.6 / 2)), height=sm.max() * 0.06)
        centers = list(pk * 2.0 + 1.0)
        med = float(np.median(np.diff(centers)))
        # fill big gaps with evenly spaced centers
        filled = [centers[0]]
        for c in centers[1:]:
            gap = c - filled[-1]
            if gap > 1.5 * med:
                n = int(round(gap / med))
                step = gap / n
                for _ in range(n - 1):
                    filled.append(filled[-1] + step)
            filled.append(c)
        return np.array(filled), med

    cxs = np.array([c["cx"] for c in comps]); cys = np.array([c["cy"] for c in comps])
    cc, pitch_x = detect_centers(cxs, pitch_x0, W)
    rc, pitch_y = detect_centers(cys, pitch_y0, H)
    n_cols, n_rows = len(cc), len(rc)
    kx = np.array([int(np.argmin(np.abs(cc - x))) for x in cxs])
    ky = np.array([int(np.argmin(np.abs(rc - y))) for y in cys])
    col_centers = cc.tolist(); row_centers = rc.tolist()
    for i, c in enumerate(comps):
        c["col"] = int(kx[i]); c["row"] = int(ky[i])
        c["span_cols"] = int(round(c["w"] / pitch_x))

    grid = {
        "W": W, "H": H, "pitch_x": pitch_x, "pitch_y": pitch_y,
        "col_centers": col_centers, "row_centers": row_centers,
        "n_cols": n_cols, "n_rows": n_rows,
    }
    (C.INTERIM / "grid.json").write_text(json.dumps(grid, indent=2))
    (C.INTERIM / "components.json").write_text(json.dumps(comps, indent=2))

    # --- summary ---
    n = len(comps)
    holes_hist = {}
    for c in comps:
        holes_hist[c["holes"]] = holes_hist.get(c["holes"], 0) + 1
    wide = sum(1 for c in comps if c["span_cols"] >= 2)
    cc = np.array(col_centers); rc = np.array(row_centers)
    res_x = np.std(cxs - cc[kx]); res_y = np.std(cys - rc[ky])
    print(f"image {W}x{H}  mask white fraction={mask.mean():.3f}")
    print(f"grid: pitch_x={pitch_x:.2f} pitch_y={pitch_y:.2f} -> {n_cols} cols x {n_rows} rows = {n_cols*n_rows} cells")
    print(f"grid-fit residual (true jitter): dx={res_x:.1f}px dy={res_y:.1f}px")
    print(f"components (area>= {MIN_AREA}): {n}")
    print(f"holes histogram: {dict(sorted(holes_hist.items()))}  (1 hole ~= '0')")
    print(f"wide components (span>=2 cols, merged blobs): {wide}")
    ws = np.array([c['w'] for c in comps]); hs = np.array([c['h'] for c in comps])
    ar = ws / hs
    print(f"width  px: median={np.median(ws):.0f} p10={np.percentile(ws,10):.0f} p90={np.percentile(ws,90):.0f}")
    print(f"height px: median={np.median(hs):.0f} p10={np.percentile(hs,10):.0f} p90={np.percentile(hs,90):.0f}")
    print(f"aspect w/h: median={np.median(ar):.2f} p10={np.percentile(ar,10):.2f} p90={np.percentile(ar,90):.2f}")

    # --- overlay on a crop: grid lines + bbox colored by hole count ---
    crop_w, crop_h = 1200, 900
    vis = cv2.cvtColor(gray[:crop_h, :crop_w], cv2.COLOR_GRAY2BGR)
    for x in col_centers:
        if x < crop_w:
            cv2.line(vis, (int(x), 0), (int(x), crop_h), (60, 60, 200), 1)
    for y in row_centers:
        if y < crop_h:
            cv2.line(vis, (0, int(y)), (crop_w, int(y)), (60, 60, 200), 1)
    for c in comps:
        if c["x"] < crop_w and c["y"] < crop_h:
            color = (0, 220, 0) if c["holes"] == 1 else (
                (0, 160, 255) if c["span_cols"] >= 2 else (0, 80, 255))
            cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), color, 2)
    cv2.imwrite(str(C.REPORTS / "s1_overlay.png"), vis)
    print("wrote reports/s1_overlay.png (green=1 hole '0', orange=merged blob, red=other)")


if __name__ == "__main__":
    main()
