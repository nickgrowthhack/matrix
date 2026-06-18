"""Stage 5: composition / layout model.

Estimates, as smooth fields over the (row, col) grid, directly from the data:
  - occupancy   P(cell has a glyph)           -> the dissolve density gradient
  - type        P(0|occ), P(1|occ), P(blob|occ) over position
  - size        per-class mean RMS-radius (px) field  + residual log-std
  - jitter      std of centroid offset from cell center
  - merge       P(a blob merges with right neighbor) field
Saved to data/model/layout.npz (+ layout_scalars.json) and a heatmap report.
"""
import json
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import common as C

SIGMA = 2.2  # cells, smoothing radius for the fields


def rms_radius_by_id(key):
    """RMS radius (px) of each ORIGINAL (centroid-centered) outline."""
    out = {}
    acc = {}
    with open(C.GLYPHS / f"outlines_{key}.csv") as f:
        next(f)
        for line in f:
            i, k, x, y = line.split(",")
            acc.setdefault(int(i), []).append((float(x), float(y)))
    for i, pts in acc.items():
        p = np.array(pts)
        p = p - p.mean(0)
        out[i] = float(np.sqrt((p ** 2).sum(1).mean()))
    return out


def main():
    grid = json.loads((C.INTERIM / "grid.json").read_text())
    comps = json.loads((C.INTERIM / "components_classified.json").read_text())
    nC, nR = grid["n_cols"], grid["n_rows"]
    px, py = grid["pitch_x"], grid["pitch_y"]

    occ = np.zeros((nR, nC))
    cnt = {"0": np.zeros((nR, nC)), "1": np.zeros((nR, nC)), "blob": np.zeros((nR, nC))}
    merge = np.zeros((nR, nC))
    dx_list, dy_list = [], []
    size_acc = {"0": [], "1": [], "blob": []}   # (row, col, rms)

    rms = {}
    for key in ("0_outer", "1", "blob"):
        rms.update(rms_radius_by_id(key))

    col_c = np.array(grid["col_centers"]); row_c = np.array(grid["row_centers"])
    for c in comps:
        r, k = c["row"], c["col"]
        if not (0 <= r < nR and 0 <= k < nC):
            continue
        base = "0" if c["cls"] == "0" else ("1" if c["cls"] == "1" else "blob")
        if c["cls"] == "blob_merged":
            # mark spanned cells as blob + merged
            span = max(1, int(round(c["w"] / px)))
            for s in range(span):
                kk = min(nC - 1, k - span // 2 + s)
                occ[r, kk] = 1; cnt["blob"][r, kk] += 1
            merge[r, k] = 1
            continue
        occ[r, k] = 1
        cnt[base][r, k] += 1
        # broaden merge detection: a single blob component wider than ~1 cell is a
        # fused pair (the width metric misses small ones, so this stays a lower bound)
        if base == "blob" and c["w"] > 1.25 * px:
            merge[r, k] = 1
        dx_list.append(c["cx"] - col_c[k]); dy_list.append(c["cy"] - row_c[r])
        # robust size: skip likely partial-merges (too wide) and clip the rms so a
        # few fused blobs don't inflate the local size field (which caused clumps)
        if c["id"] in rms and c["w"] <= 1.3 * px:
            size_acc[base].append((r, k, min(rms[c["id"]], 0.62 * px)))

    # smooth fields
    occ_s = gaussian_filter(occ, SIGMA, mode="nearest")
    cnt_s = {t: gaussian_filter(cnt[t], SIGMA, mode="nearest") for t in cnt}
    tot_s = cnt_s["0"] + cnt_s["1"] + cnt_s["blob"] + 1e-6
    type_p = np.stack([cnt_s[t] / tot_s for t in ("0", "1", "blob")])  # (3,nR,nC)
    merge_s = gaussian_filter(merge, SIGMA, mode="nearest") / (cnt_s["blob"] / (cnt_s["blob"].max() + 1e-6) + 1e-6)
    merge_s = np.clip(gaussian_filter(merge, SIGMA * 1.5, mode="nearest"), 0, 1)

    # per-class size fields (mean rms radius) + residual log std
    size_field = {}
    size_logstd = {}
    for t in ("0", "1", "blob"):
        acc = size_acc[t]
        fld = np.zeros((nR, nC)); wgt = np.zeros((nR, nC))
        for (r, k, v) in acc:
            fld[r, k] += v; wgt[r, k] += 1
        num = gaussian_filter(fld, SIGMA, mode="nearest")
        den = gaussian_filter(wgt, SIGMA, mode="nearest") + 1e-6
        mean_field = num / den
        # fill empties with global mean
        gmean = np.mean([v for _, _, v in acc]) if acc else 20.0
        mean_field[den < 1e-3] = gmean
        size_field[t] = mean_field
        # residual log-std around the local mean
        res = []
        for (r, k, v) in acc:
            res.append(np.log(v + 1e-6) - np.log(mean_field[r, k] + 1e-6))
        size_logstd[t] = float(np.std(res)) if res else 0.15

    np.savez(C.MODEL / "layout.npz",
             occ=occ_s, type_p=type_p, merge=merge_s,
             size_0=size_field["0"], size_1=size_field["1"], size_blob=size_field["blob"])
    # robust jitter: only inliers within half a pitch (excludes mis-assigned rows)
    dxa = np.array(dx_list); dya = np.array(dy_list)
    dx_in = dxa[np.abs(dxa) < px / 2]; dy_in = dya[np.abs(dya) < py / 2]
    jdx = float(np.std(dx_in)); jdy = float(np.std(dy_in))
    scalars = {
        "n_cols": nC, "n_rows": nR, "pitch_x": px, "pitch_y": py,
        "W": grid["W"], "H": grid["H"],
        "col0": int(col_c[0]), "row0": int(row_c[0]),
        "jitter_dx_std": jdx, "jitter_dy_std": jdy,
        "size_logstd": size_logstd,
        "global_occ": float(occ.mean()),
        "counts": {"0": int(cnt['0'].sum()), "1": int(cnt['1'].sum()),
                   "blob": int(cnt['blob'].sum()), "merged": int(merge.sum())},
    }
    (C.MODEL / "layout_scalars.json").write_text(json.dumps(scalars, indent=2))

    print("layout model:")
    print(f"  occupancy: min={occ_s.min():.2f} max={occ_s.max():.2f} mean={occ_s.mean():.2f}")
    print(f"  jitter std (robust): dx={jdx:.1f}px dy={jdy:.1f}px  (raw dy was {np.std(dya):.1f})")
    print(f"  size (rms px) global means: " +
          ", ".join(f"{t}={np.mean([v for _,_,v in size_acc[t]]):.1f}" for t in ('0','1','blob')))
    print(f"  size log-std: {size_logstd}")
    print(f"  merged blobs detected: {int(merge.sum())}")

    # heatmap report
    def hm(field, title):
        f = (255 * (field - field.min()) / (np.ptp(field) + 1e-6)).astype(np.uint8)
        f = cv2.resize(f, (nC * 6, nR * 6), interpolation=cv2.INTER_NEAREST)
        f = cv2.applyColorMap(f, cv2.COLORMAP_INFERNO)
        cv2.putText(f, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return f
    rows = [
        np.hstack([hm(occ_s, "occupancy"), hm(type_p[0], "P(0)")]),
        np.hstack([hm(type_p[1], "P(1)"), hm(type_p[2], "P(blob)")]),
        np.hstack([hm(size_field["blob"], "blob size"), hm(merge_s, "merge")]),
    ]
    cv2.imwrite(str(C.REPORTS / "s5_layout.png"), np.vstack(rows))
    print("wrote reports/s5_layout.png")


if __name__ == "__main__":
    main()
