"""Stage 3: extract & resample outlines for the shape model.

For each non-merged component:
  - outer outline (all classes)
  - inner outline = largest hole (class '0' only)
Outlines are centered on the component centroid (translation removed) but kept at
ORIGINAL scale (size is modeled later by the layout stage). Each is resampled to
N points by arc length, orientation normalized (CCW), and exported as CSV for R.

Outputs:
  data/glyphs/outlines_0_outer.csv, _0_inner.csv, _1.csv, _blob.csv
      columns: id,k,x,y   (x,y relative to centroid)
  data/glyphs/meta.csv    id,cls,cx,cy,w,h,area,col,row,scale
  reports/s3_outlines.png montage: resampled outlines over original glyphs
"""
import json
import numpy as np
import cv2
import common as C

N = 128  # points per outline


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def resample_closed(cnt, n=N):
    pts = cnt.astype(np.float64)
    pts = np.vstack([pts, pts[0]])
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
    d = np.concatenate([[0], np.cumsum(seg)])
    total = d[-1]
    if total == 0:
        return None
    t = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t, d, pts[:, 0])
    y = np.interp(t, d, pts[:, 1])
    out = np.column_stack([x, y])
    if signed_area(out) < 0:          # enforce CCW
        out = out[::-1]
    return out


def get_contours(region_img):
    """region_img: bool mask of the component (bbox-local). Return outer, inner|None."""
    m = (region_img.astype(np.uint8)) * 255
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    hier = hier[0]
    outer_idx = max((i for i in range(len(cnts)) if hier[i][3] == -1),
                    key=lambda i: cv2.contourArea(cnts[i]))
    outer = cnts[outer_idx][:, 0, :]
    holes = [cnts[i][:, 0, :] for i in range(len(cnts)) if hier[i][3] == outer_idx]
    inner = max(holes, key=lambda c: cv2.contourArea(c.reshape(-1, 1, 2))) if holes else None
    return outer, inner


def main():
    from skimage import measure
    comps = json.loads((C.INTERIM / "components_classified.json").read_text())
    by_id = {c["id"]: c for c in comps}
    gray = C.load_gray()
    mask = C.binarize(gray)
    labels = measure.label(mask, connectivity=2)
    props = {p.label: p for p in measure.regionprops(labels)}

    rows = {"0_outer": [], "0_inner": [], "1": [], "blob": []}
    meta = []
    for c in comps:
        if c["cls"] == "blob_merged":
            continue
        p = props.get(c["id"])
        if p is None:
            continue
        outer, inner = get_contours(p.image)
        if outer is None:
            continue
        minr, minc = p.bbox[0], p.bbox[1]
        cx, cy = c["cx"], c["cy"]
        oc = resample_closed(outer)
        if oc is None:
            continue
        # to global coords then center on centroid
        oc_g = oc + [minc, minr] - [cx, cy]
        scale = float(np.sqrt(p.area))
        meta.append([c["id"], c["cls"], cx, cy, c["w"], c["h"], c["area"], c["col"], c["row"], scale])
        key = "0_outer" if c["cls"] == "0" else c["cls"]
        for k, (x, y) in enumerate(oc_g):
            rows[key].append((c["id"], k, x, y))
        if c["cls"] == "0" and inner is not None:
            ic = resample_closed(inner)
            if ic is not None:
                ic_g = ic + [minc, minr] - [cx, cy]
                for k, (x, y) in enumerate(ic_g):
                    rows["0_inner"].append((c["id"], k, x, y))

    for key, data in rows.items():
        path = C.GLYPHS / f"outlines_{key}.csv"
        with open(path, "w") as f:
            f.write("id,k,x,y\n")
            for r in data:
                f.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.3f}\n")
        print(f"{path.name}: {len(data)//N} outlines")
    with open(C.GLYPHS / "meta.csv", "w") as f:
        f.write("id,cls,cx,cy,w,h,area,col,row,scale\n")
        for m in meta:
            f.write(",".join(str(x) for x in m) + "\n")
    print(f"meta.csv: {len(meta)} glyphs")

    # ---- validation montage ----
    def montage(cls, key, has_inner=False, ncol=8, nrow=4, cell=90, scale=2):
        ids = [c["id"] for c in comps if c["cls"] == cls and props.get(c["id"])]
        rng = np.random.default_rng(7)
        ids = rng.choice(ids, size=min(ncol * nrow, len(ids)), replace=False)
        outl = {}
        for (i, k, x, y) in rows[key]:
            outl.setdefault(i, []).append((x, y))
        inl = {}
        if has_inner:
            for (i, k, x, y) in rows["0_inner"]:
                inl.setdefault(i, []).append((x, y))
        canvas = np.full((nrow * cell * scale, ncol * cell * scale, 3), 30, np.uint8)
        for idx, gid in enumerate(ids):
            r, cc = idx // ncol, idx % ncol
            comp = by_id[gid]
            pad = 8
            x0 = max(0, comp["x"] - pad); y0 = max(0, comp["y"] - pad)
            x1 = comp["x"] + comp["w"] + pad; y1 = comp["y"] + comp["h"] + pad
            sub = gray[y0:y1, x0:x1]
            tile = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
            tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            cxl, cyl = comp["cx"] - x0, comp["cy"] - y0
            pts = (np.array(outl[gid]) + [cxl, cyl]) * scale
            cv2.polylines(tile, [pts.astype(np.int32)], True, (0, 230, 0), 1)
            if has_inner and gid in inl:
                ip = (np.array(inl[gid]) + [cxl, cyl]) * scale
                cv2.polylines(tile, [ip.astype(np.int32)], True, (0, 160, 255), 1)
            th, tw = tile.shape[:2]
            yy, xx = r * cell * scale, cc * cell * scale
            canvas[yy:yy + min(th, cell * scale), xx:xx + min(tw, cell * scale)] = \
                tile[:cell * scale, :cell * scale]
        return canvas

    m0 = montage("0", "0_outer", has_inner=True)
    m1 = montage("1", "1")
    mb = montage("blob", "blob")
    sep = np.full((6, m0.shape[1], 3), 80, np.uint8)
    full = np.vstack([m0, sep, m1, sep, mb])
    cv2.imwrite(str(C.REPORTS / "s3_outlines.png"), full)
    print("wrote reports/s3_outlines.png (top: 0 w/ inner, mid: 1, bottom: blob)")


if __name__ == "__main__":
    main()
