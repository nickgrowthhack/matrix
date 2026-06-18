"""Stage 2: classify each component as '0', '1', 'blob' (and merged blobs).

Rules:
  holes >= 1                      -> '0'   (inner counter)
  holes == 0 & narrow            -> '1'   (tall thin bar)
  holes == 0 & wide/filled       -> 'blob'
  span_cols >= 2 (or very wide)  -> 'blob_merged'

The 0-hole 1-vs-blob split is learned with KMeans(2) on [log width, log w/h]
over the non-merged 0-hole set, then validated by an overlay.
"""
import json
import numpy as np
import cv2
from sklearn.cluster import KMeans
import common as C


def main():
    comps = json.loads((C.INTERIM / "components.json").read_text())
    grid = json.loads((C.INTERIM / "grid.json").read_text())
    px = grid["pitch_x"]
    gray = C.load_gray()

    for c in comps:
        c["aspect"] = c["w"] / c["h"]
        c["fill"] = c["area"] / (c["w"] * c["h"])  # extent

    # merged blobs: clearly wider than one cell
    for c in comps:
        c["merged"] = c["w"] >= 1.45 * px

    zero = [c for c in comps if c["holes"] >= 1]
    nohole = [c for c in comps if c["holes"] == 0 and not c["merged"]]
    merged = [c for c in comps if c["holes"] == 0 and c["merged"]]

    # KMeans on the no-hole set: narrow '1' vs wide 'blob'
    feats = np.array([[np.log(c["w"]), np.log(c["aspect"])] for c in nohole])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)
    # cluster with smaller mean width is '1'
    mean_w = [np.mean([nohole[i]["w"] for i in range(len(nohole)) if km.labels_[i] == k]) for k in (0, 1)]
    one_cluster = int(np.argmin(mean_w))
    for i, c in enumerate(nohole):
        c["cls"] = "1" if km.labels_[i] == one_cluster else "blob"
    for c in zero:
        c["cls"] = "0"
    for c in merged:
        c["cls"] = "blob_merged"

    # write back
    by_id = {c["id"]: c for c in comps}
    counts = {}
    for c in comps:
        counts[c["cls"]] = counts.get(c["cls"], 0) + 1
    (C.INTERIM / "components_classified.json").write_text(json.dumps(comps, indent=2))

    print("class counts:", dict(sorted(counts.items())))
    for cls in ("0", "1", "blob", "blob_merged"):
        sub = [c for c in comps if c["cls"] == cls]
        if not sub:
            continue
        w = np.array([c["w"] for c in sub]); h = np.array([c["h"] for c in sub])
        a = np.array([c["aspect"] for c in sub]); f = np.array([c["fill"] for c in sub])
        print(f"  {cls:12s} n={len(sub):4d}  w={np.median(w):.0f}[{np.percentile(w,10):.0f}-{np.percentile(w,90):.0f}]"
              f"  h={np.median(h):.0f}  aspect={np.median(a):.2f}[{np.percentile(a,10):.2f}-{np.percentile(a,90):.2f}]"
              f"  fill={np.median(f):.2f}")

    # overlay
    crop_w, crop_h = 1200, 900
    vis = cv2.cvtColor(gray[:crop_h, :crop_w], cv2.COLOR_GRAY2BGR)
    color = {"0": (0, 220, 0), "1": (255, 130, 0), "blob": (0, 80, 255), "blob_merged": (0, 200, 255)}
    for c in comps:
        if c["x"] < crop_w and c["y"] < crop_h:
            cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), color[c["cls"]], 2)
    cv2.imwrite(str(C.REPORTS / "s2_classify.png"), vis)
    print("wrote reports/s2_classify.png (green=0, blue=1, red=blob, yellow=merged)")


if __name__ == "__main__":
    main()
