"""Stage 7: quantitative validation — re-segment the generated image and compare
its distributions against the original (the 'respect the patterns' proof)."""
import json
import numpy as np
import cv2
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C


def analyze(gray, grid):
    mask = C.binarize(gray)
    labels = measure.label(mask, connectivity=2)
    cc = np.array(grid["col_centers"]); rc = np.array(grid["row_centers"])
    nR, nC = grid["n_rows"], grid["n_cols"]
    comps = []
    for p in measure.regionprops(labels):
        if p.area < 60:
            continue
        minr, minc, maxr, maxc = p.bbox
        w, h = maxc - minc, maxr - minr
        cy, cx = p.centroid
        holes = int(1 - p.euler_number)
        cls = "0" if holes >= 1 else ("1" if w / h < 0.5 else "blob")
        comps.append(dict(w=w, h=h, cls=cls,
                          col=int(np.argmin(np.abs(cc - cx))),
                          row=int(np.argmin(np.abs(rc - cy)))))
    return comps, mask


def per_row(comps, nR, key=None, val=None):
    occ = np.zeros(nR)
    for c in comps:
        if 0 <= c["row"] < nR:
            if key is None or c[key] == val:
                occ[c["row"]] += 1
    return occ


def main():
    grid = json.loads((C.INTERIM / "grid.json").read_text())
    nR, nC = grid["n_rows"], grid["n_cols"]
    orig = json.loads((C.INTERIM / "components_classified.json").read_text())
    orig = [c for c in orig if c["cls"] != "blob_merged"] + \
           [dict(w=c["w"], h=c["h"], cls="blob", col=c["col"], row=c["row"])
            for c in orig if c["cls"] == "blob_merged"]
    gen_gray = cv2.imread(str(C.OUTPUT / "generated.png"), cv2.IMREAD_GRAYSCALE)
    gen, gmask = analyze(gen_gray, grid)
    omask = C.binarize(C.load_gray())

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 1. occupancy per row
    ax[0, 0].plot(per_row(orig, nR) / nC, label="orig", lw=2)
    ax[0, 0].plot(per_row(gen, nR) / nC, label="gen", lw=2)
    ax[0, 0].set_title("occupancy per row"); ax[0, 0].legend(); ax[0, 0].set_xlabel("row")

    # 2. type fraction per row
    for cls, col in (("0", "tab:green"), ("1", "tab:blue"), ("blob", "tab:red")):
        ax[0, 1].plot(per_row(orig, nR, "cls", cls) / nC, col, lw=2, label=f"orig {cls}")
        ax[0, 1].plot(per_row(gen, nR, "cls", cls) / nC, col, lw=1.2, ls="--", label=f"gen {cls}")
    ax[0, 1].set_title("type count per row (solid=orig, dashed=gen)"); ax[0, 1].legend(fontsize=7)

    # 3. white-pixel density per image row (downsampled)
    od = omask.reshape(omask.shape[0], -1).mean(1)
    gd = gmask.reshape(gmask.shape[0], -1).mean(1)
    ax[0, 2].plot(od, label="orig", lw=1); ax[0, 2].plot(gd, label="gen", lw=1)
    ax[0, 2].set_title("white-pixel density per pixel-row"); ax[0, 2].legend()

    # 4-6 size (height) hist per class
    for i, cls in enumerate(("0", "1", "blob")):
        oh = [c["h"] for c in orig if c["cls"] == cls]
        gh = [c["h"] for c in gen if c["cls"] == cls]
        a = ax[1, i]
        a.hist(oh, bins=30, alpha=0.5, density=True, label="orig")
        a.hist(gh, bins=30, alpha=0.5, density=True, label="gen")
        a.set_title(f"'{cls}' height dist  (orig n={len(oh)}, gen n={len(gh)})")
        a.legend()

    plt.tight_layout()
    plt.savefig(C.REPORTS / "s7_validation.png", dpi=80)
    print("wrote reports/s7_validation.png")

    # numeric summary
    def counts(cs):
        d = {}
        for c in cs:
            d[c["cls"]] = d.get(c["cls"], 0) + 1
        return d
    print("counts  orig:", counts(orig))
    print("counts  gen :", counts(gen))
    print(f"white fraction  orig={omask.mean():.4f}  gen={gmask.mean():.4f}")
    for cls in ("0", "1", "blob"):
        oh = np.array([c["h"] for c in orig if c["cls"] == cls])
        gh = np.array([c["h"] for c in gen if c["cls"] == cls])
        print(f"  '{cls}' height  orig {oh.mean():.1f}±{oh.std():.1f}   gen {gh.mean():.1f}±{gh.std():.1f}")


if __name__ == "__main__":
    main()
