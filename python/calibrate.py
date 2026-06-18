"""Auto-calibrate the generator's parameters to match the original's measured
statistics. Same measurement (findContours) is applied to the original (targets)
and to each generated candidate, so the loss is unbiased. Powell (derivative-free)
optimizes 7 params; best params -> data/model/gen_params.json (generate.py loads it).
"""
import json
import numpy as np
import cv2
from scipy.optimize import minimize
import common as C
from generate import Generator

MIN_AREA = 60


def measure(gray, col_c, row_c):
    """Measure an 8-bit image (255=glyph). Works for original and generated cov."""
    mask = (gray > 128).astype(np.uint8)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    nR, nC = len(row_c), len(col_c)
    occ = np.zeros((nR, nC))
    H = {"0": [], "1": [], "blob": []}
    n_tall = 0
    if hier is not None:
        hier = hier[0]
        for i, ct in enumerate(cnts):
            if hier[i][3] != -1:           # only outer contours
                continue
            area = cv2.contourArea(ct)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(ct)
            # hole? any child contour of real size
            child = hier[i][2]; has_hole = False
            while child != -1:
                if cv2.contourArea(cnts[child]) > 20:
                    has_hole = True
                child = hier[child][0]
            cls = "0" if has_hole else ("1" if w / h < 0.5 else "blob")
            H[cls].append(h)
            if h > 90:
                n_tall += 1
            cx, cy = x + w / 2, y + h / 2
            cc = int(np.argmin(np.abs(col_c - cx))); rr = int(np.argmin(np.abs(row_c - cy)))
            occ[rr, cc] = 1
    n = sum(len(v) for v in H.values())
    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    m = {
        "white": float(mask.mean()),
        "occ": float(occ.mean()),
        "corr_h": corr(occ[:, :-1], occ[:, 1:]),
        "corr_v": corr(occ[:-1, :], occ[1:, :]),
        "n_tall": n_tall, "n": n,
    }
    for cls in ("0", "1", "blob"):
        a = np.array(H[cls]) if H[cls] else np.array([0.0])
        m[f"{cls}_hm"] = float(a.mean()); m[f"{cls}_hs"] = float(a.std())
        m[f"{cls}_fr"] = len(H[cls]) / max(1, n)
    return m


# weights: (metric, weight). means/fracs/white/occ use relative error; corr absolute/target.
WEIGHTS = {
    "white": 8.0, "occ": 1.0, "corr_h": 1.0, "corr_v": 1.0,
    "0_hm": 2.0, "0_hs": 0.8, "1_hm": 1.5, "1_hs": 0.4, "blob_hm": 1.5, "blob_hs": 0.6,
    "0_fr": 1.0, "1_fr": 1.0, "blob_fr": 1.0,
}


def loss(m, tgt):
    L = 0.0
    for k, w in WEIGHTS.items():
        t = tgt[k]
        L += w * ((m[k] - t) / (abs(t) + 1e-9)) ** 2
    L += 0.2 * ((m["n_tall"] - tgt["n_tall"]) / (tgt["n_tall"] + 5)) ** 2    # chains -> target
    L += 0.5 * ((m["n"] - tgt["n"]) / tgt["n"]) ** 2
    return L


PARAMS = ["occ_gain", "sigma_h", "sigma_v", "blob_cap", "merge_gain", "gain1", "gainb", "var_clip"]
BOUNDS = [(0.98, 1.25), (0.20, 0.95), (0.40, 1.35), (0.44, 0.64), (0.3, 4.0), (0.92, 1.12), (0.92, 1.32), (0.20, 0.50)]
X0 = [1.07, 0.33, 0.78, 0.55, 1.47, 1.02, 1.10, 0.38]
SEED = 12345


def apply(gen, x):
    gen.occ_gain, gen.sigma_h, gen.sigma_v, gen.blob_cap, gen.merge_gain = x[:5]
    gen.gain = {"0": 1.0, "1": x[5], "blob": x[6]}
    gen.var_clip = x[7]


def main():
    gen = Generator(seed=0)
    col_c = np.array(gen.grid["col_centers"]); row_c = np.array(gen.grid["row_centers"])
    orig = cv2.imread(str(C.SRC), cv2.IMREAD_GRAYSCALE)
    tgt = measure(orig, col_c, row_c)
    print("TARGET:", {k: round(v, 4) for k, v in tgt.items()})

    hist = {"best": 1e9, "x": None, "n": 0}

    def objective(x):
        apply(gen, x)
        gen.rng = np.random.default_rng(SEED)
        cov, _ = gen.render()
        m = measure(cov, col_c, row_c)
        L = loss(m, tgt)
        hist["n"] += 1
        if L < hist["best"]:
            hist["best"] = L; hist["x"] = list(x)
            print(f"[{hist['n']:3d}] L={L:.4f} white={m['white']:.3f} cV={m['corr_v']:.2f} "
                  f"h0={m['0_hm']:.0f} hb={m['blob_hm']:.0f} tall={m['n_tall']} n={m['n']} "
                  f"x={[round(v,3) for v in x]}", flush=True)
        return L

    res = minimize(objective, X0, method="Powell", bounds=BOUNDS,
                   options={"maxfev": 160, "xtol": 1e-2, "ftol": 1e-2})

    xb = hist["x"]
    params = {"occ_gain": xb[0], "sigma_h": xb[1], "sigma_v": xb[2],
              "blob_cap": xb[3], "merge_gain": xb[4], "var_clip": xb[7],
              "gain": {"0": 1.0, "1": xb[5], "blob": xb[6]}}
    (C.MODEL / "gen_params.json").write_text(json.dumps(params, indent=2))
    print("\nSAVED gen_params.json:", json.dumps(params))

    # report best metrics vs target
    apply(gen, xb); gen.rng = np.random.default_rng(SEED)
    cov, _ = gen.render(); mb = measure(cov, col_c, row_c)
    print("\n{:10s} {:>10s} {:>10s}".format("metric", "target", "calibrated"))
    for k in ("white", "occ", "corr_h", "corr_v", "0_hm", "0_hs", "1_hm",
              "blob_hm", "blob_hs", "0_fr", "1_fr", "blob_fr", "n_tall", "n"):
        print("{:10s} {:>10.4f} {:>10.4f}".format(k, tgt[k], mb[k]))


if __name__ == "__main__":
    main()
