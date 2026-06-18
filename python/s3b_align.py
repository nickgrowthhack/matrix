"""Stage 3b: canonical pre-alignment for EFA WITHOUT rotation normalization, so
the natural upright orientation + slant of the glyphs stays in the model.

Per outline: recenter on polygon centroid, scale by RMS radius (size removed,
modeled separately), roll start point to the 'top' (angle closest to +y).
For class '0' the inner hole is transformed in the OUTER's frame so the ring
geometry (relative hole size/position) is preserved.

Outputs data/glyphs/outlines_<key>_al.csv  (key in 1, blob, 0_outer, 0_inner)
"""
import numpy as np
import common as C

N = 128


def load_outlines(key):
    import csv
    d = {}
    with open(C.GLYPHS / f"outlines_{key}.csv") as f:
        next(f)
        for line in f:
            i, k, x, y = line.split(",")
            d.setdefault(int(i), []).append((int(k), float(x), float(y)))
    for i in d:
        d[i] = np.array([p[1:] for p in sorted(d[i])])
    return d


def roll_to_top(p):
    ang = np.arctan2(p[:, 1], p[:, 0])
    # want start where the curve points "up": angle closest to +pi/2
    j = int(np.argmin(np.abs(((ang - np.pi / 2 + np.pi) % (2 * np.pi)) - np.pi)))
    return np.roll(p, -j, axis=0)


def frame(p):
    """centroid + rms scale of a polygon."""
    c = p.mean(axis=0)
    pc = p - c
    s = np.sqrt((pc ** 2).sum(axis=1).mean())
    return c, s


def write(key, data):
    path = C.GLYPHS / f"outlines_{key}_al.csv"
    with open(path, "w") as f:
        f.write("id,k,x,y\n")
        for i, p in data.items():
            for k, (x, y) in enumerate(p):
                f.write(f"{i},{k},{x:.5f},{y:.5f}\n")
    print(f"{path.name}: {len(data)} outlines")


def main():
    for key in ("1", "blob"):
        d = load_outlines(key)
        out = {}
        for i, p in d.items():
            c, s = frame(p)
            out[i] = roll_to_top((p - c) / s)
        write(key, out)

    # class 0: share outer frame
    outer = load_outlines("0_outer")
    inner = load_outlines("0_inner")
    o_out, i_out = {}, {}
    for i, p in outer.items():
        c, s = frame(p)
        o_out[i] = roll_to_top((p - c) / s)
        if i in inner:
            i_out[i] = roll_to_top((inner[i] - c) / s)
    write("0_outer", o_out)
    write("0_inner", i_out)


if __name__ == "__main__":
    main()
