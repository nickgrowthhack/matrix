"""Local FastAPI server: serves the viewer and a /generate endpoint that runs the
validated Python generator (models loaded once) to produce new faithful variations.

Run:  .venv\\Scripts\\python.exe -m uvicorn server:app --app-dir C:\\matrix --port 8000
Then: http://localhost:8000/view.html
"""
import sys
import random
import threading
import cv2
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, r"C:\matrix\python")
import common as C            # noqa: E402
from generate import Generator, render_rgba, out_name  # noqa: E402

app = FastAPI(title="Matrix — molde generativo")
_gen = Generator(seed=0)      # load the mold (shape + layout models) once
_lock = threading.Lock()      # render() mutates gen.rng -> serialize requests
_ver = {"n": 0}


def _run(mode, seed):
    if seed is None:
        seed = random.randint(0, 1_000_000_000)
    with _lock:
        rgba, placed = render_rgba(_gen, seed, mode=mode)
        out = C.OUTPUT / out_name(mode, seed)
        cv2.imwrite(str(out), rgba)
        _ver["n"] += 1
    return JSONResponse({"seed": seed, "placed": placed,
                         "total": int(sum(placed.values())),
                         "url": f"/output/{out.name}?v={_ver['n']}"})


@app.get("/generate")
def generate(seed: int | None = None):
    """New random variation (each cell's content sampled)."""
    return _run("variation", seed)


@app.get("/reconstruct")
def reconstruct(seed: int | None = None):
    """Faithful reconstruction: same type/position as the original, new shapes."""
    return _run("reconstruct", seed)


@app.get("/extract")
def extract(seed: int | None = None):
    """Extraction test (≈ copy): redraw each glyph from its exact extracted contour."""
    return _run("extract", seed)


@app.get("/outputs")
def outputs():
    """List generated PNGs in output/ NEWEST FIRST, with a readable timestamp."""
    import time
    files = sorted(C.OUTPUT.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "url": f"/output/{p.name}", "m": p.stat().st_mtime,
             "ts": time.strftime("%d/%m %H:%M", time.localtime(p.stat().st_mtime))}
            for p in files]


@app.get("/animation/shapes")
def animation_shapes(n: int = 96, seed: int = 12345):
    """Pool of mold-generated outlines per class — the reconstructor's OWN shape
    engine, exposed so /animation.html can run the same logic live in the browser.

    Each outline is a raw model-space point array (subsampled), so the JS engine
    scales it per glyph exactly like generate.py:_place_fit. 0s come as aligned
    outer+inner pairs (so the hole stays registered); blobs carry their fill
    (area/bbox) so the browser can fill-match (solid left, wispy right)."""
    import numpy as np

    def ds(o):  # 256-pt outline -> ~52 pts, rounded, as [[x,y],...]
        return [[round(float(x), 2), round(float(y), 2)] for x, y in o[::5]]

    out = {"0": [], "1": [], "blob": []}
    with _lock:
        _gen.rng = np.random.default_rng(seed)
        for _ in range(n):
            outer, inner = _gen._boot0()
            out["0"].append({"outer": ds(outer), "inner": ds(inner)})
            out["1"].append({"out": ds(_gen._boot(_gen.sm["1"], min_sol=0.0))})
            ob = _gen._boot(_gen.sm["blob"], min_sol=0.88)
            pts = ob.astype(np.float32)
            bb = float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))
            a = abs(cv2.contourArea(pts))
            out["blob"].append({"out": ds(ob), "fill": round(a / bb if bb > 0 else 0.83, 3)})
    return JSONResponse(out)


@app.get("/")
def root():
    return RedirectResponse("/view.html")


# static files last so explicit routes above take precedence
app.mount("/", StaticFiles(directory=str(C.ROOT), html=True), name="static")
