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


@app.get("/")
def root():
    return RedirectResponse("/view.html")


# static files last so explicit routes above take precedence
app.mount("/", StaticFiles(directory=str(C.ROOT), html=True), name="static")
