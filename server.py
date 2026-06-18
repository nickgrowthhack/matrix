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
from generate import Generator, render_rgba  # noqa: E402

app = FastAPI(title="Matrix — molde generativo")
_gen = Generator(seed=0)      # load the mold (shape + layout models) once
_lock = threading.Lock()      # render() mutates gen.rng -> serialize requests
_ver = {"n": 0}


@app.get("/generate")
def generate(seed: int | None = None):
    if seed is None:
        seed = random.randint(0, 1_000_000_000)
    with _lock:
        rgba, placed = render_rgba(_gen, seed)
        out = C.OUTPUT / f"generated_seed{seed}.png"
        cv2.imwrite(str(out), rgba)
        _ver["n"] += 1
    return JSONResponse({
        "seed": seed,
        "placed": placed,
        "total": int(sum(placed.values())),
        "url": f"/output/{out.name}?v={_ver['n']}",
    })


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
