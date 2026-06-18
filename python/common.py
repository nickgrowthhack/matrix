"""Shared paths and helpers for the Matrix glyph-mold pipeline."""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"C:\matrix")
SRC = ROOT / "background.png"
INTERIM = ROOT / "data" / "interim"
GLYPHS = ROOT / "data" / "glyphs"
MODEL = ROOT / "data" / "model"
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"
for _d in (INTERIM, GLYPHS, MODEL, OUTPUT, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# Source is pristine 2-color: bg #323332 (~50), glyph #ffffff (255).
THRESH = 128
BG_RGB = (50, 51, 50)  # measured
FG_RGB = (255, 255, 255)


def load_gray():
    g = cv2.imread(str(SRC), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(SRC)
    return g


def binarize(gray):
    """Return uint8 mask, 1 = glyph (white), 0 = background."""
    return (gray > THRESH).astype(np.uint8)
