"""Generative shape model: load Momocs EFA+PCA exports, sample outlines via
inverse Elliptical Fourier. Pure numpy so the generator needs no R at runtime.

coe layout (from Momocs efourier, norm=TRUE): [A1..AK, B1..BK, C1..CK, D1..DK].
Inverse EFA (centered, a0=c0=0):
  x(t) = Σ A_n cos(nt) + B_n sin(nt)
  y(t) = Σ C_n cos(nt) + D_n sin(nt)
"""
import numpy as np
from pathlib import Path

MODEL = Path(r"C:\matrix\data\model")


def _info_K():
    import csv
    with open(MODEL / "efa_info.csv") as f:
        next(f)
        return int(float(next(f).strip()))


class ShapeModel:
    def __init__(self, cls):
        self.cls = cls
        self.K = _info_K()
        self.mean = np.loadtxt(MODEL / f"efa_{cls}_mean.csv", skiprows=1)            # (4K,)
        self.rotation = np.loadtxt(MODEL / f"efa_{cls}_rotation.csv", skiprows=1, delimiter=",")  # (4K, p)
        self.sdev = np.loadtxt(MODEL / f"efa_{cls}_sdev.csv", skiprows=1)            # (p,)
        self.scores = np.loadtxt(MODEL / f"efa_{cls}_scores.csv", skiprows=1, delimiter=",")      # (n, p)
        if self.rotation.ndim == 1:
            self.rotation = self.rotation[:, None]

    def coe_to_outline(self, coe, M=256):
        K = self.K
        A, B, Cc, D = coe[0:K], coe[K:2 * K], coe[2 * K:3 * K], coe[3 * K:4 * K]
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        ang = np.outer(t, np.arange(1, K + 1))
        x = np.cos(ang) @ A + np.sin(ang) @ B
        y = np.cos(ang) @ Cc + np.sin(ang) @ D
        return np.column_stack([x, y])

    def mean_outline(self, M=256):
        return self.coe_to_outline(self.mean, M)

    def sample_coe(self, rng, npc=None, var_scale=1.0, mode="gauss"):
        p = len(self.sdev)
        npc = p if npc is None else min(npc, p)
        if mode == "gauss":
            s = np.zeros(p)
            s[:npc] = rng.standard_normal(npc) * self.sdev[:npc] * var_scale
        elif mode == "bootstrap":
            s = self.scores[rng.integers(len(self.scores))].copy()
            s[:npc] += rng.standard_normal(npc) * self.sdev[:npc] * 0.25 * var_scale
        else:
            raise ValueError(mode)
        return self.mean + s @ self.rotation.T

    def sample_outline(self, rng, M=256, **kw):
        return self.coe_to_outline(self.sample_coe(rng, **kw), M)

    def real_outline(self, i, M=256):
        """Reconstruct the i-th real glyph from its PC scores (K-harmonic)."""
        coe = self.mean + self.scores[i] @ self.rotation.T
        return self.coe_to_outline(coe, M)
