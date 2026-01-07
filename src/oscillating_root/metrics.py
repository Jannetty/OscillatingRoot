from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .config import Params

"""
Metrics computed from saved simulation frames.

All metrics are derived from frames saved by scripts/run_one.py.
Units follow the same conventions as config/state:
- time [TU], position [LU], auxin [AU].
"""


def compute_basic_metrics(frames: Dict[str, np.ndarray], p: Params) -> Dict[str, Any]:
    """
    Compute basic summary metrics for a simulation run.

    Parameters
    ----------
    frames : dict of np.ndarray
        Expected keys (Step 0):
          - "t": (n_frames,)
          - "A_L": (n_frames, n_cells)
          - "A_R": (n_frames, n_cells)
    p : Params
        Simulation parameters.

    Returns
    -------
    metrics : dict
        JSON-serializable summary metrics.
    """
    required = ["t", "A_L", "A_R"]
    for key in required:
        if key not in frames:
            raise KeyError(f"frames missing required key '{key}'")

    t = frames["t"]
    A_L = frames["A_L"]
    A_R = frames["A_R"]

    # Basic shape sanity
    if t.ndim != 1:
        raise ValueError(f"t must be 1D, got shape {t.shape}")
    if A_L.ndim != 2 or A_R.ndim != 2:
        raise ValueError("A_L and A_R must be 2D arrays (n_frames, n_cells)")

    metrics: Dict[str, Any] = {
        # ---- Run metadata ----
        "n_steps": int(p.n_steps),         # number of integration steps requested [dimensionless]
        "dt": float(p.dt),                 # timestep size [TU]
        "seed": int(p.seed),               # RNG seed [dimensionless]
        "n_cells": int(p.n_cells),         # cells per file [dimensionless]
        "n_frames": int(t.shape[0]),       # number of saved frames [dimensionless]
        "t_final": float(t[-1]),           # final saved time [TU]

        # ---- Auxin statistics: Left file ----
        "A_L_min": float(np.min(A_L)),     # min auxin amount over all frames/cells [AU]
        "A_L_max": float(np.max(A_L)),     # max auxin amount over all frames/cells [AU]
        "A_L_mean": float(np.mean(A_L)),   # mean auxin amount over all frames/cells [AU]

        # ---- Auxin statistics: Right file ----
        "A_R_min": float(np.min(A_R)),     # [AU]
        "A_R_max": float(np.max(A_R)),     # [AU]
        "A_R_mean": float(np.mean(A_R)),   # [AU]
    }

    return metrics