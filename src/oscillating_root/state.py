from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Params

"""
Simulation state (per timestep).

Conventions / units:
- Two 1D files of cells represent left/right XPP files: A_L, A_R.
- y indexes each cell's position along the root axis (tip at x=0).
- Auxin A_* is stored as per-cell auxin amount in arbitrary units (AU), unless later changed.
"""

@dataclass(slots=True)
class State:
    t: float  # current simulation time [TU]
    y: NDArray[np.float64]  # cell positions along axis, shape (n_cells,) [LU]
    A_L: NDArray[np.float64]  # auxin amount in left file cells, shape (n_cells,) [AU]
    A_R: NDArray[np.float64]  # auxin amount in right file cells, shape (n_cells,) [AU]
    step_idx: int = 0  # integer step counter (0 at init) [dimensionless]


def init_state(p: Params) -> State:
    """
    Initialize simulation state.

    Step 0 placeholder:
    - y is just cell indices (0..n_cells-1) as float
    - y is initialized as cell indices (0..n_cells-1) in LU; later y will be physical positions.
    - auxin arrays are zeros
    """
    p.validate()

    n = p.n_cells
    y = np.arange(n, dtype=np.float64)
    A_L = np.zeros(n, dtype=np.float64)
    A_R = np.zeros(n, dtype=np.float64)

    s = State(t=0.0, y=y, A_L=A_L, A_R=A_R, step_idx=0)
    validate_state(s, p)
    return s


def validate_state(state: State, p: Params) -> None:
    """
    Raise ValueError if state violates basic invariants.
    Invariant: all per-cell arrays are length n_cells and finite.
    """
    n = p.n_cells

    # Shape checks
    if state.y.shape != (n,):
        raise ValueError(f"x must have shape {(n,)}, got {state.y.shape}")
    if state.A_L.shape != (n,):
        raise ValueError(f"A_L must have shape {(n,)}, got {state.A_L.shape}")
    if state.A_R.shape != (n,):
        raise ValueError(f"A_R must have shape {(n,)}, got {state.A_R.shape}")

    # Dtype / numeric sanity
    if not np.issubdtype(state.y.dtype, np.floating):
        raise ValueError(f"x must be floating dtype, got {state.y.dtype}")
    if not np.issubdtype(state.A_L.dtype, np.floating):
        raise ValueError(f"A_L must be floating dtype, got {state.A_L.dtype}")
    if not np.issubdtype(state.A_R.dtype, np.floating):
        raise ValueError(f"A_R must be floating dtype, got {state.A_R.dtype}")

    # Finite checks
    if not np.isfinite(state.t):
        raise ValueError(f"t must be finite, got {state.t}")
    if not np.isfinite(state.y).all():
        raise ValueError("x contains NaN or inf")
    if not np.isfinite(state.A_L).all():
        raise ValueError("A_L contains NaN or inf")
    if not np.isfinite(state.A_R).all():
        raise ValueError("A_R contains NaN or inf")

    # Step index sanity
    if state.step_idx < 0:
        raise ValueError(f"step_idx must be >= 0, got {state.step_idx}")