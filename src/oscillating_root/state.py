from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Params

"""
Simulation state (per timestep).

Conventions / units:
- Two 1D files of cells represent left/right XPP files: A_L, A_R.
- y is each cell's position along the root axis (tip at y=0, increasing shootward).
- Auxin A_* is per-cell auxin amount in arbitrary units (AU).
- ids are persistent unique cell identifiers (do not depend on index).
"""


@dataclass(slots=True)
class State:
    t: float  # current simulation time [TU]
    y: NDArray[np.float64]  # cell positions along axis, shape (n_cells,) [LU]
    A_L: NDArray[np.float64]  # auxin amount in left file cells, shape (n_cells,) [AU]
    A_R: NDArray[np.float64]  # auxin amount in right file cells, shape (n_cells,) [AU]

    ids: NDArray[np.int64]  # persistent unique IDs, shape (n_cells,) [dimensionless]
    next_id: int  # next unused ID for newborn cells [dimensionless]

    L: NDArray[np.float64]  # cell length, shape (n_cells,) [LU]
    tip_buffer: float  # accumulated length at tip not yet converted to a new cell [LU]

    step_idx: int = 0  # integer step counter (0 at init) [dimensionless]


def init_state(p: Params) -> State:
    """
    Initialize simulation state.

    Milestone 1 initialization:
    - y starts as evenly spaced newborn positions: 0, insert_spacing, 2*insert_spacing, ...
    - auxin arrays are zeros
    - ids are 0..n_cells-1
    - next_id starts at n_cells
    """
    p.validate()

    n = p.n_cells
    L = np.full(n, p.newborn_length, dtype=np.float64)
    # centers from edges:
    edges = np.concatenate(([0.0], np.cumsum(L)))
    y = 0.5 * (edges[:-1] + edges[1:])

    A_L = np.zeros(n, dtype=np.float64)
    A_R = np.zeros(n, dtype=np.float64)

    ids = np.arange(n, dtype=np.int64)
    next_id = int(n)

    s = State(
        t=0.0,
        y=y,
        L=L,
        A_L=A_L,
        A_R=A_R,
        ids=ids,
        next_id=next_id,
        tip_buffer=0.0,
        step_idx=0,
    )
    validate_state(s, p)
    return s


def validate_state(state: State, p: Params) -> None:
    """
    Raise ValueError if state violates basic invariants.
    Invariant: all per-cell arrays are length n_cells, finite where applicable, and IDs are unique.
    """
    n = p.n_cells

    # Shape checks
    if state.y.shape != (n,):
        raise ValueError(f"y must have shape {(n,)}, got {state.y.shape}")
    if state.A_L.shape != (n,):
        raise ValueError(f"A_L must have shape {(n,)}, got {state.A_L.shape}")
    if state.A_R.shape != (n,):
        raise ValueError(f"A_R must have shape {(n,)}, got {state.A_R.shape}")
    if state.ids.shape != (n,):
        raise ValueError(f"ids must have shape {(n,)}, got {state.ids.shape}")
    if state.L.shape != (n,):
        raise ValueError(f"L must have shape {(n,)}, got {state.L.shape}")

    # Dtype / numeric sanity
    if not np.issubdtype(state.y.dtype, np.floating):
        raise ValueError(f"y must be floating dtype, got {state.y.dtype}")
    if not np.issubdtype(state.A_L.dtype, np.floating):
        raise ValueError(f"A_L must be floating dtype, got {state.A_L.dtype}")
    if not np.issubdtype(state.A_R.dtype, np.floating):
        raise ValueError(f"A_R must be floating dtype, got {state.A_R.dtype}")
    if not np.issubdtype(state.ids.dtype, np.integer):
        raise ValueError(f"ids must be integer dtype, got {state.ids.dtype}")

    # Finite checks
    if not np.isfinite(state.t):
        raise ValueError(f"t must be finite, got {state.t}")
    if not np.isfinite(state.y).all():
        raise ValueError("y contains NaN or inf")
    if not np.isfinite(state.A_L).all():
        raise ValueError("A_L contains NaN or inf")
    if not np.isfinite(state.A_R).all():
        raise ValueError("A_R contains NaN or inf")
    if not np.isfinite(state.L).all():
        raise ValueError("L contains NaN or inf")
    if not np.isfinite(state.tip_buffer):
        raise ValueError("tip_buffer must be finite")

    # Ordering invariant (important for neighbor coupling later)
    if not np.all(np.diff(state.y) >= 0):
        raise ValueError(
            "y must be nondecreasing (cells indexed in increasing y order)"
        )

    # IDs: must be unique and nonnegative
    if np.any(state.ids < 0):
        raise ValueError("ids must be nonnegative")
    if len(set(state.ids.tolist())) != n:
        raise ValueError("ids must be unique within a state")

    # next_id sanity
    if state.next_id < 0:
        raise ValueError(f"next_id must be >= 0, got {state.next_id}")

    # Step index sanity
    if state.step_idx < 0:
        raise ValueError(f"step_idx must be >= 0, got {state.step_idx}")

    # Length sanity
    if np.any(state.L <= 0):
        raise ValueError("All L must be > 0")

    # Tip buffer sanity
    if state.tip_buffer < 0:
        raise ValueError("tip_buffer must be >= 0")
