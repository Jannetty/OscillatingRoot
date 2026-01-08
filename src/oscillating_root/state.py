from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .config import Params
from .growth_helpers import zone_r, schedule_tau


@dataclass(slots=True)
class State:
    """
    Simulation state at a single timestep.

    Conventions:
      - Index i corresponds to the i-th cell along the file, ordered by position (tip→shoot).
      - Arrays y, L, A_L, A_R, ids, tau_grow all have shape (n_cells,).

    Interpretation:
      - y[i] is the *center position* of cell i along the root axis [µm].
      - L[i] is the *cell height/length* along y for cell i [µm].
      - A_L[i], A_R[i] are auxin *concentrations* (or concentration-like scalars) [AU].
        We keep two arrays because you model two files (left/right) with identical geometry.
      - ids[i] is a persistent unique identifier for the lineage of the cell currently occupying slot i.
        IDs move with cells as they shift shootward; indices do *not* persist.
      - tau_grow[i] is the remaining time [hours] until the next discrete growth event for cell i.
        When tau_grow[i] <= 0, the cell grows by +dy and tau_grow[i] is rescheduled.
      - tip_buffer is accumulated length at the tip [µm] not yet converted into a new newborn cell
        (only used if Params.use_tip_births=True).
    """

    t: float  # current time [hours]
    y: NDArray[np.float64]  # cell center positions along y [µm], shape (n_cells,)
    L: NDArray[np.float64]  # cell lengths/heights [µm], shape (n_cells,)

    A_L: NDArray[np.float64]  # auxin concentration in left file cells, shape (n_cells,)
    A_R: NDArray[
        np.float64
    ]  # auxin concentration in right file cells, shape (n_cells,)

    ids: NDArray[np.int64]  # persistent unique cell IDs, shape (n_cells,)
    next_id: int  # next unused ID for newly created cells

    tip_buffer: float  # tip length reservoir [µm] (optional; see Params.use_tip_births)
    tau_grow: NDArray[np.float64]  # time-to-next-growth-event [hours], shape (n_cells,)

    step_idx: int = 0  # integer step counter


def init_state(p: Params) -> State:
    p.validate()
    n = p.n_cells

    # initial lengths/heights
    L = np.full(n, p.newborn_length, dtype=np.float64)

    # centers from lengths (tip_buffer = 0 at init)
    edges = np.concatenate(([0.0], np.cumsum(L)))
    y = 0.5 * (edges[:-1] + edges[1:])

    A_L = np.zeros(n, dtype=np.float64)
    A_R = np.zeros(n, dtype=np.float64)

    ids = np.arange(n, dtype=np.int64)
    next_id = int(n)
    tip_buffer = 0.0

    # schedule initial growth timers
    r = zone_r(y, p)  # (n,)
    tau_grow = np.empty(n, dtype=np.float64)
    for i in range(n):
        tau_grow[i] = schedule_tau(float(r[i]), float(L[i]))  # returns inf if r<=0

    s = State(
        t=0.0,
        y=y,
        L=L,
        A_L=A_L,
        A_R=A_R,
        ids=ids,
        next_id=next_id,
        tip_buffer=tip_buffer,
        tau_grow=tau_grow,
        step_idx=0,
    )
    validate_state(s, p)
    return s


def validate_state(state: State, p: Params) -> None:
    n = p.n_cells

    if state.y.shape != (n,):
        raise ValueError(f"y must have shape {(n,)}, got {state.y.shape}")
    if state.L.shape != (n,):
        raise ValueError(f"L must have shape {(n,)}, got {state.L.shape}")
    if state.A_L.shape != (n,):
        raise ValueError(f"A_L must have shape {(n,)}, got {state.A_L.shape}")
    if state.A_R.shape != (n,):
        raise ValueError(f"A_R must have shape {(n,)}, got {state.A_R.shape}")
    if state.ids.shape != (n,):
        raise ValueError(f"ids must have shape {(n,)}, got {state.ids.shape}")
    if state.tau_grow.shape != (n,):
        raise ValueError(f"tau_grow must have shape {(n,)}, got {state.tau_grow.shape}")

    if not np.isfinite(state.t):
        raise ValueError("t must be finite")
    if not np.isfinite(state.y).all():
        raise ValueError("y contains NaN/inf")
    if not np.isfinite(state.L).all():
        raise ValueError("L contains NaN/inf")
    if not np.isfinite(state.A_L).all():
        raise ValueError("A_L contains NaN/inf")
    if not np.isfinite(state.A_R).all():
        raise ValueError("A_R contains NaN/inf")
    # tau_grow may be +inf for non-growing zones (r=0), but must not be NaN
    if np.isnan(state.tau_grow).any():
        raise ValueError("tau_grow contains NaN")
    # finite timers must be nonnegative in stored state
    finite_tau = np.isfinite(state.tau_grow)
    if np.any(state.tau_grow[finite_tau] < 0):
        raise ValueError("finite tau_grow must be >= 0 in stored state")
    if not np.isfinite(state.tip_buffer):
        raise ValueError("tip_buffer must be finite")

    if np.any(state.L <= 0):
        raise ValueError("All L must be > 0")
    if state.tip_buffer < 0:
        raise ValueError("tip_buffer must be >= 0")
    if np.any(state.tau_grow < 0):
        # allowed transiently inside step, but state should store nonnegative
        raise ValueError("tau_grow must be >= 0 in stored state")

    if not np.all(np.diff(state.y) >= 0):
        raise ValueError("y must be nondecreasing")

    if np.any(state.ids < 0):
        raise ValueError("ids must be nonnegative")
    if len(set(state.ids.tolist())) != n:
        raise ValueError("ids must be unique")
    if state.next_id < 0:
        raise ValueError("next_id must be >= 0")
    if state.step_idx < 0:
        raise ValueError("step_idx must be >= 0")

    # Geometry consistency: y should be centers implied by L and tip_buffer
    y_expected = 0.5 * (
        (state.tip_buffer + np.concatenate(([0.0], np.cumsum(state.L))))[:-1]
        + (state.tip_buffer + np.concatenate(([0.0], np.cumsum(state.L))))[1:]
    )
    if not np.allclose(state.y, y_expected, rtol=0, atol=1e-9):
        raise ValueError(
            "y is inconsistent with L and tip_buffer (did you forget to recompute centers?)"
        )
    if np.any(state.ids >= state.next_id):
        raise ValueError(
            "Found id >= next_id (next_id should always exceed all existing ids)"
        )
