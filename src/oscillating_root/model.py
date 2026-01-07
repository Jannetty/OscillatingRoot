from __future__ import annotations

import numpy as np

from .config import Params
from .state import State, validate_state

"""
Core simulation update rule.

- Conveyor-belt advection in 1D (y-axis).
- Persistent cell IDs.
- Insertion at tip when the first cell has moved sufficiently far from y=0.
- Truncation at far end to keep a fixed number of cells (n_cells).
"""


def v_of_y(y: np.ndarray, p: Params) -> np.ndarray:
    """
    Advection speed field v(y).

    Constant.
    """
    return p.v


def _insert_one_cell_at_tip(state: State, p: Params) -> State:
    """
    Insert one newborn cell at the tip (y ~ 0), prepend arrays, assign a new unique ID.
    """
    new_y0 = 0.0
    new_id = np.int64(state.next_id)

    y = np.concatenate(([new_y0], state.y))
    A_L = np.concatenate(([0.0], state.A_L))
    A_R = np.concatenate(([0.0], state.A_R))
    ids = np.concatenate(([new_id], state.ids))

    return State(
        t=state.t,
        y=y,
        A_L=A_L,
        A_R=A_R,
        ids=ids,
        next_id=state.next_id + 1,
        step_idx=state.step_idx,
    )


def _truncate_to_n_cells(state: State, p: Params) -> State:
    """
    Keep only the first n_cells (smallest y / closest to tip).
    This maintains a fixed-size domain and consistent shapes.
    """
    n = p.n_cells
    return State(
        t=state.t,
        y=state.y[:n],
        A_L=state.A_L[:n],
        A_R=state.A_R[:n],
        ids=state.ids[:n],
        next_id=state.next_id,
        step_idx=state.step_idx,
    )


def step(state: State, p: Params) -> State:
    """
    Advance the simulation by one timestep.

    Milestone 1 behavior:
    1) advect all cell positions: y += v(y)*dt
    2) insert newborn cells at tip while there's "room" (y[0] >= insert_spacing)
    3) truncate to keep exactly n_cells
    4) advance time and step_idx

    Notes:
    - Auxin arrays A_L/A_R remain unchanged in Milestone 1.
    - Deterministic given (state, p).
    """
    p.validate()
    validate_state(state, p)

    # 1) Advect
    y_new = state.y + v_of_y(state.y, p) * p.dt

    advected = State(
        t=state.t,  # update time at end
        y=y_new,
        A_L=state.A_L,
        A_R=state.A_R,
        ids=state.ids,
        next_id=state.next_id,
        step_idx=state.step_idx,
    )

    # 2) Insert at tip while first cell has moved beyond insert_spacing
    # (growth-ready: later you can replace insert_spacing with a function of L)
    inserted = advected
    while inserted.y[0] >= p.insert_spacing:
        inserted = _insert_one_cell_at_tip(inserted, p)

    # Optional physical cutoff: drop cells beyond y_max before truncation
    # (Not strictly needed if you always truncate to n_cells, but harmless.)
    if inserted.y.size > 0:
        keep = inserted.y <= p.y_max
        if not np.all(keep):
            inserted = State(
                t=inserted.t,
                y=inserted.y[keep],
                A_L=inserted.A_L[keep],
                A_R=inserted.A_R[keep],
                ids=inserted.ids[keep],
                next_id=inserted.next_id,
                step_idx=inserted.step_idx,
            )

    # 3) Truncate back to fixed n_cells (if we removed too many, this would be a problem;
    # for now, parameters should avoid that).
    truncated = _truncate_to_n_cells(inserted, p)

    # 4) Advance time and step index
    new_state = State(
        t=state.t + p.dt,
        y=truncated.y,
        A_L=truncated.A_L,
        A_R=truncated.A_R,
        ids=truncated.ids,
        next_id=truncated.next_id,
        step_idx=state.step_idx + 1,
    )

    validate_state(new_state, p)
    return new_state