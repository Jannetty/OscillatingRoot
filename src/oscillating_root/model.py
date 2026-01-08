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


def _recompute_centers_from_lengths(L: np.ndarray, tip_buffer: float) -> np.ndarray:
    """Given lengths and a tip offset (buffer), return cell center positions y [LU]."""
    edges = tip_buffer + np.concatenate(([0.0], np.cumsum(L)))
    return 0.5 * (edges[:-1] + edges[1:])


def W_of_y(y: np.ndarray, p: Params) -> np.ndarray:
    """OZ window function W(y). For Milestone 2: simple box window in [center-sigma, center+sigma]."""
    if p.oz_sigma <= 0:
        return np.zeros_like(y, dtype=np.float64)
    return (
        (y >= (p.oz_center - p.oz_sigma)) & (y <= (p.oz_center + p.oz_sigma))
    ).astype(np.float64)


def S_of_t(t: float, p: Params) -> float:
    """Tip-tethered oscillatory driver S(t)."""
    if p.period_T <= 0:
        return float(p.S0)
    return float(p.S0 + p.S1 * np.sin(2.0 * np.pi * t / p.period_T))


def zone_growth_targets(y, p):
    # returns L_target per cell based on zone
    Ltar = np.full_like(y, p.L_dz, dtype=float)
    Ltar[y < p.ez_end] = p.L_ez
    Ltar[y < p.tz_end] = p.L_tz
    Ltar[y < p.mz_end] = p.L_mz
    return Ltar


def zone_rates(y, p):
    k = np.full_like(y, p.k_dz, dtype=float)
    k[y < p.ez_end] = p.k_ez
    k[y < p.tz_end] = p.k_tz
    k[y < p.mz_end] = p.k_mz
    return k


def _insert_one_cell_at_tip(state: State, p: Params) -> State:
    new_id = np.int64(state.next_id)

    L = np.concatenate(([p.newborn_length], state.L))
    A_L = np.concatenate(([0.0], state.A_L))
    A_R = np.concatenate(([0.0], state.A_R))
    ids = np.concatenate(([new_id], state.ids))

    # placeholder y with correct shape; will be recomputed immediately after insertion loop
    y_placeholder = np.zeros_like(L, dtype=np.float64)

    return State(
        t=state.t,
        y=y_placeholder,
        L=L,
        A_L=A_L,
        A_R=A_R,
        ids=ids,
        next_id=state.next_id + 1,
        tip_buffer=state.tip_buffer,
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
        L=state.L[:n],
        A_L=state.A_L[:n],
        A_R=state.A_R[:n],
        ids=state.ids[:n],
        next_id=state.next_id,
        tip_buffer=state.tip_buffer,
        step_idx=state.step_idx,
    )


def step(state: State, p: Params) -> State:
    """
    Growth-driven conveyor belt step.

    1) grow each cell length:        L_i += growth_rate * dt
    2) accumulate new material at tip: tip_buffer += tip_length_accum_rate * dt
    3) while tip_buffer >= newborn_length: insert a newborn (prepend) and subtract newborn_length
    4) recompute y centers from cumulative lengths + tip_buffer
    5) truncate to exactly n_cells
    6) update auxin with OZ forcing (uses updated y)
    7) advance time and step_idx
    """
    p.validate()
    validate_state(state, p)

    t_next = state.t + p.dt

    # --- 1) grow lengths ---
    L_target = zone_growth_targets(state.y, p)
    k = zone_rates(state.y, p)
    L_new = state.L + p.dt * k * (L_target - state.L)
    L_new = np.clip(L_new, 1e-6, None)

    # --- 2) accumulate tip material ---
    tip_buffer_new = state.tip_buffer + p.tip_length_accum_rate * p.dt

    grown = State(
        t=state.t,  # time updated at end
        y=state.y,  # placeholder; will be recomputed
        L=L_new,
        A_L=state.A_L,
        A_R=state.A_R,
        ids=state.ids,
        next_id=state.next_id,
        tip_buffer=tip_buffer_new,
        step_idx=state.step_idx,
    )

    # --- 3) insert newborns as long as enough tip material exists ---
    tip_buffer = state.tip_buffer + p.tip_length_accum_rate * p.dt

    # Insert as many newborns as you can "pay for"
    inserted = grown
    while tip_buffer >= p.newborn_length:
        inserted = _insert_one_cell_at_tip(inserted, p)
        tip_buffer -= p.newborn_length

    # Recompute centers from lengths and tip_buffer
    y_new = _recompute_centers_from_lengths(inserted.L[: p.n_cells], tip_buffer)

    # --- 4) recompute y from lengths + tip buffer ---
    y_new = _recompute_centers_from_lengths(inserted.L, inserted.tip_buffer)
    repositioned = State(
        t=inserted.t,
        y=y_new,
        L=inserted.L,
        A_L=inserted.A_L,
        A_R=inserted.A_R,
        ids=inserted.ids,
        next_id=inserted.next_id,
        tip_buffer=inserted.tip_buffer,
        step_idx=inserted.step_idx,
    )

    # --- 5) truncate back to fixed n_cells ---
    truncated = _truncate_to_n_cells(repositioned, p)

    # --- 6) OZ forcing + auxin update (uses updated y) ---
    W = W_of_y(truncated.y, p)  # (n_cells,)
    S = S_of_t(t_next, p)  # scalar
    I = W * S  # (n_cells,)

    A_L_new = truncated.A_L + p.dt * (p.k_in * I - p.d * truncated.A_L)
    A_R_new = truncated.A_R + p.dt * (p.k_in * I - p.d * truncated.A_R)

    # --- 7) advance time and step index ---
    new_state = State(
        t=t_next,
        y=truncated.y,
        L=truncated.L,
        A_L=A_L_new,
        A_R=A_R_new,
        ids=truncated.ids,
        next_id=truncated.next_id,
        tip_buffer=truncated.tip_buffer,
        step_idx=state.step_idx + 1,
    )

    validate_state(new_state, p)
    return new_state
