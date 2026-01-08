from __future__ import annotations

import numpy as np
import warnings

from .config import Params
from .state import State, validate_state
from .growth_helpers import (
    recompute_centers_from_lengths,
    schedule_tau,
    zone_r,
    is_cytoplasmic_zone,
    is_mz,
)


def W_of_y(y: np.ndarray, p: Params) -> np.ndarray:
    """
    Oscillation zone spatial window W(y).

    Current implementation: box window:
      W(y)=1 if y in [oz_center-oz_sigma, oz_center+oz_sigma], else 0.
    """
    if p.oz_sigma <= 0:
        return np.zeros_like(y, dtype=np.float64)
    return (
        (y >= (p.oz_center - p.oz_sigma)) & (y <= (p.oz_center + p.oz_sigma))
    ).astype(np.float64)


def S_of_t(t: float, p: Params) -> float:
    """
    Temporal forcing S(t) used to drive auxin input in the OZ.

    If period_T <= 0, S(t)=S0 (constant).
    Otherwise S(t)=S0 + S1*sin(2π t / period_T).
    """
    if p.period_T <= 0:
        return float(p.S0)
    return float(p.S0 + p.S1 * np.sin(2.0 * np.pi * t / p.period_T))


def _pad_to_n_cells_at_tip(state: State, p: Params) -> State:
    """
    Ensure we have at least n_cells by deterministically adding newborns at the tip.

    This is a safety net for the fixed-size pipeline.
    It should rarely trigger if division/tip-births are working.
    """
    if state.L.size >= p.n_cells:
        return state

    need = int(p.n_cells - state.L.size)

    warnings.warn(
        f"_pad_to_n_cells_at_tip triggered: state has {state.L.size} cells, needs {p.n_cells}. "
        "This should be rare; check division logic / tip-birth settings.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Deterministic newborns
    new_ids = np.arange(state.next_id, state.next_id + need, dtype=np.int64)

    L = np.concatenate((np.full(need, p.newborn_length, dtype=np.float64), state.L))
    A_L = np.concatenate((np.zeros(need, dtype=np.float64), state.A_L))
    A_R = np.concatenate((np.zeros(need, dtype=np.float64), state.A_R))
    ids = np.concatenate((new_ids, state.ids))

    tau_newborn = np.full(
        need, schedule_tau(p.rgrowth_mz, p.newborn_length), dtype=np.float64
    )
    tau = np.concatenate((tau_newborn, state.tau_grow))

    y = recompute_centers_from_lengths(L, state.tip_buffer)

    return State(
        t=state.t,
        y=y,
        L=L,
        A_L=A_L,
        A_R=A_R,
        ids=ids,
        next_id=int(state.next_id + need),
        tip_buffer=state.tip_buffer,
        tau_grow=tau,
        step_idx=state.step_idx,
    )


def _truncate_to_n_cells(state: State, p: Params) -> State:
    """
    Enforce fixed domain size by keeping only the first n_cells (closest to tip).

    This approximates the paper's "remove apical cells near the domain boundary"
    but implemented as a fixed-size buffer for simplicity.
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
        tau_grow=state.tau_grow[:n],
        step_idx=state.step_idx,
    )


def _insert_newborn_at_tip(
    state: State,
    p: Params,
    *,
    newborn_A_L: float = 0.0,
    newborn_A_R: float = 0.0,
) -> State:
    """
    Prepend a newborn cell at the tip (index 0) with a new persistent ID.

    This is *not* the paper's division mechanism; it is an optional inflow mechanism.
    The newborn's growth timer is set to 0 as a placeholder and should be rescheduled
    after recomputing y and r(y).
    """
    new_id = np.int64(state.next_id)

    L = np.concatenate(([p.newborn_length], state.L))
    A_L = np.concatenate(([newborn_A_L], state.A_L))
    A_R = np.concatenate(([newborn_A_R], state.A_R))
    ids = np.concatenate(([new_id], state.ids))

    tau = np.concatenate(([0.0], state.tau_grow))  # placeholder

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
        tau_grow=tau,
        step_idx=state.step_idx,
    )


def step(state: State, p: Params) -> State:
    """
    VDB Paper-like growth + division (1D abstraction).

    Growth:
      - Each cell has a countdown tau_grow.
      - When tau_grow <= 0: add dy to its height (cap at Hmax),
        dilute A in MZ/TZ only, then reschedule tau = 1/(r*H).

    Division (MZ only):
      - If division_rule="double": divide when H >= 2 * newborn_length.
      - If division_rule="fixed":  divide when H >= Hdivision.
      - Divide into two equal daughters (each H/2), prepend one new daughter
        adjacent to the other (in 1D we model as two consecutive entries).
      - Concentrations A_L/A_R are inherited unchanged.

    Geometry:
      - After events, recompute y from cumulative lengths (this shifts shootward cells).
      - Truncate to n_cells.
    """
    p.validate()
    validate_state(state, p)

    t_next = state.t + p.dt

    # Work on copies
    L = state.L.copy()
    A_L = state.A_L.copy()
    A_R = state.A_R.copy()
    ids = state.ids.copy()
    tau = state.tau_grow.copy()
    tip_buffer = float(state.tip_buffer)

    # Optional tip births (NOT paper)
    if p.use_tip_births:
        tip_buffer += p.tip_length_accum_rate * p.dt
        while tip_buffer >= p.newborn_length:
            # Insert a newborn with zero concentration (you can change this later)
            tmp = State(
                t=state.t,
                y=np.zeros_like(L),
                L=L,
                A_L=A_L,
                A_R=A_R,
                ids=ids,
                next_id=state.next_id,
                tip_buffer=tip_buffer,
                tau_grow=tau,
                step_idx=state.step_idx,
            )
            tmp2 = _insert_newborn_at_tip(tmp, p, newborn_A_L=0.0, newborn_A_R=0.0)
            L, A_L, A_R, ids, tau = tmp2.L, tmp2.A_L, tmp2.A_R, tmp2.ids, tmp2.tau_grow
            # next_id updated in tmp2; we’ll carry it via a variable
            next_id = tmp2.next_id
            tip_buffer -= p.newborn_length
        next_id_var = next_id if "next_id" in locals() else state.next_id
    else:
        next_id_var = state.next_id

    # Recompute y before zoning decisions (uses current tip_buffer and L)
    y = recompute_centers_from_lengths(L, tip_buffer)

    # --- Growth step: decrement all taus ---
    tau = tau - p.dt

    # We must process possibly multiple events per cell per dt (rare but possible),
    # so loop until stable.
    # To keep ordering stable, we process from tip to shootward.
    while True:
        idxs = np.where(tau <= 0.0)[0]
        if idxs.size == 0:
            break

        # Process in increasing index order
        for i in idxs:
            # Recompute y locally because earlier events can shift zoning a bit via y recomputation.
            # In this 1D abstraction, zoning is primarily based on position, so we update y each pass.
            y = recompute_centers_from_lengths(L, tip_buffer)
            r_i = zone_r(np.array([y[i]], dtype=np.float64), p)[0]

            # If not in a growing zone, disable timer
            if r_i <= 0.0:
                tau[i] = np.inf
                continue

            oldH = float(L[i])
            if oldH >= p.Hmax:
                L[i] = p.Hmax
                tau[i] = np.inf
                continue

            newH = min(p.Hmax, oldH + p.dy)
            L[i] = newH

            # Dilution only for cytoplasmic growth (MZ/TZ)
            if is_cytoplasmic_zone(float(y[i]), p):
                factor = oldH / newH
                A_L[i] *= factor
                A_R[i] *= factor

            # Reschedule next growth event (add because tau could be negative)
            tau[i] = tau[i] + schedule_tau(r_i, float(L[i]))

    # After growth, recompute y again for division decision
    y = recompute_centers_from_lengths(L, tip_buffer)

    # --- Division (MZ only) ---
    if p.division_rule == "double":
        div_thresholds = 2.0 * p.newborn_length
    else:
        div_thresholds = p.Hdivision

    # Find MZ cells that exceed threshold
    # (We’ll process tip->shootward to keep deterministic)
    div_candidates = np.where((y < p.mz_end) & (L >= div_thresholds))[0]

    # Perform divisions; note this changes array sizes, so do it in a loop
    for i in div_candidates.tolist():
        # Because arrays may have grown during earlier divisions, recompute y and re-check index validity
        y = recompute_centers_from_lengths(L, tip_buffer)
        if i >= L.size:
            continue
        if not is_mz(float(y[i]), p):
            continue

        H = float(L[i])
        if H < div_thresholds:
            continue

        # Split equally
        H_d = 0.5 * H
        if H_d <= 0:
            continue

        # Mother becomes one daughter, and we insert a new daughter adjacent (at same position index)
        L[i] = H_d
        # Concentrations inherited unchanged in this representation
        A_L_i = float(A_L[i])
        A_R_i = float(A_R[i])

        # Insert new daughter *before* i (closer to tip) so it stays adjacent
        # (Choice is arbitrary; what matters is you get two adjacent daughters)
        L = np.insert(L, i, H_d)
        A_L = np.insert(A_L, i, A_L_i)
        A_R = np.insert(A_R, i, A_R_i)

        new_id = np.int64(next_id_var)
        next_id_var += 1
        ids = np.insert(ids, i, new_id)

        # Schedule tau for both daughters based on their updated positions after recompute
        tau = np.insert(tau, i, 0.0)  # placeholder; we’ll schedule after recompute

        # After insertion, recompute y and set tau for i and i+1
        y = recompute_centers_from_lengths(L, tip_buffer)
        for j in (i, i + 1):
            r_j = zone_r(np.array([y[j]], dtype=np.float64), p)[0]
            tau[j] = schedule_tau(r_j, float(L[j])) if r_j > 0 else np.inf

    # Recompute y after divisions
    y = recompute_centers_from_lengths(L, tip_buffer)

    # --- Enforce fixed n_cells using helpers ---
    new_state = State(
        t=state.t,  # still old time; we advance to t_next later
        y=y.astype(np.float64),
        L=L.astype(np.float64),
        A_L=A_L.astype(np.float64),
        A_R=A_R.astype(np.float64),
        ids=ids.astype(np.int64),
        next_id=int(next_id_var),
        tip_buffer=float(max(tip_buffer, 0.0)),
        tau_grow=tau.astype(np.float64),
        step_idx=state.step_idx,
    )

    # If we're short, pad with deterministic newborns at the tip (rare)
    new_state = _pad_to_n_cells_at_tip(new_state, p)

    # If we're long, truncate the farthest shootward cells
    new_state = _truncate_to_n_cells(new_state, p)

    # Pull arrays back out (keeps the rest of your code unchanged)
    y = new_state.y
    L = new_state.L
    A_L = new_state.A_L
    A_R = new_state.A_R
    ids = new_state.ids
    tau = new_state.tau_grow
    tip_buffer = float(new_state.tip_buffer)
    next_id_var = int(new_state.next_id)

    # --- Auxin update (uses updated y) ---
    W = W_of_y(y, p)
    S = S_of_t(t_next, p)
    I = W * S
    A_L = A_L + p.dt * (p.k_in * I - p.d * A_L)
    A_R = A_R + p.dt * (p.k_in * I - p.d * A_R)

    new_state = State(
        t=t_next,
        y=y.astype(np.float64),
        L=L.astype(np.float64),
        A_L=A_L.astype(np.float64),
        A_R=A_R.astype(np.float64),
        ids=ids.astype(np.int64),
        next_id=int(next_id_var),
        tip_buffer=float(max(tip_buffer, 0.0)),
        tau_grow=tau.astype(np.float64),
        step_idx=state.step_idx + 1,
    )
    # ensure stored finite taus are nonnegative (numerical safety)
    finite = np.isfinite(tau)
    tau[finite] = np.maximum(tau[finite], 0.0)
    validate_state(new_state, p)
    return new_state
