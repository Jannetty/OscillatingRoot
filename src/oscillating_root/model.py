from __future__ import annotations

import numpy as np

from .config import Params
from .state import State, validate_state
from .growth_helpers import (
    recompute_centers_from_lengths,
    schedule_tau,
    zone_r,
    is_mz,
)


def W_of_y(y: np.ndarray, p: Params) -> np.ndarray:
    """
    Loading zone spatial window W(y).

    Current implementation: box window:
      W(y)=1 if y in [oz_center-oz_sigma, oz_center+oz_sigma], else 0.
    """
    if p.loading_sigma <= 0:
        return np.zeros_like(y, dtype=np.float64)
    return (
        (y >= (p.loading_center - p.loading_sigma))
        & (y <= (p.loading_center + p.loading_sigma))
    ).astype(np.float64)


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
        J_prev_L=state.J_prev_L[: n - 1],
        J_prev_R=state.J_prev_R[: n - 1],
        A_R=state.A_R[:n],
        pin_rfrac=state.pin_rfrac[:n],
        W_L=state.W_L[: n - 1],
        W_R=state.W_R[: n - 1],
        ids=state.ids[:n],
        next_id=state.next_id,
        tau_grow=state.tau_grow[:n],
        step_idx=state.step_idx,
    )


def zone_piecewise(
    y: np.ndarray, p: Params, mz: float, tz: float, ez: float, dz: float
) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    out[y < p.mz_end] = mz
    out[(y >= p.mz_end) & (y < p.tz_end)] = tz
    out[(y >= p.tz_end) & (y < p.ez_end)] = ez
    out[y >= p.ez_end] = dz
    return out


def pin_tot(y: np.ndarray, p: Params) -> np.ndarray:
    return zone_piecewise(y, p, p.pin_tot_mz, p.pin_tot_tz, p.pin_tot_ez, p.pin_tot_dz)


def aux_base(y: np.ndarray, p: Params) -> np.ndarray:
    return zone_piecewise(
        y, p, p.aux_base_mz, p.aux_base_tz, p.aux_base_ez, p.aux_base_dz
    )


def aux_activity(C: np.ndarray, y: np.ndarray, p: Params) -> np.ndarray:
    base = aux_base(y, p)
    # Hill-like induction by intracellular concentration proxy
    return base * (1.0 + p.aux_alpha * (C / (p.aux_K + C + 1e-12)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pin_target_rfrac(
    C: np.ndarray,
    pin_rfrac: np.ndarray,
    y: np.ndarray,
    p: Params,
    J_prev: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns target right-fraction per cell for hypotheses A/B/C.
    - A: gradient sensing (toward higher neighbor conc)
    - B: flux reinforcement (needs interface flux J_prev, length n-1)
    - C: zone-fixed
    """
    n = C.size
    if p.pin_rule == "C":
        return zone_piecewise(
            y, p, p.pin_rfrac_mz, p.pin_rfrac_tz, p.pin_rfrac_ez, p.pin_rfrac_dz
        )

    if p.pin_rule == "A":  #
        # local forward gradient: prefer exporting to the right when C[i+1] > C[i] (up concentration gradient)
        g = np.zeros(n, dtype=np.float64)
        g[:-1] = C[1:] - C[:-1]
        return sigmoid(p.pin_beta * g)

    if p.pin_rule == "B":
        if J_prev is None:
            # fall back gracefully
            return pin_rfrac.copy()
        # For cell i, compare outgoing right interface flux vs left interface flux.
        # J_prev[j] is flux from cell j -> j+1 (positive means rightward).
        score = np.zeros(n, dtype=np.float64)
        score[0] = +J_prev[0]
        score[-1] = -J_prev[-1]
        score[1:-1] = J_prev[1:] - J_prev[:-1]
        return sigmoid(p.pin_beta * score)

    raise ValueError("Unknown pin_rule")


def transport_no_apoplast(
    A: np.ndarray, L: np.ndarray, y: np.ndarray, pin_rfrac: np.ndarray, p: Params
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns updated (A_new, J) where J is interface flux array (n-1,)
    """
    n = A.size
    eps = 1e-12
    C = A / np.maximum(L, eps)

    PINtot = pin_tot(y, p)
    PIN_R = PINtot * pin_rfrac
    PIN_L = PINtot * (1.0 - pin_rfrac)

    # Interface flux J[i] moves from i -> i+1
    J = p.k_pin * (PIN_R[:-1] * C[:-1] - PIN_L[1:] * C[1:])

    # Optional passive diffusion term
    if p.D_cell > 0:
        J += p.D_cell * (C[:-1] - C[1:])  # simple; units are "effective"

    # Flux limiting to prevent negative amounts
    dt = p.dt
    # If J>0, removing from left cell i
    cap_pos = A[:-1] / dt
    J = np.where(J > 0, np.minimum(J, cap_pos), J)
    # If J<0, removing from right cell i+1
    cap_neg = A[1:] / dt
    J = np.where(J < 0, np.maximum(J, -cap_neg), J)

    A_new = A.copy()
    A_new[:-1] -= dt * J
    A_new[1:] += dt * J
    A_new = np.maximum(A_new, 0.0)
    return A_new, J


def transport_with_apoplast(
    A: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
    pin_rfrac: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns updated (A_new, W_new, J_cell_to_right) where J_cell_to_right is net interface flux (n-1,)
    """
    n = A.size
    eps = 1e-12
    C = A / np.maximum(L, eps)
    Cw = W / max(p.V_wall, eps)

    PINtot = pin_tot(y, p)
    PIN_R = PINtot * pin_rfrac
    PIN_L = PINtot * (1.0 - pin_rfrac)

    AUX = aux_activity(C, y, p)

    dt = p.dt

    # Exports into wall interface i (between i and i+1)
    E_right = p.k_pin * PIN_R[:-1] * C[:-1]  # i -> wall(i)
    E_left = p.k_pin * PIN_L[1:] * C[1:]  # i+1 -> wall(i)

    # Imports from wall interface i into adjacent cells
    I_left = p.k_aux * AUX[:-1] * Cw  # wall(i) -> i
    I_right = p.k_aux * AUX[1:] * Cw  # wall(i) -> i+1

    # Limit exports by available cell amount
    E_right = np.minimum(E_right, A[:-1] / dt)
    E_left = np.minimum(E_left, A[1:] / dt)

    # Limit imports by available wall amount
    I_sum = I_left + I_right
    scale = np.ones_like(I_sum)
    too_big = I_sum > (W / dt + 1e-18)
    scale[too_big] = (W[too_big] / dt) / I_sum[too_big]
    I_left *= scale
    I_right *= scale

    A_new = A.copy()
    W_new = W.copy()

    # Apply exports
    A_new[:-1] -= dt * E_right
    W_new += dt * E_right

    A_new[1:] -= dt * E_left
    W_new += dt * E_left

    # Apply imports
    W_new -= dt * (I_left + I_right)
    A_new[:-1] += dt * I_left
    A_new[1:] += dt * I_right

    # Optional wall diffusion
    if p.D_wall > 0 and W_new.size >= 3:
        # simple diffusion on wall concentration then convert back
        Cw2 = W_new / max(p.V_wall, eps)
        # re-use your diffuser on centers by treating interfaces as equally spaced; or implement 1D laplacian
        # simplest: discrete laplacian with Neumann ends
        lap = np.zeros_like(Cw2)
        lap[1:-1] = Cw2[:-2] - 2 * Cw2[1:-1] + Cw2[2:]
        Cw2 = Cw2 + dt * p.D_wall * lap
        Cw2 = np.maximum(Cw2, 0.0)
        W_new = Cw2 * p.V_wall

    A_new = np.maximum(A_new, 0.0)
    W_new = np.maximum(W_new, 0.0)

    # For diagnostics: net rightward transfer across interface i is (imports into i+1 - exports from i+1 into wall?) etc.
    # A reasonable "net i -> i+1" is:
    Jnet = I_right - E_left  # positive means favors right cell
    return A_new, W_new, Jnet


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

    pin_rfrac = state.pin_rfrac.copy()
    W_L = state.W_L.copy()
    W_R = state.W_R.copy()
    J_prev_L = state.J_prev_L.copy()
    J_prev_R = state.J_prev_R.copy()

    # Get next cell id for division
    next_id_var = state.next_id

    # Recompute y before zoning decisions (uses L)
    y = recompute_centers_from_lengths(L, 0.0)

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
            y = recompute_centers_from_lengths(L, 0.0)
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

            # Amount-based auxin: no dilution on growth

            # Reschedule next growth event (add because tau could be negative)
            tau[i] = tau[i] + schedule_tau(r_i, float(L[i]))

    # After growth, recompute y again for division decision
    y = recompute_centers_from_lengths(L, 0.0)

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
        y = recompute_centers_from_lengths(L, 0.0)
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
        # (Choice is arbitrary; what matters is this results in two adjacent daughters)
        L = np.insert(L, i, H_d)
        A_L = np.insert(A_L, i, A_L_i)
        A_R = np.insert(A_R, i, A_R_i)

        new_id = np.int64(next_id_var)
        next_id_var += 1
        ids = np.insert(ids, i, new_id)

        pin_rfrac = np.insert(pin_rfrac, i, float(pin_rfrac[i]))
        W_L = np.insert(W_L, i, 0.0)
        W_R = np.insert(W_R, i, 0.0)
        J_prev_L = np.insert(J_prev_L, i, 0.0)
        J_prev_R = np.insert(J_prev_R, i, 0.0)

        # Schedule tau for both daughters based on their updated positions after recompute
        tau = np.insert(tau, i, 0.0)  # placeholder; we’ll schedule after recompute

        # After insertion, recompute y and set tau for i and i+1
        y = recompute_centers_from_lengths(L, 0.0)
        for j in (i, i + 1):
            r_j = zone_r(np.array([y[j]], dtype=np.float64), p)[0]
            tau[j] = schedule_tau(r_j, float(L[j])) if r_j > 0 else np.inf

    # Recompute y after divisions
    y = recompute_centers_from_lengths(L, 0.0)

    # --- Enforce fixed n_cells using helpers ---
    new_state = State(
        t=state.t,
        y=y.astype(np.float64),
        L=L.astype(np.float64),
        A_L=A_L.astype(np.float64),
        J_prev_L=J_prev_L.astype(np.float64),
        J_prev_R=J_prev_R.astype(np.float64),
        A_R=A_R.astype(np.float64),
        pin_rfrac=pin_rfrac.astype(np.float64),
        W_L=W_L.astype(np.float64),
        W_R=W_R.astype(np.float64),
        ids=ids.astype(np.int64),
        next_id=int(next_id_var),
        tau_grow=tau.astype(np.float64),
        step_idx=state.step_idx,
    )

    # If we're long, truncate the farthest shootward cells
    new_state = _truncate_to_n_cells(new_state, p)

    # Pull arrays back out (keeps the rest of your code unchanged)
    y = new_state.y
    L = new_state.L
    A_L = new_state.A_L
    A_R = new_state.A_R
    ids = new_state.ids
    tau = new_state.tau_grow
    next_id_var = int(new_state.next_id)

    # Be sure to track transporter arrays
    pin_rfrac = new_state.pin_rfrac
    W_L = new_state.W_L
    W_R = new_state.W_R
    J_prev_L = new_state.J_prev_L
    J_prev_R = new_state.J_prev_R
    # --- Auxin update (uses updated y and updated L) ---
    W = W_of_y(y, p)  # spatial loading window

    # size-dependent loading factor
    L0 = float(p.L0_load) if hasattr(p, "L0_load") else float(p.newborn_length)
    gamma = float(p.gamma_load) if hasattr(p, "gamma_load") else 1.0

    size_factor = (L / L0) ** gamma
    if getattr(p, "cap_load", None) is not None:
        size_factor = np.minimum(size_factor, float(p.cap_load))

    # emergent loading (larger cells can get load auxin)
    I = W * size_factor

    """
    Auxin update (amount-based, transporter-regulated source).

    We treat auxin A as an amount per cell (not a concentration).

    This update represents:
      1) Uptake of auxin from an external source in the loading zone (OZ),
         modulated by cell size and AUX/LAX-like import activity.
      2) First-order auxin turnover / degradation within the cytoplasm.

    Variables:
      - A_L, A_R : auxin amount per cell in the left/right file [AU]
      - L        : cell length [µm], used to convert amount → concentration
      - C_L, C_R : intracellular auxin concentration proxies [AU / µm]
      - AUX_L, AUX_R : effective AUX/LAX import activity per cell (dimensionless),
                       regulated by auxin concentration and zonation
      - I        : spatial loading mask × size-dependent factor
                   (defines where external auxin is available for uptake)

    Update rule (per file):
        dA/dt = k_in * I * AUX(C, y)  -  d * A

    This formulation ensures:
      - Import is localized in space (via I),
      - Import is biologically regulated (via AUX/LAX dependence),
      - Export is handled separately via PIN-mediated transport.
    """
    eps = 1e-12
    # Intracellular auxin concentration proxy (amount normalized by cell size)
    # Used for transporter regulation and PIN polarity, not stored directly
    C_L = A_L / np.maximum(L, eps)
    C_R = A_R / np.maximum(L, eps)

    # Effective AUX/LAX-like import activity per cell
    # This encodes baseline zonal expression + auxin-induced upregulation
    # Dimensionless scaling factor applied to source uptake
    AUX_L = aux_activity(C_L, y, p)
    AUX_R = aux_activity(C_R, y, p)

    # Auxin amount update:
    #   - k_in * I * AUX: transporter-regulated uptake from external source (OZ)
    #   - d * A: first-order intracellular auxin loss
    A_L = A_L + p.dt * (p.k_in * I * AUX_L - p.d * A_L)
    A_R = A_R + p.dt * (p.k_in * I * AUX_R - p.d * A_R)

    # Update PIN polarity (A/B/C)
    eps = 1e-12
    C_for_rule = A_L / np.maximum(L, eps)  # or 0.5*(A_L+A_R)/L if you prefer
    target = pin_target_rfrac(C_for_rule, pin_rfrac, y, p, J_prev=J_prev_L)
    lam = p.dt / p.pin_relax_tau
    pin_rfrac = np.clip((1.0 - lam) * pin_rfrac + lam * target, 0.0, 1.0)

    # Transport: choose backend
    if p.use_apoplast:
        A_L, W_L, J_L = transport_with_apoplast(A_L, W_L, L, y, pin_rfrac, p)
        A_R, W_R, J_R = transport_with_apoplast(A_R, W_R, L, y, pin_rfrac, p)
    else:
        A_L, J_L = transport_no_apoplast(A_L, L, y, pin_rfrac, p)
        A_R, J_R = transport_no_apoplast(A_R, L, y, pin_rfrac, p)
        # W_L/W_R stay as-is (should be zeros if you initialized that way)

    J_prev_L = J_L
    J_prev_R = J_R

    # ensure stored finite taus are nonnegative (numerical safety)
    finite = np.isfinite(tau)
    tau[finite] = np.maximum(tau[finite], 0.0)

    new_state = State(
        t=t_next,
        y=y.astype(np.float64),
        L=L.astype(np.float64),
        A_L=A_L.astype(np.float64),
        J_prev_L=J_prev_L.astype(np.float64),
        J_prev_R=J_prev_R.astype(np.float64),
        A_R=A_R.astype(np.float64),
        pin_rfrac=pin_rfrac.astype(np.float64),
        W_L=W_L.astype(np.float64),
        W_R=W_R.astype(np.float64),
        ids=ids.astype(np.int64),
        next_id=int(next_id_var),
        tau_grow=tau.astype(np.float64),
        step_idx=state.step_idx + 1,
    )
    validate_state(new_state, p)
    return new_state
