
from __future__ import annotations

from copy import copy

from .config import Params
from .state import State, validate_state

"""
Core simulation update rule.

step(state, params) advances the system by exactly one timestep dt.
No I/O, no plotting. Deterministic given (state, params) and any RNG usage.
"""

def step(state: State, p: Params) -> State:
    """
    Advance the simulation by one timestep.

    Step 0 behavior:
    - increment time by dt
    - increment step index
    - leave all state arrays unchanged
    """
    p.validate()
    validate_state(state, p)

    # Create a new State object (do not mutate input in-place)
    new_state = State(
        t=state.t + p.dt,
        y=state.y,
        A_L=state.A_L,
        A_R=state.A_R,
        step_idx=state.step_idx + 1,
    )

    validate_state(new_state, p)
    return new_state