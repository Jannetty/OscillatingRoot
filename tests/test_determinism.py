import numpy as np

from oscillating_root.config import default_params, Params
from oscillating_root.state import init_state
from oscillating_root.model import step


def _run_final_state(seed: int):
    p0 = default_params()
    # Ensure the seed is what we set (even if default changes later)
    p = Params(**{**p0.to_dict(), "seed": seed})
    s = init_state(p)
    for _ in range(p.n_steps):
        s = step(s, p)
    return s


def test_determinism_same_seed_same_final_state():
    s1 = _run_final_state(seed=123)
    s2 = _run_final_state(seed=123)

    assert s1.step_idx == s2.step_idx
    assert np.allclose(s1.y, s2.y)
    assert np.allclose(s1.A_L, s2.A_L)
    assert np.allclose(s1.A_R, s2.A_R)
    assert np.array_equal(s1.ids, s2.ids)
    assert s1.next_id == s2.next_id
    assert abs(s1.t - s2.t) < 1e-12
