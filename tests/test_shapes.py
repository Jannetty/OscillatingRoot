import numpy as np

from oscillating_root.config import default_params
from oscillating_root.state import init_state, validate_state


def test_init_state_shapes_and_validation():
    p = default_params()
    s = init_state(p)

    assert s.y.shape == (p.n_cells,)
    assert s.A_L.shape == (p.n_cells,)
    assert s.A_R.shape == (p.n_cells,)
    assert s.ids.shape == (p.n_cells,)
    assert np.issubdtype(s.ids.dtype, np.integer)
    assert len(set(s.ids.tolist())) == p.n_cells

    # Should not raise
    validate_state(s, p)

    # Also check basic finiteness
    assert np.isfinite(s.y).all()
    assert np.isfinite(s.A_L).all()
    assert np.isfinite(s.A_R).all()
