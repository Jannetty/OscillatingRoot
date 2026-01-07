from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

"""
Simulation configuration (parameters).

Conventions / units:
- TU = model time unit (later: minutes)
- LU = model length unit (later: microns)
- AU = arbitrary auxin amount unit

Coordinate system:
- x = 0 corresponds to the root tip
- x increases shootward
- all per-cell variables are indexed in increasing x order
"""

# ---- Unit conventions (documentation constants) ----
TU: str = "model time unit"        # later mapped to minutes or hours
LU: str = "model length unit"      # later mapped to microns
AU: str = "auxin amount unit"      # arbitrary auxin amount per cell


@dataclass(frozen=True, slots=True)
class Params:
    # --- Core run control ---
    seed: int = 0          # RNG seed for reproducibility [dimensionless]
    dt: float = 0.1        # timestep size [TU]
    n_steps: int = 100     # number of simulation steps to run [dimensionless]
    n_cells: int = 8       # number of cells per file (left or right) [cells]

    # ---- Placeholders for upcoming milestones (not used yet) ----
    oz_center: float = 0.0  # center of oscillation zone along x [LU]
    oz_sigma: float = 0.0   # width/scale of OZ window (e.g., Gaussian sigma or half-width) [LU]

    period_T: float = 0.0   # oscillation period of the tip-tethered forcing S(t) [TU]
    S0: float = 0.0         # baseline forcing amplitude [AU/TU] or [dimensionless]
    S1: float = 0.0         # oscillatory forcing amplitude [AU/TU] or [dimensionless]

    k_in: float = 0.0       # auxin input/import rate constant [1/TU]
    d: float = 0.0          # auxin decay/degradation rate [1/TU]
    D: float = 0.0          # diffusion coefficient along the file [LU^2/TU]

    def validate(self) -> None:
        """Raise ValueError if parameters are invalid."""
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.n_cells <= 1:
            raise ValueError(f"n_cells must be > 1, got {self.n_cells}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters to a JSON-friendly dict."""
        return asdict(self)


def default_params() -> Params:
    """Convenience factory for a sensible default parameter set."""
    p = Params(
        seed=0,
        dt=0.1,
        n_steps=400,
        n_cells=250,
        # placeholders are zero for now
    )
    p.validate()
    return p