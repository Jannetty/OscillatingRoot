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
- y = 0 corresponds to the root tip
- y increases shootward (up the screen)
- all per-cell arrays are indexed in increasing y order
"""

# ---- Unit conventions (documentation constants) ----
TU: str = "model time unit"        # later map to minutes or hours
LU: str = "model length unit"      # later map to microns
AU: str = "auxin amount unit"      # arbitrary auxin amount per cell


@dataclass(frozen=True, slots=True)
class Params:
    # --- Core run control ---
    seed: int = 0          # RNG seed for reproducibility [dimensionless]
    dt: float = 0.1        # timestep size [TU]
    n_steps: int = 100     # number of simulation steps to run [dimensionless]
    n_cells: int = 8       # number of cells per file (left or right) [cells]

    # ---- OZ ----
    oz_center: float = 50.0  # center of oscillation zone along y [LU]
    oz_sigma: float = 10.0   # width/scale of OZ window [LU]

    period_T: float = 10.0   # oscillation period of forcing S(t) [TU]
    S0: float = 1.0         # baseline forcing amplitude [AU/TU] or [dimensionless]
    S1: float = 1.0         # oscillatory forcing amplitude [AU/TU] or [dimensionless]

    # ---- Auxin ----
    k_in: float = 1.0       # auxin input/import rate constant [1/TU]
    d: float = 0.5          # auxin decay/degradation rate [1/TU]
    D: float = 0.0          # diffusion coefficient along the file [LU^2/TU]

    # --- Growth (new) ---
    growth_rate: float = 0.0          # dL/dt for each cell [LU/TU] (constant elongation per cell)
    newborn_length: float = 1.0        # length assigned to newborn cells [LU]
    tip_length_accum_rate: float = 1.0 # how fast new length is added at the tip [LU/TU]

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

        # Growth sanity
        if self.newborn_length <= 0:
            raise ValueError("newborn_length must be > 0")
        if self.growth_rate < 0:
            raise ValueError("growth_rate must be >= 0")
        if self.tip_length_accum_rate < 0:
            raise ValueError("tip_length_accum_rate must be >= 0")

        # If you rely on births to maintain a steady “conveyor”, you generally need some tip input
        if self.tip_length_accum_rate == 0 and self.growth_rate == 0:
            raise ValueError("No growth: tip_length_accum_rate and growth_rate are both 0")

        # Prevent pathological "tons of births per step" (optional but useful)
        tip_added = self.tip_length_accum_rate * self.dt
        if tip_added > 5 * self.newborn_length:
            raise ValueError(
                f"tip adds {tip_added:.3g} per step, which is >5 newborn lengths; "
                "this will spawn many cells per step and may be unstable."
        )

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
        oz_center = 50.0,
        oz_sigma = 10.0,
        newborn_length=1.0,
        growth_rate=0.0,
        tip_length_accum_rate=0.2,
    )
    p.validate()
    return p