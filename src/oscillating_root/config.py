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

    # --- Conveyor belt ---
    insert_spacing: float = 1.0   # nominal newborn spacing near tip [LU]
    y_max: float = 300.0          # far boundary (optional physical cutoff) [LU]

    v: float = 1.0                # advection speed tip [LU/TU]

    # ---- Placeholders for upcoming milestones (not used yet) ----
    oz_center: float = 0.0  # center of oscillation zone along y [LU]
    oz_sigma: float = 0.0   # width/scale of OZ window [LU]

    period_T: float = 0.0   # oscillation period of forcing S(t) [TU]
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

        # Conveyor belt sanity
        if self.insert_spacing <= 0:
            raise ValueError(f"insert_spacing must be > 0, got {self.insert_spacing}")
        if self.y_max <= self.insert_spacing:
            raise ValueError(f"y_max must be > insert_spacing, got y_max={self.y_max}")
        if self.v <= 0:
            raise ValueError(f"v must be > 0, got {self.v}")

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
        insert_spacing=1.0,
        y_max=300.0,
        v=1.0,
        # placeholders remain zero for now
        oz_center = 100.0,
        oz_sigma = 10.0
    )
    p.validate()
    return p