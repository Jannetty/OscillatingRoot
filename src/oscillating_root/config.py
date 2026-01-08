from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

# ---- Unit conventions (documentation constants) ----
TU: str = "hours"
LU: str = "microns"
AU: str = "auxin concentration (arbitrary units)"


@dataclass(frozen=True, slots=True)
class Params:
    """
    Simulation parameters.

    Units:
      - Time unit (TU): hours
      - Length unit (LU): microns (µm)
      - Auxin unit (AU): treated as a *concentration-like* scalar per cell
                         (i.e., dilution is meaningful when cell length increases).

    Geometry / indexing convention:
      - We model one 1D "file" of cells stacked along y (tip at y=0, shootward is +y).
      - Arrays in State are always ordered from tip-most (index 0, smallest y)
        to shootward (index n_cells-1, largest y).
    """

    # --- Core run control ---
    seed: int = 0  # RNG seed
    dt: float = 0.1  # timestep size [hours]
    n_steps: int = 100  # number of update steps
    n_cells: int = 250  # fixed number of cells stored per file

    # ---- Oscillation Zone (OZ) forcing window along y ----
    oz_center: float = 50.0  # OZ center position [µm from tip]
    oz_sigma: float = 10.0  # OZ half-width for box window [µm]
    # (currently: W(y)=1 in [center-sigma, center+sigma])

    # ---- Temporal forcing S(t) applied inside OZ ----
    period_T: float = 10.0  # forcing period [hours]; if <=0, forcing is constant
    S0: float = 1.0  # baseline forcing amplitude (offset) [AU-like]
    S1: float = 1.0  # oscillation amplitude [AU-like]
    # S(t) = S0 + S1*sin(2π t / T)

    # ---- Auxin dynamics (treat A as concentration) ----
    k_in: float = 1.0  # auxin input rate coefficient [1/hour] (effective)
    # (A increases in OZ as k_in * I, where I=W(y)*S(t))
    d: float = 0.5  # auxin degradation rate [1/hour]
    D: float = 0.0  # diffusion coefficient along file [µm^2/hour] (unused for now)

    # ---- Discrete growth algorithm (paper-like grid growth) ----
    dy: float = 2.0  # growth increment per event [µm]
    # corresponds to "adding one row of grid points"

    # ---- Growth rates r_zone in hr^-1 µm^-1 (from paper Table 1) ----
    rgrowth_mz: float = 0.0179  # MZ (and TZ) elemental growth parameter [hr^-1 µm^-1]
    rgrowth_ez: float = 0.00112  # EZ elemental growth parameter [hr^-1 µm^-1]

    # ---- Zonation boundaries measured from tip [µm] ----
    # Zones:
    #   MZ: [0, mz_end)
    #   TZ: [mz_end, tz_end)
    #   EZ: [tz_end, ez_end)
    #   DZ: [ez_end, ∞)
    mz_end: float = 20.0 * 8.0
    tz_end: float = 20.0 * 8.0 + 15.0 * 12.0
    ez_end: float = 20.0 * 8.0 + 15.0 * 12.0 + 10.0 * 60.0

    # ---- Division + max height (paper-like) ----
    division_rule: str = "double"  # "double": divide when H >= 2*newborn_length
    # "fixed":  divide when H >= Hdivision
    newborn_length: float = 8.0  # newborn cell height just after division [µm]
    Hdivision: float = (
        16.0  # fixed division height threshold [µm] (if division_rule="fixed")
    )
    Hmax: float = (
        144.0  # max height cap in EZ [µm]; once reached, cell stops growing (DZ-like)
    )

    # ---- Optional tip births (NOT in the paper) ----
    # This is a debugging / inflow mechanism to maintain n_cells if division is disabled,
    # or to mimic adding new cells at the tip independent of division.
    use_tip_births: bool = False
    tip_length_accum_rate: float = (
        0.0  # [µm/hour] length accumulated at the tip; converted into newborns
    )

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.n_cells <= 1:
            raise ValueError(f"n_cells must be > 1, got {self.n_cells}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

        if self.dy <= 0:
            raise ValueError("dy must be > 0")
        if self.rgrowth_mz < 0 or self.rgrowth_ez < 0:
            raise ValueError("growth rates must be >= 0")

        if not (0.0 < self.mz_end < self.tz_end < self.ez_end):
            raise ValueError("Require 0 < mz_end < tz_end < ez_end")
        if self.newborn_length <= 0:
            raise ValueError("newborn_length must be > 0")
        if self.Hmax <= self.newborn_length:
            raise ValueError("Hmax must be > newborn_length")

        if self.division_rule not in ("double", "fixed"):
            raise ValueError("division_rule must be 'double' or 'fixed'")
        if self.division_rule == "fixed" and self.Hdivision <= self.newborn_length:
            raise ValueError("Hdivision must be > newborn_length for fixed division")

        if self.use_tip_births:
            if self.tip_length_accum_rate <= 0:
                raise ValueError(
                    "tip_length_accum_rate must be > 0 when use_tip_births=True"
                )
            # prevent “tons of births per step”
            if self.tip_length_accum_rate * self.dt > 5 * self.newborn_length:
                raise ValueError(
                    "tip births too fast for dt; will spawn many cells per step"
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_params() -> Params:
    p = Params(
        seed=0,
        dt=0.1,
        n_steps=400,
        n_cells=250,
        oz_center=50.0,
        oz_sigma=10.0,
        newborn_length=8.0,
        division_rule="double",
        use_tip_births=False,
        tip_length_accum_rate=0.0,
    )
    p.validate()
    return p
