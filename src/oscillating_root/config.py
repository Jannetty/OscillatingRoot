from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

# ---- Unit conventions (documentation constants) ----
TU: str = "hours"
LU: str = "microns"
AU: str = "auxin amount (arbitrary units)"


@dataclass(frozen=True, slots=True)
class Params:
    """
    Simulation parameters.

    Units:
      - Time unit (TU): hours
      - Length unit (LU): microns (µm)
      - Auxin unit (AU): treated as a *amount-like* scalar per cell
                         (Concentration can be computed as C = A / L.)

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

    # ---- Loading Zone (OZ) loading window along y ----
    loading_center: float = 50.0  # Loading zone center position [µm from tip]
    loading_sigma: float = 20.0  # Loading zone half-width for box window [µm]
    # (currently: W(y)=1 in [center-sigma, center+sigma])

    # ---- Loading depends on cell size (emergent priming) ----
    gamma_load: float = 1.0  # exponent for size-dependent loading
    L0_load: float = 8.0  # reference length [µm] (e.g. newborn_length)
    cap_load: float | None = (
        None  # optional cap on (L/L0)^gamma to prevent huge cells dominating
    )

    # ---- Auxin dynamics (treat A as amount) ----
    # ---- Transporter / transport model toggles ----
    use_apoplast: bool = (
        True  # run with interface wall compartments or direct cell-cell
    )

    k_in: float = (
        5.0  # external loading rate into cytoplasm (OZ supply). Set 0 to disable.
    )
    d: float = 0.005  # first-order auxin loss/turnover rate

    pin_rule: str = "C"  # "A" gradient-sensing, "B" flux-reinforcement, "C" zone-fixed
    pin_relax_tau: float = 1.0  # [hr] relaxation timescale for pin_rfrac dynamics
    pin_beta: float = 5.0  # sensitivity for A and/or B (sigmoid steepness)

    # ---- AUX/LAX (import) regulation ----
    aux_alpha: float = 2.0  # fold-upreg strength
    aux_K: float = 0.5  # [conc units] half-saturation (in terms of C=A/L)

    # ---- Baseline expression profiles (piecewise by zone) ----
    pin_tot_mz: float = 1.0
    pin_tot_tz: float = 1.0
    pin_tot_ez: float = 0.5
    pin_tot_dz: float = 0.1

    aux_base_mz: float = 0.5
    aux_base_tz: float = 0.5
    aux_base_ez: float = 0.2
    aux_base_dz: float = 0.1

    # ---- Zone-fixed polarity for rule C ----
    pin_rfrac_mz: float = 0.3
    pin_rfrac_tz: float = 0.3
    pin_rfrac_ez: float = 0.3
    pin_rfrac_dz: float = 0.3

    # ---- Kinetics ----
    k_pin: float = 1.0  # export rate coefficient
    k_aux: float = 1.0  # import rate coefficient (only used if use_apoplast=True)

    # ---- Optional passive diffusion (can set to 0 to isolate transporters) ----
    D_cell: float = 0.0  # effective passive diffusion in cytoplasm (no apoplast mode)
    D_wall: float = 0.0  # diffusion along wall (apoplast mode)

    # ---- Apoplast "volume" scale for converting wall amount to wall concentration ----
    V_wall: float = 1.0

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

        if self.pin_rule not in ("A", "B", "C"):
            raise ValueError("pin_rule must be 'A','B', or 'C'")
        if self.pin_relax_tau <= 0:
            raise ValueError("pin_relax_tau must be > 0")
        if self.use_apoplast and self.k_aux < 0:
            raise ValueError("k_aux must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_params() -> Params:
    p = Params(
        seed=0,
        dt=0.1,
        n_steps=400,
        n_cells=250,
        loading_center=50.0,
        newborn_length=8.0,
        division_rule="double",
    )
    p.validate()
    return p
