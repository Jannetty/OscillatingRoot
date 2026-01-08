import numpy as np

from .config import Params


def recompute_centers_from_lengths(L: np.ndarray, tip_buffer: float) -> np.ndarray:
    """
    Convert cell lengths into cell center positions.

    Given a stack of segments of lengths L placed end-to-end starting at y=tip_buffer,
    return the center position of each segment.

    Parameters
    ----------
    L : (n,) array
        Cell lengths/heights [µm].
    tip_buffer : float
        Offset at tip [µm] (used if you have unallocated length at y=0).

    Returns
    -------
    y : (n,) array
        Cell center positions [µm] in tip→shoot order.
    """
    edges = tip_buffer + np.concatenate(([0.0], np.cumsum(L)))
    return 0.5 * (edges[:-1] + edges[1:])


def zone_r(y: np.ndarray, p: Params) -> np.ndarray:
    """
    Zone-dependent elemental growth parameter r(y) [hr^-1 µm^-1].

    MZ and TZ: r = rgrowth_mz
    EZ:        r = rgrowth_ez
    DZ:        r = 0
    """
    r = np.zeros_like(y, dtype=np.float64)
    r[y < p.tz_end] = p.rgrowth_mz
    r[(y >= p.tz_end) & (y < p.ez_end)] = p.rgrowth_ez
    return r


def schedule_tau(r_i: float, H_i: float) -> float:
    """
    Schedule time until next discrete growth event for one cell.

    Paper rule: add one row of grid points when
      t >= t_prev + 1 / (r * H)

    so tau = 1/(r*H). If r<=0, tau=inf (no growth).
    """
    if r_i <= 0.0:
        return float("inf")
    return 1.0 / (r_i * H_i)


def is_mz(y: float, p: Params) -> bool:
    """Return True if position y lies in the meristematic zone (MZ)."""
    return y < p.mz_end


def is_cytoplasmic_zone(y: float, p: Params) -> bool:
    """
    Return True if growth is cytoplasmic (MZ+TZ), where concentration should dilute with volume/length.
    """
    return y < p.tz_end
