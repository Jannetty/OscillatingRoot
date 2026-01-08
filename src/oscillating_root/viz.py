from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def plot_cell_kymograph_eulerian_raster(
    t: np.ndarray,
    y_centers: np.ndarray,
    A: np.ndarray,
    outpath: str | Path,
    *,
    cell_lengths: Optional[np.ndarray] = None,
    default_cell_length: float = 1.0,
    y_max: Optional[float] = None,
    dy: float = 1.0,  # y grid resolution [LU per pixel row]
    title: str = "Auxin kymograph (fixed y, rasterized cells)",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if t.ndim != 1:
        raise ValueError(f"t must be 1D, got {t.shape}")
    if y_centers.ndim != 2 or A.ndim != 2:
        raise ValueError("y_centers and A must be 2D (n_frames, n_cells)")
    if y_centers.shape != A.shape:
        raise ValueError(f"y_centers and A must match, got {y_centers.shape} vs {A.shape}")
    if y_centers.shape[0] != t.shape[0]:
        raise ValueError("t length must match number of frames")

    n_frames, n_cells = A.shape

    if cell_lengths is None:
        L = np.full_like(y_centers, float(default_cell_length), dtype=float)
    else:
        if cell_lengths.shape != y_centers.shape:
            raise ValueError(f"cell_lengths must match y_centers shape, got {cell_lengths.shape}")
        L = cell_lengths.astype(float)

    if y_max is None:
        y_max = float(np.nanmax(y_centers + 0.5 * L))
    if vmin is None:
        vmin = float(np.nanmin(A))
    if vmax is None:
        vmax = float(np.nanmax(A))

    # Build fixed Eulerian y grid
    n_y = int(np.ceil(y_max / dy)) + 1
    img = np.zeros((n_y, n_frames), dtype=float)

    # Paint cells into img[:, k]
    for k in range(n_frames):
        for i in range(n_cells):
            yc = float(y_centers[k, i])
            li = float(L[k, i])
            if not np.isfinite(yc) or not np.isfinite(li):
                continue
            y0 = yc - 0.5 * li
            y1 = yc + 0.5 * li
            if y1 <= 0 or y0 >= y_max:
                continue
            y0c = max(0.0, y0)
            y1c = min(y_max, y1)

            j0 = int(np.floor(y0c / dy))
            j1 = int(np.ceil(y1c / dy))
            if j1 <= j0:
                continue

            img[j0:j1, k] = float(A[k, i])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    extent = [float(t[0]), float(t[-1]), 0.0, float(y_max)]
    im = ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",  # avoid smoothing artifacts
    )

    ax.set_xlabel("time [TU]")
    ax.set_ylabel("y position [LU] (root tip at 0)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("auxin [AU]")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def plot_cell_stack_snapshot(
    y_centers: np.ndarray,
    L: np.ndarray,
    outpath: str | Path,
    *,
    title: str = "Cell stack snapshot",
    y_max: float | None = None,
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if y_centers.ndim != 1 or L.ndim != 1 or y_centers.shape != L.shape:
        raise ValueError("Expected y_centers and L as 1D arrays of same shape.")

    edges_bottom = y_centers - 0.5 * L
    edges_top = y_centers + 0.5 * L

    if y_max is None:
        y_max = float(np.max(edges_top))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # draw edges
    for b, t in zip(edges_bottom, edges_top):
        if t < 0 or b > y_max:
            continue
        ax.hlines([b, t], xmin=0, xmax=1, linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.set_xticks([])
    ax.set_ylabel("y [LU]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def plot_kymograph(
    A: np.ndarray,
    outpath: str | Path,
    title: str,
    xlabel: str = "time (frames)",
    ylabel: str = "cell index",
    t: Optional[np.ndarray] = None,
) -> Path:
    """
    Save a kymograph image of A over time.

    Parameters
    ----------
    A : np.ndarray
        Shape (n_frames, n_cells). Rows are time, columns are cell index.
    outpath : str | Path
        Output PNG path.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    t : Optional[np.ndarray]
        Optional time array of shape (n_frames,). If provided, label x-axis in time units.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D (n_frames, n_cells), got shape {A.shape}")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # If we have actual times, set extent so x-axis matches t range
    if t is not None:
        if t.ndim != 1 or t.shape[0] != A.shape[0]:
            raise ValueError(f"t must be shape ({A.shape[0]},), got {t.shape}")
        extent = [float(t[0]), float(t[-1]), 0, A.shape[1]]
        im = ax.imshow(A, aspect="auto", origin="lower", extent=extent)
        ax.set_xlabel("time")
    else:
        im = ax.imshow(A, aspect="auto", origin="lower")
        ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="auxin")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_final_state(
    y: np.ndarray,
    A_L_final: np.ndarray,
    A_R_final: np.ndarray,
    outpath: str | Path,
    title: str = "Final auxin profile",
) -> Path:
    """
    Save a 1D line plot of final auxin vs position for left/right files.

    Parameters
    ----------
    y : np.ndarray shape (n_cells,)
    A_L_final : np.ndarray shape (n_cells,)
    A_R_final : np.ndarray shape (n_cells,)
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if y.ndim != 1 or A_L_final.ndim != 1 or A_R_final.ndim != 1:
        raise ValueError("x, A_L_final, A_R_final must all be 1D arrays")
    if not (y.shape == A_L_final.shape == A_R_final.shape):
        raise ValueError(
            f"Shapes must match. Got x={y.shape}, A_L={A_L_final.shape}, A_R={A_R_final.shape}"
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y, A_L_final, label="A_L")
    ax.plot(y, A_R_final, label="A_R")
    ax.set_title(title)
    ax.set_xlabel("y position [LU]")
    ax.set_ylabel("auxin [AU]")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def plot_id_trajectories(
    t: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    outpath: str | Path,
    n_tracks: int = 12,
    seed: int = 0,
    title: str = "Tracked cell trajectories (by ID)",
) -> Path:
    """
    Plot y(t) for a subset of persistent cell IDs.

    Parameters
    ----------
    t : (n_frames,)
    y : (n_frames, n_cells)
    ids : (n_frames, n_cells)
    n_tracks : number of IDs to track
    seed : random seed for choosing which IDs to track (deterministic selection)
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if t.ndim != 1:
        raise ValueError(f"t must be 1D, got {t.shape}")
    if y.ndim != 2 or ids.ndim != 2:
        raise ValueError("y and ids must be 2D (n_frames, n_cells)")
    if y.shape != ids.shape:
        raise ValueError(f"y and ids must have same shape, got {y.shape} vs {ids.shape}")
    if y.shape[0] != t.shape[0]:
        raise ValueError("t length must match number of frames")

    # Choose IDs to track from the final frame (these exist at the end)
    final_ids = np.sort(ids[-1].astype(np.int64))
    track_ids = final_ids[:n_tracks]          # oldest remaining
    # or:
    #track_ids = final_ids[-n_tracks:]         # youngest remaining

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for cid in track_ids:
        # For each frame, find the index where this ID appears (or NaN if absent)
        y_trace = np.full(t.shape[0], np.nan, dtype=float)
        for k in range(t.shape[0]):
            match = np.where(ids[k] == cid)[0]
            if match.size > 0:
                y_trace[k] = y[k, match[0]]
        ax.plot(t, y_trace, label=f"id {int(cid)}")

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("y position")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def compute_oz_residence_times(
    t: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    oz_center: float,
    oz_sigma: float,
) -> np.ndarray:
    """
    Compute residence time in a simple OZ window [center - sigma, center + sigma]
    for each cell ID that appears in the frames.

    Returns
    -------
    residence_times : (n_ids_included,) array of residence durations [same units as t]
        Only includes IDs that both enter and exit the OZ window within the saved frames.
    """
    if oz_sigma <= 0:
        return np.array([], dtype=float)

    in_oz = (y >= (oz_center - oz_sigma)) & (y <= (oz_center + oz_sigma))

    all_ids = np.unique(ids.astype(np.int64))
    residence = []

    for cid in all_ids:
        # bool time-series of whether cid is in OZ at each frame
        in_series = np.zeros(t.shape[0], dtype=bool)
        for k in range(t.shape[0]):
            match = np.where(ids[k] == cid)[0]
            if match.size > 0:
                in_series[k] = bool(in_oz[k, match[0]])

        if not np.any(in_series):
            continue

        # Find first and last True index
        first = np.argmax(in_series)
        last = len(in_series) - 1 - np.argmax(in_series[::-1])

        # Require that it actually exits within the saved window:
        # i.e., there is at least one False after last True
        if last < (len(in_series) - 1) and not in_series[last + 1]:
            residence.append(t[last] - t[first])

    return np.asarray(residence, dtype=float)


def plot_residence_histogram(
    residence_times: np.ndarray,
    outpath: str | Path,
    title: str = "OZ residence time histogram",
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if residence_times.size == 0:
        ax.text(0.5, 0.5, "No residence times computed\n(set oz_sigma > 0 and run long enough)",
                ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(residence_times, bins=30)
        ax.set_xlabel("residence time")
        ax.set_ylabel("count")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath