from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


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
    x: np.ndarray,
    A_L_final: np.ndarray,
    A_R_final: np.ndarray,
    outpath: str | Path,
    title: str = "Final auxin profile",
) -> Path:
    """
    Save a 1D line plot of final auxin vs position for left/right files.

    Parameters
    ----------
    x : np.ndarray shape (n_cells,)
    A_L_final : np.ndarray shape (n_cells,)
    A_R_final : np.ndarray shape (n_cells,)
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if x.ndim != 1 or A_L_final.ndim != 1 or A_R_final.ndim != 1:
        raise ValueError("x, A_L_final, A_R_final must all be 1D arrays")
    if not (x.shape == A_L_final.shape == A_R_final.shape):
        raise ValueError(
            f"Shapes must match. Got x={x.shape}, A_L={A_L_final.shape}, A_R={A_R_final.shape}"
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, A_L_final, label="A_L")
    ax.plot(x, A_R_final, label="A_R")
    ax.set_title(title)
    ax.set_xlabel("position x")
    ax.set_ylabel("auxin")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath