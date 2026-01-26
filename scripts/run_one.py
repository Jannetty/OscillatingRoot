# scripts/run_one.py

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from oscillating_root.config import default_params, Params
from oscillating_root.state import init_state
from oscillating_root.model import step
from oscillating_root.io import (
    make_run_dir,
    save_frames_npz,
    save_metrics_json,
    save_params_json,
)
from oscillating_root.metrics import compute_basic_metrics
from oscillating_root.viz import (
    plot_kymograph,
    plot_final_state,
    plot_id_trajectories,
    compute_oz_residence_times,
    plot_residence_histogram,
    plot_cell_kymograph_eulerian_raster,
    plot_cell_stack_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OscillatingRoot simulation.")
    parser.add_argument(
        "--tag",
        type=str,
        default="NO_SIMTAG_SET",
        help="Run tag for output folder name",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save a frame every N steps"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Override n_steps from default params"
    )
    parser.add_argument(
        "--dt", type=float, default=None, help="Override dt from default params"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed from default params"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Create params (default + optional overrides)
    p = default_params()
    if args.steps is not None:
        p = Params(**{**p.to_dict(), "n_steps": int(args.steps)})
    if args.dt is not None:
        p = Params(**{**p.to_dict(), "dt": float(args.dt)})
    if args.seed is not None:
        p = Params(**{**p.to_dict(), "seed": int(args.seed)})
    p.validate()

    # 2) Init state
    state = init_state(p)

    # 3) Run loop, record frames
    run_dir = make_run_dir(base="runs", tag=args.tag)
    t0 = time.time()

    save_every = max(1, int(args.save_every))
    times = []
    ys = []
    ids_list = []
    A_Ls = []
    A_Rs = []
    Ls = []

    # Record initial frame (frame 0)
    times.append(state.t)
    ys.append(state.y.copy())
    ids_list.append(state.ids.copy())
    A_Ls.append(state.A_L.copy())
    A_Rs.append(state.A_R.copy())
    Ls.append(state.L.copy())

    for _ in range(p.n_steps):
        state = step(state, p)

        if state.step_idx % save_every == 0:
            times.append(state.t)
            ys.append(state.y.copy())
            ids_list.append(state.ids.copy())
            A_Ls.append(state.A_L.copy())
            A_Rs.append(state.A_R.copy())
            Ls.append(state.L.copy())

    runtime_s = time.time() - t0

    frames = {
        "t": np.asarray(times, dtype=float),
        "y": np.stack(ys, axis=0),  # (n_frames, n_cells)
        "L": np.stack(Ls, axis=0),
        "ids": np.stack(ids_list, axis=0),  # (n_frames, n_cells)
        "A_L": np.stack(A_Ls, axis=0),
        "A_R": np.stack(A_Rs, axis=0),
    }

    # 4) Compute metrics
    metrics = compute_basic_metrics(frames, p)
    metrics["runtime_s"] = float(runtime_s)
    metrics["save_every"] = int(save_every)
    metrics["run_dir"] = str(run_dir)

    # 5) Save artifacts
    save_params_json(run_dir, p)
    save_frames_npz(run_dir, frames)
    save_metrics_json(run_dir, metrics)

    # Plots
    # plot_kymograph(frames["A_L"], run_dir / "kymograph_A_L.png", title="Auxin (Left file)", t=frames["t"])
    # plot_kymograph(frames["A_R"], run_dir / "kymograph_A_R.png", title="Auxin (Right file)", t=frames["t"])
    plot_cell_kymograph_eulerian_raster(
        t=frames["t"],
        y_centers=frames["y"],
        A=frames["A_L"] / np.maximum(frames["L"], 1e-12),  # concentration
        cell_lengths=frames["L"],
        outpath=run_dir / "kymograph_cells_A_L.png",
        dy=1.0,
        title="Auxin (Left file) — fixed y, cell rectangles",
    )

    plot_cell_kymograph_eulerian_raster(
        t=frames["t"],
        y_centers=frames["y"],
        A=frames["A_L"] / np.maximum(frames["L"], 1e-12),  # concentration
        cell_lengths=frames["L"],
        outpath=run_dir / "kymograph_cells_A_R.png",
        dy=1.0,
        title="Auxin (Right file) — fixed y, cell rectangles",
    )
    plot_final_state(
        y=frames["y"][-1],
        A_L_final=frames["A_L"][-1],
        A_R_final=frames["A_R"][-1],
        outpath=run_dir / "final_auxin_profile.png",
    )

    # Evidence plots for Milestone 1
    plot_id_trajectories(
        t=frames["t"],
        y=frames["y"],
        ids=frames["ids"],
        outpath=run_dir / "trajectories.png",
        n_tracks=12,
        seed=p.seed,
    )

    # OZ residence time histogram (only meaningful if oz_sigma > 0)
    res = compute_oz_residence_times(
        t=frames["t"],
        y=frames["y"],
        ids=frames["ids"],
        oz_center=p.loading_center,
        oz_sigma=p.loading_sigma,
    )
    plot_residence_histogram(res, run_dir / "oz_residence_hist.png")

    # 6) Print run directory
    print(f"Run complete: {run_dir}")
    print(f"Artifacts: params.json, frames.npz, metrics.json, kymographs, final plot")

    plot_cell_stack_snapshot(
        frames["y"][-1],
        frames["L"][-1],
        run_dir / "cell_stack_final.png",
        y_max=120.0,
        title="Cell boundaries near tip (final frame)",
    )


if __name__ == "__main__":
    main()
