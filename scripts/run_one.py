# scripts/run_one.py

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from oscillating_root.config import default_params, Params
from oscillating_root.state import init_state
from oscillating_root.model import step
from oscillating_root.io import make_run_dir, save_frames_npz, save_metrics_json, save_params_json
from oscillating_root.metrics import compute_basic_metrics
from oscillating_root.viz import plot_kymograph, plot_final_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OscillatingRoot simulation.")
    parser.add_argument("--tag", type=str, default="NO_SIMTAG_SET", help="Run tag for output folder name")
    parser.add_argument("--save-every", type=int, default=10, help="Save a frame every N steps")
    parser.add_argument("--steps", type=int, default=None, help="Override n_steps from default params")
    parser.add_argument("--dt", type=float, default=None, help="Override dt from default params")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from default params")
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
    xs = []
    A_Ls = []
    A_Rs = []

    # Record initial frame (frame 0)
    times.append(state.t)
    xs.append(state.y.copy())
    A_Ls.append(state.A_L.copy())
    A_Rs.append(state.A_R.copy())

    for _ in range(p.n_steps):
        state = step(state, p)

        if state.step_idx % save_every == 0:
            times.append(state.t)
            xs.append(state.y.copy())
            A_Ls.append(state.A_L.copy())
            A_Rs.append(state.A_R.copy())

    runtime_s = time.time() - t0

    frames = {
        "t": np.asarray(times, dtype=float),
        "x": np.stack(xs, axis=0),      # (n_frames, n_cells)
        "A_L": np.stack(A_Ls, axis=0),  # (n_frames, n_cells)
        "A_R": np.stack(A_Rs, axis=0),  # (n_frames, n_cells)
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
    plot_kymograph(frames["A_L"], run_dir / "kymograph_A_L.png", title="Auxin (Left file)", t=frames["t"])
    plot_kymograph(frames["A_R"], run_dir / "kymograph_A_R.png", title="Auxin (Right file)", t=frames["t"])
    plot_final_state(
        x=frames["x"][-1],
        A_L_final=frames["A_L"][-1],
        A_R_final=frames["A_R"][-1],
        outpath=run_dir / "final_auxin_profile.png",
    )

    # 6) Print run directory
    print(f"Run complete: {run_dir}")
    print(f"Artifacts: params.json, frames.npz, metrics.json, kymographs, final plot")


if __name__ == "__main__":
    main()