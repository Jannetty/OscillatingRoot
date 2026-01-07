from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import Params

"""
I/O helpers for run artifacts.

Frame conventions (frames.npz):
- t: (n_frames,) time [TU]
- x: (n_frames, n_cells) positions [LU]
- A_L: (n_frames, n_cells) auxin amount [AU]
- A_R: (n_frames, n_cells) auxin amount [AU]
"""

def make_run_dir(base: str | Path = "runs", tag: Optional[str] = None) -> Path:
    """
    Create a new timestamped run directory.

    Example:
        runs/2026-01-06_14-55-01_baseline/
    """
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = (tag or "run").strip().replace(" ", "_")
    run_dir = base_path / f"{ts}_{safe_tag}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common scientific Python objects to JSON-serializable types.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_params_json(run_dir: str | Path, p: Params, filename: str = "params.json") -> Path:
    run_dir = Path(run_dir)
    outpath = run_dir / filename

    data = _to_jsonable(p)
    outpath.write_text(json.dumps(data, indent=2, sort_keys=True))
    return outpath


def save_metrics_json(
    run_dir: str | Path, metrics: Dict[str, Any], filename: str = "metrics.json"
) -> Path:
    run_dir = Path(run_dir)
    outpath = run_dir / filename

    data = _to_jsonable(metrics)
    outpath.write_text(json.dumps(data, indent=2, sort_keys=True))
    return outpath


def save_frames_npz(
    run_dir: str | Path,
    frames: Dict[str, np.ndarray],
    filename: str = "frames.npz",
    allow_pickle: bool = False,
) -> Path:
    """
    Save simulation frames into a compressed npz.

    Expected keys (Step 0):
      - "t": (n_frames,)
      - "x": (n_frames, n_cells) OR (n_cells,)
      - "A_L": (n_frames, n_cells)
      - "A_R": (n_frames, n_cells)
    """
    run_dir = Path(run_dir)
    outpath = run_dir / filename

    # Basic validation: ensure everything is an ndarray
    for k, v in frames.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(f"frames['{k}'] must be a numpy.ndarray, got {type(v)}")

    np.savez_compressed(outpath, **frames, allow_pickle=allow_pickle)
    return outpath


def load_frames_npz(run_dir: str | Path, filename: str = "frames.npz") -> Dict[str, np.ndarray]:
    """
    Load frames saved by save_frames_npz into a dict of arrays.
    """
    run_dir = Path(run_dir)
    path = run_dir / filename
    with np.load(path) as data:
        return {k: data[k] for k in data.files}