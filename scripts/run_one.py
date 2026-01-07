import argparse
from datetime import datetime
from pathlib import Path

import oscillating_root  # import works = packaging works 🎉


def create_folder(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=False)
    print(f"Created folder: {folder_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one OscillatingRoot simulation.")
    parser.add_argument("--tag", type=str, required=True, help="Simulation tag")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{timestamp}_{args.tag}"

    create_folder(run_dir)

    (run_dir / "README.txt").write_text(
        "Step 0 run created successfully.\n"
    )

    print(f"Run initialized at: {run_dir}")


if __name__ == "__main__":
    main()