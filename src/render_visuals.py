"""Render or backfill visual artifacts for completed simulation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import summarize_run
from .utils import load_config
from .visualization import create_run_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GIFs and figures for completed runs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="One run directory under outputs/runs.")
    group.add_argument("--all", action="store_true", help="Render visuals for all runs under --root.")
    parser.add_argument("--root", default="outputs/runs", help="Run root used with --all.")
    parser.add_argument("--force", action="store_true", help="Regenerate visuals even if skin_convergence.gif exists.")
    parser.add_argument("--fps", type=int, default=None, help="Override GIF frames per second.")
    parser.add_argument("--cmap", default=None, help="Override grayscale colormap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(args.run)] if args.run else sorted(Path(args.root).glob("*"))
    rendered = 0
    skipped = 0

    for run_dir in run_dirs:
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        if (run_dir / "skin_convergence.gif").exists() and not args.force:
            skipped += 1
            continue

        config = load_config(run_dir / "config.yaml")
        if not (run_dir / "pca_trajectory.npy").exists() or not (run_dir / "velocity.csv").exists():
            summarize_run(run_dir)

        fps = args.fps if args.fps is not None else int(config.get("visualization", {}).get("fps", 12))
        cmap = args.cmap if args.cmap is not None else str(config.get("visualization", {}).get("cmap", "viridis"))
        create_run_visualizations(run_dir, fps=fps, cmap=cmap)
        rendered += 1
        print(f"Rendered visuals: {run_dir}")

    print(f"Done. rendered={rendered}, skipped={skipped}")


if __name__ == "__main__":
    main()

