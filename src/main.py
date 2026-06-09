"""Command-line entry point for camouflage simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import summarize_run
from .simulation import run_simulation
from .utils import load_config
from .visualization import create_run_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuttlefish camouflage feedback optimization.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--run-name", default=None, help="Override the config run_name.")
    parser.add_argument("--output-root", default=None, help="Override output.root.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip figures and GIF generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.output_root is not None:
        config.setdefault("output", {})["root"] = args.output_root

    result = run_simulation(config)
    metrics = summarize_run(result.output_dir)

    if not args.no_visuals:
        fps = int(config.get("visualization", {}).get("fps", 12))
        cmap = str(config.get("visualization", {}).get("cmap", "viridis"))
        create_run_visualizations(result.output_dir, fps=fps, cmap=cmap)

    print(f"Run complete: {Path(result.output_dir).resolve()}")
    print(f"Final loss: {metrics.get('final_loss', float('nan')):.6f}")
    print(f"PCA path/direct ratio: {metrics.get('pca_path_to_direct_ratio', float('nan')):.3f}")
    print(f"Low-velocity fraction: {metrics.get('low_velocity_fraction', float('nan')):.3f}")


if __name__ == "__main__":
    main()
