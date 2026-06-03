"""Publication-oriented figures and GIF rendering."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio

_CACHE_ROOT = Path.cwd() / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import read_losses_csv


def make_gif(frame_dir: str | Path, output_path: str | Path, fps: int = 12) -> Path:
    """Create a GIF from saved frame images."""

    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png files found in {frame_dir}")

    duration = 1.0 / max(int(fps), 1)
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    return output_path


def plot_loss_curve(loss_csv: str | Path, output_path: str | Path) -> Path:
    """Plot loss over optimization steps."""

    rows = read_losses_csv(loss_csv)
    steps = rows[:, 0]
    losses = rows[:, 1]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(steps, losses, color="#2A6F97", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Visual Feedback Loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_pca_trajectory(pca_coords: np.ndarray, output_path: str | Path) -> Path:
    """Plot a 2D PCA trajectory of activation states."""

    pca_coords = np.asarray(pca_coords)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(pca_coords[:, 0], pca_coords[:, 1], color="#C44E52", linewidth=2, alpha=0.85)
    ax.scatter(pca_coords[0, 0], pca_coords[0, 1], color="#2E7D32", s=80, marker="o", label="Start", zorder=3)
    ax.scatter(pca_coords[-1, 0], pca_coords[-1, 1], color="#1F4E79", s=90, marker="X", label="Final", zorder=3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Activation Trajectory in PCA Space")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_velocity_curve(velocity: np.ndarray, output_path: str | Path) -> Path:
    """Plot step-wise activation velocity."""

    velocity = np.asarray(velocity)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(np.arange(1, len(velocity) + 1), velocity, color="#7A5195", linewidth=1.8)
    if len(velocity):
        threshold = np.quantile(velocity, 0.2)
        ax.axhline(threshold, color="#EF5675", linestyle="--", linewidth=1.2, label="20% low-speed threshold")
        ax.legend(frameon=False)
    ax.set_xlabel("Step")
    ax.set_ylabel("Activation Velocity")
    ax.set_title("Velocity Profile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def create_run_visualizations(run_dir: str | Path, fps: int = 12) -> dict[str, Path]:
    """Generate standard plots and GIF for a completed run."""

    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pca_path = run_dir / "pca_trajectory.npy"
    velocity_path = run_dir / "velocity.csv"
    if not pca_path.exists():
        raise FileNotFoundError("pca_trajectory.npy is missing; run evaluation first")
    if not velocity_path.exists():
        raise FileNotFoundError("velocity.csv is missing; run evaluation first")

    pca_coords = np.load(pca_path)
    velocity = _read_velocity_csv(velocity_path)

    outputs = {
        "loss_curve": plot_loss_curve(run_dir / "losses.csv", figures_dir / "loss_curve.png"),
        "pca_trajectory": plot_pca_trajectory(pca_coords, figures_dir / "pca_trajectory.png"),
        "velocity_curve": plot_velocity_curve(velocity, figures_dir / "velocity_curve.png"),
        "gif": make_gif(run_dir / "frames", run_dir / "skin_convergence.gif", fps=fps),
    }
    return outputs


def _read_velocity_csv(path: str | Path) -> np.ndarray:
    import csv

    values: list[float] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row["velocity"]))
    return np.asarray(values, dtype=np.float64)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
