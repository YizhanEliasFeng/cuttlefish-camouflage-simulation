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


def make_dynamics_gif(
    run_dir: str | Path,
    output_path: str | Path,
    fps: int = 12,
    cmap: str = "viridis",
    dpi: int = 100,
) -> Path:
    """Create a three-panel GIF with target, skin state, and PCA trajectory."""

    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_path = run_dir / "target.png"
    pca_path = run_dir / "pca_trajectory.npy"
    loss_path = run_dir / "losses.csv"
    frame_paths = sorted((run_dir / "frames").glob("frame_*.png"))

    if not target_path.exists():
        raise FileNotFoundError(f"Missing target image: {target_path}")
    if not pca_path.exists():
        raise FileNotFoundError("pca_trajectory.npy is missing; run evaluation first")
    if not loss_path.exists():
        raise FileNotFoundError(f"Missing loss CSV: {loss_path}")
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png files found in {run_dir / 'frames'}")

    target = imageio.imread(target_path)
    pca_coords = np.load(pca_path)
    loss_rows = read_losses_csv(loss_path)
    losses = loss_rows[:, 1]

    x_pad = max(0.5, 0.06 * max(np.ptp(pca_coords[:, 0]), 1e-6))
    y_pad = max(0.5, 0.06 * max(np.ptp(pca_coords[:, 1]), 1e-6))
    xlim = (float(pca_coords[:, 0].min() - x_pad), float(pca_coords[:, 0].max() + x_pad))
    ylim = (float(pca_coords[:, 1].min() - y_pad), float(pca_coords[:, 1].max() + y_pad))

    duration = 1.0 / max(int(fps), 1)
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for frame_path in frame_paths:
            step = _step_from_frame_path(frame_path)
            step = min(step, len(pca_coords) - 1)
            skin = imageio.imread(frame_path)
            loss = float(losses[min(step, len(losses) - 1)])

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)
            fig.patch.set_facecolor("#f8f9fa")

            axes[0].set_title("Target Environment", fontsize=14, fontweight="bold")
            _imshow_image(axes[0], target, cmap)
            axes[0].axis("off")

            axes[1].set_title(f"Virtual Cuttlefish Skin (Step {step})", fontsize=14, fontweight="bold")
            _imshow_image(axes[1], skin, cmap)
            axes[1].axis("off")

            axes[2].set_title(
                f"Neural Trajectory in PCA Space\nLoss: {loss:.4f}",
                fontsize=14,
                fontweight="bold",
            )
            axes[2].plot(
                pca_coords[: step + 1, 0],
                pca_coords[: step + 1, 1],
                color="#e63946",
                linewidth=2,
                alpha=0.85,
            )
            axes[2].scatter(
                pca_coords[step, 0],
                pca_coords[step, 1],
                color="#e63946",
                s=90,
                zorder=5,
            )
            axes[2].scatter(
                pca_coords[0, 0],
                pca_coords[0, 1],
                color="green",
                s=140,
                marker="*",
                zorder=6,
                label="Start",
            )
            axes[2].scatter(
                pca_coords[-1, 0],
                pca_coords[-1, 1],
                color="blue",
                s=130,
                marker="X",
                zorder=6,
                label="Goal",
            )
            axes[2].set_xlim(xlim)
            axes[2].set_ylim(ylim)
            axes[2].grid(True, linestyle="--", alpha=0.55)
            axes[2].legend(loc="lower left", frameon=True)

            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(frame)
            plt.close(fig)
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


def create_run_visualizations(run_dir: str | Path, fps: int = 12, cmap: str = "viridis") -> dict[str, Path]:
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
        "gif": make_dynamics_gif(run_dir, run_dir / "skin_convergence.gif", fps=fps, cmap=cmap),
        "skin_only_gif": make_gif(run_dir / "frames", run_dir / "skin_only.gif", fps=fps),
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


def _step_from_frame_path(frame_path: str | Path) -> int:
    stem = Path(frame_path).stem
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Frame name must end with an integer step: {frame_path}") from exc


def _imshow_image(ax: plt.Axes, image: np.ndarray, cmap: str) -> None:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        ax.imshow(array, cmap=cmap, vmin=0, vmax=255)
    else:
        ax.imshow(array)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
