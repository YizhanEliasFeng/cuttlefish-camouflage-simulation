"""Trajectory analysis for simulated skin-activation dynamics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_losses_csv, save_json


def compute_pca_trajectory(weights: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project a high-dimensional activation trajectory with PCA."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must have shape [time, n_basis]")
    if weights.shape[0] < 2:
        raise ValueError("at least two trajectory points are required")

    n_components = min(n_components, weights.shape[0], weights.shape[1])
    centered = weights - weights.mean(axis=0, keepdims=True)

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(weights)
    except ImportError:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vh[:n_components].T

    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return coords[:, :2]


def compute_velocity(weights: np.ndarray) -> np.ndarray:
    """Euclidean step speed in activation space."""

    weights = np.asarray(weights, dtype=np.float64)
    return np.linalg.norm(np.diff(weights, axis=0), axis=1)


def compute_path_length(trajectory: np.ndarray) -> float:
    """Total path length in a projected trajectory."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())


def compute_curvature_proxy(trajectory: np.ndarray) -> float:
    """Mean absolute turning angle in radians.

    This is a descriptive proxy, not a differential-geometry curvature
    estimate. Larger values indicate more meandering trajectories.
    """

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) < 3:
        return 0.0

    deltas = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = norms[:-1] * norms[1:] > 1e-12
    if not np.any(valid):
        return 0.0

    unit_a = deltas[:-1][valid] / norms[:-1][valid, None]
    unit_b = deltas[1:][valid] / norms[1:][valid, None]
    cosines = np.sum(unit_a * unit_b, axis=1).clip(-1.0, 1.0)
    angles = np.arccos(cosines)
    return float(np.mean(np.abs(angles)))


def compute_trajectory_metrics(
    weights: np.ndarray,
    losses: np.ndarray | None = None,
    pca_coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute convergence and dynamics metrics for one run."""

    weights = np.asarray(weights, dtype=np.float64)
    pca_coords = compute_pca_trajectory(weights) if pca_coords is None else np.asarray(pca_coords)
    velocity = compute_velocity(weights)

    path_length = compute_path_length(pca_coords)
    direct_distance = float(np.linalg.norm(pca_coords[-1] - pca_coords[0]))
    path_to_direct_ratio = float(path_length / direct_distance) if direct_distance > 1e-12 else 0.0
    curvature_proxy = compute_curvature_proxy(pca_coords)

    metrics: dict[str, Any] = {
        "n_steps": int(weights.shape[0] - 1),
        "n_basis": int(weights.shape[1]),
        "pca_path_length": path_length,
        "pca_direct_distance": direct_distance,
        "pca_path_to_direct_ratio": path_to_direct_ratio,
        "pca_curvature_proxy_rad": curvature_proxy,
        "mean_activation_velocity": float(np.mean(velocity)) if len(velocity) else 0.0,
        "max_activation_velocity": float(np.max(velocity)) if len(velocity) else 0.0,
    }

    if len(velocity):
        low_velocity_threshold = float(np.quantile(velocity, 0.2))
        metrics["low_velocity_threshold"] = low_velocity_threshold
        metrics["low_velocity_fraction"] = float(np.mean(velocity <= low_velocity_threshold))
    else:
        metrics["low_velocity_threshold"] = 0.0
        metrics["low_velocity_fraction"] = 0.0

    if losses is not None and len(losses):
        loss_values = np.asarray(losses, dtype=np.float64)
        if loss_values.ndim == 2:
            loss_values = loss_values[:, 1]
        initial_loss = float(loss_values[0])
        final_loss = float(loss_values[-1])
        best_loss = float(np.min(loss_values))
        target_loss = final_loss + 0.05 * max(initial_loss - final_loss, 0.0)
        convergence_candidates = np.where(loss_values <= target_loss)[0]
        convergence_step = int(convergence_candidates[0]) if len(convergence_candidates) else None
        metrics.update(
            {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "relative_loss_reduction": float((initial_loss - final_loss) / max(initial_loss, 1e-12)),
                "convergence_step_95pct": convergence_step,
            }
        )

    return metrics


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Load one run directory, compute metrics, and save analysis artifacts."""

    run_dir = Path(run_dir)
    weights = np.load(run_dir / "weights.npy")
    loss_rows = read_losses_csv(run_dir / "losses.csv")
    losses = loss_rows[:, 1]

    pca_coords = compute_pca_trajectory(weights)
    velocity = compute_velocity(weights)
    metrics = compute_trajectory_metrics(weights, losses=losses, pca_coords=pca_coords)

    np.save(run_dir / "pca_trajectory.npy", pca_coords)
    _write_vector_csv(velocity, run_dir / "velocity.csv", value_name="velocity")
    save_json(metrics, run_dir / "metrics.json")
    return metrics


def _write_vector_csv(values: np.ndarray, path: str | Path, value_name: str) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", value_name])
        for step, value in enumerate(values, start=1):
            writer.writerow([step, f"{float(value):.10f}"])

