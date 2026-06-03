"""Shared utilities for reproducible simulation runs."""

from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read config files.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a YAML config file."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to save config files.") from exc

    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_id(run_name: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run_name)
    return f"{clean_name}_{stamp}_seed{seed}"


def make_unique_dir(root: str | Path, run_id: str) -> Path:
    root = ensure_dir(root)
    candidate = root / run_id
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    index = 1
    while True:
        candidate = root / f"{run_id}_{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively updated copy of `base`."""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def tensor_to_uint8_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a `[C,H,W]`, `[H,W]`, or `[H,W,C]` image to uint8."""

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def save_image(image: torch.Tensor | np.ndarray, path: str | Path) -> None:
    """Save a normalized image tensor or array."""

    ensure_dir(Path(path).parent)
    array = tensor_to_uint8_image(image)
    Image.fromarray(array).save(path)


def load_target_image(path: str | Path, height: int, width: int, device: str | torch.device) -> torch.Tensor:
    """Load a grayscale target image as a `[1,H,W]` float tensor in `[0,1]`."""

    image = Image.open(path).convert("L").resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.float32)


def write_losses_csv(losses: list[float], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        for step, loss in enumerate(losses):
            writer.writerow([step, f"{loss:.10f}"])


def read_losses_csv(path: str | Path) -> np.ndarray:
    rows: list[tuple[int, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["step"]), float(row["loss"])))
    return np.asarray(rows, dtype=np.float64)

