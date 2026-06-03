"""Closed-loop camouflage simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .basis import create_basis_from_config, create_random_smooth_basis
from .controllers import build_controller
from .losses import get_vgg16_feature_extractor, mse_loss, perceptual_loss
from .render import render_skin
from .utils import (
    ensure_dir,
    load_target_image,
    make_run_id,
    make_unique_dir,
    save_config,
    save_image,
    save_json,
    seed_everything,
    write_losses_csv,
)


@dataclass
class SimulationResult:
    output_dir: Path
    run_id: str
    weights: np.ndarray
    losses: list[float]


def run_simulation(config: dict) -> SimulationResult:
    """Run a complete feedback-optimization simulation."""

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(config.get("device", "cpu"))

    image_config = dict(config.get("image", {}))
    image_config["device"] = device
    basis_config = dict(config.get("basis", {}))
    basis_config["device"] = device

    basis = create_basis_from_config(basis_config, image_config, seed=seed)
    target = _create_target(config, basis, device)
    weights = _initialize_weights(config, n_basis=basis.shape[0], device=device).requires_grad_(True)

    run_name = str(config.get("run_name", "experiment"))
    run_id = str(config.get("run_id") or make_run_id(run_name, seed))
    output_root = config.get("output", {}).get("root", "outputs/runs")
    output_dir = make_unique_dir(output_root, run_id)
    frames_dir = ensure_dir(output_dir / "frames")

    config_to_save = dict(config)
    config_to_save["run_id"] = output_dir.name
    save_config(config_to_save, output_dir / "config.yaml")
    save_image(target, output_dir / "target.png")
    torch.save(basis.detach().cpu(), output_dir / "basis.pt")

    loss_fn = _build_loss_fn(config, target, device)
    controller = build_controller(
        weights=weights,
        config=config.get("optimizer", {}),
        seed=seed + 3000,
    )

    steps = int(config.get("simulation", {}).get("steps", 200))
    save_every = int(config.get("simulation", {}).get("save_every", 10))
    render_activation = str(config.get("render", {}).get("activation", "sigmoid"))

    weights_history: list[np.ndarray] = []
    losses: list[float] = []

    for step in range(steps + 1):
        current_skin = render_skin(weights, basis, activation=render_activation)
        loss = loss_fn(current_skin)

        weights_history.append(weights.detach().cpu().numpy().copy())
        losses.append(float(loss.detach().cpu().item()))

        if step % save_every == 0 or step == steps:
            save_image(current_skin, frames_dir / f"frame_{step:04d}.png")

        if step == steps:
            save_image(current_skin, output_dir / "final_skin.png")
            break

        controller.zero_grad()
        loss.backward()
        controller.step(step)

    weights_array = np.asarray(weights_history, dtype=np.float32)
    np.save(output_dir / "weights.npy", weights_array)
    write_losses_csv(losses, output_dir / "losses.csv")
    save_json(
        {
            "run_id": output_dir.name,
            "seed": seed,
            "steps": steps,
            "save_every": save_every,
            "controller": controller.state.__dict__,
            "basis_shape": list(basis.shape),
            "target_type": config.get("target", {}).get("type", "synthetic_from_basis"),
        },
        output_dir / "metadata.json",
    )

    return SimulationResult(output_dir=output_dir, run_id=output_dir.name, weights=weights_array, losses=losses)


def _create_target(config: dict, basis: torch.Tensor, device: torch.device) -> torch.Tensor:
    image_config = config.get("image", {})
    target_config = config.get("target", {})
    target_type = str(target_config.get("type", "synthetic_from_basis"))
    height = int(image_config.get("height", 64))
    width = int(image_config.get("width", 64))
    seed = int(config.get("seed", 42)) + int(target_config.get("seed_offset", 1000))

    if target_type == "synthetic_from_basis":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = torch.randn(basis.shape[0], generator=generator, dtype=torch.float32)
        weights = weights.to(device) * float(target_config.get("weight_scale", 1.5))
        with torch.no_grad():
            return render_skin(weights, basis).detach()

    if target_type == "image":
        path = target_config.get("path")
        if not path:
            raise ValueError("target.path is required when target.type is image")
        return load_target_image(path, height=height, width=width, device=device)

    if target_type == "random_smooth":
        target_basis = create_random_smooth_basis(
            n_basis=1,
            height=height,
            width=width,
            low_res=int(target_config.get("low_res", 8)),
            seed=seed,
            device=device,
        )
        return (target_basis.squeeze(0) + 1.0) / 2.0

    raise ValueError(f"Unsupported target type: {target_type}")


def _initialize_weights(config: dict, n_basis: int, device: torch.device) -> torch.Tensor:
    weights_config = config.get("weights", {})
    seed = int(config.get("seed", 42)) + int(weights_config.get("seed_offset", 2000))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(n_basis, generator=generator, dtype=torch.float32)
    weights = weights * float(weights_config.get("init_scale", 1.0))
    return weights.to(device)


def _build_loss_fn(config: dict, target: torch.Tensor, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    loss_config = config.get("loss", {})
    loss_type = str(loss_config.get("type", "mse")).lower()

    if loss_type == "mse":
        return lambda current_skin: mse_loss(current_skin, target)

    if loss_type in {"perceptual", "perceptual_mse"}:
        layers = tuple(loss_config.get("perceptual_layers", ["relu3_3"]))
        extractor = get_vgg16_feature_extractor(
            layers=layers,
            pretrained=bool(loss_config.get("vgg_pretrained", True)),
            device=device,
        )
        with torch.no_grad():
            target_features = extractor(target)
        layer_weights = loss_config.get("layer_weights", {})
        perceptual_weight = float(loss_config.get("perceptual_weight", 1.0))
        mse_weight = float(loss_config.get("mse_weight", 0.0 if loss_type == "perceptual" else 1.0))

        def loss_fn(current_skin: torch.Tensor) -> torch.Tensor:
            loss = perceptual_weight * perceptual_loss(
                current_skin,
                target,
                extractor,
                layers=layers,
                layer_weights=layer_weights,
                target_features=target_features,
            )
            if mse_weight > 0:
                loss = loss + mse_weight * mse_loss(current_skin, target)
            return loss

        return loss_fn

    raise ValueError(f"Unsupported loss type: {loss_type}")

