"""Skin-pattern basis construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def create_random_smooth_basis(
    n_basis: int,
    height: int,
    width: int,
    low_res: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create deterministic smooth texture bases.

    The bases are low-resolution random fields enlarged with bicubic
    interpolation. They are not meant to be literal chromatophores, but they
    keep the model in a constrained activation-over-bases regime instead of
    free-pixel optimization.
    """

    if n_basis <= 0:
        raise ValueError("n_basis must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if low_res <= 1:
        raise ValueError("low_res must be greater than 1")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(n_basis, 1, low_res, low_res, generator=generator)
    basis = F.interpolate(raw, size=(height, width), mode="bicubic", align_corners=False)

    flat = basis.flatten(start_dim=1)
    means = flat.mean(dim=1).view(n_basis, 1, 1, 1)
    stds = flat.std(dim=1, unbiased=False).clamp_min(1e-6).view(n_basis, 1, 1, 1)
    basis = (basis - means) / stds
    basis = torch.tanh(basis)
    return basis.to(device=device, dtype=torch.float32)


def create_basis_from_config(config: dict, image_config: dict, seed: int) -> torch.Tensor:
    """Factory for basis construction from a config dictionary."""

    basis_type = config.get("type", "random_smooth")
    height = int(image_config.get("height", 64))
    width = int(image_config.get("width", 64))

    if basis_type != "random_smooth":
        raise ValueError(f"Unsupported basis type: {basis_type}")

    return create_random_smooth_basis(
        n_basis=int(config.get("n_basis", 32)),
        height=height,
        width=width,
        low_res=int(config.get("low_res", 8)),
        seed=int(seed),
        device=config.get("device", image_config.get("device", "cpu")),
    )

