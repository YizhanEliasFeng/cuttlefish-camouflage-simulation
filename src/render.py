"""Virtual skin rendering from activation weights."""

from __future__ import annotations

import torch


def render_skin(
    weights: torch.Tensor,
    basis: torch.Tensor,
    activation: str = "sigmoid",
    clamp: bool = True,
) -> torch.Tensor:
    """Render a skin image from basis activations.

    Parameters
    ----------
    weights:
        Shape `[n_basis]` for one skin state or `[batch, n_basis]`.
    basis:
        Shape `[n_basis, 1, height, width]`.
    activation:
        Output bounding rule. Supported values are `sigmoid`, `tanh`, and
        `none`.
    clamp:
        Clamp the final output into `[0, 1]`.
    """

    if basis.ndim != 4:
        raise ValueError("basis must have shape [n_basis, 1, height, width]")
    if basis.shape[1] != 1:
        raise ValueError("basis currently expects a single image channel")
    if weights.shape[-1] != basis.shape[0]:
        raise ValueError(
            f"weights last dimension ({weights.shape[-1]}) must match "
            f"basis count ({basis.shape[0]})"
        )

    if weights.ndim == 1:
        raw = torch.sum(weights.view(-1, 1, 1, 1) * basis, dim=0)
    elif weights.ndim == 2:
        raw = torch.einsum("bn,nchw->bchw", weights, basis)
    else:
        raise ValueError("weights must have shape [n_basis] or [batch, n_basis]")

    if activation == "sigmoid":
        skin = torch.sigmoid(raw)
    elif activation == "tanh":
        skin = (torch.tanh(raw) + 1.0) / 2.0
    elif activation in {"none", None}:
        skin = raw
    else:
        raise ValueError(f"Unsupported render activation: {activation}")

    if clamp:
        skin = skin.clamp(0.0, 1.0)
    return skin

