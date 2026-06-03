"""Visual feedback losses for the camouflage loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


VGG16_LAYER_INDICES = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu4_1": 18,
    "relu4_2": 20,
    "relu4_3": 22,
}


def mse_loss(rendered_skin: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared visual discrepancy."""

    return F.mse_loss(_as_batch(rendered_skin), _as_batch(target))


def _as_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError("image tensor must have shape [H,W], [C,H,W], or [B,C,H,W]")


def _prepare_vgg_input(image: torch.Tensor) -> torch.Tensor:
    image = _as_batch(image)
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    elif image.shape[1] != 3:
        raise ValueError("VGG16 input must have one or three channels")

    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return (image.clamp(0.0, 1.0) - mean) / std


class VGG16FeatureExtractor(nn.Module):
    """Frozen VGG16 feature extractor for perceptual feedback."""

    def __init__(
        self,
        layers: Sequence[str],
        pretrained: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        unknown = sorted(set(layers) - set(VGG16_LAYER_INDICES))
        if unknown:
            raise ValueError(f"Unsupported VGG16 layer names: {unknown}")

        try:
            from torchvision import models
            from torchvision.models import VGG16_Weights
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for perceptual loss. "
                "Install project dependencies from requirements.txt."
            ) from exc

        weights = VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights).features.eval()
        max_index = max(VGG16_LAYER_INDICES[layer] for layer in layers)
        self.features = nn.Sequential(*list(vgg.children())[: max_index + 1])
        self.layers = tuple(layers)
        self.layer_indices = {VGG16_LAYER_INDICES[name]: name for name in self.layers}

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.to(device)
        self.eval()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = _prepare_vgg_input(image)
        outputs: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = self.layer_indices.get(index)
            if name is not None:
                outputs[name] = x
        return outputs


def get_vgg16_feature_extractor(
    layers: Sequence[str],
    pretrained: bool = True,
    device: str | torch.device = "cpu",
) -> VGG16FeatureExtractor:
    """Build a frozen VGG16 feature extractor."""

    return VGG16FeatureExtractor(layers=layers, pretrained=pretrained, device=device)


def perceptual_loss(
    rendered_skin: torch.Tensor,
    target: torch.Tensor,
    model: VGG16FeatureExtractor,
    layers: Sequence[str] | None = None,
    layer_weights: Mapping[str, float] | None = None,
    target_features: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Feature-space loss between current skin and target."""

    selected_layers = tuple(layers or model.layers)
    weights = dict(layer_weights or {})
    skin_features = model(rendered_skin)
    if target_features is None:
        with torch.no_grad():
            target_features = model(target)

    loss = rendered_skin.new_tensor(0.0)
    for layer in selected_layers:
        layer_weight = float(weights.get(layer, 1.0))
        loss = loss + layer_weight * F.mse_loss(skin_features[layer], target_features[layer])
    return loss

