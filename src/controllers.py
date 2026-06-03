"""Optimizer-like neural feedback controllers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ControllerState:
    controller_type: str
    learning_rate: float
    noise_sigma: float = 0.0
    noise_decay: float = 1.0


class BaseController:
    """Small interface shared by all controllers."""

    def zero_grad(self) -> None:
        raise NotImplementedError

    def step(self, step_index: int) -> None:
        raise NotImplementedError

    @property
    def state(self) -> ControllerState:
        raise NotImplementedError


class OptimizerController(BaseController):
    """Torch optimizer plus optional post-update activation noise."""

    def __init__(
        self,
        weights: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        controller_type: str,
        learning_rate: float,
        noise_sigma: float = 0.0,
        noise_decay: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.weights = weights
        self.optimizer = optimizer
        self._state = ControllerState(controller_type, learning_rate, noise_sigma, noise_decay)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, step_index: int) -> None:
        self.optimizer.step()
        self._add_noise(step_index)

    @property
    def state(self) -> ControllerState:
        return self._state

    def _add_noise(self, step_index: int) -> None:
        sigma = self._state.noise_sigma * (self._state.noise_decay ** step_index)
        if sigma <= 0:
            return
        with torch.no_grad():
            noise = torch.randn(
                self.weights.shape,
                generator=self.generator,
                dtype=self.weights.dtype,
            ).to(self.weights.device)
            self.weights.add_(sigma * noise)


class LangevinController(BaseController):
    """Manual gradient update with additive Gaussian activation noise."""

    def __init__(
        self,
        weights: torch.Tensor,
        learning_rate: float,
        noise_sigma: float,
        noise_decay: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.weights = weights
        self._state = ControllerState("langevin", learning_rate, noise_sigma, noise_decay)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def zero_grad(self) -> None:
        if self.weights.grad is not None:
            self.weights.grad.zero_()

    def step(self, step_index: int) -> None:
        if self.weights.grad is None:
            raise RuntimeError("Cannot update Langevin controller without gradients")

        sigma = self._state.noise_sigma * (self._state.noise_decay ** step_index)
        with torch.no_grad():
            self.weights.add_(-self._state.learning_rate * self.weights.grad)
            if sigma > 0:
                noise = torch.randn(
                    self.weights.shape,
                    generator=self.generator,
                    dtype=self.weights.dtype,
                ).to(self.weights.device)
                self.weights.add_(sigma * noise)
        self.zero_grad()

    @property
    def state(self) -> ControllerState:
        return self._state


def build_controller(
    weights: torch.Tensor,
    config: dict,
    seed: int,
) -> BaseController:
    """Construct a controller from config."""

    controller_type = str(config.get("type", "adam")).lower()
    learning_rate = float(config.get("learning_rate", 0.1))
    noise_sigma = float(config.get("noise_sigma", 0.0))
    noise_decay = float(config.get("noise_decay", 1.0))

    if controller_type == "adam":
        optimizer = torch.optim.Adam([weights], lr=learning_rate)
        return OptimizerController(
            weights,
            optimizer,
            controller_type="adam",
            learning_rate=learning_rate,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            seed=seed,
        )
    if controller_type == "sgd":
        optimizer = torch.optim.SGD(
            [weights],
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.0)),
        )
        return OptimizerController(
            weights,
            optimizer,
            controller_type="sgd",
            learning_rate=learning_rate,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            seed=seed,
        )
    if controller_type == "langevin":
        return LangevinController(
            weights=weights,
            learning_rate=learning_rate,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            seed=seed,
        )

    raise ValueError(f"Unsupported optimizer/controller type: {controller_type}")

