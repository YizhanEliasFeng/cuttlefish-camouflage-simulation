"""Optimizer-like neural feedback controllers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class ControllerState:
    controller_type: str
    learning_rate: float
    noise_sigma: float = 0.0
    noise_decay: float = 1.0
    motor_momentum: float = 0.0
    sampling: str = "continuous"
    sample_interval: int | None = None
    sample_probability: float | None = None
    decay_tau: float | None = None


class BaseController:
    """Small interface shared by all controllers."""

    def zero_grad(self) -> None:
        raise NotImplementedError

    def step(self, step_index: int) -> None:
        raise NotImplementedError

    def wants_gradient(self, step_index: int) -> bool:
        return True

    @property
    def feedback_records(self) -> list[dict[str, float | int]]:
        return []

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


class IntermittentDecayController(BaseController):
    """Intermittent visual sampling with exponentially decaying feedback.

    On sampling steps, the current loss gradient is cached as the sensory
    feedback direction. Between sampling steps, no new gradient is required;
    the cached direction is reused with gain `exp(-steps_since_sample / tau)`.
    A motor command with inertia integrates that decaying feedback into skin
    activation updates.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        learning_rate: float,
        sample_interval: int = 8,
        decay_tau: float = 3.0,
        motor_momentum: float = 0.7,
        noise_sigma: float = 0.0,
        noise_decay: float = 1.0,
        sampling: str = "fixed",
        sample_probability: float = 0.15,
        seed: int = 0,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        if decay_tau <= 0:
            raise ValueError("decay_tau must be positive")
        if not 0.0 <= motor_momentum < 1.0:
            raise ValueError("motor_momentum must be in [0, 1)")
        if not 0.0 <= sample_probability <= 1.0:
            raise ValueError("sample_probability must be in [0, 1]")

        sampling = sampling.lower()
        if sampling not in {"fixed", "stochastic"}:
            raise ValueError("sampling must be 'fixed' or 'stochastic'")

        self.weights = weights
        self.velocity = torch.zeros_like(weights)
        self.cached_grad: torch.Tensor | None = None
        self.steps_since_sample = 0
        self._current_sample_event = True
        self._records: list[dict[str, float | int]] = []
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self._state = ControllerState(
            controller_type="intermittent_decay",
            learning_rate=learning_rate,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            motor_momentum=motor_momentum,
            sampling=sampling,
            sample_interval=sample_interval,
            sample_probability=sample_probability,
            decay_tau=decay_tau,
        )

    def wants_gradient(self, step_index: int) -> bool:
        self._current_sample_event = self._should_sample(step_index)
        if self.cached_grad is None:
            self._current_sample_event = True
        return self._current_sample_event

    def zero_grad(self) -> None:
        if self.weights.grad is not None:
            self.weights.grad.zero_()

    def step(self, step_index: int) -> None:
        sample_event = bool(self._current_sample_event)
        if sample_event:
            if self.weights.grad is None:
                raise RuntimeError("Sampling step requires a current gradient")
            self.cached_grad = self.weights.grad.detach().clone()
            self.steps_since_sample = 0

        if self.cached_grad is None:
            raise RuntimeError("Cannot update without a cached sensory gradient")

        gain = math.exp(-self.steps_since_sample / float(self._state.decay_tau))
        sigma = self._state.noise_sigma * (self._state.noise_decay ** step_index)

        with torch.no_grad():
            self.velocity.mul_(self._state.motor_momentum)
            self.velocity.add_(-self._state.learning_rate * gain * self.cached_grad)
            self.weights.add_(self.velocity)
            if sigma > 0:
                noise = torch.randn(
                    self.weights.shape,
                    generator=self.generator,
                    dtype=self.weights.dtype,
                ).to(self.weights.device)
                self.weights.add_(sigma * noise)

        self._records.append(
            {
                "step": int(step_index),
                "sample_event": int(sample_event),
                "feedback_gain": float(gain),
                "steps_since_sample": int(self.steps_since_sample),
                "motor_velocity_norm": float(torch.linalg.vector_norm(self.velocity.detach()).cpu().item()),
            }
        )
        self.steps_since_sample += 1
        self.zero_grad()

    @property
    def feedback_records(self) -> list[dict[str, float | int]]:
        return self._records

    @property
    def state(self) -> ControllerState:
        return self._state

    def _should_sample(self, step_index: int) -> bool:
        if step_index == 0:
            return True
        if self._state.sampling == "fixed":
            return step_index % int(self._state.sample_interval) == 0
        draw = torch.rand((), generator=self.generator).item()
        return draw < float(self._state.sample_probability)


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
    if controller_type in {"intermittent_decay", "decaying_intermitent", "decaying_intermittent"}:
        return IntermittentDecayController(
            weights=weights,
            learning_rate=learning_rate,
            sample_interval=int(config.get("sample_interval", 8)),
            decay_tau=float(config.get("decay_tau", 3.0)),
            motor_momentum=float(config.get("motor_momentum", 0.7)),
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            sampling=str(config.get("sampling", "fixed")),
            sample_probability=float(config.get("sample_probability", 0.15)),
            seed=seed,
        )

    raise ValueError(f"Unsupported optimizer/controller type: {controller_type}")
