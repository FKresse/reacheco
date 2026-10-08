"""Adaptive cruise control (ACC) with relative position and velocity.

State ``(p, v)``: distance to the lead vehicle and relative velocity, ``v < 0``
when closing in. The normalized action ``u in [-1, 1]`` is a relative
acceleration ``a = u * B`` for ``u >= 0`` (ego braking) and ``a = u * A``
otherwise, held for ``dt``::

    p' = p + dt v + dt^2 a / 2,    v' = v + dt a

Follows the normalized-action ACC environment of Teuber et al. (2024).
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from reacheco.environments.base import System
from reacheco.spec import SafetySpec


@dataclass(frozen=True)
class ACCConfig:
    dt: float = 0.1
    max_distance: float = 200.0
    max_acceleration: float = 100.0           # A: ego acceleration, u < 0
    max_braking_acceleration: float = 100.0   # B: ego braking, u >= 0
    reset_position_min: float = 4.0
    reset_position_max: float = 95.0
    # Share of training starts exactly on the braking boundary v = -sqrt(2 B p).
    # 0 for the paper's DWCs; 1/9 reproduces NCubeV's ACCEnv2 (Teuber et al.).
    # the DWC checkpoints reported used 0.0 for training.
    boundary_reset_probability: float = 0.0
    max_steps: int = 410

    def __post_init__(self):
        if self.dt <= 0 or self.max_acceleration <= 0 or self.max_braking_acceleration <= 0:
            raise ValueError("dt and both accelerations must be positive")
        if not 0.0 <= self.boundary_reset_probability <= 1.0:
            raise ValueError("boundary_reset_probability must be in [0, 1]")


class ACC(System):
    """ACC dynamics, interval propagators and rollouts.

    ``mode="train"`` resets to the training distribution: ``p`` uniform in
    ``[reset_position_min, reset_position_max]`` and ``v`` uniform in
    ``[-100, 100]``, or in ``[-sqrt(2 B p), 100]`` for ``p < 10``; with
    probability ``boundary_reset_probability`` instead ``v = -sqrt(2 B p)``.
    ``mode="evaluate"`` resets into the boxes ``initial_boxes`` (usually the
    verified ``X0``): area-weighted uniformly, or with probability
    ``boundary_probability`` at the lower-left corner of the box with the
    smallest position in its velocity slab, nudged in float32 until it lies
    strictly inside the braking invariant ``p > max(-v, 0)^2 / (2 B)``.
    """

    metadata = {"render_modes": []}
    config_class = ACCConfig
    state_dim = 2
    action_dim = 1

    def __init__(self, config=None, *, mode="train", initial_boxes=None,
                 boundary_probability=0.1, epsilon=1e-6):
        super().__init__()
        self.config = config or ACCConfig()
        if mode not in ("train", "evaluate"):
            raise ValueError("mode must be 'train' or 'evaluate'")
        self.mode = mode
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0
        if mode == "evaluate":
            if initial_boxes is None or not len(initial_boxes):
                raise ValueError("evaluation mode needs initial boxes")
            self._init_evaluation(initial_boxes, boundary_probability, epsilon)

    # ------------------------------------------------------------ dynamics

    def _acceleration(self, actions, reference):
        """Relative acceleration ``(N,)`` for normalized actions ``(N, 1)``."""
        if actions.ndim != 2 or actions.shape[1] != 1:
            raise ValueError("ACC actions must have shape (N, 1)")
        u = actions[:, 0].to(reference).clamp(-1.0, 1.0)
        scale = torch.where(u >= 0.0, u.new_tensor(self.config.max_braking_acceleration),
                            u.new_tensor(self.config.max_acceleration))
        return u * scale

    @torch.no_grad()
    def post(self, boxes, actions):
        """Exact successor box: the affine map is monotone in both coordinates."""
        dt = self.config.dt
        a = self._acceleration(actions, boxes)
        lower, upper = boxes[:, 0], boxes[:, 1]
        position = torch.stack((lower[:, 0] + dt * lower[:, 1] + 0.5 * dt**2 * a,
                                upper[:, 0] + dt * upper[:, 1] + 0.5 * dt**2 * a), 1)
        velocity = torch.stack((lower[:, 1] + dt * a, upper[:, 1] + dt * a), 1)
        return torch.stack((position, velocity), 2)

    @torch.no_grad()
    def tube(self, boxes, actions, segments):
        """Exact coordinate-wise hull of every trajectory per time segment.

        ``p(t) = p + v t + a t^2 / 2`` is increasing in ``p`` and ``v``, so its
        extremes over the box are attained at the lower and upper corner. Over
        a segment ``[t0, t1]`` the extreme lies at an endpoint or at the
        stationary point ``-v / a``. Velocity is affine in ``t``.
        """
        if int(segments) < 1:
            raise ValueError("segments must be positive")
        a = self._acceleration(actions, boxes)[None]
        times = torch.linspace(0.0, self.config.dt, int(segments) + 1, dtype=boxes.dtype, device=boxes.device)
        t0, t1 = times[:-1, None], times[1:, None]
        lower, upper = boxes[:, 0], boxes[:, 1]
        position_lower = _quadratic_extreme(lower[None, :, 0], lower[None, :, 1], a, t0, t1, minimum=True)
        position_upper = _quadratic_extreme(upper[None, :, 0], upper[None, :, 1], a, t0, t1, minimum=False)
        velocity_lower = torch.minimum(lower[None, :, 1] + a * t0, lower[None, :, 1] + a * t1)
        velocity_upper = torch.maximum(upper[None, :, 1] + a * t0, upper[None, :, 1] + a * t1)
        return torch.stack((torch.stack((position_lower, velocity_lower), -1),
                            torch.stack((position_upper, velocity_upper), -1)), 2)

    # ------------------------------------------------------------ rollouts

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._evaluation_start() if self.mode == "evaluate" else self._training_start()
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        config = self.config
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        a = u * (config.max_braking_acceleration if u >= 0.0 else config.max_acceleration)
        position, velocity = (float(value) for value in self.state)
        position = position + config.dt * velocity + 0.5 * config.dt * config.dt * a
        velocity = velocity + config.dt * a
        self.state = np.array([position, velocity], dtype=np.float32)
        self.steps += 1

        crash = position <= 0.0
        too_far = position > config.max_distance
        terminated = bool(crash or too_far)
        truncated = bool(self.steps >= config.max_steps and not terminated)
        reward = -2000.0 if crash else -800.0 if too_far else 10.0 * (1.0 - abs(u))
        return self.state.copy(), reward, terminated, truncated, {"crash": crash, "too_far": too_far}

    def _training_start(self):
        config = self.config
        on_boundary = float(self.np_random.random()) < config.boundary_reset_probability
        position = float(self.np_random.uniform(config.reset_position_min, config.reset_position_max))
        if on_boundary:
            velocity = -np.sqrt(2.0 * config.max_braking_acceleration * position)
        else:
            low = -np.sqrt(2.0 * config.max_braking_acceleration * position) if position < 10.0 else -100.0
            velocity = float(self.np_random.uniform(low, 100.0))
        return np.array([position, velocity], dtype=np.float32)

    def _init_evaluation(self, boxes, boundary_probability, epsilon):
        boxes = np.asarray(torch.as_tensor(boxes, dtype=torch.float64).cpu(), dtype=np.float64)
        self._lower, self._upper = boxes[:, 0], boxes[:, 1]
        area = np.prod(self._upper - self._lower, axis=1)
        self._weights = area / area.sum() if area.sum() > 0 else None
        # Lower frontier of X0: per velocity slab, the box with the smallest position.
        slabs = {}
        for index, (velocity_low, velocity_high) in enumerate(zip(self._lower[:, 1], self._upper[:, 1])):
            best = slabs.setdefault((float(velocity_low), float(velocity_high)), index)
            if self._lower[index, 0] < self._lower[best, 0]:
                slabs[(float(velocity_low), float(velocity_high))] = index
        self._boundary_boxes = np.asarray(sorted(slabs.values()), dtype=np.int64)
        self._boundary_probability = float(boundary_probability)
        self._epsilon = float(epsilon)

    def _evaluation_start(self):
        rng = self.np_random
        if rng.random() < self._boundary_probability:
            index = int(self._boundary_boxes[rng.integers(len(self._boundary_boxes))])
            position, velocity = self._lower[index]
            state = np.array([position + self._epsilon, velocity], dtype=np.float32)
            braking = self.config.max_braking_acceleration
            # this is not really necessesary and is never executed anyways, 
            # leaving it in for compatability
            for _ in range(64):
                if float(state[0]) > max(-float(state[1]), 0.0) ** 2 / (2.0 * braking):
                    break
                state[0] = np.nextafter(state[0], np.float32(np.inf))
            return state
        index = int(rng.choice(len(self._lower), p=self._weights))
        return rng.uniform(self._lower[index], self._upper[index]).astype(np.float32)


@dataclass(frozen=True, eq=False)
class ACCSpec(SafetySpec):
    """Collision avoidance (``p <= 0`` unsafe) or the braking invariant.

    In ``"braking-invariant"`` mode ``p <= max(-v, 0)^2 / (2 B)`` is unsafe,
    with ``B`` the braking acceleration of the system: the states from which
    full braking cannot avoid a collision.
    """

    state_dim = 2
    unsafe: str = "collision"
    braking_acceleration: float = 100.0

    def __post_init__(self):
        super().__post_init__()
        if self.unsafe not in ("collision", "braking-invariant"):
            raise ValueError("unsafe must be 'collision' or 'braking-invariant'")

    @classmethod
    def from_dict(cls, data, system, initial_boxes, goal_boxes):
        return cls(**cls.common_fields(data, initial_boxes, goal_boxes), unsafe=data.get("unsafe", "collision"),
                   braking_acceleration=system.config.max_braking_acceleration)

    def _unsafe_corner(self, corner):
        if self.unsafe == "collision":
            return corner[:, 0] <= 0.0
        closing = torch.clamp(-corner[:, 1], min=0.0)
        return corner[:, 0] <= closing.square() / (2.0 * self.braking_acceleration)

    def intersects_unsafe(self, boxes):
        """Decided at the lower corner: minimum position, fastest closing."""
        return self._unsafe_corner(boxes[:, 0])

    def inside_unsafe(self, boxes):
        """Decided at the upper corner: maximum position, slowest closing."""
        return self._unsafe_corner(boxes[:, 1])


def _quadratic_extreme(position, velocity, acceleration, t0, t1, *, minimum):
    """Exact minimum (or maximum) of ``p + v t + a t^2 / 2`` over ``[t0, t1]``."""
    def at(t):
        return position + velocity * t + 0.5 * acceleration * t.square()

    pick = torch.minimum if minimum else torch.maximum
    extreme = pick(at(t0), at(t1))
    nonzero = acceleration.abs() > torch.finfo(acceleration.dtype).eps
    stationary = -velocity / torch.where(nonzero, acceleration, torch.ones_like(acceleration))
    inside = nonzero & (stationary >= t0) & (stationary <= t1)
    return torch.where(inside, pick(extreme, at(stationary)), extreme)
