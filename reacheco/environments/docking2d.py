"""Planar spacecraft docking with Clohessy-Wiltshire-Hill (CWH) dynamics.

State ``(x, y, vx, vy)``: position and velocity relative to the chief. The
normalized thrust ``u in [-1, 1]^2`` is held for ``dt``. With the thrust
appended to the state, ``z = (x, y, vx, vy, ux, uy)`` follows the linear ODE
``z' = K z``, so one period is exactly ``z(t) = exp(K t) z(0)``.

Interval arithmetic runs in float64 and the bounds are rounded outward before
they are converted back to the box dtype.

Rollouts are not provided: the controllers were trained and are evaluated in
the SafeRL docking environment (RK45 integration and its reward), whose
numbers this exact-flow model would not reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from reacheco.environments.base import System
from reacheco.spec import SafetySpec

BATCH = 32_768


@dataclass(frozen=True)
class Docking2DConfig:
    dt: float = 1.0
    mass: float = 12.0
    mean_motion: float = 0.001027

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.dt, self.mass, self.mean_motion)):
            raise ValueError("dt, mass and mean_motion must be finite and positive")


class Docking2D(System):
    """CWH propagators for verification."""

    metadata = {"render_modes": []}
    config_class = Docking2DConfig
    state_dim = 4
    action_dim = 2

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Docking2DConfig()
        n, m = self.config.mean_motion, self.config.mass
        generator = torch.zeros((6, 6), dtype=torch.float64)
        generator[0, 2] = generator[1, 3] = 1.0
        generator[2, 0] = 3.0 * n * n
        generator[2, 3] = 2.0 * n
        generator[3, 2] = -2.0 * n
        generator[2, 4] = generator[3, 5] = 1.0 / m
        self.generator = generator
        self._matrices = {}
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("Docking2D rollouts use the SafeRL environment (RK45 and its reward)")

    def step(self, action):
        raise NotImplementedError("Docking2D rollouts use the SafeRL environment (RK45 and its reward)")

    @torch.no_grad()
    def post(self, boxes, actions):
        """Exact linear image of the box: ``exp(K dt)`` applied with interval arithmetic."""
        if len(boxes) > BATCH:
            return torch.cat([self.post(boxes[start:start + BATCH], actions[start:start + BATCH])
                              for start in range(0, len(boxes), BATCH)])
        z = self._augmented(boxes, actions)
        return _outward(_linear_bounds(z, self._matrix(self.config.dt, z))[:, :, :4], boxes.dtype)

    @torch.no_grad()
    def tube(self, boxes, actions, segments):
        """Per segment: the image at the segment midpoint ``c``, widened by
        ``(exp(|K| w/2) - I) |z(c)|``, which bounds ``|z(c + s) - z(c)|`` for
        ``|s| <= w/2`` (``w`` the segment width)."""
        segments = int(segments)
        if segments < 1:
            raise ValueError("segments must be positive")
        z = self._augmented(boxes, actions)
        width = self.config.dt / segments
        variation = self._matrix(width / 2, z, absolute=True)
        key = ("segments", segments, str(z.device))
        if key not in self._matrices:
            times = (torch.arange(segments, dtype=torch.float64) + 0.5) * width
            self._matrices[key] = torch.stack([self._matrix(float(t), z) for t in times.tolist()])
        matrices = self._matrices[key]
        positive, negative = matrices.clamp_min(0), matrices.clamp_max(0)
        z_lo, z_hi = z[:, 0], z[:, 1]
        lo = torch.einsum("nj,sij->sni", z_lo, positive) + torch.einsum("nj,sij->sni", z_hi, negative)
        hi = torch.einsum("nj,sij->sni", z_hi, positive) + torch.einsum("nj,sij->sni", z_lo, negative)
        radius = torch.maximum(lo.abs(), hi.abs()) @ variation.T
        tube = torch.stack((lo - radius, hi + radius), dim=2)[..., :4]
        return _outward(tube.reshape(-1, 2, 4), boxes.dtype).reshape(segments, len(boxes), 2, 4)

    def _matrix(self, time, reference, *, absolute=False):
        """``exp(K t)``, or ``exp(|K| t) - I`` clipped at 0, cached per device."""
        key = (float(time), str(reference.device), absolute)
        if key not in self._matrices:
            matrix = torch.matrix_exp((self.generator.abs() if absolute else self.generator) * time)
            if absolute:
                matrix = (matrix - torch.eye(6, dtype=torch.float64)).clamp_min(0)
            self._matrices[key] = matrix.to(reference.device)
        return self._matrices[key]

    @staticmethod
    def _augmented(boxes, actions):
        if boxes.ndim != 3 or boxes.shape[1:] != (2, 4) or actions.shape != (len(boxes), 2):
            raise ValueError("Docking2D needs boxes (N, 2, 4) and actions (N, 2)")
        thrust = actions.to(device=boxes.device, dtype=torch.float64).clamp(-1, 1)
        return torch.cat((boxes.double(), thrust[:, None].expand(-1, 2, -1)), dim=-1)


def _linear_bounds(boxes, matrix):
    positive, negative = matrix.clamp_min(0), matrix.clamp_max(0)
    lo = boxes[:, 0] @ positive.T + boxes[:, 1] @ negative.T
    hi = boxes[:, 1] @ positive.T + boxes[:, 0] @ negative.T
    return torch.stack((lo, hi), dim=1)


def _outward(boxes, dtype):
    """Pad float64 round-off, convert, and round one more step outward."""
    pad = 128 * torch.finfo(torch.float64).eps * (1 + boxes.abs().amax(1))
    lo = (boxes[:, 0] - pad).to(dtype)
    hi = (boxes[:, 1] + pad).to(dtype)
    return torch.stack((torch.nextafter(lo, torch.full_like(lo, -torch.inf)),
                        torch.nextafter(hi, torch.full_like(hi, torch.inf))), dim=1)


@dataclass(frozen=True, eq=False)
class Docking2DSpec(SafetySpec):
    """Speed limit ``||v|| <= speed_offset + 2 n ||p||`` with the system's mean motion ``n``.

    Equality is safe. Evaluated on boxes with interval norms: the box meets
    the unsafe set if its fastest speed exceeds the limit at its smallest
    distance, and lies in it if even its slowest speed exceeds the limit at its
    largest distance.
    """

    state_dim = 4
    speed_offset: float = 0.2
    speed_slope: float = 2 * 0.001027

    def __post_init__(self):
        super().__post_init__()
        if not all(np.isfinite(v) and v >= 0 for v in (self.speed_offset, self.speed_slope)):
            raise ValueError("speed_offset and speed_slope must be finite and nonnegative")

    @classmethod
    def from_dict(cls, data, system, initial_boxes, goal_boxes):
        return cls(**cls.common_fields(data, initial_boxes, goal_boxes),
                   speed_offset=float(data.get("speed_offset", 0.2)),
                   speed_slope=2 * system.config.mean_motion)

    def intersects_unsafe(self, boxes):
        smallest_distance, _ = _norm_range(boxes[:, 0, :2], boxes[:, 1, :2])
        _, largest_speed = _norm_range(boxes[:, 0, 2:], boxes[:, 1, 2:])
        return largest_speed > self.speed_offset + self.speed_slope * smallest_distance

    def inside_unsafe(self, boxes):
        _, largest_distance = _norm_range(boxes[:, 0, :2], boxes[:, 1, :2])
        smallest_speed, _ = _norm_range(boxes[:, 0, 2:], boxes[:, 1, 2:])
        return smallest_speed > self.speed_offset + self.speed_slope * largest_distance


def _norm_range(lower, upper):
    """Smallest and largest Euclidean norm over the box ``[lower, upper]``."""
    smallest = torch.where((lower <= 0) & (upper >= 0), 0.0, torch.minimum(lower.abs(), upper.abs()))
    largest = torch.maximum(lower.abs(), upper.abs())
    return torch.linalg.vector_norm(smallest, dim=-1), torch.linalg.vector_norm(largest, dim=-1)
