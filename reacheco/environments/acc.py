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
    
    
    ACC requiered exact numerics, as simple rounding buts many boxes exactly on thresholds
    this results in a larger number of additional boxes, going oom for pmax 100.
    """

    metadata = {"render_modes": []}
    config_class = ACCConfig
    state_dim = 2
    action_dim = 1

    def __init__(self, config=None, *, mode="train", initial_boxes=None,
                 boundary_probability=0.1, epsilon=1e-6):
        super().__init__()
        self.config = config or ACCConfig()
        # The propagators take dt as exactly 1 / steps and integer accelerations.
        self._steps = round(1.0 / self.config.dt)
        if (abs(self._steps * self.config.dt - 1.0) > 1e-12 or self.config.max_acceleration % 1
                or self.config.max_braking_acceleration % 1):
            raise ValueError("ACC needs dt = 1 / k for an integer k and integer accelerations")
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
        """Exact successor box: the affine map is monotone in both coordinates.

        With ``dt = 1 / k``: ``p' = (2 k^2 p + 2 k v + a) / (2 k^2)`` and
        ``v' = (k v + a) / k``, rounded outward to the dtype of ``boxes``
        (``_rounded``).
        """
        k = self._steps
        a = self._acceleration(actions, boxes.new_empty(0, dtype=torch.float64))
        # does the result fit into 64 bit?
        exact = _exact_inputs(boxes, actions, (2 * k * k, 2 * k, k), self._acceleration_bits())
        bounds = []
        for corner, upward in ((boxes[:, 0].double(), False), (boxes[:, 1].double(), True)):
            position = _rounded((2 * k * k * corner[:, 0], 2 * k * corner[:, 1], a), 2 * k * k, boxes.dtype,
                                upward=upward, exact=exact)
            velocity = _rounded((k * corner[:, 1], a), k, boxes.dtype, upward=upward, exact=exact)
            bounds.append(torch.stack((position, velocity), 1))
        return torch.stack(bounds, 1)

    @torch.no_grad()
    def tube(self, boxes, actions, segments):
        """Exact coordinate-wise hull of every trajectory per time segment.

        ``p(t) = p + v t + a t^2 / 2`` is increasing in ``p`` and ``v``, so its
        extremes over the box are attained at the lower and upper corner. Over
        a segment ``[t0, t1]`` the extreme lies at an endpoint or at the
        stationary point ``-v / a``. Velocity is affine in ``t``. Segment
        endpoints are ``j / m`` with ``m = k * segments``; bounds are rounded
        outward to the dtype of ``boxes`` (``_rounded``).
        """
        segments = int(segments)
        if segments < 1:
            raise ValueError("segments must be positive")
        m = self._steps * segments
        a = self._acceleration(actions, boxes.new_empty(0, dtype=torch.float64))[None]
        j = torch.arange(segments + 1, dtype=torch.float64, device=boxes.device)[:, None]
        t0, t1 = j[:-1] / m, j[1:] / m
        exact = _exact_inputs(boxes, actions, (2 * m * m, 2 * m * segments, m),
                              self._acceleration_bits() + (segments * segments).bit_length())
        hull = []
        for corner, upward in ((boxes[:, 0].double()[None], False), (boxes[:, 1].double()[None], True)):
            pick = torch.maximum if upward else torch.minimum
            p, v = corner[..., 0], corner[..., 1]
            position = _rounded((2 * m * m * p, 2 * m * j * v, a * j * j), 2 * m * m, boxes.dtype,
                                upward=upward, exact=exact)
            position = pick(position[:-1], position[1:])
            # The stationary point, where inside the segment (generously: an
            # extra candidate only widens the bound).
            nonzero = a != 0
            safe = torch.where(nonzero, a, torch.ones_like(a))
            stationary = -v / safe
            tolerance = 4 * EPS * (stationary.abs() + t1)
            inside = nonzero & (stationary >= t0 - tolerance) & (stationary <= t1 + tolerance)
            drop = v.square() / (2 * safe)
            extreme = _padded(p - drop, 4 * EPS * (p.abs() + drop.abs()), boxes.dtype, upward=upward)
            position = torch.where(inside, pick(position, extreme), position)
            velocity = _rounded((m * v, a * j), m, boxes.dtype, upward=upward, exact=exact)
            hull.append(torch.stack((position, pick(velocity[:-1], velocity[1:])), -1))
        return torch.stack(hull, 2)

    def _acceleration_bits(self):
        """Significand bits of ``a = u * scale`` for float32 ``u``."""
        return 24 + max(_significand_bits(self.config.max_acceleration),
                        _significand_bits(self.config.max_braking_acceleration))

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


# Relative margin for the float64 round-off of the few operations in a spec check.
ROUNDING = 16 * torch.finfo(torch.float64).eps


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

    def _unsafe_corner(self, corner, margin):
        """``margin`` moves the invariant's bound by its float64 round-off (+1 towards "unsafe", -1 away)."""
        corner = corner.double()
        if self.unsafe == "collision":
            return corner[:, 0] <= 0.0
        closing = torch.clamp(-corner[:, 1], min=0.0)
        bound = closing.square() / (2.0 * self.braking_acceleration)
        return corner[:, 0] <= bound * (1 + margin * ROUNDING)

    def intersects_unsafe(self, boxes):
        """Decided at the lower corner: minimum position, fastest closing."""
        return self._unsafe_corner(boxes[:, 0], 1)

    def inside_unsafe(self, boxes):
        """Decided at the upper corner: maximum position, slowest closing."""
        return self._unsafe_corner(boxes[:, 1], -1)


EPS = torch.finfo(torch.float64).eps


def _rounded(terms, denominator, dtype, *, upward, exact):
    """``sum(terms) / denominator`` rounded to ``dtype`` towards +inf (``upward``) or -inf.

    ``terms`` are float64; with ``exact`` they hold their values exactly
    (products of float32 values and small integers). Where their float64 sum
    is exact as well (checked with TwoSum) the result is the exact directed
    rounding, so a bound that lands on a float32 number, such as a cell
    threshold, stays on it. Elsewhere the float64 quotient is padded by its
    round-off and rounded outward.
    """
    total = terms[0]
    exact_sum = torch.full(total.shape, exact, dtype=torch.bool, device=total.device)
    for term in terms[1:]:
        # https://en.wikipedia.org/wiki/2Sum
        total, error = _two_sum(total, term)
        exact_sum = exact_sum & (error == 0)
    quotient = total / denominator
    nearest = quotient.to(dtype)
    scaled = nearest.double() * denominator      # exact: float32 times a small integer
    beyond = scaled < total if upward else scaled > total
    limit = torch.full_like(nearest, torch.inf if upward else -torch.inf)
    directed = torch.where(beyond, torch.nextafter(nearest, limit), nearest)
    # when total is not exact. i.e. exact sum is false
    magnitude = sum(term.abs() for term in terms) / denominator
    # maximum bound on the roudings happening due to the sums.
    padded = _padded(quotient, 4 * EPS * (magnitude + quotient.abs()), dtype, upward=upward)
    return torch.where(exact_sum, directed, padded)


def _padded(value, pad, dtype, *, upward):
    """``value`` moved by ``pad`` and one more step of ``dtype`` towards +inf (``upward``) or -inf."""
    moved = (value + pad if upward else value - pad).to(dtype)
    return torch.nextafter(moved, torch.full_like(moved, torch.inf if upward else -torch.inf))

# never use torch compile with this
def _two_sum(x, y):
    """``x + y`` and its rounding error (Knuth's TwoSum)."""
    total = x + y
    back = total - x
    return total, (x - (total - back)) + (y - back)


def _exact_inputs(boxes, actions, multipliers, acceleration_bits):
    """Whether the float64 terms are exact: float32 states and actions, and
    products within 53 significand bits."""
    return (boxes.dtype == torch.float32 and actions.dtype == torch.float32 and acceleration_bits <= 53
            and all(24 + int(c).bit_length() <= 53 for c in multipliers))


def _significand_bits(value):
    value = int(value)
    return (value // (value & -value)).bit_length() if value else 0
