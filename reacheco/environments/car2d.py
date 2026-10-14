"""Car2D: an acceleration-controlled unicycle on an obstacle course.

State ``(x, y, heading, speed)``; the normalized action ``(u_a, u_w)`` gives
the acceleration ``a = A u_a`` and turn rate ``w = W u_w``, held for ``dt``::

    x' = v cos(theta),  y' = v sin(theta),  theta' = w,  v' = a

The flow is integrated exactly: ``theta(t) = theta + w t``, ``v(t) = v + a t``,
and the position integrals have closed forms (``_bounds_at_time``).
Verification follows the paper's NAV-style benchmark; rollouts follow the
course the controllers were trained on (``Car2DConfig``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from reacheco.environments.base import System, outward
from reacheco.spec import SafetySpec

# The "wall" course: the lower first-row obstacle, the central block extended
# to the bottom edge, and the lower last-row obstacle, as (centre x, centre y,
# half width x, half width y).
WALL_OBSTACLES = ((3.0, 2.5, 0.6, 0.6), (5.5, 2.45, 0.6, 3.45), (8.0, 2.8, 0.6, 0.6))


@dataclass(frozen=True)
class Car2DConfig:
    # Dynamics.
    dt: float = 1.0
    max_acceleration: float = 0.25
    max_turn_rate: float = 0.5
    # Rollout task: world, unsafe set and goal of the course.
    min_speed: float = 0.25
    max_speed: float = 1.0
    world_min: float = -1.0
    world_max: float = 11.0
    heading_limit: float = math.pi
    obstacles: tuple = WALL_OBSTACLES
    goal_center: tuple = (10.0, 5.0)
    goal_half_width: float = 1.0
    # Training starts and episode length.
    start_x: tuple = (-0.5, 0.5)
    start_y: tuple = (2.0, 8.0)
    start_heading: tuple = (-math.pi / 4, math.pi / 4)
    max_steps: int = 60
    # Reward: progress towards the goal box, minus time and action penalties.
    goal_bonus: float = 10.0
    crash_penalty: float = 10.0
    time_penalty: float = 0.02
    action_penalty: float = 0.01

    def __post_init__(self):
        if self.dt <= 0 or self.max_acceleration <= 0 or self.max_turn_rate <= 0:
            raise ValueError("dt, max_acceleration and max_turn_rate must be positive")
        if self.max_turn_rate * self.dt >= math.pi:
            raise ValueError("a period must turn by less than pi")
        if not 0.0 < self.min_speed < self.max_speed:
            raise ValueError("Car2D requires 0 < min_speed < max_speed")
        object.__setattr__(self, "obstacles", tuple(tuple(float(v) for v in o) for o in self.obstacles))


class Car2D(System):
    """Car2D dynamics, interval propagators and course rollouts.

    ``mode="train"`` starts uniformly in the course start region (any speed
    in ``[min_speed, max_speed]``). ``mode="evaluate"`` starts uniformly in
    ``initial_boxes`` (``X0``) and also ends an episode when the car
    enters one of ``goal_boxes`` (the verification goal). In both modes
    the reward is the course reward and an episode ends on leaving the world,
    breaking the heading or speed limits, touching an obstacle, or reaching the
    course goal (however for verification we only check obstacle touching, and heading limits).
    """

    metadata = {"render_modes": []}
    config_class = Car2DConfig
    state_dim = 4
    action_dim = 2

    def __init__(self, config=None, *, mode="train", initial_boxes=None, goal_boxes=None):
        super().__init__()
        self.config = config or Car2DConfig()
        if mode not in ("train", "evaluate"):
            raise ValueError("mode must be 'train' or 'evaluate'")
        self.mode = mode
        c = self.config
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            np.array([c.world_min, c.world_min, -c.heading_limit, c.min_speed], dtype=np.float32),
            np.array([c.world_max, c.world_max, c.heading_limit, c.max_speed], dtype=np.float32))
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        if mode == "evaluate":
            if initial_boxes is None or not len(initial_boxes):
                raise ValueError("evaluation mode needs initial boxes")
            # Starts are float32 states, so the start box is taken in float32 too.
            boxes = np.asarray(torch.as_tensor(initial_boxes, dtype=torch.float32).cpu())
            self._lower, self._upper = boxes[:, 0], boxes[:, 1]
            area = np.prod(self._upper - self._lower, axis=1)
            self._weights = area / area.sum() if area.sum() > 0 else None
            goals = np.zeros((0, 2, 4)) if goal_boxes is None else np.asarray(
                torch.as_tensor(goal_boxes, dtype=torch.float64).cpu())
            self._goals = goals

    # ------------------------------------------------------------ dynamics

    def _controls(self, actions, reference):
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("Car2D actions must have shape (N, 2)")
        actions = actions.to(reference).clamp(-1.0, 1.0)
        return self.config.max_acceleration * actions[:, 0], self.config.max_turn_rate * actions[:, 1]

    @torch.no_grad()
    def post(self, boxes, actions):
        """Computed in float64 and rounded outward to the dtype of ``boxes``."""
        exact = boxes.double()
        acceleration, turn_rate = self._controls(actions, exact)
        return outward(_bounds_at_time(exact, acceleration, turn_rate, self.config.dt), boxes.dtype)

    @torch.no_grad()
    def tube(self, boxes, actions, segments):
        """Per segment ``[t0, t1]``: the flow box at ``t0`` widened in position by
        the largest distance the car can travel during the segment. Computed in
        float64 and rounded outward to the dtype of ``boxes``."""
        segments = int(segments)
        if segments < 1:
            raise ValueError("segments must be positive")
        exact = boxes.double()
        acceleration, turn_rate = self._controls(actions, exact)
        times = torch.linspace(0.0, self.config.dt, steps=segments + 1, dtype=exact.dtype, device=exact.device)
        count = len(boxes)
        # Rows are segment-major: row s * count + i is box i in segment s.
        tiled = exact.repeat(segments, 1, 1)
        acceleration, turn_rate = acceleration.repeat(segments), turn_rate.repeat(segments)
        starts, ends = times[:-1].repeat_interleave(count), times[1:].repeat_interleave(count)
        at_start = _bounds_at_time(tiled, acceleration, turn_rate, starts)
        speeds = torch.stack((tiled[:, 0, 3] + acceleration * starts, tiled[:, 1, 3] + acceleration * starts,
                              tiled[:, 0, 3] + acceleration * ends, tiled[:, 1, 3] + acceleration * ends))
        travel = speeds.abs().amax(dim=0) * (ends - starts)
        lower, upper = at_start[:, 0].clone(), at_start[:, 1].clone()
        lower[:, :2] -= travel[:, None]
        upper[:, :2] += travel[:, None]
        lower[:, 2] = tiled[:, 0, 2] + torch.minimum(turn_rate * starts, turn_rate * ends)
        upper[:, 2] = tiled[:, 1, 2] + torch.maximum(turn_rate * starts, turn_rate * ends)
        lower[:, 3] = torch.minimum(speeds[0], speeds[2])
        upper[:, 3] = torch.maximum(speeds[1], speeds[3])
        return outward(torch.stack((lower, upper), dim=1), boxes.dtype).reshape(segments, count, 2, 4)

    # ------------------------------------------------------------ rollouts

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng, c = self.np_random, self.config
        if self.mode == "train":
            self.state = np.array([rng.uniform(*c.start_x), rng.uniform(*c.start_y), rng.uniform(*c.start_heading),
                                   rng.uniform(c.min_speed, c.max_speed)], dtype=np.float32)
        else:
            index = 0 if len(self._lower) == 1 else int(rng.choice(len(self._lower), p=self._weights))
            self.state = rng.uniform(self._lower[index], self._upper[index]).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        c = self.config
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        before = self._goal_distance()
        box = torch.as_tensor(self.state, dtype=torch.float64)[None, None].expand(1, 2, 4)
        successor = _bounds_at_time(box, *self._controls(torch.as_tensor(action)[None], box), c.dt)[0, 0]
        self.state = successor.to(torch.float32).numpy()
        self.steps += 1
        x, y, heading, speed = successor.tolist()
        unsafe = (not (c.world_min <= x <= c.world_max and c.world_min <= y <= c.world_max)
                  or abs(heading) > c.heading_limit or not c.min_speed <= speed <= c.max_speed
                  or any(ox - hx <= x <= ox + hx and oy - hy <= y <= oy + hy for ox, oy, hx, hy in c.obstacles))
        (gx, gy), half = c.goal_center, c.goal_half_width
        course_goal = bool(gx - half <= self.state[0] <= gx + half and gy - half <= self.state[1] <= gy + half)
        success = course_goal
        if self.mode == "evaluate" and len(self._goals):
            success |= bool(((self.state >= self._goals[:, 0]) & (self.state <= self._goals[:, 1])).all(-1).any())
        reward = before - self._goal_distance() - c.time_penalty - c.action_penalty * float(np.sum(action ** 2))
        if course_goal:
            reward += c.goal_bonus
        if unsafe:
            reward -= c.crash_penalty
        terminated = bool(unsafe or success)
        truncated = bool(self.steps >= c.max_steps and not terminated)
        return self.state.copy(), float(reward), terminated, truncated, {"is_success": success, "unsafe": unsafe}

    def _goal_distance(self):
        excess = np.abs(np.asarray(self.state[:2]) - np.asarray(self.config.goal_center)) - self.config.goal_half_width
        return float(np.linalg.norm(np.maximum(excess, 0.0)))


@dataclass(frozen=True, eq=False)
class Car2DSpec(SafetySpec):
    """Touching an obstacle or exceeding the heading limit is unsafe; optionally
    also leaving ``world_bounds`` (both positions) or ``speed_bounds``.

    Obstacles are ``(centre x, centre y, half width x, half width y)`` and
    closed. Containment in a union of violations is only certified for a box
    inside a single violation.
    """

    state_dim = 4
    obstacles: tuple = ()
    heading_limit: float = math.pi
    world_bounds: tuple | None = None
    speed_bounds: tuple | None = None

    def __post_init__(self):
        super().__post_init__()
        obstacles = tuple(tuple(float(v) for v in o) for o in self.obstacles)
        if any(len(o) != 4 or o[2] < 0 or o[3] < 0 for o in obstacles):
            raise ValueError("obstacles are (centre x, centre y, half width x, half width y)")
        object.__setattr__(self, "obstacles", obstacles)
        for name in ("world_bounds", "speed_bounds"):
            bounds = getattr(self, name)
            if bounds is not None:
                object.__setattr__(self, name, (float(bounds[0]), float(bounds[1])))

    @classmethod
    def from_dict(cls, data, system, initial_boxes, goal_boxes):
        return cls(**cls.common_fields(data, initial_boxes, goal_boxes), obstacles=data["obstacles"],
                   heading_limit=float(data.get("heading_limit", math.pi)),
                   world_bounds=data.get("world_bounds"), speed_bounds=data.get("speed_bounds"))

    def intersects_unsafe(self, boxes):
        return self._unsafe(boxes, entire=False)

    def inside_unsafe(self, boxes):
        return self._unsafe(boxes, entire=True)

    def _unsafe(self, boxes, *, entire):
        # In float64, where the limits are exact. For containment even the best corner must violate a limit.
        boxes = boxes.double()
        lower = boxes[:, 1 if entire else 0]
        upper = boxes[:, 0 if entire else 1]
        unsafe = (lower[:, 2] < -self.heading_limit) | (upper[:, 2] > self.heading_limit)
        if self.world_bounds is not None:
            lo, hi = self.world_bounds
            unsafe = unsafe | (lower[:, 0] < lo) | (upper[:, 0] > hi) | (lower[:, 1] < lo) | (upper[:, 1] > hi)
        if self.speed_bounds is not None:
            lo, hi = self.speed_bounds
            unsafe = unsafe | (lower[:, 3] < lo) | (upper[:, 3] > hi)
        for x, y, hx, hy in self.obstacles:
            if entire:
                hit = ((boxes[:, 0, 0] >= x - hx) & (boxes[:, 1, 0] <= x + hx)
                       & (boxes[:, 0, 1] >= y - hy) & (boxes[:, 1, 1] <= y + hy))
            else:
                hit = ((boxes[:, 0, 0] <= x + hx) & (boxes[:, 1, 0] >= x - hx)
                       & (boxes[:, 0, 1] <= y + hy) & (boxes[:, 1, 1] >= y - hy))
            unsafe = unsafe | hit
        return unsafe


# ------------------------------------------------------- exact flow bounds

def _bounds_at_time(boxes, acceleration, turn_rate, time):
    """Box enclosing the exact flow at ``time`` from every start in ``boxes``.

    With ``phi = w t``: ``theta(t) = theta + phi``, ``v(t) = v + a t`` and
    ``x(t) = x + v t sinc(phi / 2) cos(theta + phi / 2) + a t^2 R cos(theta + delta)``
    (``y`` with ``sin``), where ``R exp(i delta) = int_0^1 u exp(i phi u) du``.
    """
    time = torch.as_tensor(time, dtype=boxes.dtype, device=boxes.device)
    if time.ndim == 0:
        time = time.expand(len(boxes))
    turn = turn_rate * time
    velocity_amplitude = time * torch.sinc(turn / (2.0 * math.pi))
    velocity_phase = 0.5 * turn
    cos_lo, cos_hi = _interval_cos(boxes[:, 0, 2] + velocity_phase, boxes[:, 1, 2] + velocity_phase)
    sin_lo, sin_hi = _interval_sin(boxes[:, 0, 2] + velocity_phase, boxes[:, 1, 2] + velocity_phase)
    c0_lo, c0_hi = _scale_interval(cos_lo, cos_hi, velocity_amplitude)
    s0_lo, s0_hi = _scale_interval(sin_lo, sin_hi, velocity_amplitude)

    acceleration_amplitude, acceleration_phase = _acceleration_integral_phase(turn)
    acceleration_amplitude = acceleration_amplitude * time.square()
    cos_lo, cos_hi = _interval_cos(boxes[:, 0, 2] + acceleration_phase, boxes[:, 1, 2] + acceleration_phase)
    sin_lo, sin_hi = _interval_sin(boxes[:, 0, 2] + acceleration_phase, boxes[:, 1, 2] + acceleration_phase)
    c1_lo, c1_hi = _scale_interval(cos_lo, cos_hi, acceleration_amplitude)
    s1_lo, s1_hi = _scale_interval(sin_lo, sin_hi, acceleration_amplitude)

    dx_velocity_lo, dx_velocity_hi = _interval_product(boxes[:, 0, 3], boxes[:, 1, 3], c0_lo, c0_hi)
    dy_velocity_lo, dy_velocity_hi = _interval_product(boxes[:, 0, 3], boxes[:, 1, 3], s0_lo, s0_hi)
    dx_acceleration_lo, dx_acceleration_hi = _scale_interval(c1_lo, c1_hi, acceleration)
    dy_acceleration_lo, dy_acceleration_hi = _scale_interval(s1_lo, s1_hi, acceleration)

    lower = torch.empty_like(boxes[:, 0])
    upper = torch.empty_like(boxes[:, 1])
    lower[:, 0] = boxes[:, 0, 0] + dx_velocity_lo + dx_acceleration_lo
    upper[:, 0] = boxes[:, 1, 0] + dx_velocity_hi + dx_acceleration_hi
    lower[:, 1] = boxes[:, 0, 1] + dy_velocity_lo + dy_acceleration_lo
    upper[:, 1] = boxes[:, 1, 1] + dy_velocity_hi + dy_acceleration_hi
    lower[:, 2] = boxes[:, 0, 2] + turn
    upper[:, 2] = boxes[:, 1, 2] + turn
    lower[:, 3] = boxes[:, 0, 3] + acceleration * time
    upper[:, 3] = boxes[:, 1, 3] + acceleration * time
    return torch.stack((lower, upper), dim=1)


def _interval_cos(lower, upper):
    """Range of ``cos`` over ``[lower, upper]``: endpoints, or +-1 if a peak is inside."""
    two_pi = 2.0 * math.pi
    at_lower, at_upper = torch.cos(lower), torch.cos(upper)
    minimum, maximum = torch.minimum(at_lower, at_upper), torch.maximum(at_lower, at_upper)
    has_maximum = torch.ceil(lower / two_pi) <= torch.floor(upper / two_pi)
    has_minimum = torch.ceil((lower - math.pi) / two_pi) <= torch.floor((upper - math.pi) / two_pi)
    return (torch.where(has_minimum, -torch.ones_like(minimum), minimum),
            torch.where(has_maximum, torch.ones_like(maximum), maximum))


def _interval_sin(lower, upper):
    return _interval_cos(lower - 0.5 * math.pi, upper - 0.5 * math.pi)


def _interval_product(first_lo, first_hi, second_lo, second_hi):
    corners = torch.stack((first_lo * second_lo, first_lo * second_hi, first_hi * second_lo, first_hi * second_hi))
    return corners.amin(dim=0), corners.amax(dim=0)


def _scale_interval(lower, upper, scale):
    first, second = lower * scale, upper * scale
    return torch.minimum(first, second), torch.maximum(first, second)


def _acceleration_integral_phase(turn):
    """``R, delta`` with ``R exp(i delta) = int_0^1 u exp(i turn u) du``.

    Sinc and a series near zero avoid the removable singularity at ``turn = 0``.
    """
    half_sinc = torch.sinc(turn / (2.0 * math.pi))
    real = torch.sinc(turn / math.pi) - 0.5 * half_sinc.square()
    safe_turn = torch.where(turn == 0.0, torch.ones_like(turn), turn)
    direct_imaginary = (torch.sin(turn) - turn * torch.cos(turn)) / safe_turn.square()
    turn2 = turn.square()
    series_imaginary = turn * (1.0 / 3.0 - turn2 / 30.0 + turn2.square() / 840.0 - turn2.square() * turn2 / 45_360.0)
    imaginary = torch.where(turn.abs() < 0.1, series_imaginary, direct_imaginary)
    return torch.hypot(real, imaginary), torch.atan2(imaginary, real)
