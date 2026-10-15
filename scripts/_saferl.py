"""SafeRL's 2D docking environment, in which the Docking2D controllers were trained.

Needs SafeRL (https://github.com/act3-ace/SafeRL, tested at 97e19caf) and the
old ray and gym it imports, hence Python 3.9::

    git clone https://github.com/act3-ace/SafeRL
    pip install --no-deps -e SafeRL     # its requirements.txt pins far more than the environment needs
    pip install "ray[rllib]==1.5.2" "gym==0.22.0" "numpy<2" "protobuf<3.21" scipy pyyaml jsonlines

``ENV_CONFIG`` is the DWCs' training configuration
(``config_aero_default_2d_unnormalized``): RK45 integration, the docking
region of radius 0.35, the velocity limit ``0.2 + 2 n d`` and the reward. Only
the deputy's random start is replaced, since every episode here starts from a
given state.
"""
from __future__ import annotations

import copy

import numpy as np


def env_config():
    from saferl.aerospace.models.cwhspacecraft.platforms.cwh import CWHSpacecraft2d
    from saferl.aerospace.tasks.docking.processors import (
        AccumulatorStatusProcessor, DockingDistanceStatusProcessor, DockingThrustDeltaVStatusProcessor,
        DockingVelocityLimit, DockingVelocityLimitViolation, FailureRewardProcessor, FailureStatusProcessor,
        InDockingStatusProcessor, RelativeVelocityConstraint, SuccessRewardProcessor, SuccessStatusProcessor)
    from saferl.environment.models.geometry import RelativeCircle
    from saferl.environment.models.platforms import AgentController
    from saferl.environment.tasks.initializers import RandBoundsInitializer
    from saferl.environment.tasks.processor.observation import AttributeObservationProcessor
    from saferl.environment.tasks.processor.reward import (
        DistanceExponentialChangeRewardProcessor, ProportionalRewardProcessor)
    # we inject states not here, so we have more control over the initalization.
    at_rest = {"initializer": RandBoundsInitializer, "x": 0, "x_dot": 0, "y": 0, "y_dot": 0}
    # uses tnah post-processing, we have this in the DWCs output post-processing.
    thrust = [{"name": name, "space": "continuous", "bounds": [-1, 1]} for name in ("thrust_x", "thrust_y")]
    # modified from normal, typically has 100 as obs norm for x and y and 0.5 for x_dot and y_dot, we set all to 1
    # as we handle scaling in the dwc.
    observe = [{"name": f"obs_{attr}", "class": AttributeObservationProcessor,
                "config": {"target": "deputy", "attr": attr, "observation_space_shape": 1, "normalization": 1}}
               for attr in ("x", "y", "x_dot", "y_dot")]
    return {
        "agent": "deputy",
        "step_size": 1,
        "env_objs": [
            {"name": "chief", "class": CWHSpacecraft2d, "config": {"init": at_rest}},
            {"name": "deputy", "class": CWHSpacecraft2d,
             "config": {"integration_method": "RK45", "controller": {"class": AgentController, "actuators": thrust},
                        "init": at_rest}},
            # radius is originially 0.5, we set it to 0.35 as in Mandal et. al. (2024)
            {"name": "docking_region", "class": RelativeCircle,
             "config": {"ref": "chief", "x_offset": 0, "y_offset": 0, "radius": 0.35,
                        "init": {"initializer": RandBoundsInitializer}}},
        ],
        "status": [
            {"name": "in_docking", "class": InDockingStatusProcessor,
             "config": {"deputy": "deputy", "docking_region": "docking_region"}},
            {"name": "docking_distance", "class": DockingDistanceStatusProcessor,
             "config": {"deputy": "deputy", "docking_region": "docking_region"}},
            {"name": "delta_v", "class": DockingThrustDeltaVStatusProcessor, "config": {"target": "deputy"}},
            {"name": "custom_metrics.delta_v_total", "class": AccumulatorStatusProcessor,
             "config": {"status": "delta_v"}},
            {"name": "max_vel_limit", "class": DockingVelocityLimit,
             "config": {"target": "deputy", "dist_status": "docking_distance", "vel_threshold": 0.2,
                        "threshold_dist": 0.5, "slope": 2}},
            {"name": "max_vel_violation", "class": DockingVelocityLimitViolation,
             "config": {"target": "deputy", "ref": "chief", "vel_limit_status": "max_vel_limit"}},
            {"name": "max_vel_constraint", "class": RelativeVelocityConstraint,
             "config": {"target": "deputy", "ref": "chief", "vel_limit_status": "max_vel_limit"}},
            {"name": "failure", "class": FailureStatusProcessor,
             "config": {"docking_distance": "docking_distance", "max_goal_distance": 40000, "timeout": 5000,
                        "in_docking_status": "in_docking", "max_vel_constraint_status": "max_vel_constraint"}},
            {"name": "success", "class": SuccessStatusProcessor,
             "config": {"in_docking_status": "in_docking", "max_vel_constraint_status": "max_vel_constraint"}},
        ],
        "observation": observe,
        "reward": [
            {"name": "dist_change_reward", "class": DistanceExponentialChangeRewardProcessor,
             "config": {"agent": "deputy", "target": "docking_region", "pivot": 100}},
            {"name": "delta_v", "class": ProportionalRewardProcessor,
             "config": {"scale": -0.01, "bias": 0, "proportion_status": "delta_v"}},
            {"name": "max_vel_constraint", "class": ProportionalRewardProcessor,
             "config": {"scale": -0.01, "bias": -0.01, "proportion_status": "max_vel_violation",
                        "cond_status": "max_vel_constraint", "cond_status_invert": True, "lower_bound": -5,
                        "lower_bound_terminal": "failure"}},
            {"name": "failure_reward", "class": FailureRewardProcessor,
             "config": {"failure_status": "failure",
                        "reward": {"crash": -1, "distance": -1, "timeout": -1,
                                   "reward_lower_bound_max_vel_constraint": 0}}},
            {"name": "success_reward", "class": SuccessRewardProcessor,
             # orignial config has 2000, 
             "config": {"reward": 1, "success_status": "success", "timeout": 5000}},
        ],
        "verbose": False,
    }


class Docking:
    """SafeRL's ``DockingEnv`` started from a given state, with the training wrappers' action and reward clipping."""

    def __init__(self):
        from saferl.aerospace.tasks.docking.task import DockingEnv
        self.env = DockingEnv(copy.deepcopy(env_config()))
        self.shapes = [space.shape for space in self.env.action_space.spaces]
        self.splits = np.cumsum([int(np.prod(shape)) for shape in self.shapes])[:-1]

    def reset(self, state):
        env = self.env
        env.reset()
        state = np.asarray(state, dtype=np.float32)
        env.sim_state.env_objs["deputy"].state.reset(vector=state, vector_deep_copy=False)
        # The injected state bypasses reset's bookkeeping (e.g. the reward's previous distance): redo it.
        env.sim_state.status = env.status_manager.reset(env.sim_state)
        env.reward_manager.reset(env.sim_state)
        env.observation_manager.reset(env.sim_state)
        env._generate_obs()
        return state

    def step(self, action):
        """``(observation, reward, done, info)``; ``info["status"]["max_vel_violation"] > 0`` is unsafe."""
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        parts = tuple(part.reshape(shape) for part, shape in zip(np.split(action, self.splits), self.shapes))
        observation, reward, done, info = self.env.step(parts)
        return np.asarray(observation, dtype=np.float32).reshape(-1), float(np.clip(reward, -10, 10)), done, info
