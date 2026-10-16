"""STLGym fine-tuning of the NCubeV ReLU PPO controller (Table 2, row STLGym; Figure 5).

Run from the repository root::

    python baselines/stlgym/finetune.py --seed 0 --output results/stlgym/seed0

STLGym (Hamilton et al., 2022) replaces the training reward by the robustness
of an STL specification. For ACC, ``G[0,420] (p > 0)`` evaluated on the sampled
states gives at step ``t`` the reward ``min_{k <= t} p_k / P`` (``P = 100``).
Episodes end on a collision (``p <= 0``), above 200 m (a safe outcome) or
after 420 steps.

- Controller: ``checkpoints/acc/ncubev_ppo.zip``, the PPO controller shipped
  with NCubeV (Teuber et al., 2024; ``ppo_acc_bigger_200000_steps.zip``, SB3
  1.4.0, ReLU MLP 4x64). Its actor and Gaussian ``log_std`` are copied into a
  fresh PPO model; the critic starts new, as the reward changed.
- Training: 200,000 steps (8 environments x 256 steps, rounded up to whole
  rollouts), learning rate 3e-4, gamma 0.99, batch 64, 10 epochs, target KL
  0.03, one CPU thread. Resets follow ``ACC``'s training distribution; 10%
  are replaced by a closing state just inside the braking boundary,
  ``p = v^2 / 200 + 1e-4`` with ``v <= 0``.
- Checks at 0%, 10%, ..., 100% of the budget (after the PPO update reaching
  each): 10,000 fresh closing starts in the braking invariant ``0 < p <= P,
  v <= 0, p >= v^2 / 200 + 1e-4`` (90% area-uniform, 10% on its boundary) and 1,000
  fixed starts from ``X_0`` (the boundary-160 initial set cut at ``P``, 10% at
  its frontier), with the ACC task reward. The starts do not depend on the seed: all
  seeds are checked on the same states. Point trajectories, no tube checks:
  empirical safety, not a certificate. (The prototype drew both velocity signs,
  following NCubeV's ``p >= v^2 / 200``; for ``v > 0`` those starts are moving
  apart and not hard, so here they are closing states only.)

``report.json``: ``seconds`` (the whole run, checks included), the checks and
the action change against the original controller on the closing box (as
``scripts/acc.py``).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from acc import closing_box  # noqa: E402
from reacheco.environments.acc import ACC, ACCConfig  # noqa: E402
from reacheco.spec import load_problem  # noqa: E402

CONTROLLER = ROOT / "checkpoints/acc/ncubev_ppo.zip"
MARGIN = 1e-4


def analytic_starts(count, rng, position_max, *, boundary=False, margin=MARGIN):
    """Closing states (``v <= 0``) of the braking invariant with a margin, ``margin <= p <= P,
    p >= v^2 / (2 B) + margin``: area-uniform, or velocity-uniform on its boundary (a start that
    only (nearly) full braking saves); float32, nudged inward where rounding leaves it."""
    reach = 2 * ACCConfig().max_braking_acceleration
    if boundary:
        velocity = -rng.random(count) * np.sqrt(reach * (position_max - margin))
        position = velocity ** 2 / reach + margin
    else:
        # The slice -sqrt(2 B (p - margin)) <= v <= 0 widens as sqrt(p - margin): area-uniform
        # positions have the distribution ((p - margin) / (P - margin))^(3/2), inverted here.
        position = margin + (position_max - margin) * rng.random(count) ** (2 / 3)
        velocity = -rng.random(count) * np.sqrt(reach * (position - margin))
    states = np.column_stack((position, velocity)).astype(np.float32)
    for _ in range(8):
        bad = np.flatnonzero(states[:, 0].astype(float) < states[:, 1].astype(float) ** 2 / reach + margin)
        if not len(bad):
            return states
        raised = np.nextafter(states[bad, 0], np.float32(np.inf))
        fits = raised <= position_max
        states[bad[fits], 0] = raised[fits]
        states[bad[~fits], 1] = np.nextafter(states[bad[~fits], 1], np.float32(0))
    raise RuntimeError("a sample could not be represented inside the invariant")


def check_starts(count, seed, position_max, boundary_fraction=0.1):
    """``count`` invariant starts, the last ``boundary_fraction`` of them on its boundary."""
    rng = np.random.default_rng(seed)
    boundary = int(round(count * boundary_fraction))
    return np.concatenate((analytic_starts(count - boundary, rng, position_max),
                           analytic_starts(boundary, rng, position_max, boundary=True)))


def x0_starts(position_max, count, seed, boundary_probability=0.1):
    """Starts from ``X_0`` as ACC's evaluation mode draws them (area-uniform, or at the frontier)."""
    problem = load_problem(ROOT / f"problems/acc/boundary160_collision_{int(position_max)}.json")
    env = ACC(mode="evaluate", initial_boxes=problem.spec.initial_boxes, boundary_probability=boundary_probability)
    starts = [env.reset(seed=seed)[0]] + [env.reset()[0] for _ in range(count - 1)]
    return np.stack(starts)


class RobustnessEnv(gym.Wrapper):
    """ACC with the STL robustness reward; 10% of resets just inside the analytic braking boundary."""

    def __init__(self, horizon, position_max, boundary_probability):
        super().__init__(ACC(ACCConfig(max_steps=horizon)))
        self.horizon, self.position_max = horizon, position_max
        self.boundary_probability = boundary_probability

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self.np_random.random() < self.boundary_probability:
            observation = analytic_starts(1, self.np_random, self.position_max, boundary=True)[0]
            self.env.state = observation.copy()
        self.minimum = float(observation[0])
        return observation, info

    def step(self, action):
        state, _, _, _, info = self.env.step(action)
        self.minimum = min(self.minimum, float(state[0]))
        terminated = bool(info["crash"] or info["too_far"] or self.env.steps >= self.horizon)
        return state, self.minimum / self.position_max, terminated, False, info


def evaluate(model, starts, horizon, config=ACCConfig()):
    """Deterministic point trajectories from ``starts``: ACC task return and collisions (above 200 m is safe)."""
    state = np.asarray(starts, dtype=np.float32).copy()
    returns, unsafe = np.zeros(len(state)), state[:, 0] <= 0
    done = unsafe.copy()
    for _ in range(horizon):
        ids = np.flatnonzero(~done)
        if not len(ids):
            break
        action = np.clip(np.asarray(model.predict(state[ids], deterministic=True)[0]).reshape(-1), -1.0, 1.0)
        old = state[ids].astype(float)
        acceleration = action * np.where(action >= 0, config.max_braking_acceleration, config.max_acceleration)
        position = old[:, 0] + config.dt * old[:, 1] + 0.5 * config.dt ** 2 * acceleration
        velocity = old[:, 1] + config.dt * acceleration
        crashed, above = position <= 0, position > config.max_distance
        returns[ids] += np.where(crashed, -2000.0, np.where(above, -800.0, 10.0 * (1.0 - np.abs(action))))
        state[ids] = np.stack((position, velocity), 1).astype(np.float32)
        unsafe[ids], done[ids] = crashed, crashed | above
    return {"episodes": len(state), "unsafe": int(unsafe.sum()), "safe_percent": 100.0 * float((~unsafe).mean()),
            "reward_mean": float(returns.mean()), "reward_std": float(returns.std())}


def make_model(env, seed, arguments):
    """A fresh PPO model whose actor (ReLU layers, action head, ``log_std``) is the NCubeV controller's.

    The archive's tensors are read directly: its pickled Gym spaces belong to SB3 1.4.0.
    """
    with zipfile.ZipFile(CONTROLLER) as archive:
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=False)
    widths = [len(state[f"mlp_extractor.policy_net.{i}.weight"]) for i in range(0, 64, 2)
              if f"mlp_extractor.policy_net.{i}.weight" in state]
    model = PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=0, learning_rate=arguments.learning_rate,
                n_steps=arguments.n_steps, batch_size=arguments.batch_size, n_epochs=arguments.n_epochs,
                gamma=arguments.gamma, target_kl=0.03,
                policy_kwargs={"net_arch": dict(pi=widths, vf=widths), "activation_fn": nn.ReLU, "ortho_init": False})
    actor = {key: value for key, value in state.items()
             if key.startswith(("mlp_extractor.policy_net.", "action_net.")) or key == "log_std"}
    target = model.policy.state_dict()
    if set(actor) != {key for key in target if key.startswith(("mlp_extractor.policy_net.", "action_net."))
                      or key == "log_std"}:
        raise ValueError("the controller's actor does not match the PPO actor")
    target.update(actor)
    model.policy.load_state_dict(target, strict=True)
    return model, actor


def actor_action(actor, states):
    """The original actor's mean action, from its tensors (ReLU layers, then the action head)."""
    x = torch.as_tensor(states, dtype=torch.float32)
    for index in range(0, 64, 2):
        if f"mlp_extractor.policy_net.{index}.weight" not in actor:
            break
        x = torch.relu(x @ actor[f"mlp_extractor.policy_net.{index}.weight"].T
                       + actor[f"mlp_extractor.policy_net.{index}.bias"])
    return x @ actor["action_net.weight"].T + actor["action_net.bias"]


class Checks(BaseCallback):
    """The checks after the PPO updates that reach 10%, 20%, ... of the budget, and at the end."""

    def __init__(self, arguments, x0, started):
        super().__init__()
        self.arguments, self.x0, self.started = arguments, x0, started
        self.decile, self.history = 1, []

    def _on_step(self):
        return True

    def check(self, model, decile):
        starts = check_starts(self.arguments.check_samples, 1000 + decile,
                              self.arguments.position_max)
        entry = {"percent": 10 * decile, "step": int(model.num_timesteps), "seconds": time.monotonic() - self.started,
                 "invariant": evaluate(model, starts, self.arguments.horizon),
                 "x0": evaluate(model, self.x0, self.arguments.horizon)}
        self.history.append(entry)
        print(f"{10 * decile}%: step {entry['step']}, invariant unsafe {entry['invariant']['unsafe']}, "
              f"X0 safe {entry['x0']['safe_percent']:.1f}%, X0 reward {entry['x0']['reward_mean']:.0f}", flush=True)

    def _on_rollout_start(self):        # after the preceding rollout's PPO update
        if self.decile < 10 and self.num_timesteps >= math.ceil(self.arguments.total_timesteps * self.decile / 10):
            self.check(self.model, self.decile)
            self.decile += 1

    def _on_training_end(self):
        self.check(self.model, 10)


def finetune(arguments):
    torch.set_num_threads(1)
    started = time.monotonic()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(RobustnessEnv(arguments.horizon, arguments.position_max, 0.1))
                       for _ in range(arguments.n_envs)])
    model, actor = make_model(env, arguments.seed, arguments)
    probe = check_starts(1000, arguments.seed, arguments.position_max)
    parity = float(np.abs(model.predict(probe, deterministic=True)[0]
                          - actor_action(actor, probe).clamp(-1, 1).numpy()).max())
    if parity > 1e-6:
        raise RuntimeError(f"the copied actor differs from the controller by {parity}")
    x0 = x0_starts(arguments.position_max, arguments.x0_episodes, 50_000)
    checks = Checks(arguments, x0, started)
    checks.check(model, 0)
    model.learn(total_timesteps=arguments.total_timesteps, callback=checks)
    seconds = time.monotonic() - started
    model.save(output / "controller.zip")
    states = closing_box(arguments.position_max).numpy()
    original = actor_action(actor, states).clamp(-1, 1).numpy().reshape(-1)
    tuned = np.clip(np.asarray(model.predict(states, deterministic=True)[0]).reshape(-1), -1, 1)
    delta = np.abs(tuned.astype(np.float64) - original.astype(np.float64))
    report = {"status": "trained", "seconds": seconds, "steps": int(model.num_timesteps),
              "change_percent": 100.0 * float((delta > 1e-6).mean()), "mean_abs_delta": float(delta.mean()),
              "checks": checks.history, "arguments": {k: str(v) for k, v in vars(arguments).items()}}
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    final = checks.history[-1]
    print(f"{seconds:.0f} s; invariant unsafe {final['invariant']['unsafe']}/{final['invariant']['episodes']}; "
          f"change {report['change_percent']:.1f}%, mean |delta| {report['mean_abs_delta']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-max", type=float, default=100.0)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--check-samples", type=int, default=10_000)
    parser.add_argument("--x0-episodes", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=420)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    finetune(parser.parse_args())


if __name__ == "__main__":
    main()
