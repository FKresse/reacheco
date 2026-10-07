"""Interface every controlled system offers to verification and to rollouts."""
from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym


class System(gym.Env, ABC):
    """A sampled-data system with zero-order hold, as a Gymnasium environment.

    Rollouts use the usual ``reset``/``step``. Verification uses the interval
    propagators below, on boxes ``(N, 2, state_dim)`` and normalized actions
    ``(N, action_dim)`` held constant for one control period.
    """

    config_class: type
    state_dim: int
    action_dim: int

    @abstractmethod
    def post(self, boxes, actions):
        """Boxes ``(N, 2, state_dim)`` containing every successor after one period."""

    @abstractmethod
    def tube(self, boxes, actions, segments):
        """Boxes ``(segments, N, 2, state_dim)`` enclosing every trajectory during the period.

        The period is split into ``segments`` equal intervals; segment ``k``
        contains all states visited during interval ``k`` from any start in the
        box. Every segment is computed from the start box directly.
        """
