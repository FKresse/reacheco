"""Controlled systems and their safety specifications, by problem-file name."""
from reacheco.environments.acc import ACC, ACCSpec
from reacheco.environments.car2d import Car2D, Car2DSpec
from reacheco.environments.docking2d import Docking2D, Docking2DSpec

ENVIRONMENTS = {
    "acc": (ACC, ACCSpec),
    "car2d": (Car2D, Car2DSpec),
    "docking2d": (Docking2D, Docking2DSpec),
}
