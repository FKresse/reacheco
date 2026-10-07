"""Controlled systems and their safety specifications, by problem-file name."""
from reacheco.environments.acc import ACC, ACCSpec

ENVIRONMENTS = {
    "acc": (ACC, ACCSpec),
}
