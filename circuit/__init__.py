"""Compile a repaired controller's patch into a circuit and measure it (see ``README.md``).

A repair is a set of grid cells, each carrying a signed delta on the
controller's popcount; it is realized as a guard decoder over the thermometer
bits feeding a small indexed memory.
"""
from .patch import ClusterPatch, cell_clusters, compile_patch, grid_code, verify_patch

__all__ = ["ClusterPatch", "cell_clusters", "compile_patch", "grid_code", "verify_patch"]
