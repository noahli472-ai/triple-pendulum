"""n-link (e.g. triple) pendulum: simulation, animation and GUI.

Python port of a MATLAB project.
Original MATLAB code: Copyright (c) 2016, Tillmann Stübler. All rights reserved.
Modifications: Lucas Zieger, Noah Lindner
"""

from .animation import PendulumArtist, animate
from .dynamics import SOLVERS, SimulationResult, energies, make_rhs, simulate

__all__ = [
    "SOLVERS",
    "SimulationResult",
    "energies",
    "make_rhs",
    "simulate",
    "PendulumArtist",
    "animate",
]
__version__ = "1.0.0"
