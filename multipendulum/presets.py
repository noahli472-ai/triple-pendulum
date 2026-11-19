"""
Parameter presets for the n-link pendulum (independent of the GUI).

Python port of ``randsetup`` from ``multipendgui.m``.

Original MATLAB code:
    Copyright (c) 2016, Tillmann Stübler. All rights reserved.
    Modifications: Lucas Zieger, Noah Lindner
Python port: Noah Lindner, 2026
License: BSD 2-Clause (see LICENSE)
"""

from __future__ import annotations

import numpy as np


def random_setup(rng: np.random.Generator | None = None) -> np.ndarray:
    """Random pendulum configuration (port of ``randsetup``).

    Returns an (n, 6) parameter matrix with columns
    ``l, m, phi0, dphi0, damping, moment``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(rng.integers(1, 8))  # 1..7 segments

    # Two random modes for lengths and masses
    if rng.integers(2) == 0:
        p = np.array([10.0 / n, 1.0]) * rng.random((n, 2))
        p[:, 0] = p.max(axis=1)
    else:
        exponents = (rng.integers(1, 3, size=2) - 1.5) * 2.0  # each +1 or -1
        p = np.arange(1, n + 1, dtype=float)[:, None] ** exponents / n

    # Two random modes for initial angles and angular velocities
    if rng.integers(2) == 0:
        p = np.hstack((p, np.array([1.0, 2.0]) * rng.standard_normal((n, 2)), np.zeros((n, 2))))
    else:
        end = (2 * rng.integers(1, 3) - 1.5) * (rng.random() + 1) * np.pi * 0.25 * (1 + (n > 1))
        phi0 = np.linspace(0.0, end, n)
        p = np.hstack((p, phi0[:, None], np.zeros((n, 3))))
    return p
