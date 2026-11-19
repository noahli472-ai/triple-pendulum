"""
Equations of motion and numerical solution of an n-link pendulum.

Python port of ``multipend.m``.

The equations of motion are derived from the Euler-Lagrange equations for a
planar chain of point masses connected by massless rigid rods. The angles
``phi_i`` are measured from the downward vertical.

Original MATLAB code:
    Copyright (c) 2016, Tillmann Stübler. All rights reserved.
    Modifications: Lucas Zieger, Noah Lindner
Python port: Noah Lindner, 2026
License: BSD 2-Clause (see LICENSE)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

#: Integrators offered by scipy.integrate.solve_ivp.
#: Rough MATLAB equivalents: RK45 ~ ode45, RK23 ~ ode23, BDF ~ ode15s,
#: Radau ~ ode23s/ode23tb, LSODA/DOP853 ~ ode113 (non-stiff, high accuracy).
SOLVERS = ("DOP853", "RK45", "RK23", "LSODA", "Radau", "BDF")


@dataclass
class SimulationResult:
    """Container for the simulation output.

    Attributes
    ----------
    t : (N,) time vector
    phi : (N, n) joint angles
    omega : (N, n) angular velocities
    T : (N,) kinetic energy
    V : (N,) potential energy
    x, y : (N, n) cartesian positions of the masses
    """

    t: np.ndarray
    phi: np.ndarray
    omega: np.ndarray
    T: np.ndarray
    V: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def E(self) -> np.ndarray:
        """Total mechanical energy T + V."""
        return self.T + self.V


def _as_vector(value: ArrayLike, n: int, name: str) -> np.ndarray:
    """Convert a scalar or sequence into a float vector of length n."""
    arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if arr.size == 1:
        arr = np.full(n, arr.item())
    if arr.size != n:
        raise ValueError(f"'{name}' must have length {n}, got {arr.size}")
    return arr


def make_rhs(
    m: ArrayLike, l: ArrayLike, K: ArrayLike = 0.0, g: float = 9.81, M_ext: ArrayLike = 0.0
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the right-hand side f(t, y) of the first-order ODE system.

    The state vector is ``y = [phi_1..phi_n, dphi_1..dphi_n]``.

    Note: as in the original MATLAB code, each equation of motion i is divided
    by l_i. Damping ``K`` and the external moment ``M_ext`` are added to that
    scaled equation, i.e. they act as generalized forces divided by l_i.
    """
    m = np.asarray(m, dtype=float).ravel()
    n = m.size
    l = _as_vector(l, n, "l")
    K = _as_vector(K, n, "K")
    M_ext = _as_vector(M_ext, n, "M_ext")

    # Helper quantities
    # mc[i]   : sum of all masses from joint i to the free end
    # mml[i,j]: l_j * mc[max(i, j)]
    mc = np.cumsum(m[::-1])[::-1]
    idx = np.arange(n)
    mm = np.maximum.outer(idx, idx)
    mml = l[np.newaxis, :] * mc[mm]

    def rhs(_t: float, y: np.ndarray):
        phi = y[:n]
        dphi = y[n:]

        # Angle differences P[i, j] = phi_j - phi_i
        P = phi[np.newaxis, :] - phi[:, np.newaxis]

        # Mass matrix and trigonometric coupling terms
        A = np.cos(P) * mml
        B = np.sin(P) * mml

        # Centrifugal terms, gravity, damping and external moment
        b = B @ dphi**2 - g * np.sin(phi) * mc - K * dphi + M_ext

        # phi'' = A^-1 * b
        d2phi = np.linalg.solve(A, b)
        return np.concatenate((dphi, d2phi))

    return rhs


def energies(
    phi: ArrayLike, omega: ArrayLike, m: ArrayLike, l: ArrayLike, g: float = 9.81
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian positions and kinetic/potential energy of the chain.

    Parameters
    ----------
    phi, omega : (N, n) arrays of angles and angular velocities

    Returns
    -------
    x, y, T, V
    """
    phi = np.atleast_2d(phi)
    omega = np.atleast_2d(omega)
    m = np.asarray(m, dtype=float).ravel()
    l = _as_vector(l, m.size, "l")

    # Positions
    x = np.cumsum(l * np.sin(phi), axis=1)
    y = np.cumsum(-l * np.cos(phi), axis=1)

    # Velocities
    vx = np.cumsum(l * np.cos(phi) * omega, axis=1)
    vy = np.cumsum(l * np.sin(phi) * omega, axis=1)

    # Energies
    T = np.sum(m * (vx**2 + vy**2) / 2.0, axis=1)
    V = np.sum(m * y * g, axis=1)
    return x, y, T, V


def simulate(
    t_span: Sequence[float] = (0.0, 10.0),
    m: Sequence[float] = (1.0, 1.0, 1.0),
    l: Sequence[float] = (1.0, 1.0, 1.0),
    K: ArrayLike = 0.0,
    g: float = 9.81,
    phi0: Sequence[float] | None = None,
    dphi0: Sequence[float] | None = None,
    M_ext: ArrayLike = 0.0,
    method: str = "DOP853",
    rtol: float | None = 1e-8,
    atol: float | None = 1e-10,
    max_step: float | None = None,
    dt_out: float | None = 1.0 / 200.0,
) -> SimulationResult:
    """Simulate an n-link pendulum.

    Parameters
    ----------
    t_span : (t0, t1) start and end time [s]
    m : masses at the joints, length n [kg]
    l : rod lengths, length n [m]
    K : damping factors (scalar or length n)
    g : gravitational acceleration [m/s^2]
    phi0 : initial angles [rad] (default: all pi/2, i.e. horizontal)
    dphi0 : initial angular velocities [rad/s] (default: zero)
    M_ext : constant external moment (scalar or length n)
    method : integrator, one of :data:`SOLVERS`
    rtol, atol, max_step : solver settings (``None`` -> scipy default)
    dt_out : sampling interval of the output. ``None`` returns the
        solver's internal steps (like the original MATLAB code).

    Returns
    -------
    SimulationResult
    """
    m = np.asarray(m, dtype=float).ravel()
    n = m.size
    l = _as_vector(l, n, "l")
    phi0 = np.full(n, np.pi / 2) if phi0 is None else _as_vector(phi0, n, "phi0")
    dphi0 = np.zeros(n) if dphi0 is None else _as_vector(dphi0, n, "dphi0")

    if method not in SOLVERS:
        raise ValueError(f"Unknown solver '{method}'. Choose one of {SOLVERS}.")

    rhs = make_rhs(m, l, K=K, g=g, M_ext=M_ext)
    y0 = np.concatenate((phi0, dphi0))
    t0, t1 = float(t_span[0]), float(t_span[1])

    options = {}
    if rtol is not None:
        options["rtol"] = rtol
    if atol is not None:
        options["atol"] = atol
    if max_step is not None:
        options["max_step"] = max_step

    t_eval = None
    if dt_out is not None:
        n_out = max(int(round((t1 - t0) / dt_out)), 1) + 1
        t_eval = np.linspace(t0, t1, n_out)

    sol = solve_ivp(rhs, (t0, t1), y0, method=method, t_eval=t_eval, **options)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    t = sol.t
    phi = sol.y[:n].T
    omega = sol.y[n:].T
    x, y, T, V = energies(phi, omega, m, l, g)
    return SimulationResult(t=t, phi=phi, omega=omega, T=T, V=V, x=x, y=y)
