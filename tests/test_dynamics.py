"""Physical sanity checks for the equations of motion (run with ``pytest``)."""

import numpy as np
import pytest

from multipendulum import simulate
from multipendulum.presets import random_setup


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_energy_is_conserved_without_damping(n):
    rng = np.random.default_rng(n)
    res = simulate(
        t_span=(0, 5),
        m=rng.uniform(0.5, 2.0, n),
        l=rng.uniform(0.5, 1.5, n),
        phi0=rng.uniform(-2, 2, n),
        dphi0=rng.uniform(-1, 1, n),
    )
    scale = np.max(np.abs(res.T)) + np.max(np.abs(res.V))
    assert np.max(np.abs(res.E - res.E[0])) / scale < 1e-6


def test_damping_dissipates_energy():
    res = simulate(t_span=(0, 10), K=[0.3, 0.3, 0.3])
    assert res.E[-1] < res.E[0]
    # total energy must never increase with damping
    assert np.all(np.diff(res.E) < 1e-9)


def test_small_angle_period_of_single_pendulum():
    l, g = 2.0, 9.81
    res = simulate(t_span=(0, 20), m=[1.0], l=[l], g=g, phi0=[0.01])
    phi = res.phi[:, 0]
    # upward zero crossings -> period
    idx = np.where((phi[:-1] < 0) & (phi[1:] >= 0))[0]
    t_cross = res.t[idx] - phi[idx] * (res.t[idx + 1] - res.t[idx]) / (phi[idx + 1] - phi[idx])
    period = np.mean(np.diff(t_cross))
    assert period == pytest.approx(2 * np.pi * np.sqrt(l / g), rel=1e-4)


def test_hanging_pendulum_stays_at_rest():
    res = simulate(t_span=(0, 5), phi0=[0, 0, 0])
    assert np.allclose(res.phi, 0.0)


def test_random_setup_shape():
    for seed in range(20):
        p = random_setup(np.random.default_rng(seed))
        assert p.ndim == 2 and p.shape[1] == 6 and 1 <= p.shape[0] <= 7
        assert np.all(p[:, :2] > 0)  # lengths and masses positive
