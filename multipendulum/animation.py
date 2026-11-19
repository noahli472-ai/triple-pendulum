"""
Drawing and animation of the n-link pendulum with matplotlib.

Python port of ``multipendanimation.m``.

Original MATLAB code:
    Copyright (c) 2016, Tillmann Stübler. All rights reserved.
    Modifications: Lucas Zieger, Noah Lindner
Python port: Noah Lindner, 2026
License: BSD 2-Clause (see LICENSE)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike


class PendulumArtist:
    """Draws the pendulum on a matplotlib Axes and updates it in place.

    Parameters
    ----------
    ax : matplotlib Axes to draw on (it is cleared)
    l : rod lengths
    m : masses; ``None`` hides the mass markers
    """

    def __init__(self, ax: Axes, l: Sequence[float], m: Sequence[float] | None = None) -> None:
        self.ax = ax
        self.l = np.asarray(l, dtype=float).ravel()
        self.m = None if m is None else np.asarray(m, dtype=float).ravel()
        self.show_masses = self.m is not None and np.all(np.isfinite(self.m))

        ax.clear()

        # Total length of the chain -> visible region [-sl, sl] x [-sl, 0.1 sl].
        # Data limits + aspect "datalim" keep a 1:1 scale and fill the whole axes.
        sl = float(np.sum(self.l))
        ax.update_datalim([(-sl, -sl), (sl, 0.1 * sl)])
        ax.margins(0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()

        # Ceiling / mounting (drawn wide so it covers the full width;
        # add_artist does not change the data limits)
        ax.add_artist(
            Rectangle(
                (-20 * sl, 0.0), 40 * sl, 20 * sl, facecolor=(0.4, 0.4, 0.4), edgecolor="none"
            )
        )

        # Rods
        (self.rods,) = ax.plot([], [], linewidth=3, color="#0072BD", zorder=2)

        # Mass points, marker size scales with sqrt(m / max(m))
        self.masses = None
        if self.show_masses:
            size = 20.0 * np.sqrt(self.m / np.max(self.m))
            self.masses = ax.scatter(
                np.zeros_like(self.m),
                np.zeros_like(self.m),
                s=size**2,
                c="k",
                edgecolors="b",
                zorder=3,
            )

    @property
    def artists(self) -> tuple:
        """Matplotlib artists that change during the animation."""
        return (self.rods,) if self.masses is None else (self.rods, self.masses)

    def positions(self, phi: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Cartesian joint positions for one set of angles."""
        phi = np.asarray(phi, dtype=float).ravel()
        x = np.cumsum(self.l * np.sin(phi))
        y = -np.cumsum(self.l * np.cos(phi))
        return x, y

    def update(self, phi: ArrayLike) -> tuple:
        """Redraw the pendulum for the angles ``phi``."""
        x, y = self.positions(phi)
        self.rods.set_data(np.r_[0.0, x], np.r_[0.0, y])
        if self.masses is not None:
            self.masses.set_offsets(np.column_stack((x, y)))
        return self.artists


def interpolate_angles(t: ArrayLike, phi: ArrayLike, t_query: float) -> np.ndarray:
    """Linear interpolation of the angle history (like MATLAB interp1)."""
    t = np.asarray(t, dtype=float)
    phi = np.atleast_2d(phi)
    return np.array([np.interp(t_query, t, phi[:, k]) for k in range(phi.shape[1])])


def animate(
    ax: Axes,
    t: ArrayLike,
    phi: ArrayLike,
    l: ArrayLike,
    m: ArrayLike | None = None,
    fps: int = 30,
    speed: float = 1.0,
    show_time: bool = True,
) -> FuncAnimation:
    """Create a matplotlib FuncAnimation of a simulation result.

    Parameters
    ----------
    ax : target Axes
    t, phi : time vector and (N, n) angle matrix, e.g. from ``simulate``
    l, m : rod lengths and masses (``m=None`` hides the masses)
    fps : frames per second of the animation
    speed : playback speed (1.0 = real time)

    Returns
    -------
    FuncAnimation  (keep a reference, otherwise it is garbage collected)
    """
    t = np.asarray(t, dtype=float)
    t = t - t[0]  # start at t = 0
    artist = PendulumArtist(ax, l, m)
    label = ax.text(0.02, 0.03, "", transform=ax.transAxes, va="bottom") if show_time else None

    n_frames = int(np.floor(t[-1] * fps / speed)) + 1
    frame_times = np.minimum(np.arange(n_frames) * speed / fps, t[-1])

    def _update(k: int):
        tk = frame_times[k]
        out = artist.update(interpolate_angles(t, phi, tk))
        if label is not None:
            label.set_text(f"t = {tk:5.2f} s")
            out = (*out, label)
        return out

    artist.update(phi[0])
    return FuncAnimation(
        ax.figure, _update, frames=n_frames, interval=1000.0 / fps, blit=True, repeat=False
    )
