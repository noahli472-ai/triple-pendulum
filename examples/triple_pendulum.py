"""
Triple pendulum without GUI: simulate, plot energy/angles and animate.

    python examples/triple_pendulum.py                 # show plots and animation
    python examples/triple_pendulum.py --save demo.gif # additionally export a GIF
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multipendulum import animate, simulate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Triple pendulum demo")
    parser.add_argument("--t-end", type=float, default=15.0, help="simulation time [s]")
    parser.add_argument("--save", metavar="FILE", help="export animation (.gif or .mp4)")
    parser.add_argument("--no-show", action="store_true", help="do not open plot windows")
    args = parser.parse_args()

    m = [1.0, 1.0, 1.0]
    l = [1.0, 1.0, 1.0]
    phi0 = [np.pi / 2, np.pi / 2, np.pi / 2]  # all links horizontal

    res = simulate(t_span=(0, args.t_end), m=m, l=l, g=9.81, phi0=phi0)
    drift = np.max(np.abs(res.E - res.E[0]))
    print(f"{len(res.t)} samples, max. energy deviation: {drift:.2e} J")

    # Energy and angles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(res.t, res.T, label=r"$E_{kin}$")
    ax1.plot(res.t, res.V, label=r"$E_{pot}$")
    ax1.plot(res.t, res.E, label=r"$E_{total}$")
    ax1.set_ylabel("E [J]")
    ax1.set_title("Energy")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    for k in range(res.phi.shape[1]):
        ax2.plot(res.t, res.phi[:, k], label=rf"$\varphi_{{{k + 1}}}$")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel(r"$\varphi$ [rad]")
    ax2.set_title("Angles")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    fig.tight_layout()

    # Animation
    fig_anim, ax = plt.subplots(figsize=(6, 4))
    fig_anim.subplots_adjust(0, 0, 1, 1)
    anim = animate(ax, res.t, res.phi, l, m, fps=30)

    if args.save:
        writer = "pillow" if args.save.lower().endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=30, dpi=80)
        print(f"Saved animation to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
