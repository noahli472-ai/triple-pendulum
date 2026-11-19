"""
Graphical user interface for the n-link pendulum simulation (tkinter + matplotlib).

Python port of ``multipendgui.m`` - run this for the full simulation:

    python -m multipendulum

Original MATLAB code:
    Copyright (c) 2016, Tillmann Stübler. All rights reserved.
    Modifications: Lucas Zieger, Noah Lindner
Python port: Noah Lindner, 2026
License: BSD 2-Clause (see LICENSE)
"""

from __future__ import annotations

import time
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .animation import PendulumArtist, interpolate_angles
from .dynamics import SOLVERS, SimulationResult, simulate
from .presets import random_setup

# Column layout of the parameter table
COLUMNS = ("l", "m", "phi(0)", "dphi/dt(0)", "Damping", "Moment")
L, M, PHI0, DPHI0, DAMP, MOMENT = range(6)
DEFAULT_ROW = np.array([1.0, 1.0, np.pi / 2, 0.0, 0.0, 0.0])


def _fmt(value: float | None) -> str:
    """Compact number formatting (similar to MATLAB num2str)."""
    return "" if value is None else f"{value:.6g}"


def _parse_optional(text: str) -> float | None:
    """Empty string -> None (solver default), otherwise float."""
    text = text.strip()
    return None if text == "" else float(text)


class MultipendGUI(tk.Tk):
    """Main window: animation panel, parameter table and solver settings."""

    def __init__(self) -> None:
        super().__init__()
        self.title("n-link pendulum")
        self.geometry("1400x850")
        self.minsize(1000, 650)

        # ---- Initial state ------------------------------------------------
        self.tspan = 10.0
        self.g = 10.0
        self.param = np.tile(DEFAULT_ROW, (3, 1))
        self.solver = "DOP853"
        self.rtol: float | None = 1e-8
        self.atol: float | None = 1e-10
        self.max_step: float | None = None
        self.result: SimulationResult | None = None
        self._anim_job = None
        self._table_entries: list[list[ttk.Entry]] = []

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.randomize()
        self.update_gui()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Animation panel
        anim_panel = ttk.LabelFrame(self, text="Animation")
        anim_panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=(6, 3))
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=anim_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Parameter panel
        param_panel = ttk.LabelFrame(self, text="Parameters and initial conditions")
        param_panel.grid(row=1, column=0, sticky="nsew", padx=(6, 3), pady=(3, 6))
        param_panel.columnconfigure(1, weight=1)

        left = ttk.Frame(param_panel)
        left.grid(row=0, column=0, sticky="nw", padx=6, pady=6)
        self.tspan_var = tk.StringVar()
        self.nseg_var = tk.StringVar()
        self.grav_var = tk.StringVar()
        for row, (label, var, cb) in enumerate(
            (
                ("Time span [s]:", self.tspan_var, self._on_tspan),
                ("Number of segments:", self.nseg_var, self._on_nseg),
                ("Gravitational acceleration [m/s²]:", self.grav_var, self._on_grav),
            )
        ):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(left, textvariable=var, width=10)
            entry.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            self._bind_commit(entry, cb)

        buttons = ttk.Frame(left)
        buttons.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Random configuration", command=self._on_random).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(buttons, text="Set m = l = 1", command=self._on_all_one).pack(side="left")

        # Scrollable parameter table
        table_outer = ttk.Frame(param_panel)
        table_outer.grid(row=0, column=1, sticky="nsw", padx=6, pady=6)
        table_outer.rowconfigure(0, weight=1)
        table_outer.columnconfigure(0, weight=1)
        self.table_canvas = tk.Canvas(table_outer, height=170, highlightthickness=0)
        scroll = ttk.Scrollbar(table_outer, orient="vertical", command=self.table_canvas.yview)
        self.table_canvas.configure(yscrollcommand=scroll.set)
        self.table_canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.table = ttk.Frame(self.table_canvas)
        self.table_canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.table.bind("<Configure>", self._on_table_resize)

        # Settings panel
        solver_panel = ttk.LabelFrame(self, text="Settings")
        solver_panel.grid(row=1, column=1, sticky="nsew", padx=(3, 6), pady=(3, 6))

        self.show_masses_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            solver_panel,
            text="Show masses as points",
            variable=self.show_masses_var,
            command=self.update_gui,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))

        ttk.Label(solver_panel, text="Solver:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.solver_var = tk.StringVar(value=self.solver)
        solver_box = ttk.Combobox(
            solver_panel, textvariable=self.solver_var, values=SOLVERS, state="readonly", width=12
        )
        solver_box.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        solver_box.bind("<<ComboboxSelected>>", self._on_solver)

        self.rtol_var = tk.StringVar()
        self.atol_var = tk.StringVar()
        self.maxstep_var = tk.StringVar()
        for row, (label, var) in enumerate(
            (
                ("Relative tolerance:", self.rtol_var),
                ("Absolute tolerance:", self.atol_var),
                ("MaxStep:", self.maxstep_var),
            ),
            start=2,
        ):
            ttk.Label(solver_panel, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
            entry = ttk.Entry(solver_panel, textvariable=var, width=14)
            entry.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            self._bind_commit(entry, self._on_solver_options)

        actions = ttk.Frame(solver_panel)
        actions.grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 2))
        self.solve_btn = ttk.Button(actions, text="Solve", command=self._on_solve)
        self.anim_btn = ttk.Button(actions, text="Animate", command=self._on_animate)
        self.plot_btn = ttk.Button(actions, text="Plot", command=self._on_plot)
        for btn in (self.solve_btn, self.anim_btn, self.plot_btn):
            btn.pack(side="left", padx=(0, 6))

        self.status_var = tk.StringVar(value="")
        ttk.Label(solver_panel, textvariable=self.status_var, foreground="#555").grid(
            row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 6)
        )

    def _on_table_resize(self, _event: tk.Event | None = None):
        """Fit the canvas width to the table and update the scroll region."""
        self.table_canvas.configure(
            scrollregion=self.table_canvas.bbox("all"), width=self.table.winfo_reqwidth()
        )

    @staticmethod
    def _bind_commit(entry: ttk.Entry, callback: Callable[[], None]) -> None:
        """Fire callback on <Return> and when the entry loses focus."""
        entry.bind("<Return>", lambda _e: callback())
        entry.bind("<FocusOut>", lambda _e: callback())

    # ------------------------------------------------------------------ #
    # Parameter table
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """Number of pendulum segments."""
        return self.param.shape[0]

    def _rebuild_table(self):
        for child in self.table.winfo_children():
            child.destroy()
        self._table_entries = []

        ttk.Label(self.table, text="#").grid(row=0, column=0, padx=3)
        for j, name in enumerate(COLUMNS):
            ttk.Label(self.table, text=name).grid(row=0, column=j + 1, padx=3)

        for i in range(self.n):
            ttk.Label(self.table, text=str(i + 1)).grid(row=i + 1, column=0, padx=3)
            row_entries = []
            for j in range(len(COLUMNS)):
                entry = ttk.Entry(self.table, width=9)
                entry.grid(row=i + 1, column=j + 1, padx=2, pady=1)
                self._bind_commit(entry, self._on_table_edit)
                row_entries.append(entry)
            self._table_entries.append(row_entries)

    def _write_table(self):
        if len(self._table_entries) != self.n:
            self._rebuild_table()
        for i, row in enumerate(self._table_entries):
            for j, entry in enumerate(row):
                entry.delete(0, "end")
                entry.insert(0, _fmt(self.param[i, j]))

    # ------------------------------------------------------------------ #
    # GUI update (port of updategui)
    # ------------------------------------------------------------------ #
    def update_gui(self) -> None:
        """Write the current state into all widgets and redraw the initial configuration."""
        self.stop_animation()

        self.tspan_var.set(_fmt(self.tspan))
        self.nseg_var.set(str(self.n))
        self.grav_var.set(_fmt(self.g))
        self._write_table()
        self.rtol_var.set(_fmt(self.rtol))
        self.atol_var.set(_fmt(self.atol))
        self.maxstep_var.set(_fmt(self.max_step))
        self.solver_var.set(self.solver)

        # Static drawing of the initial configuration
        artist = PendulumArtist(self.ax, self.param[:, L], self._masses_or_none())
        artist.update(self.param[:, PHI0])
        self.canvas.draw_idle()

        # Animate/Plot only available after a simulation
        state = "disabled" if self.result is None else "normal"
        self.anim_btn.configure(state=state)
        self.plot_btn.configure(state=state)

    def _masses_or_none(self):
        return self.param[:, M] if self.show_masses_var.get() else None

    def _invalidate(self):
        self.result = None
        self.status_var.set("")

    # ------------------------------------------------------------------ #
    # Callbacks (port of callback)
    # ------------------------------------------------------------------ #
    def _on_tspan(self):
        try:
            value = float(self.tspan_var.get())
            if value <= 0:
                raise ValueError
        except ValueError:
            pass
        else:
            if value != self.tspan:
                self.tspan = value
                self._invalidate()
        self.update_gui()

    def _on_nseg(self):
        try:
            n_new = int(float(self.nseg_var.get()))
            if n_new < 1:
                raise ValueError
        except ValueError:
            n_new = self.n
        if n_new != self.n:
            if n_new < self.n:
                # Remove surplus rows
                self.param = self.param[:n_new]
            else:
                # Append new rows with default parameters
                extra = np.tile(DEFAULT_ROW, (n_new - self.n, 1))
                self.param = np.vstack((self.param, extra))
            self._invalidate()
        self.update_gui()

    def _on_grav(self):
        try:
            value = float(self.grav_var.get())
        except ValueError:
            pass
        else:
            if value != self.g:
                self.g = value
                self._invalidate()
        self.update_gui()

    def _on_table_edit(self):
        new = self.param.copy()
        for i, row in enumerate(self._table_entries):
            for j, entry in enumerate(row):
                try:
                    new[i, j] = float(entry.get())
                except ValueError:
                    pass  # keep old value
        if np.any(new[:, L] <= 0) or np.any(new[:, M] <= 0):
            messagebox.showwarning("Invalid input", "Lengths and masses must be positive.")
            new = self.param
        if not np.array_equal(new, self.param):
            self.param = new
            self._invalidate()
        self.update_gui()

    def _on_solver(self, _event: tk.Event | None = None):
        self.solver = self.solver_var.get()
        self.update_gui()

    def _on_solver_options(self):
        for name, var in (
            ("rtol", self.rtol_var),
            ("atol", self.atol_var),
            ("max_step", self.maxstep_var),
        ):
            try:
                setattr(self, name, _parse_optional(var.get()))
            except ValueError:
                pass  # keep old value
        self.update_gui()

    def _on_random(self):
        self.randomize()
        self.update_gui()

    def _on_all_one(self):
        self.param[:, [L, M]] = 1.0
        self._invalidate()
        self.update_gui()

    def randomize(self) -> None:
        """Replace the parameters by a random configuration."""
        self.param = random_setup()
        self._invalidate()

    # ------------------------------------------------------------------ #
    # Solve / animate / plot
    # ------------------------------------------------------------------ #
    def _on_solve(self):
        self.status_var.set("Solving ...")
        self.config(cursor="watch")
        self.update_idletasks()
        p = self.param
        try:
            t_start = time.perf_counter()
            self.result = simulate(
                t_span=(0.0, self.tspan),
                m=p[:, M],
                l=p[:, L],
                K=p[:, DAMP],
                g=self.g,
                phi0=p[:, PHI0],
                dphi0=p[:, DPHI0],
                M_ext=p[:, MOMENT],
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            elapsed = time.perf_counter() - t_start
        except Exception as exc:  # show solver problems to the user
            self.result = None
            self.status_var.set("")
            messagebox.showerror("Simulation failed", str(exc))
        else:
            E = self.result.E
            scale = max(np.abs(self.result.T).max() + np.abs(self.result.V).max(), 1e-12)
            drift = (E.max() - E.min()) / scale
            self.status_var.set(f"Solved in {elapsed:.2f} s - energy variation {drift:.1e} (rel.)")
        finally:
            self.config(cursor="")
        self.update_gui()

    def _on_animate(self):
        if self.result is None:
            return
        self.stop_animation()
        res = self.result
        artist = PendulumArtist(self.ax, self.param[:, L], self._masses_or_none())
        t = res.t - res.t[0]
        start = time.perf_counter()

        def tick():
            t1 = time.perf_counter() - start
            artist.update(interpolate_angles(t, res.phi, min(t1, t[-1])))
            self.canvas.draw_idle()
            if t1 < t[-1]:
                self._anim_job = self.after(15, tick)
            else:
                self._anim_job = None

        tick()

    def stop_animation(self) -> None:
        """Cancel a running animation, if any."""
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _on_plot(self):
        if self.result is None:
            return
        res = self.result
        win = tk.Toplevel(self)
        win.title("Simulation results n-link pendulum")
        win.geometry("900x700")

        fig = Figure(figsize=(9, 7), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(res.t, res.T, label=r"$E_{kin}$")
        ax1.plot(res.t, res.V, label=r"$E_{pot}$")
        ax1.plot(res.t, res.E, label=r"$E_{total}$")
        ax1.set_title("Energy")
        ax1.set_ylabel("E [J]")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(2, 1, 2)
        for k in range(res.phi.shape[1]):
            ax2.plot(res.t, res.phi[:, k], label=rf"$\varphi_{{{k + 1}}}$")
        ax2.set_title("Angles")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel(r"$\varphi$ [rad]")
        ax2.grid(True)
        ax2.legend(loc="upper right")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        NavigationToolbar2Tk(canvas, win).pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _on_close(self):
        self.stop_animation()
        self.destroy()


def main() -> None:
    """Start the GUI."""
    MultipendGUI().mainloop()


if __name__ == "__main__":
    main()
