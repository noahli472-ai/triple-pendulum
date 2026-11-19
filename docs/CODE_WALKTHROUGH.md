# Code Walkthrough – n-Link Pendulum, Explained Step by Step

This document explains the whole project from zero. You do not need to know Lagrangian mechanics or numerical
methods to follow it. Every section first says **what** happens in plain words, then **where** it happens in the
code, then (optionally) **why** it works.

---

## 0. The big picture in three sentences

1. We describe a chain of pendulums with one number per link: its angle.
2. Physics (the Euler-Lagrange equations) tells us how fast those angles *accelerate* for any given position and
   speed.
3. A numerical solver takes tiny time steps, using those accelerations to move the pendulum forward in time; the
   result is drawn as an animation and as energy/angle plots.

```
you choose masses, lengths, start angles
        │
        ▼
dynamics.py   ── equations of motion ──► solver (scipy) ──► angles over time
        │                                                       │
        ▼                                                       ▼
energies (check that physics is respected)          animation.py (draw it)
                                                    gui.py (buttons, table, plots)
```

---

## 1. The physical setup

Picture a ceiling. A rod hangs from it, at the end of the rod is a ball (a point mass). From that ball hangs the next
rod with the next ball, and so on. With three rods you have a **triple pendulum**.

Simplifying assumptions (good to say out loud in an interview):

| Assumption | What it means |
|---|---|
| Rods are massless and rigid | All mass sits in the balls; rods never bend or stretch |
| Point masses | Balls have no size, so they cannot spin by themselves |
| Planar motion | Everything moves in one flat plane (2D) |
| Optional damping and moment | A friction-like term and a constant external torque per joint |

**Coordinates.** Each link `i` has one angle `φ_i`, measured from the line pointing straight **down**.
`φ = 0` means hanging down, `φ = π/2` (90°) means horizontal. Three links → three numbers describe the whole
pendulum. These are called *generalized coordinates*.

From the angles we get the ball positions by adding up the rods one after another:

$$x_k = \sum_{j \le k} l_j \sin\varphi_j, \qquad y_k = -\sum_{j \le k} l_j \cos\varphi_j$$

In words: ball 3 sits at "rod 1 + rod 2 + rod 3". The minus sign makes `y` negative below the ceiling.

---

## 2. Why the motion is hard (and why we need a computer)

A single pendulum swings back and forth nicely. A double or triple pendulum is **chaotic**: tiny changes in the
start angles lead to completely different motion after a few seconds. There is no closed-form solution, so we
compute the motion step by step.

---

## 3. From physics to equations (the Lagrange recipe, in plain words)

Drawing forces is messy here: every rod pushes and pulls on two balls with unknown forces. The Lagrange method avoids
those forces entirely. It only needs two energies:

- **Kinetic energy T** – energy of motion, `½ m v²` for each ball.
- **Potential energy V** – energy of height, `m g y` for each ball.

A fixed recipe (the Euler-Lagrange equation) turns `T − V` into one equation per angle. For our chain, the result can
always be written in this shape:

$$A(\varphi)\,\ddot\varphi = b(\varphi,\dot\varphi)$$

| Symbol | Meaning |
|---|---|
| $\ddot\varphi$ | the angular accelerations we want (a vector, one per link) |
| $A$ | the **mass matrix** – how much the links "resist" being accelerated, and how they drag each other along |
| $b$ | everything else: centrifugal effects, gravity, damping, external moment |

Once we have `A` and `b`, getting the accelerations is just solving a small linear system (like `3x3` for a triple
pendulum).

**A helper quantity `μ_i`** (in the code: `mc`). Link `i` has to carry its own ball *and all balls below it*. So
the relevant mass for link `i` is

$$\mu_i = \sum_{k \ge i} m_k$$

Example with three balls of 1 kg: `μ = [3, 2, 1]`. The top rod "feels" all three balls.

The full equation for link `i` (divided by its length `l_i`):

$$\sum_{j} \mu_{\max(i,j)}\, l_j \cos(\varphi_j - \varphi_i)\, \ddot\varphi_j
= \sum_{j} \mu_{\max(i,j)}\, l_j \sin(\varphi_j - \varphi_i)\, \dot\varphi_j^2
- g\,\mu_i \sin\varphi_i - K_i \dot\varphi_i + M_i$$

You do not need to memorize this. The important point: **left side = matrix × accelerations, right side = known
stuff**. The test suite checks the result against the textbook double-pendulum formula (identical to machine
precision).

---

## 4. `multipendulum/dynamics.py` – the physics engine

### 4.1 `make_rhs(m, l, K, g, M_ext)` – builds the "rule of motion"

**What it does:** returns a function `rhs(t, y)` that the solver calls thousands of times. Given the current state it
returns how the state changes.

**The state vector `y`.** Solvers only understand *first-order* equations ("how does y change?"). Our physics has
accelerations (second order). Trick: we store angles **and** angular velocities in one vector:

```
y = [φ1, φ2, φ3,  dφ1, dφ2, dφ3]
      angles        angular velocities
```

Then "how does `y` change" is:

- the change of the angles = the angular velocities (already in `y`)
- the change of the angular velocities = the accelerations (from `A φ̈ = b`)

**Step by step inside `make_rhs`:**

1. **Precompute things that never change** (done once, not in every solver call):
   ```python
   mc = np.cumsum(m[::-1])[::-1]  # μ_i: mass from link i to the end
   mm = np.maximum.outer(idx, idx)  # index max(i, j) for every pair
   mml = l[np.newaxis, :] * mc[mm]  # l_j * μ_max(i,j)
   ```
2. **Inside `rhs` (every solver call):**
   ```python
   P = phi[np.newaxis, :] - phi[:, np.newaxis]  # all angle differences φ_j − φ_i
   A = np.cos(P) * mml  # mass matrix
   B = np.sin(P) * mml  # coupling terms
   b = B @ dphi**2 - g * np.sin(phi) * mc - K * dphi + M_ext
   d2phi = np.linalg.solve(A, b)  # φ̈ = A⁻¹ b
   return np.concatenate((dphi, d2phi))
   ```
   Line by line:
   - `P` is a table of every angle difference – needed because neighbouring links pull on each other depending on
     how bent the chain is.
   - `A` couples the accelerations of all links (move one link, the others feel it).
   - `b` collects: centrifugal effects (`B @ dphi**2`), gravity (`g*sin(phi)*mc`), damping (`K*dphi`), external
     moment (`M_ext`).
   - `np.linalg.solve` finds the accelerations. (It never inverts `A` explicitly – faster and more accurate.)

> **Honest note:** damping `K` and moment `M_ext` are added to the equation *after* it was divided by `l_i`
> (same as in the original MATLAB code). Physically they therefore act as "torque divided by rod length". For
> `K = 0` and `M_ext = 0` (the default) this does not matter.

### 4.2 `simulate(...)` – runs the whole simulation

**What it does:** takes your parameters, calls the solver, returns everything in one `SimulationResult`.

Step by step:

1. **Clean up the inputs** (`_as_vector`): turns a single number like `K=0.1` into `[0.1, 0.1, 0.1]` and checks
   that all vectors have the same length.
2. **Default start:** all angles `π/2` (all rods horizontal) and zero speed, if you give nothing.
3. **Build the rule of motion:** `rhs = make_rhs(...)`.
4. **Call the solver:**
   ```python
   sol = solve_ivp(rhs, (t0, t1), y0, method=method, t_eval=t_eval, rtol=..., atol=...)
   ```
   `solve_ivp` (SciPy) moves through time in small steps. It chooses the step size itself: small steps when the
   pendulum whips around, larger steps when it is calm. `rtol`/`atol` say how accurate each step must be.
   Default solver `DOP853` is a high-order Runge-Kutta method – very accurate for smooth, non-stiff problems like
   this one.
5. **Uniform output:** `dt_out = 1/200` asks the solver for results every 5 ms, which makes plotting and animation
   easy. (`dt_out=None` returns the solver's own steps instead, like the MATLAB original.)
6. **Error check:** if the solver fails, a clear `RuntimeError` is raised instead of returning garbage.
7. **Post-processing:** `energies(...)` computes positions and energies.

### 4.3 `energies(phi, omega, m, l, g)` – the built-in lie detector

Positions are the cumulative sums from Section 1. Velocities are their time derivatives:

$$\dot x_k = \sum_{j\le k} l_j \cos\varphi_j\,\dot\varphi_j,\qquad \dot y_k = \sum_{j\le k} l_j \sin\varphi_j\,\dot\varphi_j$$

Then:

$$T = \sum_k \tfrac12 m_k(\dot x_k^2 + \dot y_k^2), \qquad V = \sum_k m_k g\, y_k$$

**Why this matters:** without damping and external moment, the **total energy `E = T + V` must stay constant**.
If the equations or the solver were wrong, `E` would drift. This is the single best sanity check of the whole
project – see the energy plot in the GUI (green line `E_total` must be flat).

### 4.4 `SimulationResult`

A simple container ("dataclass") holding `t`, `phi`, `omega`, `T`, `V`, `x`, `y` and the property `E = T + V`.

---

## 5. `multipendulum/animation.py` – drawing it

### 5.1 `PendulumArtist`

**What it does:** draws the pendulum once and afterwards only *moves* the existing lines and dots (much faster than
redrawing everything).

1. `__init__` clears the axes, sets the visible area to the total chain length `sum(l)` in every direction, draws a
   grey ceiling, one blue line for all rods and black dots for the masses. Dot size grows with
   `sqrt(m / max(m))`, so heavier balls look bigger. Pass `m=None` to hide the dots.
2. `positions(phi)` converts angles to ball coordinates (cumulative sums from Section 1).
3. `update(phi)` moves the line and the dots to the new positions.

### 5.2 `interpolate_angles(t, phi, t_query)`

The animation wants the angles at "now" (wall-clock time), which usually lies *between* two stored samples. This
function interpolates linearly between them – the Python equivalent of MATLAB's `interp1`.

### 5.3 `animate(ax, t, phi, l, m, fps, speed)`

Creates a matplotlib `FuncAnimation` (used by the example script and for the GIF): one frame every `1/fps`
seconds, each frame asks `interpolate_angles` for the current angles and calls `update`.

---

## 6. `multipendulum/gui.py` – the window with buttons

The GUI is built with **tkinter** (ships with Python) and embeds matplotlib figures.

### 6.1 Layout (`_build_layout`)

```
┌──────────────────────── Animation ─────────────────────────┐
│                     (matplotlib canvas)                    │
├──────── Parameters and initial conditions ───────┬ Settings┤
│ time span, n segments, g     │ table: l, m, φ(0),│ solver  │
│ [Random] [Set m = l = 1]     │ dφ/dt(0), damping,│ tol.    │
│                              │ moment (per row)  │ Solve / │
│                              │                   │ Animate │
│                              │                   │ / Plot  │
└──────────────────────────────┴───────────────────┴─────────┘
```

The parameter table is a grid of entry fields inside a scrollable canvas (`_rebuild_table`, `_write_table`).
Each row = one link, columns = `l, m, φ(0), dφ/dt(0), damping, moment`.

### 6.2 The central idea: one state, one refresh function

All values live in a few attributes (`self.param` – an `n × 6` array, `self.tspan`, `self.g`, solver settings,
`self.result`). Every user action follows the same pattern:

```
user edits something  →  callback updates self.<value>  →  update_gui() redraws everything
```

- `update_gui()` writes all values back into the fields, draws the pendulum in its start position and enables the
  *Animate*/*Plot* buttons only if a simulation result exists.
- `_invalidate()` throws away an old result whenever a physical parameter changes (otherwise you could animate a
  simulation that no longer matches the table).

### 6.3 What happens when you click …

| Button / action | Method | What happens |
|---|---|---|
| Edit time span / segments / g | `_on_tspan`, `_on_nseg`, `_on_grav` | Parse the number; invalid input is ignored and the old value restored. Changing `n` cuts rows or appends default rows `[1, 1, π/2, 0, 0, 0]`. |
| Edit the table | `_on_table_edit` | Reads all cells; lengths and masses must be positive, otherwise a warning appears. |
| **Random configuration** | `_on_random` → `random_setup()` (in `presets.py`) | 1–7 links; two random modes for lengths/masses and two for start angles (a port of MATLAB `randsetup`). Kept outside the GUI so it can be used and tested without tkinter. |
| **Set m = l = 1** | `_on_all_one` | All masses and lengths become 1. |
| **Solve** | `_on_solve` | Calls `simulate(...)` with the table values, measures the run time and shows the relative energy variation in the status line (should be tiny, e.g. `1e-8`). Errors appear in a message box. |
| **Animate** | `_on_animate` | Real-time animation: a `tick()` function runs every 15 ms via `self.after(...)`, measures elapsed wall-clock time and shows the interpolated angles for that moment. |
| **Plot** | `_on_plot` | Opens a new window with energies (`E_kin`, `E_pot`, `E_total`) and all angles over time. |

Edits are committed with **Enter** or when you leave the field (`_bind_commit`).

---

## 7. `examples/triple_pendulum.py` – without GUI

Runs a classic triple pendulum (all masses and lengths 1, all rods horizontal at the start), prints the maximum
energy deviation, plots energies/angles and shows an animation. `--save demo.gif` exports the GIF used in the README.

```bash
python examples/triple_pendulum.py --t-end 15 --save demo.gif
```

---

## 8. `tests/test_dynamics.py` – how we know it is right

| Test | Idea |
|---|---|
| `test_energy_is_conserved_without_damping` | For 1, 2, 3 and 5 links with random parameters, total energy changes by less than 1e-6 (relative). |
| `test_damping_dissipates_energy` | With damping, energy must go down and never up. |
| `test_small_angle_period_of_single_pendulum` | A single pendulum with a tiny swing must have the textbook period $T = 2\pi\sqrt{l/g}$. |
| `test_hanging_pendulum_stays_at_rest` | Hanging straight down with no speed: nothing may move. |
| `test_random_setup_shape` | The random generator (`presets.py`) always produces valid parameters. |

Run them with `pytest`.

---

## 9. Mapping to the MATLAB original

| MATLAB file | Python | Notes |
|---|---|---|
| `multipend.m` | `dynamics.py` (`make_rhs`, `simulate`, `energies`) | Same equations; `ode113` → `solve_ivp` (`DOP853` by default) |
| `multipendanimation.m` | `animation.py` | `tic/toc` + `interp1` → `time.perf_counter` + `np.interp` |
| `multipendgui.m` | `gui.py` | `uicontrol`/`uitable` → tkinter widgets |

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| Generalized coordinates | The minimal set of numbers that describes the system (here: one angle per link) |
| Euler-Lagrange equations | A recipe that turns energies into equations of motion without needing internal forces |
| Mass matrix `A` | Says how strongly accelerations of the links are coupled |
| State vector | Everything needed to continue the simulation: angles + angular velocities |
| ODE solver | Algorithm that advances a "rate of change" rule through time in small steps |
| `rtol` / `atol` | Allowed relative / absolute error per solver step |
| Chaos | Tiny changes in the start lead to very different outcomes later |
| Energy conservation | Without friction, `T + V` stays constant – used to check the simulation |

---

## 11. Interview cheat sheet (30 seconds)

> "I simulate an n-link pendulum. The equations of motion come from the Euler-Lagrange formalism and have the form
> mass matrix times angular accelerations equals a right-hand side with centrifugal terms and gravity. I solve this
> linear system in every step and integrate with an adaptive Runge-Kutta solver. Since the system is chaotic, I
> validate it with energy conservation and against analytic results like the small-angle period. On top there is a
> real-time animation and a GUI. The project is a Python port of a MATLAB project from my Bachelor's."
