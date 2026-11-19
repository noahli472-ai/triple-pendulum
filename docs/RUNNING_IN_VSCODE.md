# Running the Pendulum Simulation in VS Code (Windows)

A cheat sheet: one-time setup, the commands to run, and fixes for the usual errors.

---

## Quick start (after the one-time setup)

1. Open VS Code → *File → Open Folder* → the `triple-pendulum` folder
   (the one that contains `README.md` and the `multipendulum` folder).
2. *Terminal → New Terminal*. The prompt must start with `(.venv)`.
   If not: `.venv\Scripts\Activate.ps1`
3. Run:

```powershell
python -m multipendulum
```

In the GUI: **Solve** (compute) → **Animate** (watch it swing) or **Plot** (energy and angles).

---

## All commands

| Command | What it does |
|---|---|
| `python -m multipendulum` | Opens the GUI |
| `python examples/triple_pendulum.py` | Plots + animation of a triple pendulum, no GUI |
| `python examples/triple_pendulum.py --t-end 20` | Same, simulating 20 s instead of 15 s |
| `python examples/triple_pendulum.py --save demo.gif` | Additionally saves the animation as a GIF |
| `pytest` | Runs the physics tests |
| `ruff check .` | Linting (needs `pip install ruff`) |

The ▶ button in VS Code works for `examples/triple_pendulum.py`, **not** for files inside
`multipendulum/` (`gui.py`, `dynamics.py`, ...). Those are parts of a package; use
`python -m multipendulum` instead.

---

## One-time setup

### 1. Allow PowerShell scripts (once per Windows user)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm with `J` (German Windows) or `Y` (English Windows).

### 2. Get the code

```powershell
cd "$HOME\Desktop\Coding Projects"
git clone https://github.com/noahli472-ai/triple-pendulum.git
cd triple-pendulum
```

Without git: on GitHub *Code → Download ZIP*, unzip, and `cd` into the folder that directly
contains `README.md` (ZIP downloads often create a nested folder of the same name).

### 3. Create the virtual environment and install

Run these **inside the project folder**:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Check: `.venv\Scripts\python.exe -m multipendulum` opens the GUI.

### 4. Connect VS Code to the environment

1. *File → Open Folder* → the project folder.
2. Install the **Python** extension (Microsoft) if VS Code asks or it is missing.
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the entry with `.venv`
   (`.\.venv\Scripts\python.exe`). If it is not listed: *Enter interpreter path…* →
   `.venv\Scripts\python.exe`.
4. Close old terminals (trash icon) and open a new one. It should now start with `(.venv)`.

---

## Troubleshooting

| Error | Cause → fix |
|---|---|
| `[WinError 5] Access denied: 'C:\WINDOWS\System32\.venv'` | The terminal is in the Windows system folder. Open the project folder in VS Code (step 4) or `cd` into it. Do not run VS Code as administrator. |
| `...Activate.ps1 cannot be loaded because running scripts is disabled` | Run step 1 (`Set-ExecutionPolicy ...`). |
| `ModuleNotFoundError: No module named 'numpy'` | Packages are not installed in the Python that runs. Run step 3 in the project folder and select the `.venv` interpreter (step 4). |
| `ModuleNotFoundError: No module named 'multipendulum'` | Wrong folder (often the outer folder of a nested ZIP) or `pip install -e .` missing. |
| `attempted relative import with no known parent package` | A file inside `multipendulum/` was started with ▶. Use `python -m multipendulum`. |
| `.venv` created in the wrong place | Delete it (`Remove-Item -Recurse -Force .venv` in that folder) and repeat step 3 inside the project folder. |

Which Python is running?

```powershell
python -c "import sys; print(sys.executable)"
```

It must end with `...\triple-pendulum\.venv\Scripts\python.exe`.

---

## macOS / Linux

Same steps with these differences: `python3 -m venv .venv`, activate with
`source .venv/bin/activate`, and the interpreter path is `.venv/bin/python`.
On Linux, tkinter may need `sudo apt install python3-tk`.
