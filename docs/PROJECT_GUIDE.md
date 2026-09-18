# Find your way around BiC

Start in the extracted project folder, the one containing `README.md`, `pyproject.toml`, and `brain_in_computer/`. Activate the Python environment described in the README. The commands here use `python` for that environment's interpreter.

## Everyday commands

```bash
python -m brain_in_computer doctor
python -m brain_in_computer launch
```

`doctor` reads installation and model-file status. It checks that PyTorch, NumPy, and Pillow import, reports whether CUDA is available, and checks the release checkpoint files exist and are nonempty. It does not load model weights, verify their compatibility, benchmark hardware, download anything, or train. Startup performs the actual model loading and reports errors in the terminal.

`launch` starts both local interfaces using the active release in `config/checkpoints.json`:

| Interface | Address | Try first |
|---|---|---|
| Computer lab | [Open computer lab](http://127.0.0.1:8765) | `click the red button`, then `what did you click` |
| Teaching lab | [Open teaching lab](http://127.0.0.1:8766) | Select an object, teach `dax`, change scene, then `click the object left of "dax"` |

The launcher prints both addresses. Add `--open-browser` to request browser tabs on the machine running BiC. Keep the terminal open; Ctrl+C stops both servers. Each server loads its own copy of the small model. Neither interface runs a training optimizer during ordinary interaction. Teaching stores visual associations; it does not train the language model on each new name.

Useful variations:

```bash
python -m brain_in_computer launch --mode memory
python -m brain_in_computer launch --mode computer
python -m brain_in_computer launch --project-dir /path/to/brain-in-computer
python -m brain_in_computer launch --computer-port 8865 --memory-port 8866
python -m brain_in_computer launch --memory runs/my-experiment-memory.json
python -m brain_in_computer doctor --device cuda
```

Use a quoted Windows path after `--project-dir` if needed. Relative checkpoint paths always resolve from that project directory, even when the command runs elsewhere. Relative `--memory` paths resolve there too. The default memory file is `runs/personal-memory.json`; an adjacent `.scene.json` file preserves the teaching world's objects. Keep both files to preserve your names and scene across restarts. Starting the teaching lab changes the scene's appearance while keeping its object identities. Use a different `--memory` path for an independent teaching session.

CPU with one thread is the launcher default. `--device cuda` requests the installed GPU backend and fails clearly when unavailable. `--device auto` chooses CUDA when available. See [HOME_COMPUTE.md](HOME_COMPUTE.md) before interpreting GPU availability as a performance measurement.

If PyTorch is missing, the dependency-free diagnostic entry point remains available:

```bash
python -m brain_in_computer.launcher doctor
```

If a port is occupied or one server fails to start, the launcher closes the other server it started and reports the error. It does not take over an existing server. Choose different ports or stop your previous BiC session. `--startup-timeout 120` allows a longer initial model load on a slow computer.

## Repository map

| Location | Purpose | Change when |
|---|---|---|
| `README.md` | Installation, current release, first interaction | Starting or checking what the release demonstrates |
| `config/checkpoints.json` | Active computer, regional controller, and encoder paths | Deliberately selecting another compatible trained release |
| `brain_in_computer/` | Importable Python implementation | Developing the system |
| `brain_in_computer/cli.py` | User-facing command definitions | Adding a stable command |
| `brain_in_computer/launcher.py` | Installation checks and paired local launch | Improving startup behavior |
| `brain_in_computer/model.py` | Original regional networks and pathways | Studying the core brain-inspired architecture |
| `brain_in_computer/computer_use/` | Screenshot environment, grounded model, training, desktop UI | Developing desktop perception and action |
| `brain_in_computer/associative.py` | Learned visual encoder and stored exemplar associations | Developing rapid visual memory |
| `brain_in_computer/regional_memory.py` | Visual memory inputs to regional action and language pathways | Developing memory-guided behavior |
| `experiments/` | Reproducible experiment drivers and verification records | Training or investigating a claim |
| `runs/` | Checkpoints, measured results, demonstration images, personal memory | Reading evidence or saving a new experiment |
| `tests/` | Automated behavioral and integration checks | Verifying a code change |
| `docs/` | Architecture, methods, results, limitations | Understanding the science and evidence |

Read [ARCHITECTURE.md](ARCHITECTURE.md), [REGIONAL_MEMORY.md](REGIONAL_MEMORY.md), and the current release results linked from the README for the main design. [ASSOCIATIVE_MEMORY.md](ASSOCIATIVE_MEMORY.md) explains how stored visual examples differ from learned weights. Older result documents describe their own frozen checkpoints; their metrics do not automatically describe the currently selected model.

## Keep experiments understandable

Create a new descriptive output directory for each training experiment, such as `runs/my-retention-experiment/`. Preserve its protocol, settings, selected checkpoint, and evaluation report together. Do not train over a release checkpoint. Compare before and after models on the same evaluation draws, while reserving a separate final test until development choices are complete.

The existing historical directories stay in place because published measurements and checkpoint identities refer to them. The manifest provides one clear active selection without moving that evidence or duplicating trained weights. A manifest change changes which model launches; it does not train, merge, or convert checkpoints. The regional controller and encoder must be compatible, which their actual loaders enforce.

Run the automated checks from the project root:

```bash
python -m unittest discover -s tests -v
```

Passing software tests checks the implementation. Task evaluation measures the trained model's behavior. Hardware benchmarks measure computational cost. Keep these three kinds of evidence separate when describing progress.
