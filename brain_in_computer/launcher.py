"""Start the saved local labs or inspect installation status without training.

This module deliberately uses only the standard library until ``doctor`` checks
PyTorch. It can diagnose a missing dependency through its standalone entry point.
The manifest chooses checkpoint files; it never downloads or trains a model.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import webbrowser


DEFAULT_MANIFEST = "config/checkpoints.json"
CHECKPOINT_ROLES = ("computer", "regional", "encoder")
MODES = ("both", "computer", "memory")
READY_PREFIXES = {"computer": "BiC computer-use lab: ", "memory": "BiC teaching lab: "}


def load_manifest(project_dir=".", manifest=DEFAULT_MANIFEST):
    """Resolve release paths from an explicit project root, never from guesses."""
    root = Path(project_dir).expanduser().resolve()
    path = Path(manifest).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Release manifest not found: {path}. Run from the extracted project directory "
            "or supply --project-dir PATH. The release ZIP includes config/checkpoints.json."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read release manifest {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema") != "bic-release-v1":
        raise ValueError("Unsupported release manifest; expected schema bic-release-v1")
    if not isinstance(data.get("release"), str) or not data["release"].strip():
        raise ValueError("The release manifest must name its release")
    raw = data.get("checkpoints")
    if not isinstance(raw, dict):
        raise ValueError("The release manifest requires a checkpoints object")
    resolved = {}
    for role in CHECKPOINT_ROLES:
        value = raw.get(role)
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
            raise ValueError(f"Checkpoint {role} must be a nonempty project-relative path")
        checkpoint = (root / value).resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError(f"Checkpoint {role} must remain inside --project-dir")
        resolved[role] = checkpoint
    return {"project_dir": root, "manifest": path.resolve(),
            "release": data["release"], "checkpoints": resolved}


def _dependency_status():
    result = {}
    torch_module = None
    for module_name, distribution in (("torch", "torch"), ("numpy", "numpy"), ("PIL", "Pillow")):
        row = {"available": False, "version": None}
        try:
            module = importlib.import_module(module_name)
            row["available"] = True
            row["version"] = metadata.version(distribution)
            if module_name == "torch":
                torch_module = module
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        result[distribution] = row
    return result, torch_module


def doctor(project_dir=".", manifest=DEFAULT_MANIFEST, device="cpu", mode="both"):
    """Report dependencies, devices and nonempty files; do not load weights or write."""
    if device not in ("cpu", "cuda", "auto") or mode not in MODES:
        raise ValueError("Use device cpu/cuda/auto and mode both/computer/memory")
    dependencies, torch_module = _dependency_status()
    result = {
        "ready": False, "requested_device": device, "mode": mode,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "dependencies": dependencies, "cuda": {"available": False, "devices": []},
        "checkpoints": {}, "problems": [],
        "scope": "Read-only dependency/device and nonempty-file checks. No checkpoint loading, "
                 "compatibility validation, training, download, or performance measurement.",
    }
    if torch_module is not None:
        try:
            result["cuda"]["available"] = bool(torch_module.cuda.is_available())
            if result["cuda"]["available"]:
                result["cuda"]["devices"] = [
                    torch_module.cuda.get_device_name(i)
                    for i in range(torch_module.cuda.device_count())
                ]
        except Exception as error:
            result["cuda"]["error"] = f"{type(error).__name__}: {error}"
    result["selected_device"] = ("cuda" if result["cuda"]["available"] else "cpu") if device == "auto" else device
    if not all(row["available"] for row in dependencies.values()):
        result["problems"].append("Missing or broken dependencies. Activate the project environment and run python -m pip install -e .")
    if device == "cuda" and not result["cuda"]["available"]:
        result["problems"].append("CUDA is unavailable. Use --device cpu or install a compatible CUDA PyTorch build; see docs/HOME_COMPUTE.md.")
    try:
        release = load_manifest(project_dir, manifest)
        result.update({key: str(release[key]) for key in ("project_dir", "manifest", "release")})
        required = {"computer"} if mode == "computer" else {"regional", "encoder"}
        if mode == "both":
            required = set(CHECKPOINT_ROLES)
        for role, path in release["checkpoints"].items():
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            result["checkpoints"][role] = {"path": str(path), "present": exists,
                                            "bytes": size, "required": role in required}
            if role in required and not size:
                result["problems"].append(f"Missing or empty {role} checkpoint: {path}. Extract the complete release ZIP, including runs/.")
    except (OSError, ValueError) as error:
        result["problems"].append(str(error))
    result["ready"] = not result["problems"]
    return result


def _positive_integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer from {minimum}" + (f" to {maximum}" if maximum is not None else " upwards"))


def build_commands(project_dir=".", manifest=DEFAULT_MANIFEST, mode="both", device="cpu",
                   threads=1, memory=None, computer_port=8765, memory_port=8766):
    """Return shell-free commands with absolute model and persistence paths."""
    if mode not in MODES or device not in ("cpu", "cuda", "auto"):
        raise ValueError("Use mode both/computer/memory and device cpu/cuda/auto")
    _positive_integer(threads, "threads")
    for port, name in ((computer_port, "computer_port"), (memory_port, "memory_port")):
        _positive_integer(port, name, minimum=0, maximum=65535)
    if mode == "both" and computer_port == memory_port and computer_port != 0:
        raise ValueError("The two labs need different ports; use --computer-port and --memory-port")
    release = load_manifest(project_dir, manifest)
    root, paths = release["project_dir"], release["checkpoints"]
    memory_path = Path(memory).expanduser() if memory is not None else Path("runs/personal-memory.json")
    if not memory_path.is_absolute():
        memory_path = root / memory_path
    prefix = [sys.executable, "-u", "-m", "brain_in_computer"]
    common = ["--host", "127.0.0.1", "--device", device, "--threads", str(threads)]
    commands = {}
    if mode in ("computer", "both"):
        commands["computer"] = prefix + ["computer-serve", "--checkpoint", str(paths["computer"]),
                                           "--port", str(computer_port)] + common
    if mode in ("memory", "both"):
        commands["memory"] = prefix + ["memory-serve", "--checkpoint", str(paths["encoder"]),
            "--regional-checkpoint", str(paths["regional"]), "--memory", str(memory_path.resolve()),
            "--port", str(memory_port)] + common
    return release, commands


class LaunchSession:
    """Own server processes together: partial startup failure closes every child."""

    def __init__(self, project_dir=".", manifest=DEFAULT_MANIFEST, mode="both", device="cpu",
                 threads=1, memory=None, computer_port=8765, memory_port=8766,
                 startup_timeout=60, stream=None):
        if isinstance(startup_timeout, bool) or not isinstance(startup_timeout, (int, float)) or not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be a positive finite number of seconds")
        self.release, self.commands = build_commands(project_dir, manifest, mode, device, threads,
                                                     memory, computer_port, memory_port)
        self.device, self.mode = device, mode
        self.startup_timeout = float(startup_timeout)
        self.stream = sys.stdout if stream is None else stream
        self.processes = {}
        self.urls = {}
        self._events = queue.Queue()
        self._tails = {}
        self._readers = []
        self._started = False

    def _relay(self, name, process):
        try:
            for line in process.stdout:
                self._tails[name].append(line.rstrip())
                print(f"[{name}] {line.rstrip()}", file=self.stream, flush=True)
                if line.startswith(READY_PREFIXES[name]):
                    self._events.put((name, line[len(READY_PREFIXES[name]):].strip()))
        finally:
            process.stdout.close()

    def start(self):
        if self._started:
            raise RuntimeError("A LaunchSession can only start once")
        self._started = True
        status = doctor(self.release["project_dir"], self.release["manifest"], self.device, self.mode)
        if not status["ready"]:
            raise RuntimeError("Cannot start BiC:\n" + "\n".join(status["problems"]))
        try:
            for name, command in self.commands.items():
                process = subprocess.Popen(command, cwd=str(self.release["project_dir"]),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1)
                self.processes[name] = process
                self._tails[name] = deque(maxlen=20)
                reader = threading.Thread(target=self._relay, args=(name, process), daemon=True)
                self._readers.append(reader)
                reader.start()
            deadline = time.monotonic() + self.startup_timeout
            while len(self.urls) < len(self.processes):
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        tail = "\n".join(self._tails[name])
                        raise RuntimeError(f"The {name} lab exited during startup (code {process.returncode}). "
                                           "Check the terminal error and run doctor.\n" + tail)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("BiC startup timed out. Run doctor and inspect the terminal output; "
                                       "use --startup-timeout for unusually slow model loading.")
                try:
                    name, url = self._events.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                parsed = urlsplit(url)
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
                    raise RuntimeError(f"Unexpected startup address from {name}: {url}")
                self.urls[name] = url
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for reader in self._readers:
            reader.join(timeout=1)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()


def launch(project_dir=".", manifest=DEFAULT_MANIFEST, mode="both", device="cpu", threads=1,
           memory=None, computer_port=8765, memory_port=8766, open_browser=False, startup_timeout=60):
    """Run until Ctrl+C; shutdown also closes the other lab after a child exits."""
    try:
        with LaunchSession(project_dir, manifest, mode, device, threads, memory,
                           computer_port, memory_port, startup_timeout) as session:
            print(f"BiC {session.release['release']} is ready. Leave this terminal open; Ctrl+C stops both labs.", flush=True)
            if open_browser:
                for url in session.urls.values():
                    webbrowser.open(url, new=2)
            while True:
                for name, process in session.processes.items():
                    if process.poll() is not None:
                        raise RuntimeError(f"The {name} lab stopped (code {process.returncode}); all launched labs were closed.")
                time.sleep(0.1)
    except KeyboardInterrupt:
        return 0


def main(argv=None):
    """Standalone fallback that remains usable when PyTorch is not installed."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("command", choices=("doctor", "launch"))
    parser.add_argument("--project-dir", default=".", help="Extracted project directory containing runs/ and config/")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Release manifest; relative to --project-dir")
    parser.add_argument("--mode", choices=MODES, default="both", help="Labs to launch or check")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="Device to check or request")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per lab")
    parser.add_argument("--memory", help="Teaching memory path; default is runs/personal-memory.json under the project")
    parser.add_argument("--computer-port", type=int, default=8765, help="Computer lab port; 0 chooses an available port")
    parser.add_argument("--memory-port", type=int, default=8766, help="Teaching lab port; 0 chooses an available port")
    parser.add_argument("--open-browser", action="store_true", help="Request browser tabs after both selected labs start")
    parser.add_argument("--startup-timeout", type=float, default=60, help="Seconds allowed for the selected labs to load")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "doctor":
            status = doctor(**{key: args[key] for key in ("project_dir", "manifest", "device", "mode")})
            print(json.dumps(status, indent=2))
            return 0 if status["ready"] else 1
        return launch(**args)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
