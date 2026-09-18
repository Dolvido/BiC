"""Read-only Windows process gate before a separate local CUDA proof.

No Torch import, runtime configuration, waiting, retry or process mutation.
This gate confirms the fixed parent experiment has terminated; it does not
interpret its learning scores or authenticate all of its evaluation artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


SCHEMA = "bic-prefetch-parent-gate-v1"
PARENT_MANIFEST = "e477b1f6dd4d4f787288bcd6defaba73359949ef67867012caa20c2a81c2c24a"
EXPECTED_JOBS = {f"{objective}-seed{seed}" for seed in (8472, 8473, 8474)
                 for objective in ("baseline", "balanced_reply")}


def _read_pinned(path, expected):
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("explicit SHA256 pin required")
    image = Path(path).read_bytes()
    if hashlib.sha256(image).hexdigest() != expected:
        raise ValueError("parent terminal bytes differ from caller pin")
    value = json.loads(image)
    if type(value) is not dict:
        raise ValueError("parent terminal object required")
    return value


def windows_process_status(pid):
    """Query the current OS process, treating inaccessible state as unknown."""
    if type(pid) is not int or not 0 < pid < 2**32:
        raise ValueError("positive Windows process ID required")
    if os.name != "nt":
        raise RuntimeError("this local parent gate requires Windows")
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    api.GetExitCodeProcess.restype = wintypes.BOOL
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        return {"status": "absent" if error == 87 else "unknown", "windows_error": error}
    try:
        code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
            return {"status": "unknown", "windows_error": ctypes.get_last_error()}
        return {"status": "running" if code.value == 259 else "exited", "exit_code": code.value}
    finally:
        api.CloseHandle(handle)


def check_completed_parent(dispatcher_path, main_path, *, expected_dispatcher_sha256,
                           expected_main_sha256, process_status=windows_process_status):
    """Authenticate terminal metadata, then freshly refuse live/unknown PIDs.

    The injected status reader is only for isolated no-process unit fixtures;
    the local launcher uses the Windows reader above. PID reuse is conservative:
    any currently live process with a recorded ID refuses this launch.
    """
    dispatcher = _read_pinned(dispatcher_path, expected_dispatcher_sha256)
    main = _read_pinned(main_path, expected_main_sha256)
    if (dispatcher.get("schema") != "bic-objective-local-dispatch-v1"
            or dispatcher.get("manifest_sha256") != PARENT_MANIFEST
            or dispatcher.get("status") != "completed" or dispatcher.get("active") is not None
            or not dispatcher.get("ended_utc")):
        raise ValueError("fixed parent dispatcher must be terminal and completed")
    phases = dispatcher.get("phases")
    if (type(phases) is not list or len(phases) != 3
            or any(type(row) is not dict for row in phases)
            or [row.get("name") for row in phases] != ["prepare", "calibration", "main"]
            or any(row.get("status") != "completed" or type(row.get("exit_code")) is not int
                   or row["exit_code"] != 0 or not row.get("ended_utc") for row in phases)
            or phases[-1].get("terminal_sha256") != expected_main_sha256):
        raise ValueError("all fixed parent phases must complete and bind the main terminal")
    jobs = main.get("completed_jobs")
    if (main.get("schema") != "bic-foundation-objective-study-v1" or main.get("stage") != "main"
            or main.get("status") != "completed" or main.get("active_phase") is not None
            or not main.get("ended_utc") or type(jobs) is not list or len(jobs) != 6
            or any(type(name) is not str for name in jobs) or set(jobs) != EXPECTED_JOBS
            or type(main.get("physical_optimizer_updates")) is not int
            or main["physical_optimizer_updates"] != 18432):
        raise ValueError("all six fixed parent main jobs must be terminal and completed")
    pids = {"dispatcher": dispatcher.get("pid"), "main": main.get("pid"),
            "main_launcher": phases[-1].get("child_launcher_pid")}
    if any(type(pid) is not int or not 0 < pid < 2**32 for pid in pids.values()):
        raise ValueError("all parent process identities are required")
    observations = {}
    for role, pid in pids.items():
        observed = process_status(pid)
        if type(observed) is not dict or observed.get("status") not in ("absent", "exited"):
            raise RuntimeError("parent process is live or unknown: " + role)
        observations[role] = dict(pid=pid, **observed)
    # The status observation must concern the same caller-authenticated bytes.
    _read_pinned(dispatcher_path, expected_dispatcher_sha256)
    _read_pinned(main_path, expected_main_sha256)
    return dict(schema=SCHEMA, status="passed", observed_utc=datetime.now(timezone.utc).isoformat(),
        dispatcher_sha256=expected_dispatcher_sha256, main_sha256=expected_main_sha256,
        parent_manifest_sha256=PARENT_MANIFEST, process_observations=observations,
        scope="Parent termination only. No scores interpreted, CUDA initialized, process changed, waiting or retry.")
