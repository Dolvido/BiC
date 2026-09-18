"""Dispatch the frozen local capacity protocol with at most two GPU workers.

This only schedules declared commands and records process observations. It does
not inspect scores, retry failures, change recipes or implement model promotion.
An existing execution ledger is never overwritten by another dispatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dispatch(directory):
    from brain_in_computer.learning_loop import run_lock
    from experiments.capacity_study import jobs, load_protocol
    from experiments.train_cognitive import atomic_json

    directory = Path(directory).resolve()
    load_protocol(directory)
    root = Path(__file__).resolve().parents[1]
    logs = directory/"process-logs"
    logs.mkdir(exist_ok=True)
    ledger = directory/"execution.json"
    state = {"schema": "bic-capacity-dispatch-v1", "status": "running",
        "started_utc": now(), "maximum_concurrent_gpu_workers": 2,
        "protocol_sha256": digest(directory/"protocol.json"),
        "dispatcher_sha256": digest(__file__), "local_only": True,
        "network_training": False, "processes": [],
        "timing_note": "Supervisor observations include process startup and polling. Overlapping intervals are not dedicated GPU time. Worker invocation receipts preserve their narrower timing scopes."}

    def save():
        atomic_json(ledger, state)

    def run_phase(commands, concurrency, active):
        pending, failed = list(commands), False
        while active or (pending and not failed):
            while pending and len(active) < concurrency and not failed:
                name, options = pending.pop(0)
                stdout, stderr = logs/f"{name}.stdout.log", logs/f"{name}.stderr.log"
                if stdout.exists() or stderr.exists():
                    raise ValueError("process log exists; refusing an undeclared retry")
                command = [sys.executable, "-m", "experiments.capacity_study",
                           "--output", str(directory), *options]
                record = {"name": name, "command": command, "started_utc": now(), "status": "starting"}
                state["processes"].append(record)
                save()
                try:
                    with stdout.open("w", encoding="utf8") as out, stderr.open("w", encoding="utf8") as err:
                        process = subprocess.Popen(command, cwd=root, stdout=out, stderr=err,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except BaseException:
                    record.update(status="launch_failed", completed_utc=now())
                    save()
                    raise
                record.update(status="running", pid=process.pid)
                active.append((process, record, stdout, stderr, time.monotonic()))
                save()
                print(json.dumps({"started": name, "pid": process.pid}), flush=True)
            for item in list(active):
                process, record, stdout, stderr, started = item
                code = process.poll()
                if code is None:
                    continue
                record.update(status="completed" if code == 0 else "failed", exit_code=code,
                    completed_utc=now(), observed_wall_seconds=time.monotonic()-started,
                    stdout_sha256=digest(stdout), stderr_sha256=digest(stderr))
                active.remove(item)
                failed = failed or code != 0
                save()
                print(json.dumps({"completed": record["name"], "exit_code": code}), flush=True)
            if active:
                time.sleep(2)
        if failed:
            raise RuntimeError("a declared process failed; no retries or later phase launched")

    def phase(commands, concurrency):
        active = []
        try:
            run_phase(commands, concurrency, active)
        except BaseException:
            # This cleanup does not depend on another successful ledger write.
            # Windows venv executables may own a runtime child, so terminate the
            # exact Popen-owned process tree rather than only its launcher.
            for process, record, stdout, stderr, started in active:
                record["dispatcher_interrupted"] = True
                record["uncommitted_physical_work_unknown"] = process.poll() is None
                if process.poll() is None:
                    try:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                capture_output=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        else:
                            process.terminate()
                    except (OSError, subprocess.TimeoutExpired):
                        record["termination_request_failed"] = True
            for process, record, stdout, stderr, started in active:
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    record["status"] = "termination_unconfirmed"
                    continue
                record.update(status="interrupted", exit_code=code, completed_utc=now(),
                    observed_wall_seconds=time.monotonic()-started)
                for label, path in (("stdout_sha256", stdout), ("stderr_sha256", stderr)):
                    try:
                        record[label] = digest(path)
                    except OSError:
                        record[label] = None
            try:
                save()
            except OSError:
                pass
            raise

    guard = directory/"dispatcher"
    guard.mkdir(exist_ok=True)
    with run_lock(guard):
        if ledger.exists():
            raise ValueError("execution ledger already exists; inspect before any continuation")
        save()
        try:
            for stage in ("calibration", "main"):
                declared = list(jobs(stage))
                if stage == "main":
                    declared = ["w192", "w256", "w96"]
                phase([(f"{stage}-{job}", ["--phase", "train", "--stage", stage, "--job", job])
                       for job in declared], 2)
                phase([(f"verify-{stage}", ["--phase", "verify", "--stage", stage])], 1)
                if stage == "calibration":
                    phase([("select", ["--phase", "select"])], 1)
            phase([("audit", ["--phase", "audit"])], 1)
            state["status"] = "completed"
        except BaseException:
            state["status"] = "failed"
            raise
        finally:
            state["completed_utc"] = now()
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    dispatch(parser.parse_args().output)
