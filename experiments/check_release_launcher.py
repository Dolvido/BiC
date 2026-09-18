"""Check release startup, actual HTTP behavior and memory across a paired restart.

Run after the active controller's training and final evaluation reports exist.
The fixed demonstration uses the ordinary seed-7 UI scene and teaches its first
object one name. Rendering identities are read only from the saved scene to score
the returned actions; they are never submitted to either neural interface.
This is an integration demonstration, not a sealed generalization benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain_in_computer.launcher import LaunchSession, load_manifest

LABEL = "dax"
RELATIONS = ("find", "left", "right", "above", "below")
CELL_NAMES = ("top left", "top right", "bottom left", "bottom right")
DESKTOP_PROBES = (
    ("hello", "hello.", "no_action"),
    ("good morning", "hello.", "no_action"),
    ("click the red button", "clicked red.", "red_visible"),
    ("what did you click", "last was red.", "no_action"),
    ("type hi", "typed hi.", "hi_visible"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_action(reference, relation):
    """Scoring only, outside model inference."""
    if relation == "find":
        return reference
    dr, dc = {"left": (0, -1), "right": (0, 1), "above": (-1, 0), "below": (1, 0)}[relation]
    row, col = divmod(reference, 2)
    row, col = row + dr, col + dc
    return row * 2 + col if 0 <= row < 2 and 0 <= col < 2 else 10


def instruction(relation, label=LABEL):
    quoted = json.dumps(label)
    if relation == "find":
        return f"find {quoted}"
    preposition = " of" if relation in ("left", "right") else ""
    return f"click the object {relation}{preposition} {quoted}"


def verify(project_dir=ROOT, manifest="config/checkpoints.json", output=None):
    started = time.monotonic()
    root = Path(project_dir).resolve()
    output = Path(output) if output is not None else root / "experiments/launcher_v06_verification.json"
    release = load_manifest(root, manifest)
    paths = release["checkpoints"]
    # Prevent a demo from accidentally selecting a checkpoint still being tuned.
    for required in ("training_report.json", "final_evaluation.json"):
        path = paths["regional"].parent / required
        if not path.is_file():
            raise FileNotFoundError(f"Wait for completed training and evaluation before this release demonstration: {path}")
    initial_hashes = {role: digest(path) for role, path in paths.items()}
    report = {
        "schema": "bic-release-launcher-verification-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "release": release["release"],
        "checkpoint_sha256": initial_hashes,
        "manifest_sha256": digest(release["manifest"]),
        "script_sha256": digest(__file__),
        "protocol": {
            "scene": "Ordinary seed-7 teaching interface scene, first displayed object",
            "label": LABEL, "relations_before_and_after_restart": list(RELATIONS),
            "desktop_prompts_each_phase": [row[0] for row in DESKTOP_PROBES],
            "selection": "Fixed before this run; no scene or seed search",
            "generalization_claim": False,
        },
        "scope": "Public dependency diagnostic plus paired local HTTP integration. "
                 "Transport/persistence checks and observed model correctness are reported separately. "
                 "Rendering identities are used only to score memory actions, never as policy inputs.",
        "browser_visual_verification": False,
        "training_performed": False,
        "processes": [], "checks": {}, "transport": [], "desktop_cases": [], "memory_cases": [],
    }
    sessions = []
    opener = build_opener(ProxyHandler({}))

    def request(base, path, body=None, *, phase, headers=None, expected_status=200):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request_headers = {"Content-Type": "application/json"} if data is not None else {}
        request_headers.update(headers or {})
        req = Request(base + path, data=data, headers=request_headers)
        try:
            response = opener.open(req, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read()
            kind = response.headers.get_content_type()
            payload = json.loads(raw) if kind == "application/json" else raw.decode("utf-8")
            report["transport"].append({"phase": phase, "endpoint": path,
                "method": "GET" if data is None else "POST", "status": response.status,
                "expected_status": expected_status, "passed": response.status == expected_status})
            if response.status != expected_status:
                raise RuntimeError(f"{path} returned HTTP {response.status}: {payload}")
            return payload

    def desktop_cases(base, phase):
        for prompt, reply, criterion in DESKTOP_PROBES:
            payload = request(base, "/chat", {"text": prompt}, phase=phase)
            actions, state = payload["actions"], payload["state"]
            behavior = (actions == ["stop"] if criterion == "no_action" else
                        state["last_clicked"] == "red" if criterion == "red_visible" else
                        state["text"] == "hi")
            report["desktop_cases"].append({"phase": phase, "prompt": prompt,
                "reply": payload["reply"], "actions": actions, "visible_state": state,
                "terminated": payload["terminated"], "expected_reply": reply,
                "behavior_criterion": criterion, "behavior_correct": behavior,
                "reply_correct": payload["reply"] == reply,
                "joint_correct": behavior and payload["reply"] == reply,
                "correctness_scope": "Exact reply plus visible final outcome or STOP; private action targets are not inspected."})

    def memory_case(base, relation, target, phase, label=LABEL):
        command = instruction(relation, label)
        payload = request(base, "/chat", {"text": command}, phase=phase)
        result = payload["result"]
        reply = "cannot select." if target == 10 else f"selected {CELL_NAMES[target]}."
        report["memory_cases"].append({"phase": phase, "prompt": command,
            "action": result["action"], "reply": payload["reply"], "status": result["status"],
            "target_scoring_only": target, "expected_reply_scoring_only": reply,
            "action_correct": result["action"] == target, "reply_correct": payload["reply"] == reply,
            "joint_correct": result["action"] == target and payload["reply"] == reply,
            "action_source": result["action_source"], "response_source": payload["response_source"],
            "trace": result["trace"]})

    try:
        command = [sys.executable, "-S", "-m", "brain_in_computer", "doctor", "--project-dir", str(root),
                   "--manifest", str(release["manifest"])]
        probe = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        diagnosis = json.loads(probe.stdout)
        report["missing_dependency_doctor"] = {
            "command": command, "exit_code": probe.returncode, "diagnosis": diagnosis,
            "stderr": probe.stderr,
            "method": "Python -S disables site packages for this diagnostic subprocess only",
        }
        report["checks"]["public_doctor_handles_missing_dependencies"] = (
            probe.returncode == 1 and not diagnosis["ready"]
            and all(not row["available"] for row in diagnosis["dependencies"].values())
            and any("pip install" in problem for problem in diagnosis["problems"])
        )
        with tempfile.TemporaryDirectory(prefix="bic-release-launch-") as temporary:
            memory = Path(temporary) / "personal-memory.json"
            scene = memory.with_name(memory.name + ".scene.json")
            first = LaunchSession(root, release["manifest"], memory=memory,
                                  computer_port=0, memory_port=0)
            sessions.append(first)
            with first:
                report["processes"].append({"phase": "first_launch", "pids": {
                    name: process.pid for name, process in first.processes.items()}})
                for name, base in first.urls.items():
                    page = request(base, "/", phase="first_launch")
                    report["checks"][f"{name}_html_served"] = isinstance(page, str) and "<html" in page.lower()
                    state = request(base, "/state", phase="first_launch")
                    report["checks"][f"{name}_state_get_nonmutating"] = state == request(base, "/state", phase="first_launch")
                before = request(first.urls["memory"], "/state", phase="first_launch")
                report["checks"]["regional_mode_enabled"] = before["action_source"] == "regional_motor_cortex"
                request(first.urls["memory"], "/teach", {"label": "blocked"}, phase="first_launch",
                        headers={"Origin": "https://example.invalid"}, expected_status=403)
                request(first.urls["memory"], "/select", {"index": 0}, phase="first_launch")
                taught = request(first.urls["memory"], "/teach", {"label": LABEL}, phase="first_launch")
                report["checks"]["teaching_acknowledgment_is_interface_text"] = taught["response_source"] == "interface_template"
                initial_patterns = json.loads(scene.read_text())["patterns"]
                taught_pattern = initial_patterns[0]
                report["protocol"]["initial_patterns_scoring_only"] = initial_patterns
                memory_bytes = memory.read_bytes()
                for relation in RELATIONS:
                    memory_case(first.urls["memory"], relation, expected_action(0, relation), "first_launch")
                desktop_cases(first.urls["computer"], "first_launch")
            report["checks"]["both_first_children_reaped_before_restart"] = all(
                process.poll() is not None for process in first.processes.values())

            second = LaunchSession(root, release["manifest"], memory=memory,
                                   computer_port=0, memory_port=0)
            sessions.append(second)
            with second:
                report["processes"].append({"phase": "restart_without_teaching", "pids": {
                    name: process.pid for name, process in second.processes.items()}})
                after = request(second.urls["memory"], "/state", phase="restart_without_teaching")
                new_patterns = json.loads(scene.read_text())["patterns"]
                report["checks"]["label_loaded_without_reteaching"] = after["labels"] == [LABEL]
                report["checks"]["four_distinct_child_processes"] = len({
                    process.pid for session in sessions for process in session.processes.values()}) == 4
                reference = new_patterns.index(taught_pattern)
                report["checks"]["taught_object_changed_pixels"] = (
                    before["objects"][0]["image"] != after["objects"][reference]["image"])
                report["checks"]["scene_advanced_after_restart"] = after["scene_number"] > before["scene_number"]
                report["protocol"]["restarted_patterns_scoring_only"] = new_patterns
                for relation in RELATIONS:
                    memory_case(second.urls["memory"], relation, expected_action(reference, relation), "restart_without_teaching")
                memory_case(second.urls["memory"], "find", 10, "unknown_name", "never_taught")
                desktop_cases(second.urls["computer"], "restart_without_teaching")
                report["checks"]["memory_bytes_unchanged_by_queries_and_restart"] = memory.read_bytes() == memory_bytes
                report["memory_sha256"] = hashlib.sha256(memory_bytes).hexdigest()
            report["teaching_calls_after_restart"] = 0
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for session in sessions:
            session.close()
        report["checks"]["all_started_children_reaped"] = bool(sessions) and all(
            process.poll() is not None for session in sessions for process in session.processes.values())
        report["checks"]["release_checkpoint_bytes_unchanged"] = all(
            digest(paths[role]) == value for role, value in initial_hashes.items())
        report["checks"]["memory_actions_and_replies_use_neural_pathways"] = bool(report["memory_cases"]) and all(
            case["action_source"] == "regional_motor_cortex" and case["response_source"] == "neural_byte_decoder"
            and "<name>" in case["trace"]["instruction_to_network"] for case in report["memory_cases"])
        report["transport_checks_passed"] = bool(report["transport"]) and all(
            row["passed"] for row in report["transport"])
        report["structural_checks_passed"] = "error" not in report and all(report["checks"].values()) and report["transport_checks_passed"]
        report["model_results"] = {
            "desktop": {"total": len(report["desktop_cases"]), "joint_correct": sum(
                case["joint_correct"] for case in report["desktop_cases"])},
            "regional_memory": {"total": len(report["memory_cases"]), "joint_correct": sum(
                case["joint_correct"] for case in report["memory_cases"])},
            "interpretation": "A fixed interface demonstration; errors do not become transport failures or justify model tuning on these cases.",
        }
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--manifest", default="config/checkpoints.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify(**vars(args))
    print(json.dumps({key: report[key] for key in (
        "structural_checks_passed", "transport_checks_passed", "model_results", "elapsed_seconds")}, indent=2))
    return 0 if report["structural_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
