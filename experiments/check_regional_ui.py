"""Exercise trained regional memory through the public HTTP teaching interface.

The protocol is fixed before the demonstration: seed 2026, the first four new
sealed identities, four new symbolic names, and every supported relation.
Rendering identities are read only to construct the world and score outputs.
No identity, reference index, relation class, or expected action enters inference.
This is a reproducible integration demonstration, not an additional sealed test.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Share HTTP transport and image typography with the v0.4 verification utility.
from experiments.check_teaching_lab import font, request, sha256, stop_server, success
LABELS = ("huxen", "vablo", "nirpet", "moskin")
RELATIONS = ("find", "left", "right", "above", "below")
CELL_NAMES = ("top left", "top right", "bottom left", "bottom right")


def start_server(encoder, checkpoint, memory):
    command = [sys.executable, "-m", "brain_in_computer", "memory-serve",
               "--checkpoint", str(encoder), "--regional-checkpoint", str(checkpoint),
               "--memory", str(memory), "--port", "0", "--seed", "2026",
               "--device", "cpu", "--threads", "1"]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    output, lines = queue.Queue(), []

    def consume():
        for line in process.stdout:
            lines.append(line.rstrip())
            output.put(line)
        output.put(None)

    threading.Thread(target=consume, daemon=True).start()
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            try:
                line = output.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                raise RuntimeError("Server exited before startup: " + "\n".join(lines))
            match = re.search(r"BiC teaching lab: (http://127\.0\.0\.1:\d+)", line)
            if match:
                return process, match.group(1)
        raise RuntimeError("Server did not announce its localhost URL within 30 seconds")
    except BaseException:
        stop_server(process)
        raise


def expected_action(reference, relation):
    """Scoring only: this function is never passed to the server or neural agent."""
    if relation == "find":
        return reference
    dr, dc = {"left": (0, -1), "right": (0, 1), "above": (-1, 0), "below": (1, 0)}[relation]
    row, col = divmod(reference, 2)
    row, col = row + dr, col + dc
    return row * 2 + col if 0 <= row < 2 and 0 <= col < 2 else 10


def instruction(label, relation):
    quoted = json.dumps(label)
    return f"find {quoted}" if relation == "find" else (
        f"select the object {relation}{' of' if relation in ('left', 'right') else ''} {quoted}")


def run_case(url, label, relation, target, phase, use_find_button=False):
    text = instruction(label, relation)
    response = (success(url, "/find", {"label": label}) if use_find_button
                else success(url, "/chat", {"text": text}))
    result = response["result"]
    expected_reply = "cannot select." if target == 10 else f"selected {CELL_NAMES[target]}."
    return {"phase": phase, "instruction": text, "endpoint": "/find" if use_find_button else "/chat",
            "target_scoring_only": target, "expected_reply_scoring_only": expected_reply,
            "action": result["action"], "selected_index": response["selected"],
            "status": result["status"], "reply": response["reply"],
            "action_correct": result["action"] == target,
            "reply_correct": response["reply"] == expected_reply,
            "joint_correct": result["action"] == target and response["reply"] == expected_reply,
            "action_source": result["action_source"], "response_source": response["response_source"],
            "trace": result["trace"]}


def contact_sheet(before, after, labels_by_phase, cases, output):
    """Illustration of recorded HTTP images and responses; not a browser screenshot."""
    image = Image.new("RGB", (1420, 1135), "#0d1013")
    draw = ImageDraw.Draw(image)
    draw.text((35, 23), "BiC v0.5 | Memory informs regional actions", fill="#edf0f3", font=font(30, True))
    draw.text((35, 70), "Recorded HTTP images and neural byte replies; separate server processes", fill="#a9b6c4", font=font(17))
    for column, (state, phase) in enumerate(((before, "after_teaching"), (after, "after_restart"))):
        x = 35 + column * 710
        draw.text((x, 111), "Teach, then act" if column == 0 else "Restart, then act without teaching",
                  fill="#d4eee0", font=font(21, True))
        for item in state["objects"]:
            i = item["index"]
            left, top = x + (i % 2) * 320, 155 + (i // 2) * 260
            draw.rounded_rectangle((left, top, left + 300, top + 240), radius=8,
                                   fill="#151c21", outline="#485462", width=2)
            crop = Image.open(io.BytesIO(base64.b64decode(item["image"].split(",", 1)[1]))).convert("RGB")
            image.paste(crop.resize((168, 168), Image.Resampling.NEAREST), (left + 66, top + 14))
            draw.text((left + 15, top + 190), f'{CELL_NAMES[i]} · "{labels_by_phase[phase][i]}"',
                      fill="#e0eee7", font=font(17, True))
            draw.text((left + 15, top + 218), "Name from teaching log (scoring only)", fill="#a7b7c4", font=font(12))
        sample = [case for case in cases if case["phase"] == phase][:5]
        draw.text((x, 687), 'Commands referring to "huxen"', fill="#d4eee0", font=font(18, True))
        for row, case in enumerate(sample):
            y = 725 + row * 55
            short = case["instruction"].replace("select the object ", "")
            draw.text((x, y), short, fill="#a7b7c4", font=font(15))
            marker = "PASS" if case["joint_correct"] else "FAIL"
            draw.text((x + 15, y + 23), f'{case["reply"]!r}   action {case["action"]}   {marker}',
                      fill="#d4eee0" if case["joint_correct"] else "#ffaf9f", font=font(15, True))
    passed = sum(case["joint_correct"] for case in cases)
    draw.text((35, 1030), f"All recorded cases: {passed}/{len(cases)} correct action + reply. Full transcript is in the JSON report.",
              fill="#c2cdd7", font=font(18))
    draw.text((35, 1070), "Quoted names address saved memory. Direction words and actions are learned; teaching and identify use interface text.",
              fill="#9cabb8", font=font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/regional-memory-v05/checkpoint.pt")
    parser.add_argument("--encoder", type=Path, default=ROOT / "runs/associative/encoder.pt")
    parser.add_argument("--protocol", type=Path, default=ROOT / "runs/regional-memory-v05/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/regional_ui_verification.json")
    parser.add_argument("--image", type=Path, default=ROOT / "runs/regional-memory-v05/ui_demo.png")
    args = parser.parse_args()
    for name in ("checkpoint", "encoder", "protocol", "output", "image"):
        setattr(args, name, getattr(args, name).resolve())
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    patterns = protocol["identity_partitions"]["sealed"][:4]
    checkpoint_hash, encoder_digest = sha256(args.checkpoint), sha256(args.encoder)
    report = {"schema": "bic-regional-http-verification-v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.checkpoint.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
              "encoder_sha256": encoder_digest, "protocol_sha256": sha256(args.protocol),
              "selection": "First four new sealed identities; seed 2026 and 50 cases fixed before model evaluation",
              "identities_scoring_only": patterns, "labels": LABELS, "policy_patched": False,
              "model_inputs": "Explicit quoted name, instruction with name masked, pixel-derived memory similarities",
              "browser_visual_verification": False, "illustration_is_browser_screenshot": False,
              "checks": {}, "processes": [], "cases": []}
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="bic-regional-http-") as directory:
            memory = Path(directory) / "memory.json"
            scene = memory.with_name(memory.name + ".scene.json")
            scene.write_text(json.dumps({"schema": "bic-teaching-scene-v1", "seed": 2026,
                                         "scene_number": 0, "patterns": patterns}))
            process, url = start_server(args.encoder, args.checkpoint, memory)
            first_pid = process.pid
            report["processes"].append({"phase": "teaching_and_actions", "pid": first_pid})
            before = success(url, "/state")
            report["checks"]["regional_mode_enabled"] = before["action_source"] == "regional_motor_cortex"
            report["checks"]["get_state_nonmutating"] = before == success(url, "/state")
            page = success(url, "/")
            script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
            js = subprocess.run(["node", "--check", "-"], input=script, text=True, capture_output=True)
            report["checks"]["frontend_javascript_syntax"] = js.returncode == 0
            report["checks"]["frontend_controls_served"] = all(marker in page for marker in (
                'id="command-help"', 'id="teach-form"', 'id="trace-box"', 'id="chat-form"'))
            report["checks"]["cross_origin_rejected"] = request(url, "/teach", {"label": "blocked"},
                                                                {"Origin": "https://example.invalid"})[0] == 403
            report["checks"]["multiple_quoted_names_rejected"] = request(url, "/chat", {"text": 'find "a" "b"'})[0] == 400
            teach_patterns = json.loads(scene.read_text())["patterns"]
            mapping = dict(zip(LABELS, teach_patterns))
            for index, label in enumerate(LABELS):
                success(url, "/select", {"index": index})
                result = success(url, "/teach", {"label": label})
                assert result["response_source"] == "interface_template"
            memory_bytes = memory.read_bytes()
            for label in LABELS:
                for relation in RELATIONS:
                    target = expected_action(teach_patterns.index(mapping[label]), relation)
                    report["cases"].append(run_case(url, label, relation, target, "after_teaching", relation == "find"))
            identified = success(url, "/identify", {})
            report["checks"]["identify_remains_matcher_and_template"] = (
                identified["response_source"] == "interface_template"
                and identified["result"]["action_source"] == "visual_matcher")
            stop_server(process)
            report["checks"]["first_process_stopped_before_restart"] = process.poll() is not None
            process = None
            process, url = start_server(args.encoder, args.checkpoint, memory)
            report["processes"].append({"phase": "restart_without_teaching", "pid": process.pid})
            after = success(url, "/state")
            recall_patterns = json.loads(scene.read_text())["patterns"]
            report["checks"]["distinct_os_processes"] = first_pid != process.pid
            report["checks"]["names_loaded_without_teaching"] = after["labels"] == list(LABELS)
            before_images = {p: before["objects"][i]["image"] for i, p in enumerate(teach_patterns)}
            report["checks"]["all_objects_changed_pixels"] = all(
                before_images[p] != after["objects"][i]["image"] for i, p in enumerate(recall_patterns))
            report["checks"]["object_order_changed"] = recall_patterns != teach_patterns
            for label in LABELS:
                for relation in RELATIONS:
                    target = expected_action(recall_patterns.index(mapping[label]), relation)
                    report["cases"].append(run_case(url, label, relation, target, "after_restart", relation == "find"))
            for relation in RELATIONS:
                report["cases"].append(run_case(url, "never_taught", relation, 10, "unknown_name"))
            success(url, "/scene", {"new_objects": True})
            for relation in RELATIONS:
                report["cases"].append(run_case(url, LABELS[0], relation, 10, "absent_object"))
            report["checks"]["memory_bytes_unchanged_by_inference_and_restart"] = memory.read_bytes() == memory_bytes
            report["checks"]["checkpoint_bytes_unchanged"] = sha256(args.checkpoint) == checkpoint_hash
            report["checks"]["encoder_bytes_unchanged"] = sha256(args.encoder) == encoder_digest
            report["checks"]["all_actions_and_replies_are_neural"] = all(
                case["action_source"] == "regional_motor_cortex" and case["response_source"] == "neural_byte_decoder"
                for case in report["cases"])
            report["checks"]["all_queries_passed_masked_name_to_network"] = all(
                "<name>" in case["trace"]["instruction_to_network"] for case in report["cases"])
            report["checks"]["regional_activity_and_four_scores_recorded"] = all(
                len(case["trace"]["memory_cosine_similarities"]) == 4
                and len(case["trace"]["regional_activity_norms"]) == 13 for case in report["cases"])
            report["teaching_calls_after_restart"] = 0
            report["memory_sha256"] = hashlib.sha256(memory_bytes).hexdigest()
            labels_by_phase = {phase: [next(label for label, value in mapping.items() if value == pattern)
                                      for pattern in positions]
                               for phase, positions in (("after_teaching", teach_patterns), ("after_restart", recall_patterns))}
            contact_sheet(before, after, labels_by_phase, report["cases"], args.image)
            report["illustration"] = str(args.image.relative_to(ROOT))
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        if process is not None:
            stop_server(process)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["total"] = len(report["cases"])
        report["action_correct"] = sum(case["action_correct"] for case in report["cases"])
        report["reply_correct"] = sum(case["reply_correct"] for case in report["cases"])
        report["joint_correct"] = sum(case["joint_correct"] for case in report["cases"])
        report["structural_checks_passed"] = "error" not in report and all(report["checks"].values())
        report["all_cases_correct"] = report["total"] == 50 and report["joint_correct"] == 50
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({key: report[key] for key in (
            "structural_checks_passed", "all_cases_correct", "total", "action_correct", "reply_correct", "joint_correct")}))
    if not report["structural_checks_passed"]:
        raise SystemExit("HTTP integration has a structural failure; see the recorded report")


if __name__ == "__main__":
    main()
