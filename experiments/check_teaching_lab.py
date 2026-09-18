"""Exercise the actual teaching CLI across two separate server processes.

Object IDs are used only to initialize the rendered world and score outcomes.
All teaching and retrieval requests go through the normal HTTP interface.
"""
from __future__ import annotations

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
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/associative/encoder.pt"
REPORT = ROOT / "experiments/teaching_ui_verification.json"
DEMO = ROOT / "runs/associative/restart_demo.png"
LABELS = ("dax", "wug", "toma", "kiki")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_server(memory_file):
    command = [sys.executable, "-m", "brain_in_computer", "memory-serve",
               "--checkpoint", str(CHECKPOINT), "--memory", str(memory_file),
               "--port", "0", "--seed", "2026", "--device", "cpu", "--threads", "1"]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    output = queue.Queue()
    lines = []

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
                return process, match.group(1), lines
        raise RuntimeError("Server did not announce its localhost URL within 30 seconds")
    except BaseException:
        stop_server(process)
        raise


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if process.stdout:
        process.stdout.close()


def request(url, path, body=None, headers=None):
    req = urllib.request.Request(url + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = response.read()
            return response.status, (json.loads(payload) if response.headers.get_content_type() == "application/json"
                                     else payload.decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def success(url, path, body=None):
    status, response = request(url, path, body)
    if status != 200:
        raise RuntimeError(f"{path} returned {status}: {response}")
    return response


def font(size, bold=False):
    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf"),
                  Path("C:/Windows/Fonts/arial" + ("bd" if bold else "") + ".ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def contact_sheet(before, after, results, pids):
    canvas = Image.new("RGB", (1200, 860), "#0d1013")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 25), "BiC v0.4 | Teach, restart, recall", fill="#edf0f3", font=font(30, True))
    draw.text((35, 70), "Actual local HTTP requests to two independent Python server processes", fill="#a9b6c4", font=font(17))
    draw.text((35, 112), f"1. Teach four names  |  Process {pids[0]}", fill="#d4eee0", font=font(19, True))
    rows = ((before, 155, list(LABELS)), (after, 490, None))
    for state, top, names in rows:
        for item in state["objects"]:
            index = item["index"]
            left = 35 + index * 293
            draw.rounded_rectangle((left, top, left + 255, top + 270), radius=8,
                                   fill="#151c21", outline="#485462", width=2)
            image = Image.open(io.BytesIO(base64.b64decode(item["image"].split(",", 1)[1]))).convert("RGB")
            image = image.resize((180, 180), Image.Resampling.NEAREST)
            canvas.paste(image, (left + 38, top + 20))
            if names is not None:
                caption = f'Taught: "{names[index]}"'
                detail = f"Selected object {index + 1}"
            else:
                assigned = [r["label"] for r in results if r["observed_index"] == index]
                caption = "Found: " + (", ".join(assigned) if assigned else "no accepted name")
                passed = all(r["correct"] for r in results if r["observed_index"] == index)
                detail = "Verified correct" if assigned and passed else "See recorded results"
            draw.text((left + 14, top + 214), caption, fill="#e0eee7", font=font(18, True))
            draw.text((left + 14, top + 244), detail, fill="#a7b7c4", font=font(13))
    draw.text((35, 449), f"2. Restart and find names in changed views  |  Process {pids[1]}",
              fill="#d4eee0", font=font(19, True))
    draw.text((35, 785), "Saved neural visual embeddings were loaded; no teaching occurred after restart.",
              fill="#c2cdd7", font=font(17))
    draw.text((35, 817), "Names and replies use interface templates. This demonstration does not use the English decoder.",
              fill="#9cabb8", font=font(15))
    DEMO.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(DEMO)


def main():
    started = time.monotonic()
    checkpoint_sha = sha256(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    patterns = checkpoint["test_identities"][:4]
    report = {"schema": "bic-teaching-http-restart-verification-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": "runs/associative/encoder.pt", "checkpoint_sha256": checkpoint_sha,
              "selection": "First four sealed test identities; seed 2026 fixed before this run",
              "policy_patched": False, "model_input": "RGB pixels and explicitly taught symbolic labels",
              "scoring_only_ids": True, "browser_visual_verification": False,
              "browser_note": "Frontend served over real HTTP; no browser access bypass attempted",
              "template_responses": True, "english_decoder_used": False,
              "processes": [], "retrieval": [], "checks": {}}
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="bic-teaching-restart-") as temporary:
            memory = Path(temporary) / "memory.json"
            scene_file = memory.with_name(memory.name + ".scene.json")
            scene_file.write_text(json.dumps({"schema": "bic-teaching-scene-v1", "seed": 2026,
                                              "scene_number": 0, "patterns": patterns}), encoding="utf-8")
            process, url, log = start_server(memory)
            report["processes"].append({"phase": "teaching", "pid": process.pid, "url": url})
            before = success(url, "/state")
            page = success(url, "/")
            script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
            js = subprocess.run(["node", "--check", "-"], input=script, text=True, capture_output=True)
            report["checks"]["frontend_javascript_syntax"] = js.returncode == 0
            report["checks"]["frontend_controls_served"] = all(
                marker in page for marker in ('id="teach-form"', 'id="find-form"', 'id="chat-form"', 'id="new-scene"'))
            report["checks"]["get_state_nonmutating"] = before == success(url, "/state")
            report["checks"]["unknown_endpoint_404"] = request(url, "/unknown")[0] == 404
            report["checks"]["invalid_selection_400"] = request(url, "/select", {"index": True})[0] == 400
            report["checks"]["cross_origin_rejected_403"] = request(
                url, "/teach", {"label": "not_saved"}, {"Origin": "https://example.invalid"})[0] == 403
            teach_patterns = json.loads(scene_file.read_text())["patterns"]
            name_to_pattern = dict(zip(LABELS, teach_patterns))
            for index, label in enumerate(LABELS):
                success(url, "/select", {"index": index})
                result = success(url, "/teach", {"label": label})
                assert result["result"]["label"] == label
            saved_memory = memory.read_bytes()
            report["saved_memory_sha256"] = hashlib.sha256(saved_memory).hexdigest()
            first_pid = process.pid
            stop_server(process)
            report["processes"][0]["stopped_before_restart"] = process.poll() is not None
            process = None

            process, url, log = start_server(memory)
            report["processes"].append({"phase": "recall_only", "pid": process.pid, "url": url})
            after = success(url, "/state")
            recall_patterns = json.loads(scene_file.read_text())["patterns"]
            report["checks"]["distinct_os_processes"] = first_pid != process.pid
            report["checks"]["labels_loaded_without_teaching"] = after["labels"] == list(LABELS)
            report["checks"]["scene_order_changed"] = teach_patterns != recall_patterns
            before_images = {p: before["objects"][i]["image"] for i, p in enumerate(teach_patterns)}
            report["checks"]["all_objects_changed_pixels"] = all(
                before_images[p] != after["objects"][i]["image"] for i, p in enumerate(recall_patterns))
            for label in LABELS:
                response = success(url, "/find", {"label": label})
                result = response["result"]
                index = result["index"]
                actual = recall_patterns[index] if index is not None else None
                report["retrieval"].append({"label": label, "expected_pattern_scoring_only": name_to_pattern[label],
                                            "observed_pattern_scoring_only": actual, "observed_index": index,
                                            "correct": actual == name_to_pattern[label],
                                            "status": result["status"], "cosine_similarity": result["confidence"],
                                            "margin": result["margin"], "reply": response["reply"]})
            unknown = success(url, "/find", {"label": "never_taught"})["result"]
            report["checks"]["unknown_label_abstained"] = unknown["status"] == "unknown_label" and unknown["index"] is None
            success(url, "/select", {"index": 0})
            identified = success(url, "/chat", {"text": "what is this?"})
            expected = next(label for label, p in name_to_pattern.items() if p == recall_patterns[0])
            report["identification"] = {"expected_label": expected, "result": identified["result"],
                                        "correct": identified["result"]["label"] == expected,
                                        "reply": identified["reply"]}
            report["checks"]["memory_bytes_unchanged_after_restart_and_recall"] = memory.read_bytes() == saved_memory
            report["checks"]["checkpoint_bytes_unchanged"] = sha256(CHECKPOINT) == checkpoint_sha
            report["teaching_calls_after_restart"] = 0
            report["recall_count"] = sum(r["correct"] for r in report["retrieval"])
            report["recall_total"] = len(LABELS)
            contact_sheet(before, after, report["retrieval"], [first_pid, process.pid])
            report["contact_sheet"] = "runs/associative/restart_demo.png"
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        if process is not None:
            stop_server(process)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["passed"] = ("error" not in report and all(report["checks"].values())
                            and report.get("recall_count") == 4 and report.get("identification", {}).get("correct", False))
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "recall_correct": report.get("recall_count"),
                          "recall_total": report.get("recall_total"), "report": str(REPORT), "demo": str(DEMO)}))
    if not report["passed"]:
        raise SystemExit("Restart verification recorded a failure; see the unchanged report")


if __name__ == "__main__":
    main()
