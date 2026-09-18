"""Exercise the local HTTP UI with an actual saved computer-use policy.

    python experiments/check_computer_ui.py --checkpoint runs/computer/checkpoint.pt \
        --output experiments/computer_ui_verification.json

Assertions cover transport, response structure, and direct human desktop input.
Model replies and actions are recorded without asserting that the model follows
every instruction. No teacher, goal generator, or oracle is called.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch

from brain_in_computer.computer_use.environment import ACTION_NAMES, MiniDesktop
from brain_in_computer.computer_use.training import load_computer_checkpoint, run_turn
from brain_in_computer.computer_use.ui import LabSession, create_server
from brain_in_computer.training import atomic_json


def require(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_summary(payload):
    """Verify the public response and omit PNG bytes from the report."""
    allowed = {"reply", "actions", "image", "state", "terminated", "screen_size"}
    require(not set(payload) - allowed, "Response exposes fields outside the visible-state API")
    require(set(payload["state"]) == {"text", "last_clicked"}, "State exposes unexpected fields")
    require(isinstance(payload["state"]["text"], str), "Visible text is not a string")
    require(type(payload["terminated"]) is bool, "Termination is not a boolean")
    prefix = "data:image/png;base64,"
    require(payload["image"].startswith(prefix), "Screenshot is not a PNG data URL")
    png = base64.b64decode(payload["image"][len(prefix):], validate=True)
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        dimensions = {"width": image.width, "height": image.height}
        require(image.format == "PNG" and image.size == (96, 96), "Screenshot has unexpected format or dimensions")
    require(payload["screen_size"] == dimensions, "Reported screen dimensions differ from the PNG")
    summary = {key: value for key, value in payload.items() if key != "image"}
    summary["image"] = {"format": "PNG", **dimensions, "bytes": len(png),
                        "sha256": hashlib.sha256(png).hexdigest()}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/computer_ui_verification.json"))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = {
        "kind": "local_http_integration_with_saved_policy",
        "interpretation": "Transport and UI-state assertions are separate from model capability. Probe replies/actions are recorded as observed, including any mistakes.",
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_digest(args.checkpoint)},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "threads": torch.get_num_threads()},
        "teacher_or_oracle_called_by_this_script": False,
        "structural_checks": [],
        "capability_probes": [],
        "manual_interactions": [],
        "browser_visual_verification": {"completed": False,
                                        "limitation": "Cloud browser access to the loopback UI was blocked with ERR_BLOCKED_BY_CLIENT; these HTTP checks do not verify visual rendering."},
    }
    model = load_computer_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    session = LabSession(model, MiniDesktop, run_turn, ACTION_NAMES, seed=7)
    server = create_server(session, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    # Do not let environment proxy configuration redirect these local checks.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None, headers=None):
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request_headers = {"Content-Type": "application/json"} if encoded is not None else {}
        request_headers.update(headers or {})
        req = urllib.request.Request(base + path, data=encoded, headers=request_headers)
        try:
            response = opener.open(req, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read()
            content_type = response.headers.get_content_type()
            payload = json.loads(raw) if content_type == "application/json" else raw.decode("utf-8")
            return response.status, content_type, payload

    started = time.perf_counter()
    failure = None
    try:
        status, kind, page = request("/")
        require(status == 200 and kind == "text/html", "GET / did not serve HTML successfully")
        require("BiC computer-use lab" in page and "Small trained lab model; limited English" in page,
                "The lab title or capability limitation is missing from HTML")
        report["structural_checks"].append({"endpoint": "GET /", "status": status, "result": "passed"})

        status, _, payload = request("/state")
        require(status == 200, "GET /state failed")
        report["initial_state"] = response_summary(payload)
        report["structural_checks"].append({"endpoint": "GET /state", "status": status, "result": "passed"})

        for prompt in ("hello", "click red", "what did you click", "type hi"):
            turn_started = time.perf_counter()
            status, _, payload = request("/chat", {"text": prompt})
            require(status == 200, f"POST /chat failed for {prompt!r}: {payload}")
            require(isinstance(payload.get("reply"), str), "Model reply is not text")
            require(isinstance(payload.get("actions"), list)
                    and all(isinstance(action, str) for action in payload["actions"]),
                    "Model action names are not a list of strings")
            report["capability_probes"].append({"prompt": prompt, "http_status": status,
                                                "seconds": time.perf_counter() - turn_started,
                                                "observed": response_summary(payload)})
        report["structural_checks"].append({"endpoint": "POST /chat", "turns": 4, "result": "passed"})

        status, _, payload = request("/reset", {"seed": 23})
        require(status == 200 and payload["state"]["text"] == "", "Reset did not clear visible text")
        report["manual_interactions"].append({"event": "reset seed 23", "observed": response_summary(payload)})
        for event in ({"kind": "click", "x": 32, "y": 83}, {"kind": "key", "key": "ok"}):
            status, _, payload = request("/event", event)
            require(status == 200, f"Manual event failed: {event}")
            report["manual_interactions"].append({"event": event, "observed": response_summary(payload)})
        require(payload["state"]["text"] == "ok", "Human focus/type input did not update the visible field")
        report["structural_checks"].append({"endpoint": "POST /reset and /event", "result": "passed"})

        for path, body in (("/chat", {"text": ""}),
                           ("/event", {"kind": "click", "x": 96, "y": 0}),
                           ("/event", {"kind": "key", "key": "z"})):
            status, _, payload = request(path, body)
            require(status == 400 and isinstance(payload.get("error"), str), "Invalid input was not rejected with a JSON 400 error")
            report["structural_checks"].append({"endpoint": f"POST {path}", "invalid_input": body,
                                                "status": status, "error": payload["error"], "result": "passed"})
        status, _, payload = request("/chat", {"text": "hello"}, {"Origin": "https://example.invalid"})
        require(status == 403 and isinstance(payload.get("error"), str), "Cross-origin request was not rejected")
        report["structural_checks"].append({"endpoint": "POST /chat", "origin": "https://example.invalid",
                                            "status": status, "result": "passed"})
        report["structural_result"] = "passed"
    except Exception as error:
        failure = error
        report["structural_result"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(args.output, report)
    print(f"HTTP integration: {report['structural_result']}; saved {args.output}", flush=True)
    for probe in report["capability_probes"]:
        observed = probe["observed"]
        print(json.dumps({"prompt": probe["prompt"], "reply": observed["reply"],
                          "actions": observed["actions"], "visible_state": observed["state"]}), flush=True)
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    sys.exit(main())
