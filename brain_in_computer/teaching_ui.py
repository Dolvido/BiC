"""Local teaching interface for saved associations and optional regional actions.

Teaching and identification use explicit interface templates. In regional mode,
quoted commands use the regional policy and its neural byte decoder. Object
identities drive rendering only; inference receives image crops and taught names.
"""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import random
import socket
import threading
import traceback
from urllib.parse import urlsplit

import torch
from PIL import Image

from .associative import (
    AssociativeMemory, distinct_patterns, encoder_hash, load_calibration,
    load_encoder_checkpoint, render_glyph,
)
from .computer_use.ui import _integer, _loopback

MAX_BODY_BYTES = 4096
SCENE_SCHEMA = "bic-teaching-scene-v1"


def _image_url(pixels):
    array = pixels.detach().cpu().permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().numpy()
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class LabSession:
    """An encoder, persistent association bank, and four rendered object crops.

    A separate scene file remembers which objects to render after a restart.
    Pattern IDs never enter ``teach``, ``find``, ``name``, or encoder inputs.
    Restarting changes the objects' positions, scale, color, and translation.
    """

    def __init__(self, encoder, memory_file, threshold, margin, seed=7, regional_policy=None):
        self.encoder = encoder.eval()
        self.regional_policy = regional_policy
        if regional_policy is not None and encoder_hash(regional_policy.encoder) != encoder_hash(encoder):
            raise ValueError("The regional policy and memory must use the same visual encoder")
        self.memory_file = Path(memory_file)
        self.scene_file = self.memory_file.with_name(self.memory_file.name + ".scene.json")
        self.lock = threading.RLock()
        self.memory = (AssociativeMemory.load(self.memory_file, encoder)
                       if self.memory_file.exists()
                       else AssociativeMemory(encoder, threshold=threshold, margin=margin))
        self.seed = _integer(seed, "seed", 0, 2**32 - 1)
        self.scene_number = 0
        self.patterns = distinct_patterns(4, self.seed)
        if self.scene_file.exists():
            if self.scene_file.stat().st_size > 4096:
                raise ValueError("The saved teaching scene is too large")
            scene = json.loads(self.scene_file.read_text(encoding="utf-8"))
            if not isinstance(scene, dict) or scene.get("schema") != SCENE_SCHEMA:
                raise ValueError("Unsupported teaching scene format")
            patterns = scene.get("patterns")
            if (not isinstance(patterns, list) or len(patterns) != 4
                    or any(type(p) is not int or not 0 < p < 65536 for p in patterns)
                    or len(set(patterns)) != 4):
                raise ValueError("The teaching scene must contain four distinct object patterns")
            self.seed = _integer(scene.get("seed"), "saved seed", 0, 2**32 - 1)
            self.scene_number = _integer(scene.get("scene_number"), "saved scene number", 0, 2**31 - 2)
            self.patterns = patterns
        self.selected = 0
        self.result = None
        self.response_source = "interface_template"
        self._new_scene(False)

    def _save_scene(self):
        self.scene_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.scene_file.with_name(self.scene_file.name + ".tmp")
        temporary.write_text(json.dumps({"schema": SCENE_SCHEMA, "seed": self.seed,
                                        "scene_number": self.scene_number,
                                        "patterns": self.patterns}), encoding="utf-8")
        temporary.replace(self.scene_file)

    def _new_scene(self, new_objects):
        self.scene_number += 1
        rng = random.Random(self.seed + self.scene_number * 104729)
        if new_objects:
            self.patterns = distinct_patterns(4, rng.randrange(2**32), exclude=self.patterns)
        rng.shuffle(self.patterns)
        self.pixels = [render_glyph(p, seed=rng.randrange(2**32), variation=True)
                       for p in self.patterns]
        self.images = [_image_url(p) for p in self.pixels]
        self.selected = 0
        self.result = None
        self.response_source = "interface_template"
        self._save_scene()

    def snapshot(self):
        """Read-only: repeated GET requests do not change scenes or memory."""
        with self.lock:
            return {"objects": [{"index": i, "image": img} for i, img in enumerate(self.images)],
                    "selected": self.selected, "scene_number": self.scene_number,
                    "labels": self.memory.labels, "capacity": self.memory.capacity,
                    "examples_per_label": self.memory.examples_per_label,
                    "memory_saved": self.memory_file.exists(),
                    "threshold": self.memory.threshold, "margin": self.memory.margin,
                    "result": self.result,
                    "response_source": self.response_source,
                    "action_source": ("regional_motor_cortex" if self.regional_policy is not None
                                      else "visual_matcher"),
                    "learning_mode": ("regional_policy_with_persistent_visual_evidence"
                                      if self.regional_policy is not None
                                      else "fixed_neural_encoder_and_persistent_exemplars")}

    def _reply(self, text, result=None, response_source="interface_template"):
        self.result = result
        self.response_source = response_source
        return {**self.snapshot(), "reply": text}

    def select(self, body):
        index = _integer(body.get("index"), "index", 0, 3)
        with self.lock:
            self.selected = index
            self.result = None
            return self.snapshot()

    def teach(self, body):
        with self.lock:
            result = self.memory.teach(body.get("label"), self.pixels[self.selected])
            self.memory.save(self.memory_file)
            return self._reply(f'Saved "{result["label"]}" with {result["examples"]} visual '
                               f'example(s). Change the scene and try finding it.',
                               {"kind": "teach", **result})

    def find(self, body):
        label = body.get("label")
        if self.regional_policy is not None:
            if not isinstance(label, str):
                raise ValueError("Enter a saved name as text")
            return self.regional_command("find " + json.dumps(label, ensure_ascii=False))
        with self.lock:
            result = self.memory.find(label, self.pixels)
            if result["index"] is not None:
                self.selected = result["index"]
                text = f'Visual match for "{label}": object {self.selected + 1}.'
            elif result["status"] == "unknown_label":
                text = f'I have no saved association for "{label}". Select an object and teach its name.'
            elif result["status"] == "ambiguous":
                text = "The visual matches are too similar to choose reliably. Teach another example."
            else:
                text = "No object in this scene matched strongly enough."
            return self._reply(text, {"kind": "find", "query": label, **result})

    def regional_command(self, instruction):
        """Send pixels and memory evidence through the learned regional policy.

        No scene identities, desired relation, target location, or matcher-selected
        index is supplied to the agent. The grid highlight follows its motor action.
        """
        with self.lock:
            result = self.regional_policy.act(self.memory, instruction, self.pixels)
            index = result["index"]
            if type(index) is int and 0 <= index <= 3:
                self.selected = index
                status = "selected"
            else:
                status = "stopped" if result["stopped"] else "unsupported_action"
            trace = {"memory_cosine_similarities": result["scores"],
                     "name_in_memory": result["known_label"],
                     "instruction_to_network": result["normalized_instruction"],
                     "regional_activity_norms": result["region_activity"]}
            return self._reply(result["reply"], {"kind": "regional", "query": instruction,
                                                **result, "status": status, "trace": trace},
                               response_source=result["response_source"])

    def identify(self, body):
        with self.lock:
            result = self.memory.name(self.pixels[self.selected])
            if result["label"] is not None:
                text = f'Object {self.selected + 1} visually matches "{result["label"]}".'
            elif result["status"] == "empty_memory":
                text = "No names have been taught yet. Select an object and teach its name."
            elif result["status"] == "ambiguous":
                text = "More than one saved name matches this object. I cannot choose reliably."
            else:
                text = "This object does not match a saved name strongly enough."
            return self._reply(text, {"kind": "identify", "action_source": "visual_matcher", **result})

    def scene(self, body):
        new_objects = body.get("new_objects", False)
        if type(new_objects) is not bool:
            raise ValueError("new_objects must be true or false")
        with self.lock:
            self._new_scene(new_objects)
            return self._reply("Four new objects are ready. Saved names are retained."
                               if new_objects else "The same objects have new positions and appearances.")

    def forget(self, body):
        with self.lock:
            removed = self.memory.forget(body.get("label"))
            if removed:
                self.memory.save(self.memory_file)
            return self._reply("Association removed and memory saved."
                               if removed else "That name was not in memory.")

    def chat(self, body):
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 160:
            raise ValueError("Enter a command between 1 and 160 characters")
        text = text.strip()
        lower = text.lower()
        if lower.startswith("remember "):
            return self.teach({"label": text[9:].strip()})
        if self.regional_policy is not None and '"' in text:
            return self.regional_command(text)
        if lower.startswith("find "):
            return self.find({"label": text[5:].strip()})
        if lower.rstrip("?") == "what is this":
            return self.identify({})
        with self.lock:
            if self.regional_policy is not None:
                return self._reply('Try find "dax" or select the object left of "dax". '
                                   'Use a quoted saved name for regional commands. '
                                   'Remember <name> and what is this? use the teaching interface.')
            return self._reply("Supported commands: remember <name>, find <name>, and what is this? "
                               "These commands use interface templates and the visual matcher.")


def create_server(session, host="127.0.0.1", port=8766):
    """Create the loopback server without starting its event loop."""
    if not isinstance(host, str) or not _loopback(host):
        raise ValueError("The teaching lab accepts only loopback hosts")
    _integer(port, "port", 0, 65535)
    page = Path(__file__).with_name("teaching_viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "BiC-Teaching-Lab"

        def log_message(self, format, *args):
            return

        def send_bytes(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                             "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                             "connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, data):
            self.send_bytes(status, json.dumps(data, allow_nan=False).encode("utf-8"),
                            "application/json; charset=utf-8")

        def valid_origin(self):
            host_header = self.headers.get("Host", "")
            try:
                parsed = urlsplit("//" + host_header)
                valid = (parsed.hostname and _loopback(parsed.hostname)
                         and parsed.port == self.server.server_address[1]
                         and parsed.username is None and parsed.password is None
                         and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                valid = False
            if not valid:
                self.send_json(403, {"error": "Use the teaching lab's exact localhost address and port"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host_header:
                self.send_json(403, {"error": "Cross-origin requests are not accepted"})
                return False
            return True

        def do_GET(self):
            if not self.valid_origin():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send_bytes(200, page, "text/html; charset=utf-8")
            elif path == "/state":
                self.send_json(200, session.snapshot())
            elif path == "/favicon.ico":
                self.send_bytes(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "Unknown teaching lab endpoint"})

        def do_POST(self):
            if not self.valid_origin():
                return
            routes = {"/select": session.select, "/teach": session.teach,
                      "/find": session.find, "/identify": session.identify,
                      "/scene": session.scene, "/forget": session.forget, "/chat": session.chat}
            path = urlsplit(self.path).path
            if path not in routes:
                self.send_json(404, {"error": "Unknown teaching lab endpoint"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Send JSON with Content-Type: application/json"})
                return
            length = self.headers.get("Content-Length")
            if length is None:
                self.send_json(411, {"error": "Content-Length is required"})
                return
            try:
                length = int(length)
                if length < 0:
                    raise ValueError
            except ValueError:
                self.send_json(400, {"error": "Invalid Content-Length"})
                return
            if length > MAX_BODY_BYTES:
                self.send_json(413, {"error": "Request body exceeds 4096 bytes"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("The request body must be a JSON object")
                self.send_json(200, routes[path](body))
            except (ValueError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                traceback.print_exc()
                self.send_json(500, {"error": "Could not complete this action; see the server terminal"})

        def do_OPTIONS(self):
            self.send_json(405, {"error": "Use GET or POST on the teaching lab endpoints"})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), Handler)


def serve(checkpoint, memory_file="runs/teaching/memory.json", host="127.0.0.1", port=8766,
          device="cpu", seed=7, regional_checkpoint=None):
    """Serve an existing trained visual encoder and durable, user-taught memory."""
    if not isinstance(host, str) or not _loopback(host):
        raise ValueError("The teaching lab accepts only loopback hosts")
    torch.set_num_threads(1)
    encoder = load_encoder_checkpoint(checkpoint, device=device)
    calibration = load_calibration(checkpoint)
    regional_policy = None
    if regional_checkpoint is not None:
        from .regional_memory import load_regional_checkpoint
        regional_policy = load_regional_checkpoint(regional_checkpoint, device=device)
    session = LabSession(encoder, memory_file, calibration["threshold"], calibration["margin"], seed,
                         regional_policy=regional_policy)
    server = create_server(session, host=host, port=port)
    printable_host = f"[{host}]" if ":" in host else host
    print(f"BiC teaching lab: http://{printable_host}:{server.server_address[1]}", flush=True)
    print("Teach saves visual associations. "
          + ("Quoted commands use the regional policy and neural replies. " if regional_policy is not None
             else "Command replies use interface templates. ")
          + "Teaching and identification use interface text. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
