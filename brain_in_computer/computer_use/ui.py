"""Local browser interface for the trained computer-use policy.

This server calls the actual policy and simulated desktop. It never calls an
expert/teacher, trains a model, downloads weights, or controls the real OS.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
from pathlib import Path
import socket
import threading
import traceback
from urllib.parse import urlsplit


MAX_BODY_BYTES = 8192
SCREEN_SIZE = 96


def _loopback(host):
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


class LabSession:
    """Shared desktop and policy; a lock makes each turn an atomic UI action."""

    def __init__(self, model, environment_factory, run_turn, action_names=(), seed=7):
        self.model = model
        self.environment_factory = environment_factory
        self.run_turn = run_turn
        self.action_names = action_names
        self.env = environment_factory(seed=seed)
        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            frame = self.env.render()
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG")
            # Deliberately do not serialize the complete environment state:
            # only human-visible text and interaction history belong in the UI.
            clicked = getattr(self.env, "last_clicked", None)
            if clicked is not None and not isinstance(clicked, (str, int, float, bool, list, tuple)):
                clicked = str(clicked)
            return {
                "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "state": {"text": str(getattr(self.env, "text", "")), "last_clicked": clicked},
                "terminated": bool(getattr(self.env, "terminated", False)),
                "screen_size": {"width": frame.width, "height": frame.height},
            }

    def action_name(self, action):
        try:
            if type(action) is int:
                name = self.action_names[action]
                return str(name)
        except (IndexError, KeyError, TypeError):
            pass
        return str(action)

    def chat(self, body):
        prompt = body.get("text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a nonempty message in the text field")
        if len(prompt) > 512:
            raise ValueError("Messages must be at most 512 characters for this small lab model")
        with self.lock:
            result = self.run_turn(self.model, self.env, prompt.strip(), max_actions=8)
            if not isinstance(result, dict) or not isinstance(result.get("reply"), str):
                raise RuntimeError("The model turn did not return a text reply")
            # Re-render the environment after the policy acts. PIL frames and
            # training/evaluation metadata in result are never exposed as JSON.
            return {**self.snapshot(), "reply": result["reply"],
                    "actions": [self.action_name(a) for a in result.get("actions", ())],
                    "terminated": bool(result.get("terminated", False))}

    def reset(self, body):
        with self.lock:
            if "seed" in body:
                seed = _integer(body["seed"], "seed", 0, 2**32 - 1)
                self.env = self.environment_factory(seed=seed)
            else:
                self.env.reset()
            return self.snapshot()

    def event(self, body):
        kind = body.get("kind")
        if kind == "click":
            x = _integer(body.get("x"), "x", 0, SCREEN_SIZE - 1)
            y = _integer(body.get("y"), "y", 0, SCREEN_SIZE - 1)
        elif kind == "key":
            key = body.get("key")
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("key must contain between 1 and 128 characters")
            keys = ["backspace"] if key.lower() == "backspace" else list(key)
            if any(character not in ("h", "i", "o", "k", "backspace") for character in keys):
                raise ValueError("This desktop supports h, i, o, k, and Backspace; the text field holds two characters")
        else:
            raise ValueError("event kind must be 'click' or 'key'")
        with self.lock:
            if getattr(self.env, "terminated", False):
                self.env.begin_turn()
            if kind == "click":
                self.env.click(x, y)
            else:
                for character in keys:
                    if getattr(self.env, "terminated", False):
                        self.env.begin_turn()
                    self.env.key(character)
            return self.snapshot()


def create_server(session, host="127.0.0.1", port=8765):
    """Build the server without starting it; useful for integration checks."""
    if not isinstance(host, str) or not _loopback(host):
        raise ValueError("The computer-use lab can bind only to a loopback host (127.0.0.1 or localhost)")
    _integer(port, "port", 0, 65535)
    page = Path(__file__).with_name("viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "BiC-Lab"

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
            self.send_bytes(status, json.dumps(data, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def valid_origin(self):
            host_header = self.headers.get("Host", "")
            try:
                hostname = urlsplit("//" + host_header).hostname
            except ValueError:
                hostname = None
            if not hostname or not _loopback(hostname):
                self.send_json(403, {"error": "Use the lab through its localhost address"})
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
                try:
                    self.send_json(200, session.snapshot())
                except Exception:
                    traceback.print_exc()
                    self.send_json(500, {"error": "Could not render the desktop; see the server terminal"})
            elif path == "/favicon.ico":
                self.send_bytes(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "Unknown lab endpoint"})

        def do_POST(self):
            if not self.valid_origin():
                return
            path = urlsplit(self.path).path
            if path not in ("/chat", "/reset", "/event"):
                self.send_json(404, {"error": "Unknown lab endpoint"})
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
                self.send_json(413, {"error": "Request body exceeds 8192 bytes"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("The request body must be a JSON object")
                result = {"/chat": session.chat, "/reset": session.reset, "/event": session.event}[path](body)
                self.send_json(200, result)
            except (ValueError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                traceback.print_exc()
                self.send_json(500, {"error": "The model could not complete this action; see the server terminal"})

        def do_OPTIONS(self):
            self.send_json(405, {"error": "Use GET or POST on the documented lab endpoints"})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), Handler)


def serve(checkpoint, host="127.0.0.1", port=8765, device="cpu"):
    """Load an existing local checkpoint and serve the interactive lab."""
    if not isinstance(host, str) or not _loopback(host):
        raise ValueError("The computer-use lab accepts only loopback hosts")
    from . import environment
    from .training import load_computer_checkpoint, run_turn

    model = load_computer_checkpoint(checkpoint, device=device)
    model.eval()
    session = LabSession(model, environment.MiniDesktop, run_turn,
                         action_names=getattr(environment, "ACTION_NAMES", ()))
    server = create_server(session, host=host, port=port)
    bound_port = server.server_address[1]
    printable_host = f"[{host}]" if ":" in host else host
    print(f"BiC computer-use lab: http://{printable_host}:{bound_port}", flush=True)
    print("Using the saved local policy. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
