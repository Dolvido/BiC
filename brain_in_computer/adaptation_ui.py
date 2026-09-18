"""Local browser lab for experimental symbolic feedback adaptation.

Only public observations enter AdaptiveSession. The hidden world checkpoint is
saved on disk for continuity and is never included in a browser response.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
import traceback
from urllib.parse import urlsplit
import uuid
import webbrowser

import torch

from .adaptation import RuleWorld
from .adaptive_agent import AdaptiveSession, load_agent, model_digest
from .computer_use.ui import _integer, _loopback
from .model import Brain

COLORS = ("red", "blue", "green", "yellow")
CELL_NAMES = ("top left", "top right", "bottom left", "bottom right")
SAVE_SCHEMA = "bic-learning-lab-v1"
MAX_BODY_BYTES = 4096


def _json_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON fields are not accepted")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("JSON numbers must be finite")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class LearningLab:
    """A locked world and one recurrent stream; queries never advance either."""

    def __init__(self, model, *, session_path=None, checkpoint_path=None, seed=7007,
                 horizon=40, reversal_step=None):
        self.lock = threading.RLock()
        self.model = model.eval()
        self.agent = AdaptiveSession(model)
        self.horizon = _integer(horizon, "horizon", 2, 200)
        self.reversal_step = reversal_step
        self.seed = _integer(seed, "seed", 0, 2**31 - 1)
        self.world = RuleWorld(self.seed, self.horizon, self.reversal_step)
        self.session_path = Path(session_path).expanduser().resolve() if session_path else None
        self.world_path = (self.session_path.with_name(self.session_path.name + ".world.json")
                           if self.session_path else None)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        if self.session_path and self.session_path.suffix.lower() != ".pt":
            raise ValueError("The saved session path must end in .pt")
        if self.checkpoint_path and self.checkpoint_path in (self.session_path, self.world_path):
            raise ValueError("A session must not overwrite its model checkpoint")
        self.saved = False
        self.notice = "A new episode is ready."
        if self.session_path and (self.session_path.exists() or self.world_path.exists()):
            self.reload({})

    def _history(self):
        # World history contains scoring labels. Select completed, observable
        # outcomes explicitly; never hand the full records to browser or policy.
        return [{"trial": entry["step_index"] + 1, "layout": list(entry["layout"]),
                 "cell": entry["action"], "cell_name": CELL_NAMES[entry["action"]],
                 "category": entry["chosen_category"], "color": COLORS[entry["chosen_category"]],
                 "success": entry["reward"] == 1.0}
                for entry in self.world.export_history()]

    def snapshot(self):
        with self.lock:
            observation = self.world.observe()
            history = self._history()
            return {
                "lab": "BiC Learning Lab",
                "model": "BiC regional network" if isinstance(self.model, Brain) else "GRU reference network",
                "scope": "Experimental symbolic feedback learning; separate from the desktop model.",
                "learning": "Experience changes recurrent memory during this episode; weights stay fixed.",
                "objects": [{"cell": index, "category": category, "color": COLORS[category],
                             "position": CELL_NAMES[index]}
                            for index, category in enumerate(observation["layout"])],
                "completed_trials": len(history), "horizon": self.horizon,
                "done": self.world.done, "successes": sum(row["success"] for row in history),
                "history": history, "last_result": history[-1] if history else None,
                "can_save": self.session_path is not None,
                "has_saved_session": self.saved,
                "notice": self.notice,
            }

    def step(self, body):
        if body:
            raise ValueError("Step does not accept parameters")
        with self.lock:
            if self.world.done:
                raise ValueError("This episode is complete. Start a new episode.")
            observation = self.world.observe()
            action = self.agent.act(observation)
            self.world.step(action)
            self.notice = "BiC received success or failure feedback from its choice."
            return self.snapshot()

    def new(self, body):
        if set(body) - {"seed"}:
            raise ValueError("New episode accepts only an optional seed")
        seed = _integer(body.get("seed", secrets.randbits(31)), "seed", 0, 2**31 - 1)
        with self.lock:
            self.world = RuleWorld(seed, self.horizon, self.reversal_step)
            self.agent = AdaptiveSession(self.model)
            self.seed = seed
            self.notice = "New episode. The earlier episode's memory has been reset."
            return self.snapshot()

    def save(self, body):
        if body:
            raise ValueError("Save does not accept parameters")
        with self.lock:
            if self.session_path is None:
                raise ValueError("Start the lab with --session to enable saved sessions")
            agent = self.agent.snapshot()
            world = self.world.snapshot()
            if agent["steps"] != world["step_index"]:
                raise RuntimeError("World and neural session action counts differ")
            document = {
                "schema": SAVE_SCHEMA, "save_id": uuid.uuid4().hex,
                "model_sha256": agent["model_sha256"],
                "configuration": {"horizon": self.horizon, "reversal_step": self.reversal_step},
                "world": world,
            }
            raw = _json_bytes(document)
            payload = {"schema": SAVE_SCHEMA, "save_id": document["save_id"],
                       "world_sha256": hashlib.sha256(raw).hexdigest(), "agent": agent}
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            state_tmp = self.session_path.with_name(self.session_path.name + "." + token + ".tmp")
            world_tmp = self.world_path.with_name(self.world_path.name + "." + token + ".tmp")
            try:
                world_tmp.write_bytes(raw)
                torch.save(payload, state_tmp)
                # Hash-linked pair: an interrupted two-file replacement is
                # detected on reload, rather than silently resuming wrong state.
                world_tmp.replace(self.world_path)
                state_tmp.replace(self.session_path)
            finally:
                state_tmp.unlink(missing_ok=True)
                world_tmp.unlink(missing_ok=True)
            self.saved = True
            unit = "trial" if agent["steps"] == 1 else "trials"
            self.notice = f"Saved after {agent['steps']} {unit}. Both world and memory will resume."
            return self.snapshot()

    def reload(self, body):
        if body:
            raise ValueError("Reload does not accept parameters")
        with self.lock:
            if not self.session_path or not self.session_path.is_file() or not self.world_path.is_file():
                raise ValueError("No complete saved session was found. Save this episode first.")
            if self.session_path.stat().st_size > 8 * 1024 * 1024 or self.world_path.stat().st_size > 1024 * 1024:
                raise ValueError("The saved session exceeds the size limit")
            raw = self.world_path.read_bytes()
            document = _json_loads(raw.decode("utf-8"))
            payload = torch.load(self.session_path, map_location="cpu", weights_only=True)
            if (not isinstance(document, dict) or set(document) !=
                    {"schema", "save_id", "model_sha256", "configuration", "world"}
                    or document["schema"] != SAVE_SCHEMA
                    or not isinstance(payload, dict) or set(payload) !=
                    {"schema", "save_id", "world_sha256", "agent"} or payload["schema"] != SAVE_SCHEMA
                    or payload["save_id"] != document["save_id"]
                    or payload["world_sha256"] != hashlib.sha256(raw).hexdigest()):
                raise ValueError("Saved world and memory do not form a matching session")
            if document["model_sha256"] != model_digest(self.model):
                raise ValueError("This saved session belongs to a different model")
            config = document["configuration"]
            if not isinstance(config, dict) or set(config) != {"horizon", "reversal_step"}:
                raise ValueError("Invalid saved session configuration")
            horizon = _integer(config["horizon"], "saved horizon", 2, 200)
            # Validate configuration and bound history before deterministic replay.
            RuleWorld(0, horizon, config["reversal_step"])
            saved_world = document["world"]
            if (not isinstance(saved_world, dict) or saved_world.get("horizon") != horizon
                    or not isinstance(saved_world.get("history"), list)
                    or len(saved_world["history"]) > horizon
                    or (config["reversal_step"] is not None
                        and saved_world.get("reversal_step") != config["reversal_step"])):
                raise ValueError("Invalid saved world configuration or action count")
            restored_world = RuleWorld.from_snapshot(saved_world)
            snapshot = payload["agent"]
            if (not isinstance(snapshot, dict) or set(snapshot) != {"schema", "model_sha256", "steps", "state"}
                    or snapshot.get("steps") != saved_world["step_index"]
                    or (snapshot["state"] is None) != (saved_world["step_index"] == 0)):
                raise ValueError("Saved neural memory and world are at different trials")
            restored_agent = AdaptiveSession(self.model)
            restored_agent.restore(snapshot)
            # Commit only after every component has passed validation.
            self.world, self.agent = restored_world, restored_agent
            self.horizon, self.reversal_step = horizon, config["reversal_step"]
            self.seed = saved_world["seed"]
            self.saved = True
            unit = "trial" if self.agent.steps == 1 else "trials"
            self.notice = f"Resumed the saved episode after {self.agent.steps} {unit}."
            return self.snapshot()


def create_server(lab, host="127.0.0.1", port=8767):
    if not isinstance(host, str) or not _loopback(host):
        raise ValueError("The learning lab accepts only loopback hosts")
    _integer(port, "port", 0, 65535)
    page = Path(__file__).with_name("adaptation_viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "BiC-Learning-Lab"

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format, *args):
            pass

        def send_bytes(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; "
                             "frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status, value):
            self.send_bytes(status, _json_bytes(value), "application/json; charset=utf-8")

        def valid_origin(self):
            host_header = self.headers.get("Host", "")
            try:
                parsed = urlsplit("//" + host_header)
                valid = (len(self.headers.get_all("Host", [])) == 1
                         and parsed.hostname and _loopback(parsed.hostname)
                         and parsed.port == self.server.server_address[1]
                         and parsed.username is None and parsed.password is None
                         and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                valid = False
            if not valid:
                self.send_json(403, {"error": "Use this lab's exact localhost address and port"})
                return False
            origins = self.headers.get_all("Origin", [])
            if len(origins) > 1 or (origins and origins[0] != "http://" + host_header):
                self.send_json(403, {"error": "Cross-origin requests are not accepted"})
                return False
            return True

        def do_GET(self):
            if not self.valid_origin():
                return
            if self.path == "/":
                self.send_bytes(200, page, "text/html; charset=utf-8")
            elif self.path == "/state":
                self.send_json(200, lab.snapshot())
            elif self.path == "/favicon.ico":
                self.send_bytes(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "Unknown learning lab endpoint"})

        def do_POST(self):
            if not self.valid_origin():
                return
            routes = {"/step": lab.step, "/new": lab.new, "/save": lab.save, "/reload": lab.reload}
            if self.path not in routes:
                self.send_json(404, {"error": "Unknown learning lab endpoint"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Use Content-Type: application/json"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                self.send_json(411, {"error": "Content-Length is required"})
                return
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                self.send_json(400, {"error": "Use one Content-Length and an ordinary JSON body"})
                return
            try:
                length = int(lengths[0])
                if not 0 <= length <= MAX_BODY_BYTES:
                    self.send_json(413 if length > MAX_BODY_BYTES else 400,
                                   {"error": "Request body must be at most 4096 bytes"})
                    return
            except ValueError:
                self.send_json(400, {"error": "Invalid Content-Length"})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = _json_loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object")
                self.send_json(200, routes[self.path](body))
            except (ValueError, TypeError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                traceback.print_exc()
                self.send_json(500, {"error": "Action failed; see the lab terminal for details"})

        def do_OPTIONS(self):
            self.send_json(405, {"error": "Use GET or POST on the learning lab endpoints"})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--session", type=Path, default=Path("runs/learning-session.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--seed", type=int, default=7007)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not _loopback(args.host):
            raise ValueError("The learning lab accepts only loopback hosts")
        torch.set_num_threads(1)
        model, _ = load_agent(args.checkpoint, device=args.device)
        lab = LearningLab(model, session_path=args.session, checkpoint_path=args.checkpoint, seed=args.seed)
        server = create_server(lab, args.host, args.port)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    host, port = server.server_address[:2]
    address = f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
    print(f"BiC learning lab: {address}", flush=True)
    print("Experimental symbolic feedback adaptation. Weights stay fixed during use. Ctrl+C closes the lab.", flush=True)
    if args.open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
