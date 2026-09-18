"""Actual local HTTP and persistent neural-session integration checks."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import torch

from brain_in_computer.adaptation import OBSERVATION_KEYS
from brain_in_computer.adaptive_agent import build_model
from brain_in_computer.adaptation_ui import LearningLab, create_server


class AdaptationUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(707)
        self.model = build_model("bic", hidden_size=8)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "session.pt"
        self.lab = LearningLab(self.model, session_path=self.path, seed=73, horizon=8, reversal_step=4)
        self.server = create_server(self.lab, port=0)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.close_server)
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])
        self.opener = build_opener(ProxyHandler({}))

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)

    def request(self, path, body=None, headers=None, raw=None):
        payload = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        request_headers = {"Content-Type": "application/json"} if payload is not None else {}
        request_headers.update(headers or {})
        request = Request(self.base + path, data=payload, headers=request_headers)
        try:
            response = self.opener.open(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            content = json.loads(data) if response.headers.get_content_type() == "application/json" else data
            return response.status, content

    def test_get_is_read_only_and_hidden_world_never_enters_public_state(self):
        before = self.lab.world.snapshot()
        status, state = self.request("/state")
        self.assertEqual(status, 200)
        self.assertEqual(state, self.request("/state")[1])
        self.assertEqual(before, self.lab.world.snapshot())
        self.assertEqual(self.lab.agent.steps, 0)
        serialized = json.dumps(state)
        for hidden in ("target_category", "initial_target", "reversed_target", "reversal_step", "rng_state"):
            self.assertNotIn(hidden, serialized)
        self.assertEqual(len(state["objects"]), 4)
        self.assertEqual(self.request("/")[0], 200)
        self.assertFalse(self.path.exists())

    def test_action_loop_sends_only_observations_and_finishes_episode(self):
        with patch.object(self.lab.agent, "act", wraps=self.lab.agent.act) as act:
            for trial in range(1, 9):
                status, state = self.request("/step", {})
                self.assertEqual(status, 200)
                self.assertEqual(state["completed_trials"], trial)
                self.assertEqual(self.lab.agent.steps, trial)
                self.assertEqual(set(act.call_args.args[0]), OBSERVATION_KEYS)
                self.assertEqual(state["successes"], sum(row["success"] for row in state["history"]))
                self.assertEqual(state["last_result"]["trial"], trial)
        self.assertTrue(state["done"])
        self.assertEqual(self.request("/step", {})[0], 400)
        self.assertEqual(self.request("/new", {"seed": 74})[1]["completed_trials"], 0)
        self.assertIsNone(self.lab.agent.state)

    def test_save_restart_and_reload_preserve_exact_future_choices(self):
        for _ in range(3):
            self.lab.step({})
        self.assertEqual(self.request("/save", {})[0], 200)
        self.assertTrue(self.path.exists())
        self.assertTrue(self.lab.world_path.exists())
        reloaded = LearningLab(copy.deepcopy(self.model), session_path=self.path)
        self.assertEqual(reloaded.world.snapshot(), self.lab.world.snapshot())
        for _ in range(3):
            original, resumed = self.lab.step({}), reloaded.step({})
            self.assertEqual(original["history"], resumed["history"])
            self.assertEqual(original["objects"], resumed["objects"])
        status, state = self.request("/reload", {})
        self.assertEqual(status, 200)
        self.assertEqual(state["completed_trials"], 3)
        self.assertEqual(self.lab.agent.steps, 3)

    def test_save_pair_corruption_fails_without_mutating_live_session(self):
        self.lab.step({})
        self.lab.save({})
        before = self.lab.world.snapshot()
        world = json.loads(self.lab.world_path.read_text())
        world["world"]["layout"].reverse()
        self.lab.world_path.write_text(json.dumps(world))
        self.assertEqual(self.request("/reload", {})[0], 400)
        self.assertEqual(before, self.lab.world.snapshot())
        self.assertEqual(self.lab.agent.steps, 1)

    def test_model_mismatch_rejected_and_checkpoint_path_protected(self):
        self.lab.save({})
        different = copy.deepcopy(self.model)
        with torch.no_grad():
            next(different.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "different model"):
            LearningLab(different, session_path=self.path)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            LearningLab(self.model, session_path=self.path, checkpoint_path=self.path)

    def test_strict_http_rejects_cross_origin_wrong_host_and_bad_json(self):
        before = self.lab.world.snapshot()
        self.assertEqual(self.request("/step", {}, {"Origin": "https://example.invalid"})[0], 403)
        self.assertEqual(self.request("/step", {}, {"Host": "127.0.0.1:1"})[0], 403)
        self.assertEqual(self.request("/step", {}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("/step", raw=b'{"seed":1,"seed":2}')[0], 400)
        self.assertEqual(self.request("/step", raw=b'{"x":NaN}')[0], 400)
        self.assertEqual(self.request("/step", raw=b"[]" )[0], 400)
        self.assertEqual(self.request("/step", raw=b" " * 4097)[0], 413)
        self.assertEqual(self.request("/step", {"unexpected": 1})[0], 400)
        self.assertEqual(before, self.lab.world.snapshot())
        self.assertEqual(self.lab.agent.steps, 0)
        with self.assertRaises(ValueError):
            create_server(self.lab, host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()

