"""Persistence and action-contract tests, using tiny untrained local models."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.adaptation import RuleWorld, split_layouts
from brain_in_computer.adaptive_agent import (
    AdaptiveSession, brain_config, build_model, export_state,
    load_agent, model_digest, save_agent,
)
from brain_in_computer.model import Brain
from experiments.learn_adaptation import check_restart


def tensor_roundtrip(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True, map_location="cpu")


class AdaptiveAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(473)

    def assert_payload_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                with self.subTest(key=key):
                    self.assert_payload_equal(left[key], right[key])
        else:
            self.assertEqual(left, right)

    def test_session_resume_into_fresh_same_weights_model_preserves_future_exactly(self):
        layouts, _ = split_layouts(7001)
        for kind in ("bic", "gru"):
            with self.subTest(kind=kind):
                model = build_model(kind, hidden_size=4)
                original_digest = model_digest(model)
                world = RuleWorld(9241, horizon=14, reversal_step=7, allowed_layouts=layouts)
                session = AdaptiveSession(model)
                for _ in range(5):
                    world.step(session.act(world.observe()))
                saved = tensor_roundtrip(session.snapshot())
                other_model = build_model(kind, hidden_size=4)
                other_model.load_state_dict(model.state_dict(), strict=True)
                resumed = AdaptiveSession(other_model)
                resumed.restore(saved)
                other_world = RuleWorld.from_snapshot(json.loads(json.dumps(world.snapshot())))
                self.assertEqual(resumed.steps, 5)
                self.assert_payload_equal(session.snapshot(), resumed.snapshot())
                while not world.done:
                    action = session.act(world.observe())
                    other_action = resumed.act(other_world.observe())
                    self.assertEqual(action, other_action)
                    self.assertEqual(world.step(action), other_world.step(other_action))
                    self.assert_payload_equal(session.snapshot(), resumed.snapshot())
                self.assertEqual(session.steps, 14)
                self.assertEqual(model_digest(model), original_digest)
                self.assertEqual(model_digest(other_model), original_digest)

    def test_restore_rejects_other_weights_kind_and_same_shape_other_config(self):
        model = build_model("bic", hidden_size=4)
        session = AdaptiveSession(model)
        session.act(RuleWorld(8).observe())
        payload = session.snapshot()
        changed_weights = copy.deepcopy(model)
        with torch.no_grad():
            next(changed_weights.parameters()).add_(.125)
        changed_config = Brain(replace(brain_config(4), memory_slots=3))
        changed_config.load_state_dict(model.state_dict(), strict=True)
        for other in (changed_weights, changed_config, build_model("gru", 4)):
            with self.subTest(model=type(other).__name__), self.assertRaises(ValueError):
                AdaptiveSession(other).restore(payload)

    def test_invalid_step_metadata_and_state_presence_leave_existing_session_unchanged(self):
        for kind in ("bic", "gru"):
            model = build_model(kind, hidden_size=4)
            session = AdaptiveSession(model)
            session.act(RuleWorld(3).observe())
            before = session.snapshot()
            corrupt = [dict(before, steps=value) for value in (-1, True, 1.0, "1", 0)]
            corrupt += [dict(before, schema=999), dict(before, state=None),
                        {"schema": 1, "model_sha256": model_digest(model), "steps": 1, "state": None}]
            for payload in corrupt:
                with self.subTest(kind=kind, steps=payload["steps"]), self.assertRaises(ValueError):
                    session.restore(payload)
                self.assert_payload_equal(before, session.snapshot())
            empty = AdaptiveSession(model)
            empty.restore(tensor_roundtrip(empty.snapshot()))
            self.assertEqual(empty.steps, 0)
            self.assertIsNone(empty.state)

    def test_session_rejects_multi_stream_and_nonfinite_activity(self):
        for kind in ("bic", "gru"):
            model = build_model(kind, hidden_size=4)
            session = AdaptiveSession(model)
            session.act(RuleWorld(13).observe())
            payload = session.snapshot()
            if kind == "bic":
                multiple = model.state_to_dict(model.initial_state(2))
                corrupt = copy.deepcopy(payload["state"])
                corrupt["previous_prefrontal"].fill_(float("nan"))
            else:
                multiple = payload["state"].repeat(1, 2, 1)
                corrupt = torch.full_like(payload["state"], float("nan"))
            for state in (multiple, corrupt):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    session.restore(dict(payload, state=state))

    def test_snapshot_and_restored_activity_have_independent_storage(self):
        for kind in ("bic", "gru"):
            model = build_model(kind, 4)
            session = AdaptiveSession(model)
            session.act(RuleWorld(4).observe())
            saved = session.snapshot()
            untouched = tensor_roundtrip(saved)
            resumed = AdaptiveSession(copy.deepcopy(model))
            resumed.restore(saved)
            if kind == "bic":
                saved["state"]["previous_motor"].fill_(123)
            else:
                saved["state"].fill_(123)
            self.assert_payload_equal(session.snapshot(), untouched)
            self.assert_payload_equal(resumed.snapshot(), untouched)

    def test_category_argmax_maps_to_visible_cell_and_steps_advance_once(self):
        model = build_model("bic", 4)
        for category in range(4):
            logits = torch.full((1, 1, 4), -10.)
            logits[0, 0, category] = 10
            layout = [2, 0, 3, 1]
            observation = {"layout": layout, "previous_category": None, "reward": None}
            session = AdaptiveSession(model)
            with patch.object(model, "forward_with_state", return_value=({"logits": logits}, None)) as forward:
                cell = session.act(observation)
            self.assertEqual(cell, layout.index(category))
            self.assertEqual(session.steps, 1)
            self.assertEqual(forward.call_count, 1)
            encoded = forward.call_args.args[0]
            self.assertEqual(set(encoded), {"visual", "auditory", "body", "feedback", "tokens"})

    def test_saved_agents_load_identical_weights_kind_config_and_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind in ("bic", "gru"):
                model = build_model(kind, 4)
                path = Path(temporary) / f"{kind}.pt"
                save_agent(model, path, seed=19, hidden_size=4, metadata={"purpose": "unit test"})
                restored, data = load_agent(path)
                self.assertEqual(model_digest(model), model_digest(restored))
                self.assertIs(type(model), type(restored))
                self.assertEqual(data["kind"], kind)
                self.assertEqual(data["brain_hidden_size"], 4)
                self.assertEqual(data["metadata"], {"purpose": "unit test"})
                self.assertFalse(restored.training)
                first, second = AdaptiveSession(model), AdaptiveSession(restored)
                world = RuleWorld(791, horizon=5, reversal_step=2)
                while not world.done:
                    observation = world.observe()
                    action = first.act(observation)
                    self.assertEqual(action, second.act(observation))
                    world.step(action)
                self.assert_payload_equal(first.snapshot(), second.snapshot())

    def test_checkpoint_loader_is_strict_about_schema_weights_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.pt"
            model = build_model("bic", 4)
            save_agent(model, path, seed=9, hidden_size=4)
            original = torch.load(path, weights_only=True)
            variants = []
            variants.append(dict(original, schema="unsupported"))
            variants.append(dict(original, brain_hidden_size=5))
            missing = copy.deepcopy(original)
            missing["state_dict"].pop(next(iter(missing["state_dict"])))
            variants.append(missing)
            extra = copy.deepcopy(original)
            extra["state_dict"]["unexpected.weight"] = torch.zeros(1)
            variants.append(extra)
            for index, data in enumerate(variants):
                torch.save(data, path)
                with self.subTest(corruption=index), self.assertRaises((ValueError, RuntimeError)):
                    load_agent(path)

    def test_supplied_restart_check_passes_with_random_models_without_weight_changes(self):
        layouts, _ = split_layouts(7001)
        for kind in ("bic", "gru"):
            model = build_model(kind, 4)
            before = model_digest(model)
            result = check_restart(model, layouts, seed=34001)
            self.assertEqual(result, {"exact_actions_rewards_observations": True,
                                      "exact_final_neural_state": True, "restart_after_steps": 13})
            self.assertEqual(model_digest(model), before)


if __name__ == "__main__":
    unittest.main()
