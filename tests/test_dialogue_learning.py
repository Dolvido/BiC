"""English learning-controller gates, transaction boundaries, and audit isolation."""

from contextlib import contextmanager, ExitStack, redirect_stdout
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer import dialogue_learning as learning


def tiny_config(**changes):
    options = dict(steps=1, batch_size=1, lessons=2, replay_per_focus=2,
                   dev_per_focus=1, candidates=2)
    options.update(changes)
    return learning.DialogueConfig(**options)


def metrics(accuracy=.5, loss=1.0):
    return {"accuracy": accuracy, "query_accuracy": accuracy, "loss": loss,
            "per_focus": {focus: {"accuracy": accuracy, "query_accuracy": accuracy,
                                    "loss": loss, "reply_exact_accuracy": 0.0}
                          for focus in learning.FOCI}}


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))


@contextmanager
def fake_engine():
    def generate(seed, count, split="train", focus="grounding"):
        return [{"seed": seed + index, "focus": focus, "split": split} for index in range(count)]

    def train(weights, episodes, *, steps, learning_rate, optimizer_state=None, **options):
        if any(episode["split"] != "train" for episode in episodes):
            raise ValueError("held-out leakage")
        updates = steps + (0 if optimizer_state is None else optimizer_state["updates"])
        return {"state_dict": {"weight": weights["weight"].clone() + learning_rate * steps * 10},
                "optimizer_state": {"updates": updates}, "updates": steps,
                "episodes_seen": steps * options["batch_size"], "loss": 1.0,
                "training_seconds": 0.01, "deadline_reached": False}

    def assess(model, seed, count, split="dev", **options):
        progress = float(model.weight.detach().item())
        return metrics(.5 + progress, 1.0 - progress)

    with ExitStack() as stack:
        stack.enter_context(patch.object(learning, "build_dialogue_student", side_effect=lambda *_args, **_kwargs: TinyStudent()))
        stack.enter_context(patch.object(learning, "generate_dialogues", side_effect=generate))
        stack.enter_context(patch.object(learning, "train_dialogue_candidate", side_effect=train))
        stack.enter_context(patch.object(learning, "assess", side_effect=assess))
        stack.enter_context(patch.object(learning, "evaluate_dialogues", return_value={"query_accuracy": .5}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield


class DialogueLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_restart_matches_uninterrupted_weights_optimizer_and_explorer(self):
        with tempfile.TemporaryDirectory() as root, fake_engine():
            full = learning.DialogueLoop(Path(root) / "full", tiny_config())
            split = learning.DialogueLoop(Path(root) / "split", tiny_config())
            full.run(.01, max_cycles=2)
            split.run(.01, max_cycles=1)
            resumed = learning.DialogueLoop(split.directory, resume=True)
            resumed.run(.01, max_cycles=1)
            self.assertEqual(full.state["promotions"], 2)
            self.assert_tree_equal(full.weights, resumed.weights)
            self.assert_tree_equal(full.optimizer, resumed.optimizer)
            self.assert_tree_equal(full.explorer, resumed.explorer)
            for key in ("cycle", "promotions", "updates", "development", "anchors", "replay", "last", "history"):
                self.assert_tree_equal(full.state[key], resumed.state[key])

    def test_retention_rejection_keeps_champion_but_preserves_exploration(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            loop = learning.DialogueLoop(directory, tiny_config())
            initial = copy.deepcopy(loop.weights)
            loop.state["anchors"] = {"grounding": .7}

            def assessed(model, seed, _count, **_options):
                if not float(model.weight.detach().item()):
                    return metrics(.5)
                return metrics(.4 if seed == loop.config.seed + 20_000_000 else .65, .8)

            with patch.object(learning, "assess", side_effect=assessed):
                loop.run(.01, max_cycles=1)
            self.assertEqual(loop.state["promotions"], 0)
            self.assert_tree_equal(loop.weights, initial)
            self.assertIsNotNone(loop.explorer)
            self.assertFalse(torch.equal(loop.explorer["weights"]["weight"], initial["weight"]))
            self.assertTrue(all(item["reason"] == "retention:grounding" for item in loop.state["history"][-1]["candidates"]))

    def test_nonfinite_nested_metrics_cannot_promote_or_poison_checkpoint(self):
        for location in ("development", "retention"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory, fake_engine():
                loop = learning.DialogueLoop(directory, tiny_config())
                loop.state["anchors"] = {"grounding": .6}
                initial = copy.deepcopy(loop.weights)

                def assessed(model, seed, _count, **_options):
                    result = metrics(.65, .8) if float(model.weight.detach().item()) else metrics(.5)
                    is_retention = seed == loop.config.seed + 20_000_000
                    if float(model.weight.detach().item()) and is_retention == (location == "retention"):
                        result["per_focus"]["grounding"]["query_accuracy"] = math.nan
                    return result

                with patch.object(learning, "assess", side_effect=assessed):
                    try:
                        loop.run(.01, max_cycles=1)
                    except ValueError:
                        pass  # Aborting an invalid evaluation is also fail-closed.
                restored = learning.DialogueLoop(directory, resume=True)
                self.assertEqual(restored.state["promotions"], 0)
                self.assert_tree_equal(restored.weights, initial)
                json.dumps(restored.state, allow_nan=False)

    def test_interruption_after_partial_install_rolls_back_every_component(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            loop = learning.DialogueLoop(directory, tiny_config())
            before = copy.deepcopy(loop.snapshot())

            def interrupt(_deadline):
                loop.weights = {"weight": torch.tensor([999.0])}
                loop.optimizer = {"partial": True}
                loop.explorer = {"partial": True}
                raise KeyboardInterrupt

            with patch.object(loop, "cycle", side_effect=interrupt):
                loop.run(.01, max_cycles=1)
            restored = learning.DialogueLoop(directory, resume=True)
            self.assertEqual(restored.state["cycle"], 0)
            self.assertEqual(restored.state["status"], "interrupted")
            for name in ("weights", "optimizer", "explorer"):
                self.assert_tree_equal(getattr(restored, name), before[name])

    def test_stale_writer_and_auditor_cannot_overwrite_or_mislabel_newer_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            current = learning.DialogueLoop(directory, tiny_config())
            stale = learning.DialogueLoop(directory, resume=True)
            current.run(.01, max_cycles=1)
            before = hashlib.sha256(current.path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "advanced"):
                stale.run(.01, max_cycles=1)
            with self.assertRaisesRegex(RuntimeError, "advanced"):
                stale.audit(count=1)
            self.assertEqual(hashlib.sha256(current.path.read_bytes()).hexdigest(), before)

    def test_resume_hash_is_bound_to_the_loaded_bytes_during_concurrent_commit(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            original = learning.DialogueLoop(directory, tiny_config())
            load = torch.load

            def advance_after_loading(*args, **kwargs):
                payload = load(*args, **kwargs)
                changed = copy.deepcopy(payload)
                changed["state"]["cycle"] = 77
                torch.save(changed, original.path)
                return payload

            with patch.object(torch, "load", side_effect=advance_after_loading):
                stale = learning.DialogueLoop(directory, resume=True)
            with self.assertRaisesRegex(RuntimeError, "advanced"):
                stale.run(.01, max_cycles=1)
            self.assertEqual(load(original.path, weights_only=True)["state"]["cycle"], 77)

    def test_audit_is_immutable_and_all_controls_use_audit_split(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            loop = learning.DialogueLoop(directory, tiny_config())
            loop.run(.01, max_cycles=1)
            snapshot = copy.deepcopy(loop.snapshot())
            before = hashlib.sha256(loop.path.read_bytes()).hexdigest()
            calls = []

            def record(model, seed, count, split="dev", **ablations):
                calls.append((split, ablations))
                return metrics()

            with patch.object(learning, "assess", side_effect=record):
                report = loop.audit(count=1)
            self.assertTrue(report["checkpoint_unchanged"])
            self.assertEqual(report["checkpoint_sha256"], before)
            self.assertEqual(hashlib.sha256(loop.path.read_bytes()).hexdigest(), before)
            self.assert_tree_equal(loop.snapshot(), snapshot)
            self.assertEqual(calls, [("audit", {}), ("audit", {"reset_each_turn": True}),
                                     ("audit", {"blank_text": True}), ("audit", {})])
            with self.assertRaises(FileExistsError):
                loop.audit(count=1)

    def test_expired_cycle_and_changed_source_do_not_commit(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            loop = learning.DialogueLoop(directory, tiny_config())
            snapshot = copy.deepcopy(loop.snapshot())
            self.assertIsNone(loop.cycle(time.monotonic() - 1))
            self.assert_tree_equal(snapshot, loop.snapshot())
            before = hashlib.sha256(loop.path.read_bytes()).hexdigest()
            with patch.object(learning, "fingerprint", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    loop.run(.01, max_cycles=1)
            self.assertEqual(hashlib.sha256(loop.path.read_bytes()).hexdigest(), before)

    def test_training_attempt_precedes_help_and_perfect_attempt_needs_no_tutor(self):
        for accuracy in (.5, 1.0):
            with self.subTest(accuracy=accuracy), tempfile.TemporaryDirectory() as directory, fake_engine():
                loop = learning.DialogueLoop(directory, tiny_config(candidates=1))
                events = []
                original_teach = loop.tutor.teach

                def attempt(_model, episodes, **options):
                    self.assertTrue(all(episode["split"] == "train" for episode in episodes))
                    self.assertFalse(options["score_replies"])
                    events.append("attempt")
                    return {"query_accuracy": accuracy}

                def teach(**options):
                    self.assertEqual(events, ["attempt"])
                    events.append("tutor")
                    return original_teach(**options)

                with patch.object(learning, "evaluate_dialogues", side_effect=attempt), \
                        patch.object(loop.tutor, "teach", side_effect=teach) as tutor:
                    loop.run(.01, max_cycles=1)
                self.assertEqual(tutor.call_count, int(accuracy < 1.0))
                self.assertEqual(loop.state["history"][-1]["attempt_query_accuracy"], accuracy)

    def test_replay_and_history_remain_bounded_over_many_cycles(self):
        with tempfile.TemporaryDirectory() as directory, fake_engine():
            loop = learning.DialogueLoop(directory, tiny_config(candidates=1))
            with patch.object(learning, "assess", return_value=metrics()):
                loop.run(.01, max_cycles=70)
            self.assertEqual(loop.state["cycle"], 70)
            self.assertLessEqual(len(loop.state["history"]), 64)
            for focus in learning.FOCI:
                self.assertLessEqual(len(loop.state["replay"][focus]), 2)
                self.assertTrue(all(type(seed) is int for seed in loop.state["replay"][focus]))

    def test_real_byte_language_student_completes_a_tiny_learning_and_audit_cycle(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            loop = learning.DialogueLoop(directory, tiny_config(candidates=1, lessons=1))
            loop.run(.01, max_cycles=1)
            self.assertEqual(loop.state["cycle"], 1)
            self.assertEqual(loop.state["updates"], 1)
            restarted = learning.DialogueLoop(directory, resume=True)
            self.assert_tree_equal(loop.weights, restarted.weights)
            report = restarted.audit(count=1)
            self.assertTrue(report["checkpoint_unchanged"])
            self.assertTrue(math.isfinite(report["student"]["loss"]))
            self.assertIn("query_accuracy", report["blank_english"])


if __name__ == "__main__":
    unittest.main()
