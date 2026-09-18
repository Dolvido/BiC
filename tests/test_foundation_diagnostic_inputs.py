"""Synthetic boundary checks only, never historical authentication/completion proof.

One fresh tiny CPU template, two tiny safe inventory deserializations, no forward,
backward, optimizer, real checkpoint load, or historical/formal inventory load.
Private synthetic handles and mocked loads exercise reporting/refusal plumbing.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments import foundation_diagnostic_inputs as inputs


def setUpModule():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


class DiagnosticInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.inventory_loads = 0
        cls.model = build_sequence_student(831906799, device="cpu",
            config=SequenceConfig(width=8, layers=1, heads=2, feedforward=16, max_turns=12))
        cls.weights = {name: value.detach().clone() for name, value in cls.model.state_dict().items()}
        cls.digest = checkpoint_digest(cls.model)
        cls.forward_guard = cls.model.register_forward_pre_hook(
            lambda *args: (_ for _ in ()).throw(AssertionError("fixture permits no model forward")))

    @classmethod
    def tearDownClass(cls):
        cls.forward_guard.remove()
        value = dict(schema="bic-diagnostic-inputs-cpu-proof-v1", model_constructions=1,
            model_seed=831906799, configuration=vars(cls.model.config),
            inventory_deserializations=cls.inventory_loads, checkpoint_deserializations=0,
            encoder_episodes=0, decoder_row_steps=0, backward_evaluations=0, optimizer_updates=0,
            historical_or_formal_inputs=False, synthetic_public_loads_are_mocked=True,
            wall_seconds=time.monotonic() - cls.started,
            within_approved_bounds=cls.inventory_loads == 2)
        print("DIAGNOSTIC_INPUTS_CPU_WORK=" + json.dumps(value, sort_keys=True))
        path = os.environ.get("BIC_DIAGNOSTIC_INPUTS_ACCOUNTING")
        if path:
            with Path(path).open("x", encoding="utf8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
        if not value["within_approved_bounds"]:
            raise AssertionError("two-inventory CPU fixture bound differs")

    def tearDown(self):
        self.assertEqual(checkpoint_digest(self.model), self.digest)
        self.assertIs(self.model.tokens.weight, self.model.observation_head.weight)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def owner(self, root, *, image=b"synthetic image", digest=None):
        selected = dict(inputs.STATES["initial"], path="fixture.pt",
            file_sha256=hashlib.sha256(image).hexdigest(), weights_sha256=digest or self.digest)
        metadata = dict(source_sha256={}, runtime={}, states={"initial": selected})
        return inputs.AuthenticatedInputs(inputs._KEY, root, metadata, ("a" * 64,))

    def test_authenticated_bytes_and_canonical_path_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "small.bin").write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            self.assertEqual(inputs._read_bound(root, "small.bin", digest), b"original")
            for name in ("../small.bin", "./small.bin", "x\\small.bin", "/small.bin"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    inputs._read_bound(root, name, digest)
            with self.assertRaises(ValueError):
                inputs._read_bound(root, "small.bin", digest, maximum=3)
            (root / "small.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "byte mismatch"):
                inputs._read_bound(root, "small.bin", digest)
            with self.assertRaises(ValueError):
                inputs._read_bound(root, "small.bin", digest.upper())

    def test_two_safe_inventory_images_and_ordering(self):
        values = ["1" * 64, "a" * 64]
        for rows, valid in ((values, True), (list(reversed(values)), False)):
            buffer = io.BytesIO()
            torch.save(rows, buffer)
            type(self).inventory_loads += 1
            if valid:
                self.assertEqual(inputs._decode_inventory(buffer.getvalue(),
                    semantic_sha256=inputs._hash(values), count=2), (values, inputs._hash(values)))
            else:
                with self.assertRaisesRegex(ValueError, "sorted unique"):
                    inputs._decode_inventory(buffer.getvalue())

    def test_exact_finite_weights_and_tied_copy_consistency(self):
        self.assertEqual(inputs._load_weights(self.model, self.weights, self.digest), self.digest)
        for kind in ("missing", "extra", "shape", "dtype", "nan", "gradient", "tied", "digest"):
            weights = copy.deepcopy(self.weights)
            name = "tokens.weight"
            if kind == "missing":
                weights.pop(name)
            elif kind == "extra":
                weights["not_a_weight"] = weights[name]
            elif kind == "shape":
                weights[name] = weights[name][:-1]
            elif kind == "dtype":
                weights[name] = weights[name].double()
            elif kind == "nan":
                weights[name][0, 0] = float("nan")
            elif kind == "gradient":
                weights[name].requires_grad_(True)
            elif kind == "tied":
                weights[name][0, 0] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs._load_weights(self.model, weights, "0" * 64 if kind == "digest" else self.digest)

    def test_exact_checkpoint_metadata_without_checkpoint_load(self):
        selected = inputs.STATES["initial"]
        history = dict(execution_profile={"synthetic": True}, trainer_source_sha256={"example.py": "1" * 64},
            plan_sha256="2" * 64, protected_sha256="3" * 64, protected_count=2)
        recipe = dict(schema="bic-foundation-trainer-v1", curriculum_version="bic-shared-foundation-v1",
            source_sha256=history["trainer_source_sha256"], plan_sha256=history["plan_sha256"],
            order="curriculum", seed=6101, config=inputs.CONFIG,
            protected_transcripts_sha256=history["protected_sha256"], protected_transcripts_count=2,
            micro_batch_size=32, family_order=["color", "count", "switch"], learning_rate=.001,
            optimizer=dict(name="AdamW", param_groups=[]),
            objective="mean of three unchanged sequence_objective microbatches", gradient_clip=1.)
        saved = dict(schema="bic-foundation-order-pilot-v2", arm="curriculum", protocol_sha256=inputs.PROTOCOL_SHA256,
            weights_sha256=selected["weights_sha256"], execution_profile=history["execution_profile"],
            learner=dict(schema="bic-foundation-trainer-v1", recipe=recipe, weights={"opaque": "not loaded"},
                optimizer={"opaque": "not restored"}, cursor=0, evidence={"cursor": 0}, timing={}))
        self.assertEqual(inputs._envelope(saved, selected, {"historical": history}), saved["learner"]["weights"])
        for kind in ("extra", "arm", "cursor", "boolean_cursor", "boolean_evidence", "config", "recipe", "weight_pin"):
            malformed = copy.deepcopy(saved)
            if kind == "extra": malformed["unexpected"] = True
            elif kind == "arm": malformed["arm"] = "mixed"
            elif kind == "cursor": malformed["learner"]["cursor"] = 1
            elif kind == "boolean_cursor": malformed["learner"]["cursor"] = False
            elif kind == "boolean_evidence": malformed["learner"]["evidence"]["cursor"] = False
            elif kind == "config": malformed["learner"]["recipe"]["config"]["width"] = 8
            elif kind == "recipe": malformed["learner"]["recipe"]["source_sha256"] = {}
            elif kind == "weight_pin": malformed["weights_sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inputs._envelope(malformed, selected, {"historical": history})

    def test_wrong_pin_refuses_before_deserialization_and_no_second_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            owner = self.owner(root)
            (root / "fixture.pt").write_bytes(b"different")
            with patch.object(inputs, "_root", return_value=root), patch.object(inputs, "_guard"), \
                    patch.object(inputs.torch, "load") as load, patch.object(inputs, "build_sequence_student") as build:
                with self.assertRaisesRegex(ValueError, "byte mismatch") as caught:
                    inputs.load_state(root, "initial", owner)
                self.assertEqual(caught.exception.input_report["checkpoint_loads_started"], 0)
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    inputs.load_state(root, "initial", owner)
                load.assert_not_called()
                build.assert_not_called()

    def test_interrupted_load_keeps_original_type_and_partial_accounting(self):
        root = inputs._SELF.parents[1]
        owner = self.owner(root)
        with patch.object(inputs, "_guard"), patch.object(inputs, "_read_bound", return_value=b"synthetic image"), \
                patch.object(inputs.torch, "load", side_effect=KeyboardInterrupt("fixture")) as load, \
                patch.object(inputs, "build_sequence_student") as build:
            with self.assertRaises(KeyboardInterrupt) as caught:
                inputs.load_state(root, "initial", owner)
            report = caught.exception.input_report
            self.assertEqual((report["checkpoint_byte_images"], report["checkpoint_loads_started"],
                report["checkpoint_loads_completed"]), (1, 1, 0))
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(load.call_args.kwargs, dict(map_location="cpu", weights_only=True))
            build.assert_not_called()
            self.assertEqual(owner.load_reports[-1], report)

    def test_mocked_public_load_records_digest_and_detached_reports(self):
        root = inputs._SELF.parents[1]
        owner = self.owner(root)
        with patch.object(inputs, "_guard"), patch.object(inputs, "_read_bound", return_value=b"synthetic image"), \
                patch.object(inputs.torch, "load", return_value={}) as load, \
                patch.object(inputs, "_envelope", return_value=self.weights), \
                patch.object(inputs, "build_sequence_student", return_value=self.model) as build:
            self.assertIs(inputs.load_state(root, "initial", owner), self.model)
            self.assertEqual(load.call_args.args[0].getvalue(), b"synthetic image")
            self.assertEqual(load.call_args.kwargs, dict(map_location="cpu", weights_only=True))
            self.assertEqual(build.call_count, 1)  # The constructor is mocked; no second template exists.
        report = owner.load_reports[-1]
        self.assertEqual(report["loaded_weights_sha256"], self.digest)
        self.assertEqual(report["optimizer_restores"], 0)
        report["status"] = "changed"
        self.assertEqual(owner.load_reports[-1]["status"], "completed")
        detached = owner.metadata
        detached["states"].clear()
        self.assertIn("initial", owner.metadata["states"])
        self.assertEqual(owner.exclusions, frozenset(("a" * 64,)))
        with self.assertRaises(TypeError): pickle.dumps(owner)

    def test_runtime_and_import_resolution_without_neural_work(self):
        runtime = inputs._runtime()
        self.assertEqual(runtime["solver_sha256"], inputs.SOLVER_SHA256)
        with patch.object(inputs.torch, "get_num_threads", return_value=2), self.assertRaises(ValueError):
            inputs._runtime()
        root = inputs._SELF.parents[1]
        names = ["experiments/foundation_diagnostic_inputs.py", "brain_in_computer/__init__.py"]
        self.assertEqual(len(inputs._imports(root, names)), 2)
        module = sys.modules["experiments.foundation_diagnostic_inputs"]
        with patch.object(module, "__file__", str(root / "wrong.py")), self.assertRaisesRegex(ValueError, "another source"):
            inputs._imports(root, names)
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            inputs._root(folder)


if __name__ == "__main__":
    unittest.main()
