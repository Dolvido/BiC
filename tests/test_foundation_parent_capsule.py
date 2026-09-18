"""Synthetic codec/tensor checks only; these fixtures prove no completed work.

Actual captured-token export/load belongs to the separate repeated-cycle test.
No forward pass, backward pass, optimizer update, or model scoring occurs here.
"""
import copy
import hashlib
import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from experiments import foundation_parent_capsule as capsule


class UnsupportedPayload:
    pass


class CapsuleCodecTests(unittest.TestCase):
    def setUp(self):
        self.parts = {name: b"synthetic-no-completion-proof" for name in capsule.MEMBERS if name != "manifest.json"}
        self.raw = capsule._encode(self.parts)
        self.sha = hashlib.sha256(self.raw).hexdigest()

    def test_synthetic_codec_roundtrip_not_completion(self):
        self.assertEqual(capsule._decode(self.raw, self.sha, capsule.MAX_BYTES), self.parts)

    def test_pin_and_limit_must_be_external_valid_inputs(self):
        for digest, limit in (("0" * 64, capsule.MAX_BYTES), (self.sha, len(self.raw) - 1),
                              (None, capsule.MAX_BYTES), (self.sha, True)):
            with self.subTest(digest=digest, limit=limit), self.assertRaises(ValueError):
                capsule._decode(self.raw, digest, limit)

    def test_rejects_unlisted_duplicate_or_compressed_members(self):
        for names, compression in ((capsule.MEMBERS + ("../outside",), zipfile.ZIP_STORED),
                                   (capsule.MEMBERS[:-1] + ("metadata.json",), zipfile.ZIP_STORED),
                                   (capsule.MEMBERS, zipfile.ZIP_DEFLATED)):
            with self.subTest(names=names, compression=compression):
                out = io.BytesIO()
                with zipfile.ZipFile(out, "w", compression=compression) as archive:
                    for name in names:
                        archive.writestr(name, b"synthetic")
                raw = out.getvalue()
                with self.assertRaisesRegex(ValueError, "exact uncompressed"):
                    capsule._decode(raw, hashlib.sha256(raw).hexdigest(), capsule.MAX_BYTES)

    def test_internal_manifest_hash_tamper_rejected_even_if_repinned(self):
        out = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(self.raw)) as old, zipfile.ZipFile(out, "w") as new:
            for name in capsule.MEMBERS:
                new.writestr(name, b"changed" if name == "learner.pt" else old.read(name))
        raw = out.getvalue()
        with self.assertRaisesRegex(ValueError, "member hashes"):
            capsule._decode(raw, hashlib.sha256(raw).hexdigest(), capsule.MAX_BYTES)

    def test_json_duplicate_nonfinite_and_noncanonical_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{ "a": 1 }'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                capsule._json(raw)

    def test_export_refuses_arbitrary_receipt_or_checkpoint(self):
        for value in ({"status": "completed"}, object(), None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "exact process-captured"):
                capsule.export_parent_capsule(value, "unused-capsule-path")

    def _load_synthetic(self, parts, digest=None):
        raw = capsule._encode(parts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.capsule"
            path.write_bytes(raw)
            return capsule.load_parent_capsule(path, expected_sha256=digest or hashlib.sha256(raw).hexdigest(),
                                                evaluation_banks={})

    def test_wrong_pin_precedes_tensor_deserialization(self):
        with patch.object(capsule.torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "caller pin"):
                self._load_synthetic(self.parts, "0" * 64)
            loader.assert_not_called()

    def test_source_mismatch_precedes_tensor_deserialization(self):
        self.parts["sources.json"] = b"{}"
        with patch.object(capsule.torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "current source closure"):
                self._load_synthetic(self.parts)
            loader.assert_not_called()

    def test_runtime_mismatch_precedes_tensor_deserialization(self):
        self.parts["sources.json"] = capsule.cycles._encoded(capsule.cycles.source_hashes())
        self.parts["metadata.json"] = capsule.cycles._encoded({"provider_identity": {"runtime": {}}, "device": "cpu"})
        with patch.object(capsule.torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "runtime"):
                self._load_synthetic(self.parts)
            loader.assert_not_called()

    def test_unsafe_pickle_globals_are_not_allowed(self):
        self.parts["sources.json"] = capsule.cycles._encoded(capsule.cycles.source_hashes())
        self.parts["metadata.json"] = capsule.cycles._encoded({
            "provider_identity": {"runtime": capsule.providers.runtime_identity("cpu")}, "device": "cpu"})
        image = io.BytesIO()
        torch.save(UnsupportedPayload(), image)
        self.parts["learner.pt"] = image.getvalue()
        with self.assertRaises(pickle.UnpicklingError):
            self._load_synthetic(self.parts)


class CapsuleTensorTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.learner = dict(weights=dict(self.model.state_dict()), optimizer={
            "param_groups": self.optimizer.state_dict()["param_groups"],
            "state": {index: {"step": torch.tensor(7.), "exp_avg": torch.zeros_like(parameter),
                               "exp_avg_sq": torch.zeros_like(parameter)}
                      for index, parameter in enumerate(self.model.parameters())}})

    def validate(self, learner=None):
        capsule._validate_tensor_state(learner or self.learner, self.model, self.optimizer, 7)

    def test_synthetic_tensor_invariants_do_not_prove_training(self):
        self.validate()

    def test_optimizer_lifetime_moment_shape_and_nonnegative_checks(self):
        mutations = (
            lambda value: value["optimizer"]["state"][0].update(step=torch.tensor(6.)),
            lambda value: value["optimizer"]["state"][0].update(exp_avg=torch.zeros(99)),
            lambda value: value["optimizer"]["state"][0].update(exp_avg_sq=-torch.ones_like(self.model.weight)),
            lambda value: value["optimizer"]["state"].pop(0),
            lambda value: value["optimizer"]["param_groups"][0].update(lr=.002),
        )
        for change in mutations:
            value = copy.deepcopy(self.learner)
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(value)

    def test_weights_dtype_shape_finite_and_grad_boundaries(self):
        for weight in (torch.zeros(99), self.model.weight.detach().double(),
                       torch.full_like(self.model.weight, float("nan")),
                       self.model.weight.detach().clone().requires_grad_(True)):
            value = copy.deepcopy(self.learner)
            value["weights"]["weight"] = weight
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.validate(value)

    def test_tied_weights_must_agree(self):
        self.model.alias = self.model.weight
        value = copy.deepcopy(self.learner)
        value["weights"]["alias"] = torch.ones_like(self.model.weight) * 100
        with self.assertRaisesRegex(ValueError, "tied weights"):
            self.validate(value)

    def test_reference_count_comparison_uses_metric_direction(self):
        best = dict(correct=3, total=4, rate=.75)
        lower = dict(correct=2, total=4, rate=.5)
        self.assertTrue(capsule._reference_not_worse(best, lower, "paired_action"))
        self.assertFalse(capsule._reference_not_worse(lower, best, "paired_action"))
        fewer = dict(count=1, total=4, rate=.25)
        more = dict(count=2, total=4, rate=.5)
        self.assertTrue(capsule._reference_not_worse(fewer, more, "unsupported_ask"))
        self.assertFalse(capsule._reference_not_worse(more, fewer, "unsupported_ask"))


if __name__ == "__main__":
    unittest.main()
