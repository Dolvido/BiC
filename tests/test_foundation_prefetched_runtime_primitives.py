"""Five bounded primitive checks, never the CUDA proof or a learner.

The nested roundtrip uses10 CPU tensor elements plus10 decoded elements. The
metadata test allocates21 fixture elements; its sole noncontiguous digest adds
at most4 contiguous temporary elements. Other tests allocate no tensors. There
are no model, data/index, optimizer, gradient, CUDA or historical-input calls.
"""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from experiments import foundation_prefetched_runtime_probe as probe


CPU_TENSOR_ELEMENT_BOUNDS = {
    "nested_roundtrip": 20,
    "tensor_metadata_and_bytes": 25,
    "scalar_and_container_types": 0,
    "journal_success": 0,
    "journal_failures_and_deadline": 0,
}


class PrefetchedRuntimePrimitiveTests(unittest.TestCase):
    def assert_distinct(self, left, right):
        self.assertFalse(probe._same(left, right))
        self.assertFalse(probe._same(right, left))
        self.assertNotEqual(probe._digest(left), probe._digest(right))

    def journal(self, directory):
        started = time.monotonic()
        return probe._Journal(Path(directory), started, time.process_time(), started+30.)

    def test_exact_nested_cpu_tree_weights_only_byte_roundtrip(self):
        tree = dict(main=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            nested=([torch.tensor([True, False], dtype=torch.bool)],
                    {0: torch.tensor([0., -0.], dtype=torch.float64)}),
            metadata=[True, 1, None, "fixture", .125])
        image = io.BytesIO()
        torch.save(tree, image)
        restored = torch.load(io.BytesIO(image.getvalue()), map_location="cpu", weights_only=True)
        self.assertTrue(probe._same(tree, restored))
        self.assertEqual(probe._digest(tree), probe._digest(restored))
        # Mapping insertion order is not a semantic difference; typed contents are.
        reordered = dict(reversed(list(restored.items())))
        self.assertTrue(probe._same(tree, reordered))
        self.assertEqual(probe._digest(tree), probe._digest(reordered))

    def test_tensor_bytes_dtype_shape_stride_offset_and_grad_flag_are_distinct(self):
        base = torch.zeros((2, 2), dtype=torch.float32)  #4 elements
        content = base.clone()  #4
        content[0, 0] = 1.
        wider = base.to(dtype=torch.float64)  #4
        storage = torch.zeros(5, dtype=torch.float32)  #5
        offset = storage.as_strided((2, 2), (2, 1), storage_offset=1)
        negative_zero = base.clone()  #4; total21 fixture elements
        negative_zero[0, 0] = -0.
        cases = dict(content=content, dtype=wider, shape=base.reshape(4), stride=base.t(),
            storage_offset=offset, requires_grad=base.detach().requires_grad_(True), signed_zero=negative_zero)
        self.assertTrue(torch.equal(base, negative_zero))
        self.assertTrue(torch.equal(base, cases["stride"]))
        self.assertTrue(torch.equal(base, offset))
        for name, changed in cases.items():
            with self.subTest(difference=name):
                self.assert_distinct(base, changed)

    def test_scalar_container_and_dictionary_key_types_remain_distinct(self):
        for left, right in ((True, 1), (1, 1.), ([1, True], (1, True)), (0., -0.),
                            ({True: "same"}, {1: "same"}), ({0.: "same"}, {0: "same"}),
                            ({0.: "same"}, {-0.: "same"})):
            with self.subTest(left=repr(left), right=repr(right)):
                self.assert_distinct(left, right)

    def test_journal_flushes_intent_before_callback_and_chains_completion(self):
        with tempfile.TemporaryDirectory(prefix="bic-prefetch-journal-success-") as directory:
            journal = self.journal(directory)
            fsync = probe.os.fsync
            with patch.object(probe.os, "fsync", wraps=fsync) as sync:
                def operation():
                    lines = journal.path.read_bytes().splitlines(keepends=True)
                    self.assertEqual(len(lines), 1)
                    self.assertEqual(json.loads(lines[0])["event"], "intent")
                    self.assertEqual(sync.call_count, 1)
                    self.assertEqual(journal.active["kind"], "fixture_operation")
                    return {"value": 7}
                result = journal.perform("fixture_operation", operation, context={"case": "small"},
                    summarize=lambda value: {"returned": value["value"]})
                self.assertEqual(sync.call_count, 2)
            self.assertEqual(result, {"value": 7})
            lines = journal.path.read_bytes().splitlines(keepends=True)
            self.assertEqual(len(lines), 2)
            intent, completion = map(json.loads, lines)
            self.assertEqual([intent["sequence"], completion["sequence"]], [0, 1])
            self.assertIsNone(intent["previous_event_sha256"])
            self.assertEqual(completion["previous_event_sha256"], hashlib.sha256(lines[0]).hexdigest())
            self.assertEqual([intent["event"], completion["event"]], ["intent", "completed"])
            self.assertEqual(intent["operation"], completion["operation"])
            self.assertEqual(intent["context"], completion["context"])
            self.assertEqual(completion["details"], {"returned": 7})
            self.assertEqual(journal.previous, hashlib.sha256(lines[1]).hexdigest())
            self.assertEqual(journal.sequence, 2)
            self.assertEqual(journal.counts["fixture_operation"], dict(attempted=1, completed=1, failed=0))
            self.assertIsNone(journal.active)
            self.assertFalse(journal.unknown)
            self.assertTrue(all(value == 0 for value in journal.work.values()))

    def test_journal_io_failures_stay_unknown_and_expired_deadline_never_calls_work(self):
        for failure in ("append", "flush", "completion_fsync"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="bic-prefetch-journal-failure-") as directory:
                journal = self.journal(directory)
                operation = Mock(return_value={"small": True})
                if failure == "append":
                    injection = patch.object(Path, "open", side_effect=OSError("injected append failure"))
                elif failure == "flush":
                    handle = MagicMock()
                    handle.__enter__.return_value = handle
                    handle.write.side_effect = lambda image: len(image)
                    handle.flush.side_effect = OSError("injected flush failure")
                    injection = patch.object(Path, "open", return_value=handle)
                else:
                    injection = patch.object(probe.os, "fsync",
                        side_effect=[None, OSError("injected completion fsync failure"), None])
                with injection, self.assertRaises(OSError):
                    journal.perform("fixture_operation", operation)
                self.assertTrue(journal.unknown)
                if failure == "completion_fsync":
                    operation.assert_called_once_with()
                    self.assertEqual(journal.counts["fixture_operation"], dict(attempted=1, completed=1, failed=1))
                    # A successful later failure append must never clear the
                    # uncertainty of the earlier failed completion flush.
                    self.assertIsNone(journal.active)
                else:
                    operation.assert_not_called()
                    self.assertEqual(journal.sequence, 0)
                    self.assertEqual(journal.counts["fixture_operation"], dict(attempted=0, completed=0, failed=0))
        with tempfile.TemporaryDirectory(prefix="bic-prefetch-journal-expired-") as directory:
            journal = self.journal(directory)
            journal.deadline = time.monotonic()-1.
            operation = Mock()
            with self.assertRaises(TimeoutError):
                journal.perform("must_not_start", operation)
            operation.assert_not_called()
            self.assertEqual(journal.path.read_bytes(), b"")
            self.assertEqual(journal.counts, {})
            self.assertEqual(journal.sequence, 0)
            self.assertIsNone(journal.active)
            self.assertFalse(journal.unknown)


if __name__ == "__main__":
    unittest.main()
