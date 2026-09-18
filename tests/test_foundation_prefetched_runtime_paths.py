"""Bounded real Windows publication checks; no model, data or CUDA work."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import torch

from experiments import foundation_prefetched_runtime_probe as probe


OBSERVATIONS = []


@unittest.skipUnless(os.name == "nt", "real Windows namespace checks")
class PrefetchedRuntimePathTests(unittest.TestCase):
    def fixture(self, name, *, long=False):
        root = os.environ.get("BIC_PREFETCH_PATH_FIXTURES")
        if root is None:
            root = tempfile.mkdtemp(prefix="bic-path-check-")
        ordinary = Path(root).absolute()/name
        if long:
            ordinary = ordinary/("nested-"+"x"*80)/("nested-"+"y"*80)
        native = probe._native_path(ordinary)
        native.mkdir(parents=True, exist_ok=False)
        self.assertEqual(probe._native_path(native), native)
        if long:
            self.assertGreater(len(str(ordinary)), 260)
        return ordinary, native

    def assert_immutable(self, path, image):
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            probe._publish(path, image+b"rejected overwrite")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(before, image)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_uuid_suffix_limit_reproduced_then_identical_bytes_published(self):
        ordinary, native = self.fixture("legacy")
        stem_length = 252-len(str(ordinary))-1-len(".bin")
        self.assertGreater(stem_length, 0)
        target = ordinary/("a"*stem_length+".bin")
        self.assertEqual(len(str(target)), 252)
        image = b"immutable path regression fixture\n"
        # Same old helper and filesystem, without changing machine settings.
        with self.assertRaises(FileNotFoundError):
            probe._publish_image(target, image)
        self.assertFalse((native/target.name).exists())
        result = probe._publish(target, image)
        self.assertEqual(result, hashlib.sha256(image).hexdigest())
        self.assert_immutable(native/target.name, image)
        OBSERVATIONS.append(dict(case="legacy_uuid_suffix", destination_chars=len(str(target)),
            temporary_chars=len(str(target))+37, old_publication_refused=True,
            wrapped_publication_passed=True, bytes=len(image), sha256=result))

    def test_long_source_copy_hash_relative_identity_and_overwrite_refusal(self):
        ordinary, native = self.fixture("source", long=True)
        relative = "sources/experiments/foundation_prefetched_runtime_probe.py"
        target = ordinary/relative
        probe._native_path(target.parent).mkdir(parents=True)
        image = (probe.ROOT/"experiments/foundation_prefetched_runtime_probe.py").read_bytes()
        result = probe._publish(target, image)
        native_target = native/relative
        self.assertEqual(result, hashlib.sha256(image).hexdigest())
        self.assertEqual(native_target.relative_to(native).as_posix(), relative)
        inventory = {path.relative_to(native).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in native.rglob("*") if path.is_file()}
        self.assertEqual(inventory, {relative: result})
        self.assert_immutable(native_target, image)
        OBSERVATIONS.append(dict(case="source_copy", destination_chars=len(str(target)),
            bytes=len(image), relative_name=relative, sha256=result))

    def test_long_json_and_torch_archive_preserve_images_and_refuse_overwrite(self):
        ordinary, native = self.fixture("serialized", long=True)
        payload = {"schema": "path-only-fixture-v1", "nested": [{"count": 7}, True, None, -.0]}
        json_image = probe._bytes(payload)+b"\n"
        json_digest = probe._json(ordinary/"receipt.json", payload)
        self.assertEqual(json_digest, hashlib.sha256(json_image).hexdigest())
        self.assertEqual(json.loads((native/"receipt.json").read_bytes()), payload)
        with self.assertRaises(FileExistsError):
            probe._json(ordinary/"receipt.json", {"refused": True})
        self.assertEqual((native/"receipt.json").read_bytes(), json_image)
        # The production checkpoint path serializes into BytesIO, then _publish.
        # This fixture contains only scalar/container data, with zero tensors.
        stream = io.BytesIO()
        torch.save(payload, stream)
        archive_image = stream.getvalue()
        archive_digest = probe._publish(ordinary/"midpoint.pt", archive_image)
        restored = torch.load(io.BytesIO((native/"midpoint.pt").read_bytes()),
                              map_location="cpu", weights_only=True)
        self.assertTrue(probe._same(payload, restored))
        self.assert_immutable(native/"midpoint.pt", archive_image)
        self.assertEqual((native/"midpoint.pt").relative_to(native).as_posix(), "midpoint.pt")
        OBSERVATIONS.append(dict(case="json_and_scalar_archive", destination_chars=len(str(ordinary/"midpoint.pt")),
            json_bytes=len(json_image), json_sha256=json_digest,
            archive_bytes=len(archive_image), archive_sha256=archive_digest, fixture_tensor_elements=0))

    def test_long_journal_append_chain_and_native_namespace_idempotence(self):
        ordinary, native = self.fixture("journal", long=True)
        started = time.monotonic()
        journal = probe._Journal(ordinary, started, time.process_time(), started+30.)
        journal.perform("scalar_fixture", lambda: 7, summarize=lambda value: {"value": value})
        lines = (native/"operations.jsonl").read_bytes().splitlines(keepends=True)
        self.assertEqual(len(lines), 2)
        first, last = map(json.loads, lines)
        self.assertEqual(first["event"], "intent")
        self.assertEqual(last["event"], "completed")
        self.assertEqual(last["previous_event_sha256"], hashlib.sha256(lines[0]).hexdigest())
        self.assertEqual(journal.path.relative_to(native).as_posix(), "operations.jsonl")
        self.assertFalse(journal.unknown)
        self.assertTrue(all(value == 0 for value in journal.work.values()))
        self.assertEqual(str(probe._native_path(Path(r"\\server\share\folder\file"))),
                         r"\\?\UNC\server\share\folder\file")
        OBSERVATIONS.append(dict(case="journal", destination_chars=len(str(ordinary/"operations.jsonl")),
            durable_events=2, native_namespace_idempotent=True, unc_spelling_only=True))


if __name__ == "__main__":
    unittest.main()
