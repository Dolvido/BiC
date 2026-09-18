"""Small, caller-pinned byte fixtures; no model, tensor, or archived-code loads."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.foundation_historical_archive import ExpectedEvidence, _receipt_destination, file_hash, verify_archive


class HistoricalArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.study = self.root / "runs/pilot"
        self.study.mkdir(parents=True)
        self.sources = {"experiments/old.py": b"old source\n", "docs/old.md": b"old protocol note\n"}
        for name, content in self.sources.items():
            self.write(self.root / name, content)
            self.write(self.study / "source" / name, content)
        self.write(self.study / "checkpoint.pt", b"opaque bytes; must never be deserialized")
        self.contract = dict(schema="bic-foundation-order-pilot-v2", version="fixture-v1",
                             arms=["curriculum", "mixed"], checkpoints=[0, 2], updates=2,
                             automatic_promotion=False, evaluation_during_training=False)
        self.protocol = dict(contract=self.contract, source_sha256={
            name: hashlib.sha256(content).hexdigest() for name, content in self.sources.items()})
        self.write_json("protocol.json", self.protocol)
        protocol_hash = file_hash(self.study / "protocol.json")
        self.write_json("preparation.json", dict(status="prepared", protocol_sha256=protocol_hash,
                                                neural_training_or_inference=False))
        receipts = {}
        for arm in self.contract["arms"]:
            receipts[arm] = dict(schema=self.contract["schema"], status="completed", arm=arm,
                                 protocol_sha256=protocol_hash, retained_updates=2,
                                 physical_work_unknown=False, automatic_promotion=False)
            self.write_json(arm + "/receipt.json", receipts[arm])
        self.write_json("verification.json", dict(schema="bic-foundation-results-v1", status="completed",
            protocol_sha256=protocol_hash, automatic_promotion=False, neural_training_or_inference=False,
            arms={arm: dict(exact_official_checkpoint_restores=True, receipt=value)
                  for arm, value in receipts.items()}))
        self.write_json("evaluation/report.json", dict(schema="bic-foundation-results-v1", status="completed",
            protocol_sha256=protocol_hash, verification_sha256=file_hash(self.study / "verification.json"),
            automatic_promotion=False, base_checkpoints_unchanged=True, results={arm: {} for arm in receipts}))
        self.summary = dict(schema="bic-foundation-pilot-summary-v1", status="completed_descriptive",
            automatic_promotion=False, contract=self.contract, arms={arm: {} for arm in receipts},
            integrity=dict(canonical_replay_repeated=False, neural_training_or_inference=False,
                           source_sha256=self.protocol["source_sha256"], input_file_sha256={}))
        self.refresh_inputs()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def write_json(self, name, value):
        self.write(self.study / name, (json.dumps(value, sort_keys=True) + "\n").encode())

    def refresh_inputs(self):
        self.summary["integrity"]["input_file_sha256"] = {
            str(path): file_hash(path) for path in self.root.rglob("*")
            if path.is_file() and path != self.study / "evaluation/summary.json"}
        self.pin()

    def pin(self):
        self.write_json("evaluation/summary.json", self.summary)
        # Fixed fixture contract, deliberately not len(manifest) as a trust anchor.
        self.expected = ExpectedEvidence(file_hash(self.study / "evaluation/summary.json"),
                                         input_count=11, source_count=2,
                                         version="fixture-v1", checkpoints=(0, 2))

    def verify(self, expected=None):
        return verify_archive(self.study, repository_root=self.root, expected=expected or self.expected)

    def test_archive_preserves_all_bindings_and_explicit_remapping(self):
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["logical_input_count"], 11)
        self.assertEqual(result["remapped_source_input_count"], 2)
        self.assertEqual(result["unchanged_input_count"], 9)
        self.assertEqual(result["physical_file_count"], 9)
        self.assertEqual(result["input_file_sha256"], self.summary["integrity"]["input_file_sha256"])
        for name in self.sources:
            row = result["source_input_remapping"][str(self.root / name)]
            self.assertEqual(row["archived_path"], str(self.study / "source" / name))
            self.assertEqual(row["sha256"], self.protocol["source_sha256"][name])
        for flag in ("neural_training_or_inference", "checkpoint_deserialization",
                     "canonical_replay_repeated", "artifact_migration", "automatic_promotion"):
            self.assertIs(result[flag], False)

    def test_live_source_edits_and_deletions_are_compatible(self):
        (self.root / "experiments/old.py").write_bytes(b"future implementation")
        (self.root / "docs/old.md").unlink()
        self.assertEqual(self.verify()["remapped_source_input_count"], 2)

    def test_archive_tamper_rejected(self):
        (self.study / "source/experiments/old.py").write_bytes(b"tampered archive")
        with self.assertRaisesRegex(ValueError, "bound evidence changed"):
            self.verify()

    def test_non_source_tamper_cannot_use_archive_fallback(self):
        self.write(self.study / "source/checkpoint.pt", (self.study / "checkpoint.pt").read_bytes())
        (self.study / "checkpoint.pt").write_bytes(b"tamper")
        with self.assertRaisesRegex(ValueError, "bound evidence changed"):
            self.verify()

    def test_summary_requires_external_pin(self):
        self.summary["status"] = "forged"
        self.write_json("evaluation/summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "caller-pinned"):
            self.verify()

    def test_explicit_counts_and_protocol_identity(self):
        for expected in (replace(self.expected, input_count=12), replace(self.expected, source_count=1),
                         replace(self.expected, version="wrong"), replace(self.expected, checkpoints=(0, 3))):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.verify(expected)

    def test_protocol_must_be_an_unchanged_bound_input(self):
        (self.study / "protocol.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "bound evidence changed"):
            self.verify()

    def test_unbound_protocol_rejected_even_with_repin(self):
        del self.summary["integrity"]["input_file_sha256"][str(self.study / "protocol.json")]
        self.pin()
        with self.assertRaisesRegex(ValueError, "not summary-bound"):
            self.verify(replace(self.expected, input_count=10))

    def test_protocol_and_summary_source_maps_must_agree(self):
        self.summary["integrity"]["source_sha256"] = dict(self.protocol["source_sha256"])
        self.summary["integrity"]["source_sha256"]["experiments/old.py"] = "0" * 64
        self.pin()
        with self.assertRaisesRegex(ValueError, "protocol source map differs"):
            self.verify()

    def test_source_input_digest_must_equal_archive_digest(self):
        self.summary["integrity"]["input_file_sha256"][str(self.root / "experiments/old.py")] = "0" * 64
        self.pin()
        with self.assertRaisesRegex(ValueError, "source/archive digest differs"):
            self.verify()

    def test_archive_copy_must_already_be_declared(self):
        del self.summary["integrity"]["input_file_sha256"][str(self.study / "source/experiments/old.py")]
        self.pin()
        with self.assertRaisesRegex(ValueError, "both be summary-bound"):
            self.verify(replace(self.expected, input_count=10))

    def test_source_relative_aliases_rejected(self):
        original = self.summary["integrity"]["source_sha256"]
        for alias in ("../old.py", "experiments/./old.py", "experiments//old.py", "experiments\\old.py",
                      "/old.py", "C:/old.py", "experiments/old.py:stream", "experiments/old.py.", "CON"):
            with self.subTest(alias=alias):
                self.summary["integrity"]["source_sha256"] = {alias: original["experiments/old.py"],
                                                               "docs/old.md": original["docs/old.md"]}
                self.pin()
                with self.assertRaisesRegex(ValueError, "canonical relative"):
                    self.verify()

    def test_case_aliases_in_source_map_rejected(self):
        value = self.protocol["source_sha256"]["experiments/old.py"]
        self.summary["integrity"]["source_sha256"] = {"experiments/old.py": value, "Experiments/old.py": value}
        self.pin()
        with self.assertRaisesRegex(ValueError, "ambiguous case alias"):
            self.verify()

    def test_absolute_input_alias_rejected(self):
        inputs = self.summary["integrity"]["input_file_sha256"]
        original = str(self.study / "checkpoint.pt")
        alias = str(self.study) + os.sep + "." + os.sep + "checkpoint.pt"
        inputs[alias] = inputs.pop(original)
        self.pin()
        with self.assertRaisesRegex(ValueError, "canonical absolute"):
            self.verify()

    def test_duplicate_json_key_rejected(self):
        path = self.study / "evaluation/summary.json"
        raw = path.read_bytes().replace(b'{', b'{"schema":"duplicate",', 1)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "duplicate JSON member"):
            self.verify(replace(self.expected, summary_sha256=file_hash(path)))

    def test_incomplete_summary_rejected_even_when_pinned(self):
        self.summary["status"] = "running"
        self.pin()
        with self.assertRaisesRegex(ValueError, "completed descriptive"):
            self.verify()

    def test_incomplete_arm_rejected_even_when_pinned(self):
        name = "curriculum/receipt.json"
        value = json.loads((self.study / name).read_text())
        value["status"] = "failed"
        self.write_json(name, value)
        self.refresh_inputs()
        with self.assertRaisesRegex(ValueError, "completed arm"):
            self.verify()

    def test_wrong_completion_protocol_binding_rejected(self):
        name = "evaluation/report.json"
        value = json.loads((self.study / name).read_text())
        value["protocol_sha256"] = "0" * 64
        self.write_json(name, value)
        self.refresh_inputs()
        with self.assertRaisesRegex(ValueError, "completed protocol-bound evaluation"):
            self.verify()

    def test_missing_evidence_rejected(self):
        (self.study / "checkpoint.pt").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.verify()

    def test_symlink_escape_rejected(self):
        target = self.study / "source/experiments/old.py"
        content = target.read_bytes()
        target.unlink()
        external = self.root / "outside.py"
        external.write_bytes(content)
        try:
            target.symlink_to(external)
        except (OSError, NotImplementedError) as error:
            if os.name != "nt":
                self.skipTest("OS does not permit test symlink: " + str(error))
            # Windows junctions exercise the same escape boundary without the
            # privileged symlink right. Both directories are fixture-owned.
            outside_directory = self.root / "outside-directory"
            outside_directory.mkdir()
            (outside_directory / target.name).write_bytes(content)
            target.parent.rmdir()
            linked = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(target.parent),
                                     str(outside_directory)], capture_output=True, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
        with self.assertRaisesRegex(ValueError, "symlink or junction"):
            self.verify()

    def test_hardlink_alias_rejected(self):
        target = self.study / "checkpoint-alias.pt"
        os.link(self.study / "checkpoint.pt", target)
        self.refresh_inputs()
        with self.assertRaisesRegex(ValueError, "hardlinked evidence aliases"):
            self.verify(replace(self.expected, input_count=12))

    def test_receipt_must_be_new_and_outside_evidence(self):
        with self.assertRaisesRegex(ValueError, "outside the historical"):
            _receipt_destination(self.study / "new-receipt.json", self.study)
        with self.assertRaisesRegex(ValueError, "already exists"):
            _receipt_destination(self.root / "docs/old.md", self.study)
        destination = self.root / "validation/new-receipt.json"
        self.assertEqual(_receipt_destination(destination, self.study), destination)
        self.assertFalse(destination.parent.exists())

    def test_verifier_identity_is_receipted(self):
        result = self.verify()
        self.assertEqual(file_hash(result["verifier_path"]), result["verifier_sha256"])


if __name__ == "__main__":
    unittest.main()
