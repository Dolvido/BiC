"""Pure filesystem checks; no learner, torch, checkpoint or evaluation calls."""
from collections import Counter
import copy
from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path
import pickle
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import foundation_source_inventory as source


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SourceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "brain_in_computer").mkdir()
        (self.root / "experiments").mkdir()
        self.expected = {}
        for name, raw in (("brain_in_computer/a.py", b"a = 1\n"),
                          ("brain_in_computer/.hidden.py", b"hidden = 1\n"),
                          ("experiments/static.py", b"static = 1\n")):
            (self.root / name).write_bytes(raw)
            self.expected[name] = sha(raw)
        self.specs = [source.SourceMembership("brain_in_computer", "*.py"),
                      source.SourceMembership("experiments", "__init__.py")]
        self.pin = sha(Path(source.__file__).read_bytes())

    def make(self, **changes):
        arguments = dict(root=self.root, expected_sha256=self.expected,
                         memberships=self.specs, guard_sha256=self.pin)
        arguments.update(changes)
        return source.SourceInventory(**arguments)

    def assert_poisoned(self, inventory):
        self.assertTrue(inventory.poisoned)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("poisoned guard read bytes")):
            with self.assertRaises(source.SourceInventoryPoisoned):
                inventory.check()

    def test_detached_expected_membership_and_results(self):
        inventory = self.make()
        original = self.expected.copy()
        self.expected.clear()
        self.specs.clear()
        inventory.expected_sha256.clear()
        inventory.declaration["memberships"].clear()
        result = inventory.check()
        result.clear()
        self.assertEqual(inventory.check(), original)
        self.assertEqual(len(inventory.memberships), 2)
        with self.assertRaises(FrozenInstanceError):
            inventory._expected = ()
        with self.assertRaises(FrozenInstanceError):
            inventory.memberships[0].pattern = "*"

    def test_reads_every_source_once_on_construction_and_each_check(self):
        actual_read = Path.read_bytes
        counts = Counter()
        def counted(path):
            counts[path] += 1
            return actual_read(path)
        with patch.object(Path, "read_bytes", counted):
            inventory = self.make()
            self.assertEqual(inventory.check(), self.expected)
            self.assertEqual(inventory.check(), self.expected)
        expected_paths = {self.root / name for name in self.expected} | {Path(source.__file__).resolve()}
        self.assertEqual(set(counts), expected_paths)
        self.assertTrue(all(count == 3 for count in counts.values()))

    def test_guard_in_inventory_is_not_read_twice(self):
        guard = Path(source.__file__).resolve()
        expected = {guard.name: self.pin}
        actual_read = Path.read_bytes
        counts = Counter()
        def counted(path):
            counts[path] += 1
            return actual_read(path)
        with patch.object(Path, "read_bytes", counted):
            inventory = self.make(root=guard.parent, expected_sha256=expected, memberships=())
            self.assertEqual(inventory.check(), expected)
        self.assertEqual(counts, {guard: 2})
        with self.assertRaises(source.SourceInventoryError):
            self.make(root=guard.parent, expected_sha256={guard.name: "0" * 64}, memberships=())

    def test_same_size_same_mtime_byte_drift_and_repair_cannot_unpoison(self):
        inventory = self.make()
        path = self.root / "brain_in_computer/a.py"
        before, metadata = path.read_bytes(), path.stat()
        path.write_bytes(b"a = 2\n")
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        with self.assertRaisesRegex(source.SourceInventoryError, "bytes changed"):
            inventory.check()
        path.write_bytes(before)
        self.assert_poisoned(inventory)

    def test_missing_static_file_poisoned(self):
        inventory = self.make()
        path = self.root / "experiments/static.py"
        raw = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            inventory.check()
        path.write_bytes(raw)
        self.assert_poisoned(inventory)

    def test_unreadable_source_poisoned(self):
        inventory = self.make()
        actual_read = Path.read_bytes
        def unreadable(path):
            if path == self.root / "experiments/static.py":
                raise PermissionError("injected unreadable source")
            return actual_read(path)
        with patch.object(Path, "read_bytes", unreadable):
            with self.assertRaises(PermissionError):
                inventory.check()
        self.assert_poisoned(inventory)

    def test_add_remove_rename_and_nonregular_membership_rejected(self):
        for mutation in ("add", "remove", "rename", "directory"):
            with self.subTest(mutation=mutation):
                inventory = self.make()
                original = self.root / "brain_in_computer/a.py"
                extra = self.root / "brain_in_computer/extra.py"
                if mutation == "add":
                    extra.write_bytes(b"pass\n")
                elif mutation == "remove":
                    original.unlink()
                elif mutation == "rename":
                    original.rename(extra)
                else:
                    extra.mkdir()
                with self.assertRaisesRegex(source.SourceInventoryError, "membership changed"):
                    inventory.check()
                if mutation == "directory":
                    extra.rmdir()
                elif extra.exists():
                    extra.unlink()
                original.write_bytes(b"a = 1\n")
                self.assert_poisoned(inventory)

    def test_dynamic_inventory_must_be_complete_at_construction(self):
        reduced = dict(self.expected)
        del reduced["brain_in_computer/.hidden.py"]
        with self.assertRaisesRegex(source.SourceInventoryError, "membership changed"):
            self.make(expected_sha256=reduced)
        (self.root / "brain_in_computer/directory.py").mkdir()
        declared = {**self.expected, "brain_in_computer/directory.py": "0" * 64}
        with self.assertRaisesRegex(source.SourceInventoryError, "file type"):
            self.make(expected_sha256=declared)

    def test_optional_initializer_appearance_rejected(self):
        inventory = self.make()
        (self.root / "experiments/__init__.py").write_bytes(b"")
        with self.assertRaisesRegex(source.SourceInventoryError, "membership changed"):
            inventory.check()
        self.assert_poisoned(inventory)

    def test_nonrecursive_membership_ignores_unrelated_files(self):
        inventory = self.make()
        (self.root / "brain_in_computer/nested").mkdir()
        (self.root / "brain_in_computer/nested/new.py").write_bytes(b"new\n")
        (self.root / "brain_in_computer/readme.txt").write_bytes(b"new\n")
        (self.root / "experiments/unrelated.py").write_bytes(b"new\n")
        self.assertEqual(inventory.check(), self.expected)

    def test_native_pattern_case_and_hidden_file_semantics(self):
        directory = self.root / "brain_in_computer"
        (directory / "upper.PY").write_bytes(b"new\n")
        expected_glob = {p.relative_to(self.root).as_posix() for p in directory.glob("*.py")}
        spec = self.specs[0]
        self.assertEqual(expected_glob, {p.relative_to(self.root).as_posix()
            for p in directory.iterdir() if spec.includes(p.relative_to(self.root).as_posix())})
        self.assertIn("brain_in_computer/.hidden.py", expected_glob)
        complete = {**self.expected, **{name: sha((self.root / name).read_bytes()) for name in expected_glob}}
        self.assertEqual(self.make(expected_sha256=complete).check(), complete)

    def test_after_scan_catches_addition_during_reads(self):
        inventory = self.make()
        actual_read = Path.read_bytes
        def add_during_read(path):
            raw = actual_read(path)
            if path == self.root / "experiments/static.py":
                (self.root / "brain_in_computer/late.py").write_bytes(b"late\n")
            return raw
        with patch.object(Path, "read_bytes", add_during_read):
            with self.assertRaisesRegex(source.SourceInventoryError, "membership changed"):
                inventory.check()
        self.assert_poisoned(inventory)

    def test_scan_failure_poisoned(self):
        inventory = self.make()
        with patch.object(source.os, "scandir", side_effect=PermissionError("unreadable directory")):
            with self.assertRaises(PermissionError):
                inventory.check()
        self.assert_poisoned(inventory)

    def test_implementation_pin_and_disk_drift(self):
        with self.assertRaisesRegex(source.SourceInventoryError, "imported implementation"):
            self.make(guard_sha256="0" * 64)
        inventory = self.make()
        actual_read = Path.read_bytes
        def altered_guard(path):
            return b"changed guard" if path == Path(source.__file__).resolve() else actual_read(path)
        with patch.object(Path, "read_bytes", altered_guard):
            with self.assertRaisesRegex(source.SourceInventoryError, "implementation changed"):
                inventory.check()
        self.assert_poisoned(inventory)

    def test_noncanonical_names_hashes_patterns_and_ambiguous_case_rejected(self):
        for name in ("../escape.py", "/absolute.py", "a//b.py", "a/./b.py", "a/../b.py",
                     "C:/a.py", "a\\b.py", "a.py:stream", "a. /b.py", "a./b.py", "a/CON.py", "a/?.py"):
            with self.subTest(name=name), self.assertRaises(source.SourceInventoryError):
                self.make(expected_sha256={name: "0" * 64})
        with self.assertRaises(source.SourceInventoryError):
            self.make(expected_sha256={"A.py": "0" * 64, "a.py": "1" * 64})
        for value in ("A" * 64, "0" * 63, 123, None):
            with self.subTest(digest=value), self.assertRaises(source.SourceInventoryError):
                self.make(expected_sha256={"a.py": value})
        for pattern in ("**/*.py", "../*.py", "**", "", "a\\*.py"):
            with self.subTest(pattern=pattern), self.assertRaises(source.SourceInventoryError):
                source.SourceMembership("brain_in_computer", pattern)
        with self.assertRaises(source.SourceInventoryError):
            self.make(memberships=self.specs * 2)
        with self.assertRaises(source.SourceInventoryError):
            self.make(memberships=[{"directory": "brain_in_computer", "pattern": "*.py"}])

    def test_directory_and_reparse_ancestor_rejected(self):
        inventory = self.make()
        path = self.root / "experiments/static.py"
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(source.SourceInventoryError, "file type"):
            inventory.check()
        self.assert_poisoned(inventory)
        path.rmdir()
        path.write_bytes(b"static = 1\n")
        inventory = self.make()
        original_lstat = Path.lstat
        def reparse(path, *args, **kwargs):
            if path == self.root / "brain_in_computer":
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original_lstat(path, *args, **kwargs)
        with patch.object(Path, "lstat", reparse):
            with self.assertRaisesRegex(source.SourceInventoryError, "reparse"):
                inventory.check()
        self.assert_poisoned(inventory)

    def test_process_owner_and_serialization(self):
        inventory = self.make()
        for operation in (pickle.dumps, copy.copy, copy.deepcopy):
            with self.assertRaises(TypeError):
                operation(inventory)
        with patch.object(source.os, "getpid", return_value=-1):
            with self.assertRaisesRegex(source.SourceInventoryError, "different process"):
                inventory.check()
        self.assert_poisoned(inventory)

    def test_subclass_cannot_override_constructor_validation(self):
        called = []
        class ForgedInventory(source.SourceInventory):
            def check(self):
                called.append(True)
                return {}
        with self.assertRaisesRegex(source.SourceInventoryError, "exact SourceInventory"):
            ForgedInventory(self.root, self.expected, memberships=self.specs, guard_sha256=self.pin)
        self.assertEqual(called, [])

    def test_interrupted_check_permanently_poisoned(self):
        inventory = self.make()
        with patch.object(Path, "read_bytes", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                inventory.check()
        self.assert_poisoned(inventory)


if __name__ == "__main__":
    unittest.main()
