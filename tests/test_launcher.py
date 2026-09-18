"""Check path stability, read-only diagnosis and coupled process lifetime."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brain_in_computer.launcher import LaunchSession, build_commands, doctor, load_manifest


class FakeProcess:
    def __init__(self, output, returncode=None):
        self.stdout = io.StringIO(output)
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        (self.root / "runs").mkdir()
        self.data = {"schema": "bic-release-v1", "release": "test", "checkpoints": {
            "computer": "runs/controller.pt", "regional": "runs/controller.pt",
            "encoder": "runs/encoder.pt"}}
        (self.root / "runs/controller.pt").write_bytes(b"fixture")
        (self.root / "runs/encoder.pt").write_bytes(b"fixture")
        self.write_manifest()

    def write_manifest(self):
        (self.root / "config/checkpoints.json").write_text(json.dumps(self.data))

    def test_commands_use_explicit_root_and_separate_saved_memory(self):
        release, commands = build_commands(self.root)
        self.assertEqual(release["project_dir"], self.root.resolve())
        self.assertEqual(set(commands), {"computer", "memory"})
        for command in commands.values():
            self.assertIn("127.0.0.1", command)
            self.assertIn(str(self.root / "runs/controller.pt"), command)
            self.assertNotIn("train", command)
        memory_command = commands["memory"]
        self.assertEqual(memory_command[memory_command.index("--memory") + 1], str(self.root / "runs/personal-memory.json"))
        self.assertEqual(set(build_commands(self.root, mode="computer")[1]), {"computer"})
        self.assertFalse((self.root / "runs/personal-memory.json").exists())

    def test_wrong_working_directory_and_escaping_paths_have_remediation(self):
        with self.assertRaisesRegex(FileNotFoundError, "--project-dir"):
            load_manifest(self.root / "elsewhere")
        for value in ("../outside.pt", str(self.root / "runs/controller.pt"), "", None):
            self.data["checkpoints"]["computer"] = value
            self.write_manifest()
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_manifest(self.root)

    def test_doctor_does_not_load_model_or_write_and_reports_missing_dependencies(self):
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        missing = {"torch": {"available": False, "version": None},
                   "numpy": {"available": True, "version": "test"},
                   "Pillow": {"available": True, "version": "test"}}
        with patch("brain_in_computer.launcher._dependency_status", return_value=(missing, None)):
            status = doctor(self.root, device="cuda")
        self.assertFalse(status["ready"])
        self.assertEqual(len(status["problems"]), 2)
        self.assertFalse(status["cuda"]["available"])
        self.assertIn("No checkpoint loading", status["scope"])
        self.assertEqual(before, {str(path.relative_to(self.root)): path.read_bytes()
                                for path in self.root.rglob("*") if path.is_file()})

    def test_doctor_only_requires_checkpoints_for_the_selected_lab(self):
        (self.root / "runs/encoder.pt").unlink()
        okay = {"torch": {"available": True}, "numpy": {"available": True}, "Pillow": {"available": True}}
        with patch("brain_in_computer.launcher._dependency_status", return_value=(okay, None)):
            self.assertTrue(doctor(self.root, mode="computer")["ready"])
            self.assertFalse(doctor(self.root, mode="memory")["ready"])

    def test_invalid_ports_and_budgets_fail_before_starting_a_process(self):
        with patch("brain_in_computer.launcher.subprocess.Popen") as popen:
            for args in ({"computer_port": -1}, {"threads": 0}, {"threads": True},
                         {"computer_port": 8766}, {"startup_timeout": float("nan")},
                         {"startup_timeout": 0}, {"mode": "unknown"}):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    LaunchSession(self.root, **args)
            popen.assert_not_called()

    def test_ready_processes_are_owned_and_both_terminated_on_exit(self):
        children = [FakeProcess("BiC computer-use lab: http://127.0.0.1:50101\n"),
                    FakeProcess("BiC teaching lab: http://127.0.0.1:50102\n")]
        with patch("brain_in_computer.launcher.doctor", return_value={"ready": True}), patch(
            "brain_in_computer.launcher.subprocess.Popen", side_effect=children
        ) as popen:
            with LaunchSession(self.root, computer_port=0, memory_port=0, stream=io.StringIO()) as session:
                self.assertEqual(session.urls, {"computer": "http://127.0.0.1:50101", "memory": "http://127.0.0.1:50102"})
                self.assertFalse(any(child.terminated for child in children))
            self.assertTrue(all(child.terminated for child in children))
            self.assertEqual(popen.call_args.kwargs["cwd"], str(self.root))
            self.assertNotIn("shell", popen.call_args.kwargs)

    def test_failed_second_lab_closes_successful_first_lab(self):
        children = [FakeProcess("BiC computer-use lab: http://127.0.0.1:50101\n"),
                    FakeProcess("Address already in use\n", returncode=1)]
        with patch("brain_in_computer.launcher.doctor", return_value={"ready": True}), patch(
            "brain_in_computer.launcher.subprocess.Popen", side_effect=children
        ), self.assertRaisesRegex(RuntimeError, "memory lab exited"):
            LaunchSession(self.root, stream=io.StringIO()).start()
        self.assertTrue(children[0].terminated)

    def test_timeout_closes_children_and_preflight_failure_starts_none(self):
        child = FakeProcess("")
        with patch("brain_in_computer.launcher.doctor", return_value={"ready": True}), patch(
            "brain_in_computer.launcher.subprocess.Popen", return_value=child
        ), self.assertRaisesRegex(RuntimeError, "timed out"):
            LaunchSession(self.root, mode="computer", startup_timeout=0.01, stream=io.StringIO()).start()
        self.assertTrue(child.terminated)
        with patch("brain_in_computer.launcher.doctor", return_value={"ready": False, "problems": ["missing file"]}), patch(
            "brain_in_computer.launcher.subprocess.Popen"
        ) as popen, self.assertRaisesRegex(RuntimeError, "missing file"):
            LaunchSession(self.root).start()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
