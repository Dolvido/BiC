"""Dispatcher failure/barrier tests using fake children, never model training."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import run_capacity_study as dispatcher
from experiments.train_cognitive import atomic_json


class FakeProcess:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code
        self.wait_timeouts = []

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.code is None:
            raise subprocess.TimeoutExpired("fake child", timeout)
        return self.code


class CapacityDispatcherTests(unittest.TestCase):
    @contextmanager
    def fixture(self, directory, popen, taskkill=None, writer=atomic_json):
        path = Path(directory)
        (path / "protocol.json").write_text("{}", encoding="utf8")
        with ExitStack() as stack:
            stack.enter_context(patch("experiments.capacity_study.load_protocol", return_value={}))
            # Replace only the dispatcher's os binding, not global os.name/Path.
            stack.enter_context(patch.object(dispatcher, "os", SimpleNamespace(name="nt")))
            launches = stack.enter_context(patch.object(dispatcher.subprocess, "Popen", side_effect=popen))
            kills = stack.enter_context(patch.object(dispatcher.subprocess, "run", side_effect=taskkill))
            stack.enter_context(patch("experiments.train_cognitive.atomic_json", side_effect=writer))
            stack.enter_context(patch.object(dispatcher.time, "sleep", return_value=None))
            stack.enter_context(patch("builtins.print"))
            yield path, launches, kills

    def test_second_launch_exception_stops_owned_tree_and_records_unknown_work(self):
        first = FakeProcess(51001)
        launched = []
        def launch(command, **options):
            launched.append(command)
            if len(launched) == 2:
                raise OSError("second launch failed")
            return first
        def stop(command, **options):
            self.assertEqual(command, ["taskkill", "/PID", "51001", "/T", "/F"])
            self.assertEqual(options["timeout"], 10)
            first.code = -9
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory, self.fixture(directory, launch, stop) as (path, launches, kills):
            with self.assertRaisesRegex(OSError, "second launch failed"):
                dispatcher.dispatch(path)
            state = json.loads((path / "execution.json").read_text())
            self.assertEqual(launches.call_count, 2)
            self.assertEqual(kills.call_count, 1)
            self.assertEqual(first.wait_timeouts, [10])
            self.assertEqual(state["status"], "failed")
            self.assertEqual([row["status"] for row in state["processes"]], ["interrupted", "launch_failed"])
            self.assertTrue(state["processes"][0]["uncommitted_physical_work_unknown"])
            self.assertTrue(state["processes"][0]["dispatcher_interrupted"])
            self.assertEqual(state["processes"][0]["exit_code"], -9)
            self.assertTrue(all(row["name"].startswith("calibration-") for row in state["processes"]))

    def test_failed_calibration_verification_blocks_selection_main_and_audit(self):
        commands = []
        def launch(command, **options):
            commands.append(command)
            phase = command[command.index("--phase") + 1]
            return FakeProcess(52000 + len(commands), 7 if phase == "verify" else 0)
        with tempfile.TemporaryDirectory() as directory, self.fixture(directory, launch) as (path, launches, kills):
            with self.assertRaisesRegex(RuntimeError, "no retries or later phase"):
                dispatcher.dispatch(path)
            state = json.loads((path / "execution.json").read_text())
            self.assertEqual(launches.call_count, 10)
            kills.assert_not_called()
            self.assertEqual([command[command.index("--phase") + 1] for command in commands], ["train"] * 9 + ["verify"])
            self.assertTrue(all(command[command.index("--stage") + 1] == "calibration" for command in commands))
            self.assertEqual(len({command[command.index("--job") + 1] for command in commands[:-1]}), 9)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["processes"][-1]["name"], "verify-calibration")
            self.assertEqual(state["processes"][-1]["exit_code"], 7)

    def test_ledger_failure_cannot_skip_active_process_cleanup(self):
        first = FakeProcess(53001)
        failed = False
        def writer(path, value):
            nonlocal failed
            if len(value["processes"]) == 2 and not failed:
                failed = True
                raise OSError("ledger unavailable")
            atomic_json(path, value)
        def stop(command, **options):
            first.code = -9
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory, self.fixture(directory, lambda *a, **k: first, stop, writer) as (path, launches, kills):
            with self.assertRaisesRegex(OSError, "ledger unavailable"):
                dispatcher.dispatch(path)
            self.assertEqual(launches.call_count, 1)
            self.assertEqual(kills.call_count, 1)
            self.assertEqual(first.wait_timeouts, [10])
            state = json.loads((path / "execution.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["processes"][0]["status"], "interrupted")

    def test_existing_execution_ledger_never_relaunches_jobs(self):
        with tempfile.TemporaryDirectory() as directory, self.fixture(directory, lambda *a, **k: self.fail("unexpected launch")) as (path, launches, kills):
            (path / "execution.json").write_text("{\"status\":\"failed\"}", encoding="utf8")
            original = (path / "execution.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "ledger already exists"):
                dispatcher.dispatch(path)
            launches.assert_not_called()
            kills.assert_not_called()
            self.assertEqual((path / "execution.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
