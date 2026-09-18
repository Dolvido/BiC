"""Small terminal/process-gate fixtures; no learner, CUDA or study-data access.

All experiment receipts are invented temporary JSON files. Process observations
are injected except one read-only query of this test process on Windows. No
process is started, stopped, waited on or otherwise modified by these tests.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from experiments import foundation_prefetch_gate as gate


ROLES = ("dispatcher", "main", "main_launcher")
PIDS = dict(zip(ROLES, (41001, 41002, 41003)))
ENDED = "2026-09-17T12:00:00+00:00"
PARENT_MANIFEST = "e477b1f6dd4d4f787288bcd6defaba73359949ef67867012caa20c2a81c2c24a"
EXPECTED_JOBS = {f"{objective}-seed{seed}" for seed in (8472, 8473, 8474)
                 for objective in ("baseline", "balanced_reply")}


def digest(image):
    return hashlib.sha256(image).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class PrefetchGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bic-prefetch-gate-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dispatcher_path = self.directory / "dispatcher.json"
        self.main_path = self.directory / "main.json"

    def records(self):
        dispatcher = dict(schema="bic-objective-local-dispatch-v1", manifest_sha256=PARENT_MANIFEST,
            status="completed", active=None, ended_utc=ENDED, pid=PIDS["dispatcher"],
            phases=[dict(name=name, status="completed", exit_code=0, ended_utc=ENDED)
                    for name in ("prepare", "calibration", "main")])
        dispatcher["phases"][-1]["child_launcher_pid"] = PIDS["main_launcher"]
        main = dict(schema="bic-foundation-objective-study-v1", stage="main", status="completed",
            active_phase=None, ended_utc=ENDED, pid=PIDS["main"],
            completed_jobs=sorted(EXPECTED_JOBS), physical_optimizer_updates=18432)
        return dispatcher, main

    def write(self, dispatcher=None, main=None, *, bind_terminal=True):
        default_dispatcher, default_main = self.records()
        dispatcher = copy.deepcopy(default_dispatcher if dispatcher is None else dispatcher)
        main = copy.deepcopy(default_main if main is None else main)
        main_image = encoded(main)
        if bind_terminal:
            # Malformed phase fixtures may intentionally have no final mapping.
            phases = dispatcher.get("phases")
            if isinstance(phases, list) and phases and isinstance(phases[-1], dict):
                phases[-1]["terminal_sha256"] = digest(main_image)
        dispatcher_image = encoded(dispatcher)
        self.main_path.write_bytes(main_image)
        self.dispatcher_path.write_bytes(dispatcher_image)
        return dict(expected_dispatcher_sha256=digest(dispatcher_image), expected_main_sha256=digest(main_image))

    def check(self, pins, reader):
        return gate.check_completed_parent(self.dispatcher_path, self.main_path, process_status=reader, **pins)

    def refused_before_process_queries(self, dispatcher, main):
        reader = Mock(return_value={"status": "absent"})
        with self.assertRaises(ValueError):
            self.check(self.write(dispatcher, main), reader)
        reader.assert_not_called()

    def test_completed_parent_accepts_fresh_absent_and_exited_observations(self):
        for statuses in (("absent",)*3, ("exited",)*3, ("absent", "exited", "absent")):
            with self.subTest(statuses=statuses):
                pins = self.write()
                images = (self.dispatcher_path.read_bytes(), self.main_path.read_bytes())
                observations = [{"status": status, **({"exit_code": 0} if status == "exited" else {"windows_error": 87})}
                                for status in statuses]
                reader = Mock(side_effect=copy.deepcopy(observations))
                result = self.check(pins, reader)
                self.assertEqual([call.args[0] for call in reader.call_args_list], list(PIDS.values()))
                self.assertEqual(result["schema"], gate.SCHEMA)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["dispatcher_sha256"], pins["expected_dispatcher_sha256"])
                self.assertEqual(result["main_sha256"], pins["expected_main_sha256"])
                self.assertEqual(result["parent_manifest_sha256"], PARENT_MANIFEST)
                self.assertEqual(result["process_observations"], {
                    role: dict(pid=PIDS[role], **value) for role, value in zip(ROLES, observations)})
                self.assertTrue(result["observed_utc"])
                self.assertEqual((self.dispatcher_path.read_bytes(), self.main_path.read_bytes()), images)

    def test_running_unknown_or_malformed_process_observation_refuses_every_role(self):
        for index, role in enumerate(ROLES):
            for bad in ({"status": "running", "exit_code": 259}, {"status": "unknown", "windows_error": 5},
                        {}, None, {"status": "unrecognized"}):
                with self.subTest(role=role, observation=bad):
                    observations = [{"status": "absent"} for _ in ROLES]
                    observations[index] = bad
                    reader = Mock(side_effect=observations)
                    with self.assertRaisesRegex(RuntimeError, role):
                        self.check(self.write(), reader)
                    self.assertEqual([call.args[0] for call in reader.call_args_list], list(PIDS.values())[:index+1])

    def test_incomplete_active_or_wrong_terminal_and_phase_contracts_refuse(self):
        terminal_cases = (
            ("dispatcher", "schema", "other"), ("dispatcher", "manifest_sha256", "f"*64),
            ("dispatcher", "status", "running"), ("dispatcher", "status", "failed"),
            ("dispatcher", "active", "main"), ("dispatcher", "ended_utc", None),
            ("main", "schema", "other"), ("main", "stage", "calibration"),
            ("main", "status", "running"), ("main", "status", "failed"),
            ("main", "active_phase", "score/job"), ("main", "ended_utc", None))
        for target, field, value in terminal_cases:
            with self.subTest(target=target, field=field, value=value):
                dispatcher, main = self.records()
                (dispatcher if target == "dispatcher" else main)[field] = value
                self.refused_before_process_queries(dispatcher, main)
        for index in range(3):
            for field, value in (("status", "running"), ("exit_code", 1), ("exit_code", True), ("ended_utc", None)):
                with self.subTest(phase=index, field=field, value=value):
                    dispatcher, main = self.records()
                    dispatcher["phases"][index][field] = value
                    self.refused_before_process_queries(dispatcher, main)
        for kind in ("missing", "reordered", "duplicate", "not_mapping"):
            with self.subTest(phases=kind):
                dispatcher, main = self.records()
                if kind == "missing":
                    dispatcher["phases"].pop()
                elif kind == "reordered":
                    dispatcher["phases"].reverse()
                elif kind == "duplicate":
                    dispatcher["phases"][1]["name"] = "prepare"
                else:
                    dispatcher["phases"][1] = None
                self.refused_before_process_queries(dispatcher, main)

    def test_exact_job_budget_and_all_three_process_identities_are_required(self):
        expected = sorted(EXPECTED_JOBS)
        for jobs in (expected[:-1], expected+[expected[0]], expected[:-1]+[expected[0]],
                     expected[:-1]+["different-seed"], expected[:-1]+[17], None):
            with self.subTest(jobs=jobs):
                dispatcher, main = self.records()
                main["completed_jobs"] = jobs
                self.refused_before_process_queries(dispatcher, main)
        for updates in (18431, 18433, True, "18432"):
            with self.subTest(updates=updates):
                dispatcher, main = self.records()
                main["physical_optimizer_updates"] = updates
                self.refused_before_process_queries(dispatcher, main)
        for role in ROLES:
            for invalid in (None, True, 0, -1, 2**32, "41001"):
                with self.subTest(role=role, pid=invalid):
                    dispatcher, main = self.records()
                    if role == "main_launcher":
                        dispatcher["phases"][-1]["child_launcher_pid"] = invalid
                    else:
                        (dispatcher if role == "dispatcher" else main)["pid"] = invalid
                    self.refused_before_process_queries(dispatcher, main)

    def test_caller_pins_main_terminal_binding_and_changed_bytes_refuse_before_query(self):
        for target in ("dispatcher", "main"):
            key = "expected_"+target+"_sha256"
            for invalid in ("f"*64, "A"*64, "0"*63, None):
                with self.subTest(pin=target, invalid=invalid):
                    pins, reader = self.write(), Mock(return_value={"status": "absent"})
                    pins[key] = invalid
                    with self.assertRaises(ValueError):
                        self.check(pins, reader)
                    reader.assert_not_called()
            with self.subTest(changed_bytes=target):
                pins, reader = self.write(), Mock(return_value={"status": "absent"})
                path = self.dispatcher_path if target == "dispatcher" else self.main_path
                path.write_bytes(path.read_bytes()+b"\n")  # Valid JSON, different authenticated bytes.
                with self.assertRaises(ValueError):
                    self.check(pins, reader)
                reader.assert_not_called()
        dispatcher, main = self.records()
        dispatcher["phases"][-1]["terminal_sha256"] = "f"*64
        pins, reader = self.write(dispatcher, main, bind_terminal=False), Mock(return_value={"status": "absent"})
        with self.assertRaises(ValueError):
            self.check(pins, reader)
        reader.assert_not_called()

    def test_receipt_change_during_process_query_is_caught_by_final_rehash(self):
        for target in ("dispatcher", "main"):
            with self.subTest(mutated_receipt=target):
                pins, calls = self.write(), []
                path = self.dispatcher_path if target == "dispatcher" else self.main_path
                def observe(pid):
                    calls.append(pid)
                    if len(calls) == 1:
                        path.write_bytes(path.read_bytes()+b" ")
                    return {"status": "absent"}
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    self.check(pins, observe)
                self.assertEqual(calls, list(PIDS.values()))

    @unittest.skipUnless(os.name == "nt", "real process status reader is Windows-only")
    def test_windows_reader_reports_this_live_process_running(self):
        observed = gate.windows_process_status(os.getpid())
        self.assertEqual(observed["status"], "running")
        self.assertEqual(observed["exit_code"], 259)


if __name__ == "__main__":
    unittest.main()
