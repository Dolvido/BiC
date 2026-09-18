"""CLI orchestration only: fake owner, no model/optimizer/forward construction."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments import foundation_cycle_run as cli


PIN = "a" * 64
NEW_PIN = "b" * 64


class CycleRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.owner = self.root / "owner" / "owner.pt"
        self.receipt = self.root / "result.json"
        self.args = ["--owner", str(self.owner), "--expected-sha256", PIN,
                     "--receipt", str(self.receipt)]
        self.configuration = patch.object(cli, "_configure_runtime")
        self.configured = self.configuration.start()
        self.addCleanup(self.configuration.stop)

    def test_requires_explicit_pin_and_finite_allowance_before_loading(self):
        with patch.object(cli.CycleOwner, "load") as load, contextlib.redirect_stderr(io.StringIO()):
            for suffix in ([], ["--hours", "nan"], ["--hours", "inf"], ["--hours", "0"],
                           ["--max-updates", "-1"], ["--max-cycles", "1.5"]):
                with self.assertRaises(SystemExit):
                    cli.main(self.args + suffix)
            with self.assertRaises(SystemExit):
                cli.main(["--owner", str(self.owner), "--receipt", str(self.receipt), "--hours", "1"])
            load.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_passes_bounds_and_pin_and_returns_new_pin(self):
        owner = Mock(last_report=None)
        owner.run.return_value = dict(status="cycle_bound", owner_sha256=NEW_PIN)
        def after_configuration(*args, **kwargs):
            self.configured.assert_called_once()
            return owner
        with patch.object(cli.CycleOwner, "load", side_effect=after_configuration) as load, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(self.args + ["--max-updates", "10", "--max-cycles", "2", "--min-generation", "3"]), 0)
        load.assert_called_once_with(self.owner.resolve(), expected_sha256=PIN,
            min_generation=3, device="cpu", recovery_policy="stop")
        owner.run.assert_called_once_with(max_updates=10, max_cycles=2, deadline=None)
        self.assertEqual(json.loads(output.getvalue())["owner_sha256"], NEW_PIN)
        self.assertEqual(json.loads(self.receipt.read_text())["run"]["owner_sha256"], NEW_PIN)

    def test_hours_include_loading_and_do_not_reset_after_load(self):
        owner = Mock(last_report=None)
        owner.run.return_value = dict(status="deadline", owner_sha256=PIN)
        with patch.object(cli.CycleOwner, "load", return_value=owner), patch.object(cli.time, "monotonic", side_effect=[100., 125., 130.]), contextlib.redirect_stdout(io.StringIO()):
            cli.main(self.args + ["--hours", "0.01"])
        owner.run.assert_called_once_with(max_updates=None, max_cycles=None, deadline=136.)
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result["load_seconds"], 25.)
        self.assertEqual(result["wall_seconds"], 30.)

    def test_failure_preserves_owner_report_without_manufacturing_pin(self):
        owner = Mock(last_report=dict(status="failed", publication_uncertain=True))
        owner.run.side_effect = RuntimeError("injected update failure")
        with patch.object(cli.CycleOwner, "load", return_value=owner):
            with self.assertRaisesRegex(RuntimeError, "injected update"):
                cli.main(self.args + ["--max-updates", "1"])
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["run"]["publication_uncertain"])
        self.assertNotIn("owner_sha256", result)

    def test_unknown_load_requires_explicit_recovery_option(self):
        with patch.object(cli.CycleOwner, "load", side_effect=RuntimeError("unknown work")) as load:
            with self.assertRaisesRegex(RuntimeError, "unknown work"):
                cli.main(self.args + ["--max-updates", "0", "--recovery-policy", "acknowledge_unknown"])
        self.assertEqual(load.call_args.kwargs["recovery_policy"], "acknowledge_unknown")
        self.assertIsNone(json.loads(self.receipt.read_text())["run"])

    def test_existing_result_or_intent_prevents_work(self):
        for path in (self.receipt, self.receipt.with_name(self.receipt.name + ".intent.json")):
            path.write_text("existing")
            with patch.object(cli.CycleOwner, "load") as load:
                with self.assertRaises(FileExistsError):
                    cli.main(self.args + ["--max-updates", "1"])
                load.assert_not_called()
            self.assertEqual(path.read_text(), "existing")
            path.unlink()

    def test_receipt_cannot_modify_owner_directory(self):
        args = self.args[:-1] + [str(self.owner.parent / "result.json")]
        with patch.object(cli.CycleOwner, "load") as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(args + ["--max-updates", "1"])
            load.assert_not_called()

    def test_failed_publication_never_exposes_partial_final_json(self):
        with patch.object(cli.os, "link", side_effect=OSError("injected publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                cli._publish(self.receipt, {"complete": True})
        self.assertFalse(self.receipt.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cuda_configuration_precedes_thread_setup_without_cuda_work(self):
        self.configuration.stop()
        args = cli.parser().parse_args(self.args + ["--device", "cuda:0", "--interop-threads", "2"])
        events = []
        with patch.object(cli, "configure_strict_profile", side_effect=lambda: events.append("strict")), \
             patch.object(cli.torch, "get_num_interop_threads", return_value=12), \
             patch.object(cli.torch, "set_num_interop_threads", side_effect=lambda n: events.append(("interop", n))), \
             patch.object(cli.torch, "set_num_threads", side_effect=lambda n: events.append(("threads", n))):
            cli._configure_runtime(args)
        self.assertEqual(events, ["strict", ("interop", 2), ("threads", 1)])

    def test_explicit_device_spelling_is_preserved_and_unsupported_devices_reject(self):
        for device in ("cpu", "cuda", "cuda:0"):
            args = cli.parser().parse_args(self.args + ["--device", device])
            self.assertEqual(args.device, device)
        for device in ("mps", "cuda:-1", "cuda:00"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(self.args + ["--device", device])

    def test_runtime_setup_failure_stops_before_load(self):
        self.configured.side_effect = RuntimeError("runtime setup rejected")
        with patch.object(cli.CycleOwner, "load") as load:
            with self.assertRaisesRegex(RuntimeError, "runtime setup rejected"):
                cli.main(self.args + ["--max-cycles", "0"])
            load.assert_not_called()
        self.assertEqual(json.loads(self.receipt.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
