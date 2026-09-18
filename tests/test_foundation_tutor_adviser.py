"""Eight local decision slots; transport is always mocked, no lessons or learner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import foundation_tutor_adviser as adviser


class TutorAdviserTests(unittest.TestCase):
    work = dict(decision_slots=0, mock_api_calls=0, mock_metadata_calls=0,
                mock_chat_calls=0, mock_returned_chat_envelopes=0,
                real_network_calls=0, canonical_episode_generations=0,
                learner_models=0, optimizer_updates=0)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=adviser._native(tempfile.gettempdir()))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.block_network = mock.patch("http.client.HTTPConnection", side_effect=AssertionError("real network forbidden"))
        self.block_network.start(); self.addCleanup(self.block_network.stop)
        self.block_transport = mock.patch.object(adviser.transport, "_request_json", side_effect=AssertionError("unmocked transport forbidden"))
        self.transport = self.block_transport.start(); self.addCleanup(self.block_transport.stop)

    def request(self, *, mode="local"):
        config = dict(stage_updates=12, final_updates=18, micro_batch_size=2, rehearsal_every=2)
        metrics = {key: dict(count=0, total=2) for key in adviser.METRICS}
        current = dict(weights_sha256="1"*64, lifetime_updates=90,
                       by_family={family: deepcopy(metrics) for family in adviser.FAMILIES})
        return dict(schema=adviser.REQUEST_SCHEMA,
            parent=dict(identity_sha256="2"*64, weights_sha256="1"*64, cycle=1, lifetime_updates=90),
            catalogue=dict(schema=adviser.CATALOGUE_SCHEMA, profiles=[
                dict(id="foundation", order="curriculum", layout="original", plan_config=deepcopy(config)),
                dict(id="mixed_layout", order="mixed", layout="varied", plan_config=deepcopy(config))]),
            development=dict(schema=adviser.EVIDENCE_SCHEMA, role="development", current=current,
                             reference=deepcopy(current)), procedural_profile_id="foundation", mode=mode,
            teacher=dict(model="fixture:tiny", sha256="3"*64) if mode == "local" else None,
            max_seconds=10, source_sha256=adviser.source_hashes())

    def fake_service(self, *, content='{"profile_id":"mixed_layout"}', failure=None,
                     before_digest="3"*64, after_digest="3"*64):
        calls, tag_count = [], 0
        def service(route, payload, deadline):
            nonlocal tag_count
            self.work["mock_api_calls"] += 1
            calls.append((route, deepcopy(payload), deadline))
            if route == "/api/tags":
                self.work["mock_metadata_calls"] += 1
                tag_count += 1
                return dict(models=[dict(name="fixture:tiny", digest=before_digest if tag_count == 1 else after_digest)])
            if route == "/api/show":
                self.work["mock_metadata_calls"] += 1
                return dict(details={"family": "fixture"})
            self.assertEqual(route, "/api/chat")
            self.work["mock_chat_calls"] += 1
            self.assertEqual(payload["keep_alive"], 0)
            self.assertIs(payload["stream"], False)
            self.assertEqual(set(payload["format"]["properties"]), {"profile_id"})
            self.assertNotIn("withdrawn", payload["format"]["properties"]["profile_id"]["enum"])
            self.assertEqual({entry[2] for entry in calls}, {deadline})
            if failure is not None:
                raise failure
            self.work["mock_returned_chat_envelopes"] += 1
            return dict(model="fixture:tiny", done=True, done_reason="stop",
                message=dict(role="assistant", content=content), prompt_eval_count=40, eval_count=4,
                total_duration=100, load_duration=10, prompt_eval_duration=20, eval_duration=60)
        self.transport.side_effect = service
        return calls

    def decide(self, request, slot="decision"):
        self.work["decision_slots"] += 1
        return adviser.decide(self.root/slot, request=request,
                              expected_request_sha256=adviser.request_sha256(request))

    def test_accepted_response_is_durable_reusable_and_tamper_evident(self):
        request = self.request(); original = deepcopy(request)
        calls = self.fake_service()
        slot = "/".join(["long_path_"+"x"*40]*6)
        result = self.decide(request, slot)
        directory = adviser._native(self.root/slot)
        self.assertEqual(result["decision"]["outcome"], "local_accepted")
        self.assertEqual(result["decision"]["profile_id"], "mixed_layout")
        self.assertEqual(result["decision"]["coverage"]["updates"], 90)
        self.assertEqual(len(result["decision"]["coverage"]["mixed_tail_depth_turn_bundles"]), 18)
        raw = (directory/"response.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result["decision"]["teacher_cost"]["response_file_sha256"])
        self.assertEqual(json.loads(raw)["message"]["content"], '{"profile_id":"mixed_layout"}')
        self.assertEqual(request, original)
        before_calls = len(calls)
        with self.assertRaises(adviser.DecisionPinRequired):
            adviser.decide(directory, request=request, expected_request_sha256=adviser.request_sha256(request))
        reused = adviser.decide(directory, request=request, expected_request_sha256=adviser.request_sha256(request),
                                expected_decision_sha256=result["decision_sha256"])
        self.assertTrue(reused["reused"])
        self.assertEqual(reused["decision"], result["decision"])
        self.assertEqual(reused["invocation_cost"]["api_attempts"], 0)
        self.assertEqual(len(calls), before_calls)
        (directory/"response.json").write_bytes(raw+b" ")
        with self.assertRaises(ValueError):
            adviser.load_decision(directory/"decision.json", expected_sha256=result["decision_sha256"],
                                  expected_request_sha256=adviser.request_sha256(request))

    def test_invalid_selection_falls_back_and_preserves_raw_cost(self):
        request = self.request(); self.fake_service(content='{"profile_id":"withdrawn"}')
        result = self.decide(request)
        decision = result["decision"]
        self.assertEqual(decision["outcome"], "procedural_fallback")
        self.assertEqual(decision["profile_id"], "foundation")
        self.assertEqual(decision["rejection_reason"], "selection_unavailable_or_invalid")
        self.assertEqual(decision["teacher_cost"]["api_attempts"], 4)
        self.assertEqual(decision["teacher_cost"]["usage"]["eval_count"], 4)
        self.assertEqual(adviser.load_decision(self.root/"decision"/"decision.json",
            expected_sha256=result["decision_sha256"], expected_request_sha256=adviser.request_sha256(request)), decision)

    def test_unknown_timeout_and_interrupt_retain_intent_and_refuse_replay(self):
        for index, error in enumerate((TimeoutError("fixture timeout"), KeyboardInterrupt("fixture interruption"))):
            with self.subTest(error=type(error).__name__):
                request = self.request(); calls = self.fake_service(failure=error)
                expected = adviser.UnknownTutorRequest if isinstance(error, TimeoutError) else KeyboardInterrupt
                with self.assertRaises(expected) as raised:
                    self.decide(request, str(index))
                report = raised.exception.tutor_report
                self.assertTrue(report["teacher_cost"]["physical_teacher_work_unknown"])
                self.assertEqual(report["teacher_cost"]["chat_attempts"], 1)
                self.assertFalse((self.root/str(index)/"decision.json").exists())
                self.assertTrue((self.root/str(index)/"uncommitted.json").exists())
                count = len(calls)
                with self.assertRaises(adviser.UnknownTutorRequest):
                    adviser.decide(self.root/str(index), request=request,
                                   expected_request_sha256=adviser.request_sha256(request))
                self.assertEqual(len(calls), count)

    def test_caller_withdrawal_makes_no_teacher_request(self):
        request = self.request(mode="withdrawn")
        result = self.decide(request)
        self.transport.assert_not_called()
        decision = result["decision"]
        self.assertEqual(decision["outcome"], "teacher_withdrawn")
        self.assertEqual(decision["profile_id"], "foundation")
        self.assertEqual(decision["teacher_cost"]["api_attempts"], 0)
        self.assertEqual(decision["teacher_cost"]["chat_attempts"], 0)
        self.assertFalse((self.root/"decision"/"response.json").exists())

    def test_actual_catalogue_coverage_rehearsal_and_evidence_gate(self):
        original = self.request()
        for case in ("coverage", "rehearsal", "unequal_cost", "audit", "source", "hash"):
            request = deepcopy(original)
            if case == "coverage":
                for profile in request["catalogue"]["profiles"]:
                    profile["plan_config"]["final_updates"] = 6
            elif case == "rehearsal":
                for profile in request["catalogue"]["profiles"]:
                    profile["plan_config"]["stage_updates"] = 4
            elif case == "unequal_cost":
                request["catalogue"]["profiles"][1]["plan_config"]["micro_batch_size"] = 4
            elif case == "audit":
                request["development"]["role"] = "audit"
            elif case == "source":
                request["source_sha256"]["experiments/foundation_plan.py"] = "0"*64
            pin = "0"*64 if case == "hash" else adviser.request_sha256(request)
            with self.subTest(case=case), self.assertRaises(ValueError):
                adviser.decide(self.root/case, request=request, expected_request_sha256=pin)
            self.assertFalse((self.root/case/"intent.json").exists())
        self.transport.assert_not_called()

    def test_source_drift_after_response_leaves_uncommitted_evidence(self):
        request = self.request(); calls = self.fake_service()
        changed = deepcopy(request["source_sha256"])
        changed["experiments/foundation_tutor_adviser.py"] = "0"*64
        with mock.patch.object(adviser, "source_hashes", side_effect=[request["source_sha256"], changed]):
            with self.assertRaisesRegex(RuntimeError, "source changed") as raised:
                self.decide(request)
        self.assertFalse(raised.exception.tutor_report["teacher_cost"]["physical_teacher_work_unknown"])
        self.assertTrue((self.root/"decision"/"response.json").exists())
        self.assertFalse((self.root/"decision"/"decision.json").exists())
        count = len(calls)
        with self.assertRaises(adviser.UnknownTutorRequest):
            adviser.decide(self.root/"decision", request=request, expected_request_sha256=adviser.request_sha256(request))
        self.assertEqual(len(calls), count)

    def test_installed_identity_is_required_before_and_after_chat(self):
        for index, before, after, expected_calls in ((0, "0"*64, "3"*64, 1), (1, "3"*64, "0"*64, 4)):
            with self.subTest(stage="before" if index == 0 else "after"):
                request = self.request()
                calls = self.fake_service(before_digest=before, after_digest=after)
                result = self.decide(request, str(index))
                self.assertEqual(result["decision"]["outcome"], "procedural_fallback")
                self.assertEqual(result["decision"]["profile_id"], "foundation")
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(result["decision"]["teacher_cost"]["chat_attempts"], index)
                self.assertEqual(result["decision"]["teacher_cost"]["keep_alive"], 0)


if __name__ == "__main__":
    unittest.main()
