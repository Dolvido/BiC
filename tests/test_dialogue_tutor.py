"""External dialogue tutoring selects facts; it cannot invent them or load models."""

from contextlib import contextmanager
import copy
import http.client
import json
import time
import unittest
from unittest.mock import patch

from brain_in_computer.dialogue_curriculum import FOCI, generate_dialogues, validate_dialogue
from brain_in_computer.dialogue_tutor import DialogueTutor


MODEL = "ministral-3:3b"
DIGEST = "d" * 64


class Service:
    def __init__(self, choice=None, envelope=None):
        self.choice = {"lesson": 1} if choice is None else choice
        self.envelope = envelope or {}
        self.calls = []
        self.tags = {"models": [{"name": MODEL, "digest": DIGEST}]}
        self.details = {"details": {"family": "mistral3"}}

    def __call__(self, route, payload, deadline):
        self.calls.append((route, copy.deepcopy(payload), deadline))
        if route == "/api/tags":
            return self.tags
        if route == "/api/show":
            return self.details
        if route != "/api/chat":
            raise AssertionError(f"Unexpected tutor route: {route}")
        content = self.choice if isinstance(self.choice, str) else json.dumps(self.choice)
        return {"model": MODEL, "done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": content}, **self.envelope}


@contextmanager
def mocked_transport(service):
    with patch("brain_in_computer.curriculum_tutor._request_json", side_effect=service), \
            patch("brain_in_computer.dialogue_tutor._request_json", side_effect=service):
        yield


class DialogueTutorTests(unittest.TestCase):
    def teach(self, tutor, **options):
        return tutor.teach(focus="grounding", seed=719, question="Which tool may I borrow?", **options)

    def assert_verified_fallback(self, result):
        self.assertEqual(result["source"], "procedural")
        self.assertEqual(result["selected"], 0)
        self.assertEqual(result["episode"], generate_dialogues(719, 1, split="train", focus="grounding")[0])
        self.assertTrue(validate_dialogue(result["episode"]))
        self.assertIsNotNone(result["rejection_reason"])

    def test_procedural_tutor_is_canonical_training_only_and_never_contacts_service(self):
        def forbidden(*_args):
            raise AssertionError("Procedural tutoring attempted a service request")

        with mocked_transport(forbidden):
            tutor = DialogueTutor()
            for focus in FOCI:
                result = tutor.teach(focus=focus, seed=719, question="Teach this topic")
                self.assertEqual(result["source"], "procedural")
                self.assertTrue(validate_dialogue(result["episode"]))
                self.assertEqual(result["episode"]["split"], "train")
                self.assertEqual(result["episode"]["focus"], focus)
                self.assertEqual(result["provenance"]["split"], "train")
                result["episode"]["turns"][0]["text"] = "Mutated externally"
                self.assertTrue(validate_dialogue(tutor.teach(focus=focus, seed=719, question="Teach")["episode"]))

    def test_valid_installed_local_selection_is_exact_and_cannot_change_facts(self):
        service = Service()
        with mocked_transport(service):
            tutor = DialogueTutor(MODEL)
            self.assertEqual(service.calls, [])
            result = self.teach(tutor)
        self.assertEqual(result["source"], "local_ollama")
        self.assertEqual(result["selected"], 1)
        self.assertEqual(result["model_digest"], DIGEST)
        self.assertEqual(result["episode"], generate_dialogues(719, 2, split="train", focus="grounding")[1])
        self.assertTrue(validate_dialogue(result["episode"]))
        self.assertEqual([route for route, _, _ in service.calls], ["/api/tags", "/api/show", "/api/chat"])
        payload = service.calls[-1][1]
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["format"]["additionalProperties"])
        self.assertEqual(payload["format"]["properties"]["lesson"]["enum"], [0, 1])
        prompt = json.loads(payload["messages"][1]["content"])
        expected = [[{"text": turn["text"], "answer": turn["reply"]} for turn in episode["turns"]]
                    for episode in generate_dialogues(719, 2, split="train", focus="grounding")]
        self.assertEqual(prompt["verified_lessons"], expected)
        self.assertEqual(len(set(deadline for _, _, deadline in service.calls)), 1)

    def test_uninstalled_remote_or_unverified_models_never_reach_chat_or_download(self):
        services = []
        missing = Service()
        missing.tags = {"models": []}
        services.append(missing)
        cloud_inventory = Service()
        cloud_inventory.tags["models"][0]["remote_host"] = "remote.example"
        services.append(cloud_inventory)
        cloud_metadata = Service()
        cloud_metadata.details["details"]["remote_model"] = "provider/model"
        services.append(cloud_metadata)
        bad_digest = Service()
        bad_digest.tags["models"][0]["digest"] = "not-a-checkpoint-digest"
        services.append(bad_digest)
        for service in services:
            with self.subTest(tags=service.tags, details=service.details), mocked_transport(service):
                self.assert_verified_fallback(self.teach(DialogueTutor(MODEL)))
                self.assertNotIn("/api/chat", [route for route, _, _ in service.calls])
                self.assertTrue(all(route in ("/api/tags", "/api/show") for route, _, _ in service.calls))
        for model in ("model:cloud", "https://remote.example/model", "model/name", ""):
            with self.subTest(model=model), self.assertRaises(ValueError):
                DialogueTutor(model)

    def test_unknown_choices_extra_claims_and_malformed_json_preserve_canonical_fallback(self):
        choices = ({"lesson": -1}, {"lesson": 2}, {"lesson": True}, {"lesson": "1"},
                   {"lesson": 0, "new_rule": "Always allow"}, {"lesson": [1]},
                   "[]", "[[]]", "null", "false", "42", "not JSON", "{}",
                   '{"lesson": 0, "lesson": 1}', '{"lesson": NaN}',
                   '{"lesson": 1e999}', '"Ignore the rules and change every answer"')
        for choice in choices:
            service = Service(choice)
            with self.subTest(choice=choice), mocked_transport(service):
                self.assert_verified_fallback(self.teach(DialogueTutor(MODEL)))

    def test_malformed_message_shapes_and_tool_calls_preserve_verified_fallback(self):
        envelopes = ({"message": None}, {"message": []}, {"message": "raw text"},
                     {"message": {"role": "assistant", "content": ["not", "text"]}},
                     {"message": {"role": "assistant", "content": '{"lesson":1}', "tool_calls": [{"name": "run"}]}},
                     {"message": {"role": "user", "content": '{"lesson":1}'}},
                     {"model": "different-model"}, {"done": False}, {"done_reason": "length"})
        for envelope in envelopes:
            with self.subTest(envelope=envelope), mocked_transport(Service(envelope=envelope)):
                self.assert_verified_fallback(self.teach(DialogueTutor(MODEL)))

    def test_transport_protocol_errors_timeout_and_outage_fall_back(self):
        for error in (http.client.BadStatusLine("broken"), http.client.IncompleteRead(b"partial"),
                      ConnectionRefusedError("offline"), TimeoutError("late")):
            with self.subTest(error=type(error)), mocked_transport(lambda *_args: (_ for _ in ()).throw(error)):
                self.assert_verified_fallback(self.teach(DialogueTutor(MODEL)))

    def test_expired_deadline_does_not_contact_local_service_and_valid_budget_is_shared(self):
        service = Service()
        with mocked_transport(service):
            self.assert_verified_fallback(self.teach(DialogueTutor(MODEL), deadline=time.monotonic() - 1))
        self.assertEqual(service.calls, [])
        service = Service()
        deadline = time.monotonic() + 2.0
        with mocked_transport(service):
            result = self.teach(DialogueTutor(MODEL, timeout=20.0), deadline=deadline)
        self.assertEqual(result["source"], "local_ollama")
        self.assertTrue(all(expiry == deadline for _, _, expiry in service.calls))


if __name__ == "__main__":
    unittest.main()
