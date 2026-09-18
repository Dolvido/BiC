"""Causal cross-turn language wiring with unchanged legacy checkpoint structure."""

import copy
import io
import unittest

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec, LanguageBrain, LanguageConfig
from brain_in_computer.model import Brain, BrainConfig, REGION_NAMES
from brain_in_computer.tasks import TaskStream


class LanguageStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(409)
        self.model = LanguageBrain(
            Brain(BrainConfig(hidden_size=8, memory_slots=3)),
            LanguageConfig(hidden_size=12, embedding_size=6,
                           max_input_bytes=48, max_output_bytes=16),
        ).double()
        raw = TaskStream(809).sample(2, delay=3).observations
        self.observations = {name: value if name == "tokens" else value.double()
                             for name, value in raw.items()}
        self.ids, self.lengths = self.model.prepare_inputs(["The circle is red.", "The box is blue."])
        self.next_ids, self.next_lengths = self.model.prepare_inputs(["What color was it?", "What color was it?"])
        self.decoder = torch.tensor([[ByteCodec.BOS, 117, 104], [ByteCodec.BOS, 101, 111]])

    def chunk(self, start, stop):
        return {name: value[:, start:stop] for name, value in self.observations.items()}

    def assert_tree_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name, value in left.items():
            if isinstance(value, dict):
                self.assert_tree_equal(value, right[name])
            else:
                self.assertTrue(torch.equal(value, right[name]), name)

    def call(self, observations=None, state=None, *, model=None, next_turn=False, decoder=None, **options):
        model = self.model if model is None else model
        return model.forward_with_state(
            self.observations if observations is None else observations,
            self.next_ids if next_turn else self.ids,
            self.next_lengths if next_turn else self.lengths,
            self.decoder if decoder is None else decoder, state, **options,
        )

    def legacy_computation(self):
        # Reproduce the original independent-episode computation to catch more
        # than the new forward and forward_with_state agreeing with each other.
        comprehension, last = self.model.posterior_temporal(self.ids, self.lengths)
        context = self.model.semantic_bridge.comprehend(last)
        outputs = self.model.brain(
            self.observations, return_activity=True,
            language_context=context[:, None, :].expand(-1, self.observations["tokens"].shape[1], -1),
        )
        activity = outputs["region_activity"]
        regional = torch.cat([activity[name][:, -1]
                              for name in ("prefrontal_cortex", "parietal_association", "hippocampus")], dim=-1)
        production = self.model.semantic_bridge.formulate(regional)
        return {**outputs, "language_logits": self.model.inferior_frontal(self.decoder, production),
                "concept_context": context, "production_context": production,
                "comprehension_states": comprehension}

    def test_fresh_forward_is_bitwise_legacy_and_checkpoint_keys_do_not_change(self):
        weights = copy.deepcopy(self.model.state_dict())
        keys = set(weights)
        with torch.no_grad():
            old = self.legacy_computation()
            fresh = self.model(self.observations, self.ids, self.lengths, self.decoder)
            explicit, _ = self.call()
            self.assert_tree_equal(old, fresh)
            self.assert_tree_equal(old, explicit)
        self.assertEqual(set(self.model.state_dict()), keys)
        self.assert_tree_equal(weights, self.model.state_dict())
        recreated = LanguageBrain(Brain(self.model.brain.config), self.model.config).double()
        result = recreated.load_state_dict(weights, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        with torch.no_grad():
            self.assert_tree_equal(old, recreated(self.observations, self.ids, self.lengths, self.decoder))

    def test_legacy_forward_preserves_brain_module_hooks_once(self):
        events = []
        before = self.model.brain.register_forward_pre_hook(lambda *_: events.append("before"))
        after = self.model.brain.register_forward_hook(lambda *_: events.append("after"))
        try:
            self.model(self.observations, self.ids, self.lengths, self.decoder)
        finally:
            before.remove()
            after.remove()
        self.assertEqual(events, ["before", "after"])

    def test_same_utterance_whole_episode_equals_observation_chunks(self):
        memory_context = torch.randn(2, 5, 8, dtype=torch.float64)
        for ablate in ((), ("hippocampus",), ("prefrontal_cortex", "cerebellum")):
            with self.subTest(ablate=ablate), torch.no_grad():
                full, end_full = self.call(memory_context=memory_context, ablate=ablate)
                parts, state = [], None
                for start, stop in ((0, 1), (1, 3), (3, 5)):
                    part, state = self.call(self.chunk(start, stop), state,
                                            memory_context=memory_context[:, start:stop], ablate=ablate)
                    parts.append(part)
                for name in ("logits", "visual_logits", "auditory_logits", "prediction", "value"):
                    self.assertTrue(torch.equal(full[name], torch.cat([part[name] for part in parts], dim=1)), name)
                for region in REGION_NAMES:
                    self.assertTrue(torch.equal(full["region_activity"][region],
                                                torch.cat([part["region_activity"][region] for part in parts], dim=1)), region)
                for name in ("language_logits", "concept_context", "production_context", "comprehension_states"):
                    self.assertTrue(torch.equal(full[name], parts[-1][name]), name)
                self.assert_tree_equal(self.model.brain.state_to_dict(end_full),
                                       self.model.brain.state_to_dict(state))

    def test_conversation_serialization_restores_bitwise_next_turn(self):
        with torch.no_grad():
            _, state = self.call(self.chunk(0, 2))
            original_state = self.model.brain.state_to_dict(state)
            expected, expected_next = self.call(self.chunk(2, 5), state, next_turn=True)
            payload = {"model": self.model.state_dict(), "state": original_state}
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location="cpu", weights_only=True)
            restarted = LanguageBrain(Brain(self.model.brain.config), self.model.config).double()
            restarted.load_state_dict(saved["model"], strict=True)
            restored = restarted.brain.state_from_dict(saved["state"])
            actual, actual_next = self.call(self.chunk(2, 5), restored, model=restarted, next_turn=True)
            self.assert_tree_equal(expected, actual)
            self.assert_tree_equal(self.model.brain.state_to_dict(expected_next),
                                   restarted.brain.state_to_dict(actual_next))
            self.assert_tree_equal(original_state, self.model.brain.state_to_dict(state))

    def test_prior_text_and_state_causally_affect_next_turn(self):
        altered_ids, altered_lengths = self.model.prepare_inputs(["The circle is green.", "The box is yellow."])
        with torch.no_grad():
            _, previous = self.call(self.chunk(0, 2))
            _, changed_previous = self.model.forward_with_state(
                self.chunk(0, 2), altered_ids, altered_lengths, self.decoder)
            continued, _ = self.call(self.chunk(2, 5), previous, next_turn=True)
            changed, _ = self.call(self.chunk(2, 5), changed_previous, next_turn=True)
            reset, _ = self.call(self.chunk(2, 5), next_turn=True)
            for name in ("logits", "production_context", "language_logits"):
                self.assertFalse(torch.equal(continued[name], changed[name]), name)
                self.assertFalse(torch.equal(continued[name], reset[name]), name)
            # The utterance encoder itself is deliberately unchanged by regional
            # context; prior text affects integration in the recurrent brain.
            self.assertTrue(torch.equal(continued["concept_context"], changed["concept_context"]))
            self.assertTrue(torch.equal(continued["comprehension_states"], reset["comprehension_states"]))

    def test_current_text_lesion_removes_its_effect_even_with_carried_state(self):
        altered_ids, altered_lengths = self.model.prepare_inputs(["Where is the box?", "Where is the circle?"])
        with torch.no_grad():
            _, state = self.call(self.chunk(0, 2))
            first, first_state = self.call(self.chunk(2, 5), state, next_turn=True, ablate=("temporal_language",))
            changed, changed_state = self.model.forward_with_state(
                self.chunk(2, 5), altered_ids, altered_lengths, self.decoder, state,
                ablate=("temporal_language",))
            self.assertTrue(torch.equal(first["language_logits"], changed["language_logits"]))
            self.assertTrue(torch.equal(first["logits"], changed["logits"]))
            self.assert_tree_equal(self.model.brain.state_to_dict(first_state),
                                   self.model.brain.state_to_dict(changed_state))

    def test_decoder_inputs_cannot_change_carried_brain_state(self):
        changed_decoder = self.decoder.clone()
        changed_decoder[:, 1:] += 20
        with torch.no_grad():
            first, first_state = self.call(self.chunk(0, 2))
            changed, changed_state = self.call(self.chunk(0, 2), decoder=changed_decoder)
            self.assertFalse(torch.equal(first["language_logits"], changed["language_logits"]))
            self.assert_tree_equal(self.model.brain.state_to_dict(first_state),
                                   self.model.brain.state_to_dict(changed_state))
            after_first, _ = self.call(self.chunk(2, 5), first_state, next_turn=True)
            after_changed, _ = self.call(self.chunk(2, 5), changed_state, next_turn=True)
            self.assert_tree_equal(after_first, after_changed)

    def test_next_turn_loss_reaches_first_turn_language_and_regional_state(self):
        first, state = self.call(self.chunk(0, 2))
        first["comprehension_states"].retain_grad()
        first["concept_context"].retain_grad()
        state.previous_prefrontal.retain_grad()
        output, _ = self.call(self.chunk(2, 5), state, next_turn=True)
        targets = torch.tensor([[120, 121, ByteCodec.EOS], [124, 125, ByteCodec.EOS]])
        F.cross_entropy(output["language_logits"].flatten(0, 1), targets.flatten()).backward()
        self.assertIsNotNone(first["concept_context"].grad)
        self.assertGreater(float(first["concept_context"].grad.abs().sum()), 0.0)
        self.assertGreater(float(state.previous_prefrontal.grad.abs().sum()), 0.0)
        for name, module in (("encoder", self.model.posterior_temporal),
                             ("decoder", self.model.inferior_frontal),
                             ("bridge", self.model.semantic_bridge),
                             ("regional core", self.model.brain)):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), name)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0, name)

    def test_detach_explicitly_cuts_prior_turn_graph_and_state_schema_is_validated(self):
        first, state = self.call(self.chunk(0, 2))
        first["concept_context"].retain_grad()
        output, _ = self.call(self.chunk(2, 5), state.detached(), next_turn=True)
        output["language_logits"].square().mean().backward()
        self.assertIsNone(first["concept_context"].grad)
        with self.assertRaises(ValueError):
            self.call(state={})
        with self.assertRaises(ValueError):
            self.call(state=self.model.brain.initial_state(1))


if __name__ == "__main__":
    unittest.main()
