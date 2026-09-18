"""English-path, continuing-state and supervision-boundary checks."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_curriculum import generate_dialogues
from brain_in_computer.dialogue_student import (
    DialogueSession, build_dialogue_student, checkpoint_digest, encode_dialogues,
    evaluate_dialogues, train_dialogue_candidate, _generate_reply_tokens, _run_turn,
)
from brain_in_computer.language import ByteCodec


class DialogueStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.model = build_dialogue_student(42)
        self.episodes = generate_dialogues(100, 4, 'train', focus='grounding')

    def test_reproducible_initialization_preserves_external_rng(self):
        before = torch.random.get_rng_state().clone()
        same = build_dialogue_student(42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(checkpoint_digest(same), checkpoint_digest(self.model))
        self.assertNotEqual(checkpoint_digest(build_dialogue_student(43)), checkpoint_digest(self.model))
        self.assertEqual(self.model.brain.config.vocab_size, 1)

    def test_text_bytes_are_actual_inputs_and_labels_never_become_observations(self):
        batches = encode_dialogues(self.model, self.episodes)
        self.assertEqual(len(batches), 6)
        for position, batch in enumerate(batches):
            for row, episode in enumerate(self.episodes):
                self.assertEqual(self.model.codec.decode(batch['text_ids'][row]), episode['turns'][position]['text'])
                self.assertEqual(self.model.codec.decode(torch.cat((batch['decoder_input_ids'][row, :1],
                                                                    batch['reply_targets'][row]))),
                                 episode['turns'][position]['reply'])
            self.assertEqual(set(batch['observations']), {'visual', 'auditory', 'body', 'feedback', 'tokens'})
            self.assertTrue(all(not bool(value.any()) for value in batch['observations'].values()))
            self.assertTrue(torch.equal(batch['decoder_input_ids'][:, 1:], batch['reply_targets'][:, :-1]))
        blank = encode_dialogues(self.model, self.episodes, blank_text=True)
        self.assertTrue(all(batch['text_ids'].shape[1] == 2 for batch in blank))

    def test_cross_turn_action_loss_reaches_earlier_english_and_memory(self):
        batches = encode_dialogues(self.model, self.episodes)
        state = None
        first_context = None
        for batch in batches:
            output, state = _run_turn(self.model, batch, state)
            if first_context is None:
                first_context = output['concept_context']
                first_context.retain_grad()
        F.cross_entropy(output['logits'][:, -1], batches[-1]['targets']).backward()
        self.assertIsNotNone(first_context.grad)
        self.assertGreater(float(first_context.grad.abs().sum()), 0)
        for module in (self.model.posterior_temporal, self.model.semantic_bridge,
                       self.model.brain.regions['hippocampus']):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)

    def test_blank_text_and_reset_state_remove_paired_rule_information(self):
        episodes = generate_dialogues(100, 2, 'dev', focus='grounding')
        normal = encode_dialogues(self.model, episodes)
        blank = encode_dialogues(self.model, episodes, blank_text=True)
        state = blank_state = None
        saw_memory_effect = False
        with torch.inference_mode():
            for batch, blank_batch in zip(normal, blank):
                continued, state = _run_turn(self.model, batch, state)
                forgotten, _ = _run_turn(self.model, batch, None)
                no_english, blank_state = _run_turn(self.model, blank_batch, blank_state)
                torch.testing.assert_close(no_english['logits'][0], no_english['logits'][1], rtol=1e-6, atol=1e-7)
                if bool(batch['targets'].ne(3).all()):
                    torch.testing.assert_close(forgotten['logits'][0], forgotten['logits'][1], rtol=1e-6, atol=1e-7)
                    saw_memory_effect |= not torch.equal(continued['logits'][0], continued['logits'][1])
        self.assertTrue(saw_memory_effect)
        metrics = evaluate_dialogues(self.model, episodes, blank_text=True)
        self.assertEqual(metrics['counterfactual_accuracy'], 0)
        self.assertEqual(metrics['counterfactual_prediction_flip_rate'], 0)

    def test_evaluation_never_uses_teacher_or_updates_weights_or_modes(self):
        episodes = generate_dialogues(100, 4, 'dev', focus='revision')
        before = checkpoint_digest(self.model)
        self.model.train()
        self.model.posterior_temporal.eval()
        modes = [module.training for module in self.model.modules()]
        with patch('brain_in_computer.dialogue_curriculum.dialogue_oracle', side_effect=AssertionError('teacher leaked')), \
             patch('brain_in_computer.dialogue_curriculum.validate_dialogue', side_effect=AssertionError('teacher leaked')):
            metrics = evaluate_dialogues(self.model, episodes)
        self.assertEqual(before, checkpoint_digest(self.model))
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertEqual(metrics['total'], 24)
        self.assertEqual(sum(row['total'] for row in metrics['per_target'].values()), 24)
        self.assertEqual(sum(row['total'] for row in metrics['per_kind'].values()), metrics['query_total'])
        self.assertFalse(metrics['teacher_used_for_policy'])
        self.assertTrue(metrics['free_running_replies'])

    def test_reply_scoring_can_be_skipped_without_changing_action_metrics(self):
        full = evaluate_dialogues(self.model, self.episodes)
        with patch('brain_in_computer.dialogue_student._generate_reply_tokens', side_effect=AssertionError('reply scoring should be skipped')):
            fast = evaluate_dialogues(self.model, self.episodes, score_replies=False)
        for key in ('accuracy', 'query_accuracy', 'loss', 'per_kind', 'per_target', 'counterfactual_accuracy'):
            self.assertEqual(full[key], fast[key])
        self.assertIsNone(fast['reply_exact_accuracy'])
        self.assertFalse(fast['free_running_replies'])

    def test_free_reply_generation_equals_full_decoder_autoregression(self):
        batch = encode_dialogues(self.model, self.episodes[:1])[0]
        with torch.inference_mode():
            output, _ = _run_turn(self.model, batch, None)
            context = output['production_context']
            generated = _generate_reply_tokens(self.model, context)
            for index in range(1, generated.shape[1]):
                expected = self.model.inferior_frontal(generated[:, :index], context)[:, -1].argmax(dim=-1)
                self.assertTrue(torch.equal(generated[:, index], expected))
                if int(generated[0, index]) == ByteCodec.EOS:
                    break

    def test_training_updates_language_and_core_and_resumes_adam_exactly(self):
        initial = copy.deepcopy(self.model.state_dict())
        args = dict(seed=17, batch_size=2, learning_rate=.002)
        full = train_dialogue_candidate(initial, self.episodes, steps=2, **args)
        first = train_dialogue_candidate(initial, self.episodes, steps=1, **args)
        second = train_dialogue_candidate(first['state_dict'], self.episodes, steps=1,
                                          optimizer_state=first['optimizer_state'],
                                          sampler_state=first['sampler_state'], **args)
        self.assertEqual(full['updates'], 2)
        self.assertEqual(full['episodes_seen'], 4)
        self.assertEqual(full['turns_seen'], 24)
        self.assertTrue(torch.equal(full['sampler_state'], second['sampler_state']))
        for key, value in initial.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]), key)
            self.assertTrue(torch.equal(full['state_dict'][key], second['state_dict'][key]), key)
        for prefix in ('posterior_temporal.', 'inferior_frontal.', 'semantic_bridge.', 'brain.regions.prefrontal_cortex.'):
            self.assertTrue(any(not torch.equal(value, first['state_dict'][key])
                                for key, value in initial.items() if key.startswith(prefix)), prefix)

    def test_training_refuses_held_out_tampered_and_nonfinite_data(self):
        args = dict(seed=17, steps=1, batch_size=2)
        for split in ('dev', 'audit'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'held-out'):
                train_dialogue_candidate(self.model.state_dict(), generate_dialogues(100, 2, split), **args)
        forged = copy.deepcopy(self.episodes)
        forged[0]['turns'][0]['observations']['feedback'][0][0] = 1
        with self.assertRaises(ValueError):
            train_dialogue_candidate(self.model.state_dict(), forged, **args)
        initial = copy.deepcopy(self.model.state_dict())
        next(iter(initial.values())).fill_(float('nan'))
        with self.assertRaises(ValueError):
            train_dialogue_candidate(initial, self.episodes, **args)

    def test_expired_deadline_preserves_weights_and_performs_no_updates(self):
        result = train_dialogue_candidate(self.model.state_dict(), self.episodes, seed=17,
                                           steps=10, batch_size=2, deadline=time.monotonic() - 1)
        self.assertEqual(result['updates'], 0)
        self.assertIsNone(result['loss'])
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, result['state_dict'][key]))

    def test_session_checkpoint_restores_next_turn_bitwise_without_transcript(self):
        episode = self.episodes[0]
        session = DialogueSession(self.model)
        for turn in episode['turns'][:3]:
            session.step(turn['observations'], turn['text'], generate_reply=False)
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'session.pt'
            session.save(file)
            resumed = DialogueSession.load(file)
            self.assertEqual(session.checkpoint_sha256, resumed.checkpoint_sha256)
            self.assertEqual(session.turns, resumed.turns)
            for turn in episode['turns'][3:]:
                expected = session.step(turn['observations'], turn['text'])
                actual = resumed.step(turn['observations'], turn['text'])
                self.assertEqual(expected, actual)
            for key, value in session.state_dict()['brain_state'].items():
                self.assertTrue(torch.equal(value, resumed.state_dict()['brain_state'][key]), key)

    def test_session_rejects_incompatible_weights_or_unexpected_input_fields(self):
        session = DialogueSession(self.model)
        saved = session.state_dict()
        with self.assertRaisesRegex(ValueError, 'binding'):
            DialogueSession(build_dialogue_student(43)).load_state_dict(saved)
        turn = self.episodes[0]['turns'][0]
        with self.assertRaises(ValueError):
            session.step(dict(turn['observations'], target=turn['target']), turn['text'])
        with torch.no_grad():
            next(self.model.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            session.step(turn['observations'], turn['text'])


if __name__ == '__main__':
    unittest.main()
