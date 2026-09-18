"""Tiny CPU architecture checks; explicit physical work ledger, no study scores."""
import copy
import io
import json
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from brain_in_computer.language import ByteCodec
from experiments.composition_data import pack_composition_episodes, pack_observations
from experiments.foundation_curriculum import generate_pair, validate_pair
from experiments.hierarchical_sequence_student import (
    ARCHITECTURE, HierarchicalSequenceStudent, build_hierarchical_sequence_student)
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective


class HierarchicalSequenceStudentTests(unittest.TestCase):
    accounting = {"forward_attempts": 0, "completed_forwards": 0,
        "rejected_or_failed_forwards": 0, "completed_forward_episodes": 0,
        "completed_forward_turns": 0, "completed_forward_observation_bytes": 0,
        "gradient_enabled_forwards": 0, "backward_calls": 0,
        "optimizer_attempts": 0, "completed_optimizer_updates": 0,
        "completed_optimizer_episode_exposures": 0}

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        cls.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        cls.config = SequenceConfig(width=16, layers=4, heads=2, feedforward=32, max_turns=12)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic, warn_only=cls.warn_only)
        print("HIERARCHICAL_CPU_ACCOUNTING=" + json.dumps(cls.accounting, sort_keys=True))

    def model(self, seed=731):
        return build_hierarchical_sequence_student(seed, config=self.config)

    def inputs(self, rows, *, reply="Yes.", padding=0):
        inputs = pack_observations(rows)
        if padding:
            inputs["token_ids"] = F.pad(inputs["token_ids"], (0, padding))
            inputs["valid_mask"] = inputs["token_ids"].ne(ByteCodec.PAD)
        prefix = torch.tensor(ByteCodec().encode(reply)[:-1], dtype=torch.long)
        inputs["decoder_input_ids"] = prefix[None, None, :].expand(len(rows), len(rows[0]), -1).clone()
        return inputs

    def forward(self, model, inputs, *, grad=False):
        type(self).accounting["forward_attempts"] += 1
        try:
            with torch.set_grad_enabled(grad):
                output = model(**inputs)
        except BaseException:
            type(self).accounting["rejected_or_failed_forwards"] += 1
            raise
        counts = type(self).accounting
        counts["completed_forwards"] += 1
        counts["completed_forward_episodes"] += inputs["token_ids"].shape[0]
        counts["completed_forward_turns"] += inputs["eos_positions"].numel()
        counts["completed_forward_observation_bytes"] += int(inputs["token_ids"].ge(3).sum())
        counts["gradient_enabled_forwards"] += int(grad)
        return output

    def backward(self, loss):
        type(self).accounting["backward_calls"] += 1
        loss.backward()

    def tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_exact_parameter_inventory_initial_tensors_and_rng_preservation(self):
        before = torch.random.get_rng_state().clone()
        flat = build_sequence_student(731, config=self.config)
        first, second = self.model(), self.model()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(first.parameter_counts(), flat.parameter_counts())
        self.assertEqual([(name, p.shape) for name, p in first.named_parameters()],
                         [(name, p.shape) for name, p in flat.named_parameters()])
        self.tree_equal(first.state_dict(), flat.state_dict())
        self.tree_equal(first.state_dict(), second.state_dict())
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in first.state_dict().values()))
        self.assertIs(first.tokens.weight, first.observation_head.weight)
        self.assertEqual(first.architecture, ARCHITECTURE)
        for layers in (1, 2, 3, 5):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "four blocks"):
                HierarchicalSequenceStudent(SequenceConfig(layers=layers))

    def test_actual_parallel_local_repacking_preserves_global_positions(self):
        model = self.model()
        rows = [["A", "BC", ""], ["longer", "x", "YZ"]]
        inputs = self.inputs(rows, padding=3)
        captured = []
        handles = [block.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
                   for block in model.blocks]
        try:
            output = self.forward(model, inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual([tuple(x.shape) for x in captured], [(6, 8, 16), (6, 8, 16), (2, 3, 16), (2, 3, 16)])
        for session, texts in enumerate(rows):
            start = 0
            for turn, text in enumerate(texts):
                ids = torch.tensor(ByteCodec().encode(text))
                positions = torch.arange(start, start+len(ids))
                expected = model.tokens(ids) + model.positions(positions)
                torch.testing.assert_close(captured[0][session*3+turn, :len(ids)], expected, rtol=0, atol=0)
                start += len(ids)
        self.assertNotIn("context_states", output)
        self.assertEqual(output["logits"].shape, (2, 3, 4))
        self.assertEqual(output["language_logits"].shape, (2, 3, 5, ByteCodec.VOCAB_SIZE))
        self.assertEqual(output["observation_language_logits"].shape, (*inputs["token_ids"].shape, ByteCodec.VOCAB_SIZE))
        self.assertEqual(output["local_byte_states"].shape, (*inputs["token_ids"].shape, 16))
        self.assertEqual(output["turn_context_states"].shape, (2, 3, 16))
        self.assertEqual(output["production_context"].shape, (2, 3, 128))

    def test_future_byte_and_turn_causality_in_training_and_inference_paths(self):
        model = self.model()
        for evaluation in (False, True):
            model.train(not evaluation)
            context = torch.inference_mode() if evaluation else torch.no_grad()
            with context:
                first = self.forward(model, self.inputs([["abcX", "question", "future"]]))
                byte_changed = self.forward(model, self.inputs([["abcY", "question", "future"]]))
                future_changed = self.forward(model, self.inputs([["abcX", "question", "a much longer ending"]]))
            for key in ("local_byte_states", "observation_language_logits"):
                torch.testing.assert_close(first[key][:, :4], byte_changed[key][:, :4], rtol=0, atol=1e-6)
            for key in ("logits", "language_logits", "turn_context_states", "production_context"):
                torch.testing.assert_close(first[key][:, :2], future_changed[key][:, :2], rtol=0, atol=1e-6)
            self.assertFalse(torch.equal(first["utterance_summaries"][:, 0], byte_changed["utterance_summaries"][:, 0]))

    def test_full_prefix_equivalence_and_local_auxiliary_receptive_field(self):
        model = self.model().eval()
        rows = ["AAAA", "BC", "a longer observation", "query"]
        full = self.forward(model, self.inputs([rows]))
        for length in range(1, len(rows)):
            inputs = self.inputs([rows[:length]], padding=4)
            prefix = self.forward(model, inputs)
            for key in ("logits", "language_logits", "turn_context_states"):
                torch.testing.assert_close(full[key][:, :length], prefix[key], rtol=0, atol=1e-6)
            end = int(inputs["lengths"][0])
            torch.testing.assert_close(full["local_byte_states"][:, :end], prefix["local_byte_states"][:, :end], rtol=0, atol=1e-6)
        changed = self.forward(model, self.inputs([["ZZZZ", *rows[1:]]]))
        boundary = len(ByteCodec().encode(rows[0]))
        torch.testing.assert_close(full["local_byte_states"][:, boundary:], changed["local_byte_states"][:, boundary:], rtol=0, atol=0)
        self.assertFalse(torch.equal(full["logits"][:, -1], changed["logits"][:, -1]))

    def test_padding_blank_and_other_sessions_do_not_change_visible_outputs(self):
        model = self.model()
        normal = self.inputs([["A.", "What?", ""], ["", "", ""]])
        padded = self.inputs([["A.", "What?", ""], ["", "", ""]], padding=37)
        changed = self.inputs([["A.", "What?", ""], ["Another session", "long", "observation"]])
        a, b, c = (self.forward(model, value) for value in (normal, padded, changed))
        for key in ("logits", "language_logits", "turn_context_states"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=1e-6)
            torch.testing.assert_close(a[key][0], c[key][0], rtol=0, atol=1e-6)
        self.assertEqual(float(b["local_byte_states"][~padded["valid_mask"]].abs().sum()), 0.)
        self.assertTrue(all(torch.isfinite(value).all() for value in b.values()))
        wide = build_hierarchical_sequence_student(732, config=SequenceConfig(
            width=8, layers=4, heads=2, feedforward=16, max_turns=12, max_positions=1560))
        maximal = self.inputs([["x"*128]*12, [""]*12])
        result = self.forward(wide, maximal)
        self.assertEqual(result["local_byte_states"].shape, (2, 1560, 8))
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))

    def test_decoder_prefix_and_supervision_cannot_enter_encoder(self):
        model = self.model()
        a = self.forward(model, self.inputs([["fact", "question"]], reply="Yes."))
        b = self.forward(model, self.inputs([["fact", "question"]], reply="No."))
        for key in set(a)-{"language_logits"}:
            self.assertTrue(torch.equal(a[key], b[key]))
        inputs = self.inputs([["fact", "question"]])
        inputs["decoder_input_ids"] = torch.full((1, 2, 1), ByteCodec.BOS, dtype=torch.long)
        self.assertEqual(self.forward(model, inputs)["language_logits"].shape, (1, 2, 1, 259))
        for metadata in ("family", "depth", "targets", "recipe", "observations"):
            bad = {**inputs, metadata: "must not enter"}
            with self.subTest(metadata=metadata), self.assertRaises(TypeError):
                self.forward(model, bad)

    def test_later_action_gradient_reaches_earlier_local_encoder_without_decoder(self):
        model = self.model()
        captured = []
        def capture(module, args, output):
            output.retain_grad()
            captured.append(output)
        handle = model.blocks[1].register_forward_hook(capture)
        try:
            output = self.forward(model, self.inputs([["Z earlier fact", "second", "query"]]), grad=True)
            output["utterance_summaries"].retain_grad()
            self.backward(F.cross_entropy(output["logits"][:, -1], torch.tensor([1])))
        finally:
            handle.remove()
        self.assertGreater(float(output["utterance_summaries"].grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(captured[0].grad[0].abs().sum()), 0.)
        self.assertGreater(float(model.tokens.weight.grad[ord("Z")+3].abs().sum()), 0.)
        for block in model.blocks:
            self.assertGreater(float(block.self_attn.in_proj_weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.inferior_frontal.parameters()))

    def canonical_batch(self, family="color", seed=981000001, depth=2):
        rows = generate_pair(family, seed, depth=depth, turns=8)
        return pack_composition_episodes(rows, training=True, pair_validator=validate_pair)

    def objective_forward(self, model, batch):
        return self.forward(model, {**batch["inputs"],
            "decoder_input_ids": batch["supervision"]["reply_decoder_input_ids"]}, grad=True)

    def test_unchanged_objective_contract_and_canonical_input_boundary(self):
        model, batch = self.model(), self.canonical_batch()
        with patch("experiments.foundation_curriculum.generate_pair", side_effect=AssertionError("oracle during inference")):
            output = self.objective_forward(model, batch)
        losses = sequence_objective(output, batch)
        self.assertEqual(set(losses), {"loss", "action_loss", "reply_loss", "observation_language_loss"})
        self.assertTrue(all(value.ndim == 0 and torch.isfinite(value) for value in losses.values()))
        torch.testing.assert_close(losses["loss"], losses["action_loss"]+.1*losses["reply_loss"]+.1*losses["observation_language_loss"], rtol=0, atol=0)
        self.backward(losses["loss"])
        for module in (model.tokens, model.norm, model.action_head, model.reply_context, model.inferior_frontal):
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checked_weight_envelope_rejects_architecture_config_and_alias_tampering(self):
        model, target = self.model(), self.model(734)
        saved = model.weight_checkpoint()
        target.load_weight_checkpoint(saved)
        self.tree_equal(model.state_dict(), target.state_dict())
        saved["weights"]["action_head.bias"].add_(1)
        self.tree_equal(model.state_dict(), target.state_dict())
        for kind in ("architecture", "config", "alias", "nan", "missing"):
            bad = model.weight_checkpoint()
            if kind == "architecture": bad["architecture"] = "flat-sequence"
            elif kind == "config": bad["config"]["width"] = float(self.config.width)
            elif kind == "alias": bad["weights"]["observation_head.weight"].add_(1)
            elif kind == "nan": bad["weights"]["action_head.bias"][0] = float("nan")
            else: del bad["weights"]["norm.bias"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                target.load_weight_checkpoint(bad)
            self.tree_equal(model.state_dict(), target.state_dict())
        with self.assertRaises(ValueError):
            target.load_weight_checkpoint(model.state_dict())
        # Ordinary PyTorch copying remains available, explicitly outside public
        # checkpoint validation, for a future architecture-aware trainer.
        target.load_state_dict(model.state_dict(), strict=True)

    def optimizer_step(self, model, optimizer, batch):
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_objective(self.objective_forward(model, batch), batch)["loss"]
        self.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        type(self).accounting["optimizer_attempts"] += 1
        optimizer.step()
        type(self).accounting["completed_optimizer_updates"] += 1
        type(self).accounting["completed_optimizer_episode_exposures"] += batch["inputs"]["token_ids"].shape[0]

    def test_tiny_cpu_optimizer_serialized_split_resume_is_exact(self):
        batches = [self.canonical_batch(family, 981000010+index, depth=index)
                   for index, family in enumerate(("color", "count", "switch"))]
        reference, first = self.model(), self.model()
        a = torch.optim.AdamW(reference.parameters(), lr=.001)
        b = torch.optim.AdamW(first.parameters(), lr=.001)
        for batch in batches:
            self.optimizer_step(reference, a, batch)
        self.optimizer_step(first, b, batches[0])
        stream = io.BytesIO()
        torch.save({"learner": first.weight_checkpoint(), "optimizer": b.state_dict(), "cursor": 1}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        resumed = self.model()
        resumed.load_weight_checkpoint(saved["learner"])
        optimizer = torch.optim.AdamW(resumed.parameters(), lr=.001)
        optimizer.load_state_dict(saved["optimizer"])
        for batch in batches[saved["cursor"]:]:
            self.optimizer_step(resumed, optimizer, batch)
        self.tree_equal(reference.state_dict(), resumed.state_dict())
        self.tree_equal(a.state_dict(), optimizer.state_dict())

    def test_input_boundaries_reject_padding_metadata_and_malformed_turns(self):
        model = self.model()
        original = self.inputs([["A", "B"]], padding=2)
        changes = (
            lambda x: x["token_ids"].__setitem__((0, 1), ByteCodec.PAD),
            lambda x: x["eos_positions"].__setitem__((0, 0), 0),
            lambda x: x["lengths"].__setitem__(0, 1),
            lambda x: x["valid_mask"].__setitem__((0, 0), False),
            lambda x: x["decoder_input_ids"].__setitem__((0, 0, 0), ByteCodec.PAD))
        for change in changes:
            bad = copy.deepcopy(original); change(bad)
            with self.assertRaises(ValueError):
                self.forward(model, bad)


if __name__ == "__main__":
    unittest.main()
