"""Training scaffolding preserves causal supervision and inference boundaries."""
import copy
import unittest

import torch

from brain_in_computer.dialogue_curriculum import (
    ALIASES, COLORS, generate_dialogues, parse_sentence,
)
from brain_in_computer.dialogue_scaffolding import (
    HEAD_NAMES, NONE_ALIAS, ROLE_NAMES, UNKNOWN_COLOR, ScaffoldHeads, causal_targets,
)
from brain_in_computer.dialogue_student import build_dialogue_student, encode_dialogues, _run_turn


class DialogueScaffoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_labels_follow_only_current_text_and_preceding_memory(self):
        episodes = generate_dialogues(40, 12, split="train", focus="revision")
        original = copy.deepcopy(episodes)
        targets = causal_targets(episodes)
        self.assertEqual(episodes, original)
        for name in HEAD_NAMES:
            shape = (6, 12, 8) if name == "bindings" else (6, 12)
            self.assertEqual(tuple(targets[name].shape), shape)
            self.assertEqual(targets[name].dtype, torch.long)
            self.assertTrue(bool(targets["masks"][name].all()))
        for row, episode in enumerate(episodes):
            known, permitted = {}, UNKNOWN_COLOR
            for position, turn in enumerate(episode["turns"]):
                role, facts = parse_sentence(turn["text"])
                alias = ALIASES.index(facts["alias"]) if "alias" in facts else NONE_ALIAS
                color = COLORS.index(facts["color"]) if "color" in facts else UNKNOWN_COLOR
                self.assertEqual(int(targets["role"][position, row]), ROLE_NAMES.index(role))
                self.assertEqual(int(targets["alias"][position, row]), alias)
                self.assertEqual(int(targets["color"][position, row]), color)
                if role == "definition":
                    known[alias] = color
                elif role in ("rule", "correction"):
                    permitted = color
                self.assertEqual(int(targets["permitted"][position, row]), permitted)
                self.assertEqual(targets["bindings"][position, row].tolist(),
                                 [known.get(index, UNKNOWN_COLOR) for index in range(8)])

    def test_changing_canonical_future_rule_cannot_change_prefix_targets(self):
        episodes = generate_dialogues(0, 64, split="train", focus="grounding")
        checked_prefixes = 0
        for left, right in zip(episodes[::2], episodes[1::2]):
            difference = next(i for i, (a, b) in enumerate(zip(left["turns"], right["turns"]))
                              if a["text"] != b["text"])
            if not difference:
                continue
            targets = causal_targets([left, right])
            for name in HEAD_NAMES:
                self.assertTrue(torch.equal(targets[name][:difference, 0], targets[name][:difference, 1]))
            self.assertNotEqual(int(targets["permitted"][difference, 0]),
                                int(targets["permitted"][difference, 1]))
            checked_prefixes += 1
        self.assertGreater(checked_prefixes, 0)

    def test_training_boundary_rejects_heldout_and_noncanonical_episodes(self):
        for split in ("dev", "audit"):
            with self.assertRaisesRegex(ValueError, "only train"):
                causal_targets(generate_dialogues(0, 2, split=split))
        tampered = generate_dialogues(0, 1, split="train")
        tampered[0]["turns"][0]["target"] = 0
        tampered[0]["turns"][0]["reply"] = "No."
        with self.assertRaises(ValueError):
            causal_targets(tampered)
        with self.assertRaises(ValueError):
            causal_targets([])

    def test_auxiliary_loss_reaches_concepts_state_encoder_and_memory(self):
        model, heads = build_dialogue_student(37), ScaffoldHeads()
        episodes = generate_dialogues(20, 4, split="train", focus="grounding")
        targets = causal_targets(episodes)
        output, _ = _run_turn(model, encode_dialogues(model, episodes)[0], None)
        output["concept_context"].retain_grad()
        for name in ("prefrontal_cortex", "hippocampus"):
            output["region_activity"][name].retain_grad()
        losses = heads.losses(output, targets, 0)
        self.assertEqual(set(losses), set(HEAD_NAMES))
        torch.stack(tuple(losses.values())).mean().backward()
        self.assertGreater(float(output["concept_context"].grad.abs().sum()), 0)
        for name in ("prefrontal_cortex", "hippocampus"):
            self.assertGreater(float(output["region_activity"][name].grad.abs().sum()), 0)
        for module in (model.posterior_temporal, model.brain.regions["hippocampus"]):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_removing_or_restoring_heads_cannot_change_policy_output(self):
        model, heads = build_dialogue_student(37), ScaffoldHeads()
        batch = encode_dialogues(model, generate_dialogues(20, 2, split="train"))[0]
        before_keys = tuple(model.state_dict())
        before, _ = _run_turn(model, batch, None)
        logits_before = before["logits"].detach().clone()
        auxiliary = heads(before)
        restored = ScaffoldHeads()
        restored.load_state_dict(heads.state_dict())
        for name, logits in restored(before).items():
            self.assertTrue(torch.equal(logits, auxiliary[name]))
        del heads, restored
        after, _ = _run_turn(model, batch, None)
        self.assertTrue(torch.equal(logits_before, after["logits"]))
        self.assertEqual(tuple(model.state_dict()), before_keys)

    def test_masks_allow_missing_supervision_and_text_only_heads_need_no_state(self):
        heads = ScaffoldHeads(include_state=False)
        concept = torch.randn(2, 32, requires_grad=True)
        output = {"concept_context": concept}
        targets = causal_targets(generate_dialogues(20, 2, split="train"))
        for name in ("role", "alias", "color"):
            targets["masks"][name][0] = False
            targets[name][0] = -999
        loss = heads.loss(output, targets, 0)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(concept.grad).all())
        self.assertEqual(float(concept.grad.abs().sum()), 0.)
        self.assertEqual(set(heads(output)), {"role", "alias", "color"})
        with self.assertRaises(ValueError):
            heads.loss(output, targets, 6)


if __name__ == "__main__":
    unittest.main()
