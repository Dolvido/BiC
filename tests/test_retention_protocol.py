"""Checks split integrity and measurement boundaries without training a model."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from brain_in_computer.computer_use.environment import COLORS, POSITIONS, TASKS
from experiments import repair_retention as repair
from experiments import train_regional_memory as memory


ROOT = Path(__file__).resolve().parents[1]


def sentences(bank):
    return {
        template.format(color=color, position=position.replace("_", " "), word=word)
        for templates in bank.values()
        for template, color, position, word in product(templates, COLORS, POSITIONS, ("hi", "ok"))
    }


class SensoryEncoder(nn.Module):
    """A deterministic stand-in keeps target-isolation checks weight independent."""

    def forward(self, pixels):
        return F.normalize(F.adaptive_avg_pool2d(pixels, (4, 4)).flatten(1), dim=-1)


class MetricAgent(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.specs = specs

    def forward_evidence(self, scores, known, prompts, **kwargs):
        logits = torch.zeros(len(prompts), 1, 11)
        for index, spec in enumerate(self.specs):
            # Half the actions are correct, and the complementary half of
            # replies are correct. No episode succeeds jointly.
            choice = spec["target"] if index % 2 == 0 else (spec["target"] + 1) % 11
            logits[index, 0, choice] = 1
        return {"logits": logits}

    def respond_evidence(self, scores, known, prompts, **kwargs):
        return ["wrong reply" if index % 2 == 0 else spec["reply"]
                for index, spec in enumerate(self.specs)]


class RetentionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fresh_rendered_wording_is_disjoint_from_all_rehearsal(self):
        rehearsal = repair.repair_phrases()
        self.assertEqual(set(rehearsal), set(TASKS))
        self.assertEqual(set(repair.FRESH_PHRASES), set(TASKS))
        self.assertFalse(sentences(rehearsal) & sentences(repair.FRESH_PHRASES))
        # Former held-out templates are explicitly development/rehearsal now.
        for task in TASKS:
            self.assertTrue(set(repair.desktop.V04_TEMPLATES["sealed"][task]) <= set(rehearsal[task]))
        self.assertTrue(all(len(text.encode("utf8")) <= 128 for text in sentences(repair.FRESH_PHRASES)))

    def test_fresh_evaluation_uses_temporary_bank_and_restores_it_on_error(self):
        original = repair.desktop.V04_TEMPLATES

        def fail_after_check(*args, **kwargs):
            self.assertEqual(kwargs["split"], "v06_fresh")
            self.assertEqual(repair.desktop.V04_TEMPLATES["v06_fresh"], repair.FRESH_PHRASES)
            raise RuntimeError("evaluation interrupted")

        with patch.object(repair.desktop, "evaluate", side_effect=fail_after_check):
            with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
                repair.evaluate_desktop(object(), seed=1, split="v06_fresh")
        self.assertIs(repair.desktop.V04_TEMPLATES, original)
        self.assertNotIn("v06_fresh", original)

    def test_teacher_metadata_cannot_change_sensory_evidence(self):
        specs = memory.build_specs(list(range(1000, 1020)), "train", 52, 40)
        changed = deepcopy(specs)
        for row in changed:
            row.update(target=999, reply="poisoned target", relation="poisoned relation",
                       category="poisoned category", reference=999, name="different label",
                       prompt="poisoned instruction")
        original = memory.encode_specs(specs, SensoryEncoder())
        altered = memory.encode_specs(changed, SensoryEncoder())
        self.assertTrue(torch.equal(original["scores"], altered["scores"]))
        self.assertTrue(torch.equal(original["known"], altered["known"]))
        self.assertFalse(torch.equal(original["targets"], altered["targets"]))

    def test_arbitrary_label_bytes_are_removed_before_relation_comprehension(self):
        first = [{"prompt": 'select the object left of "dax"'}]
        second = [{"prompt": 'select the object left of "new arbitrary name"'}]
        self.assertEqual(memory.normalized_prompts(first), memory.normalized_prompts(second))
        self.assertNotEqual(memory.normalized_prompts(first),
                            memory.normalized_prompts([{"prompt": 'select the object right of "dax"'}]))

    def test_joint_metric_requires_action_and_reply_in_same_episode(self):
        specs = memory.build_specs(list(range(1000, 1020)), "sealed", 74, 40)
        data = {"specs": specs, "scores": torch.zeros(40, 4), "known": torch.ones(40, 1)}
        result = memory.evaluate_memory(MetricAgent(specs), data, batch_size=40)
        self.assertEqual(result["action_accuracy"], .5)
        self.assertEqual(result["reply_exact"], .5)
        self.assertEqual(result["joint_success"], 0)
        self.assertEqual(result["all_stop_accuracy"], .5)

    def test_frozen_fresh_identities_exclude_every_earlier_partition(self):
        path = ROOT / "runs/retention-v06/protocol.json"
        if not path.exists():
            self.skipTest("the independently frozen experiment artifact is not present")
        protocol = json.loads(path.read_text())
        earlier = json.loads((ROOT / "runs/regional-memory-v05/protocol.json").read_text())
        excluded = {value for partition in earlier["identity_partitions"].values() for value in partition}
        fresh = protocol["fresh_memory_identities"]
        self.assertEqual(len(fresh), len(set(fresh)))
        self.assertFalse(set(fresh) & excluded)
        self.assertEqual(protocol["excluded_identity_count"], len(excluded))
        self.assertFalse(protocol["fresh_test_used_for_selection"])
        self.assertFalse(protocol["local_llm_invoked"])


if __name__ == "__main__":
    unittest.main()
