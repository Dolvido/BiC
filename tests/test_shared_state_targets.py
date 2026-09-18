"""CPU-only causal supervision tests; preserved fixtures, no lesson generation."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import unittest

import torch

from experiments import shared_state_targets as state
from experiments.cognitive_curriculum import ALIASES, REPLIES, ASK, ACK, ALLOW, DENY

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"runs/foundation-layout-curriculum-validation-local/attempt-001/fixtures.json"
FIXTURE_SHA256 = "9cb18c2032ca5492347d482432daab77877450f11ec6f667fcfe14b214fe5c0c"
TEST_WORK = dict(pack_attempts=0, pack_returns=0, packed_rows=0, packed_turns=0,
    packed_state_entries=0, fixture_rows=0, rejected_rows=0, loss_attempts=0,
    loss_returns=0, rejected_losses=0, tensor_backwards=0, models=0, updates=0,
    generated_lessons=0, cuda_calls=0)


def row(family, sentences):
    return dict(family=family, split="train", structure_partition="train",
        turns=[dict(text=text, target=answer, reply=REPLIES[answer]) for text, answer in sentences])


def examples():
    return {
        "color": row("color", [
            ("Is the color of dax red?", ASK),
            ("Move the color of fep one step in cycle red, green, blue, yellow.", ACK),
            ("Set the color of dax to yellow.", ACK),
            ("Copy the color of dax to wug.", ACK),
            ("Move the color of dax one step in cycle red, green, blue, yellow.", ACK),
            ("Set the color of dax to blue.", ACK),
            ("Is the color of wug yellow?", ALLOW),
            ("Copy the color of fep to wug.", ACK),
            ("Is the color of wug yellow?", ASK),
            ("Set the color of pim to green.", ACK),
            ("Is the color of dax red?", DENY),
            ("Set the color of fep to red.", ACK)]),
        "count": row("count", [
            ("Is the count of dax 99?", ASK), ("Increase the count of fep by 3.", ACK),
            ("Set the count of dax to 0.", ACK), ("Copy the count of dax to wug.", ACK),
            ("Increase the count of dax by 3.", ACK), ("Decrease the count of dax by 2.", ACK),
            ("Is the count of wug 0?", ALLOW), ("Copy the count of fep to wug.", ACK),
            ("Is the count of wug 0?", ASK), ("Set the count of pim to 99.", ACK),
            ("Decrease the count of pim by 1.", ACK), ("Is the count of dax 0?", DENY)]),
        "switch": row("switch", [
            ("Is the switch of dax on?", ASK), ("Toggle the switch of fep.", ACK),
            ("Set the switch of dax to off.", ACK), ("Copy the switch of dax to wug.", ACK),
            ("Toggle the switch of dax.", ACK), ("Set the switch of dax to off.", ACK),
            ("Is the switch of wug off?", ALLOW), ("Copy the switch of fep to wug.", ACK),
            ("Is the switch of wug off?", ASK), ("Set the switch of pim to on.", ACK),
            ("Toggle the switch of pim.", ACK), ("Is the switch of dax on?", DENY)])}


def pack(rows):
    TEST_WORK["pack_attempts"] += 1
    value = state.pack_state_targets(rows)
    TEST_WORK["pack_returns"] += 1
    TEST_WORK["packed_rows"] += value.shape[0]
    TEST_WORK["packed_turns"] += value.shape[0]*value.shape[1]
    TEST_WORK["packed_state_entries"] += value.numel()
    return value


def loss(logits, targets):
    TEST_WORK["loss_attempts"] += 1
    value = state.state_loss(logits, targets)
    TEST_WORK["loss_returns"] += 1
    return value


class StateTargetsTests(unittest.TestCase):
    def test_hand_authored_prefix_states(self):
        expected = {
            "color": [{}, {}, {"dax":4}, {"dax":4,"wug":4}, {"dax":1,"wug":4},
                {"dax":3,"wug":4}, {"dax":3,"wug":4}, {"dax":3}, {"dax":3},
                {"dax":3,"pim":2}, {"dax":3,"pim":2}, {"dax":3,"pim":2,"fep":1}],
            "count": [{}, {}, {"dax":5}, {"dax":5,"wug":5}, {"dax":8,"wug":5},
                {"dax":6,"wug":5}, {"dax":6,"wug":5}, {"dax":6}, {"dax":6},
                {"dax":6,"pim":104}, {"dax":6,"pim":103}, {"dax":6,"pim":103}],
            "switch": [{}, {}, {"dax":105}, {"dax":105,"wug":105}, {"dax":106,"wug":105},
                {"dax":105,"wug":105}, {"dax":105,"wug":105}, {"dax":105}, {"dax":105},
                {"dax":105,"pim":106}, {"dax":105,"pim":105}, {"dax":105,"pim":105}]}
        for family, item in examples().items():
            with self.subTest(family=family):
                actual = pack([item])
                wanted = torch.tensor([[[values.get(alias,0) for alias in ALIASES]
                    for values in expected[family]]], dtype=torch.long)
                self.assertEqual(actual.device.type, "cpu")
                self.assertFalse(actual.requires_grad)
                self.assertTrue(torch.equal(actual, wanted))

    def test_future_and_stale_metadata_do_not_change_prefix(self):
        replacements = {"color":"Set the color of pim to blue.",
            "count":"Set the count of pim to 98.", "switch":"Set the switch of pim to off."}
        for family, item in examples().items():
            future, stale = deepcopy(item), deepcopy(item)
            future["turns"][9]["text"] = replacements[family]
            stale.update(program=[{"op":"set", "name":"dax", "value":"wrong"}],
                recipe={"future_aliases":["not-an-alias"], "oracle_answer":"wrong"}, latent_state={"dax":99})
            for turn in stale["turns"]:
                turn.update(observation="wrong", kind="wrong", value="wrong")
            targets = pack([item, future, stale])
            self.assertEqual(tuple(targets.shape), (3,12,12))
            self.assertTrue(torch.equal(targets[0,:9], targets[1,:9]))
            self.assertFalse(torch.equal(targets[0,9:], targets[1,9:]))
            self.assertTrue(torch.equal(targets[0], targets[2]))
            self.assertTrue(bool((targets[:,:9,ALIASES.index("pim")] == 0).all()))

    def test_preserved_original_and_varied_fixtures(self):
        image = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(image).hexdigest(), FIXTURE_SHA256)
        groups = {}
        cells = set()
        for entry in json.loads(image)["layout_pairs"]:
            family, depth, turns, layout, seed = entry["key"]
            if (layout, seed) not in (("original",0),("varied",1)): continue
            cells.add((family,depth,turns,layout))
            groups.setdefault((family,turns), []).extend(entry["pair"])
        self.assertEqual(len(cells),108)
        self.assertEqual(len(groups),9)
        for (family, turns), rows in sorted(groups.items()):
            actual = pack(rows)
            self.assertEqual(tuple(actual.shape), (24,turns,12))
            TEST_WORK["fixture_rows"] += len(rows)

    def test_malformed_or_nontraining_rows_rejected(self):
        items = examples()
        cases = [[], [items["color"],items["count"]], [items["switch"],{**items["switch"],"turns":[]}]]
        for field, value in (("split","dev"),("structure_partition","audit"),("family","other")):
            changed = deepcopy(items["color"]); changed[field] = value; cases.append([changed])
        for index, field, value in ((1,"target",ALLOW),(0,"target",ALLOW),(6,"reply","No."),
            (1,"text","Toggle the switch of fep."),(0,"target",True)):
            changed = deepcopy(items["color"]); changed["turns"][index][field] = value; cases.append([changed])
        cases.extend([
            [row("count",[("Set the count of dax to 99.",ACK),("Increase the count of dax by 1.",ACK)])],
            [row("count",[("Set the count of dax to 0.",ACK),("Decrease the count of dax by 1.",ACK)])],
            [row("count",[("Set the count of dax to 100.",ACK)])],
            [row("color",[("Set the color of dax to purple.",ACK)])],
            [row("switch",[("Set the switch of dax to maybe.",ACK)])],
            [row("color",[("Set the color of somebody to red.",ACK)])]])
        for index, invalid in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError): pack(invalid)
            TEST_WORK["rejected_rows"] += 1

    def test_balanced_loss_is_per_example_turn(self):
        # Flattened B=2,T=2: mixed groups; all unknown; all known; mixed groups.
        labels = torch.zeros((2,2,12),dtype=torch.long)
        labels[0,0,:2] = torch.tensor([1,2]); labels[1,0] = 5; labels[1,1,0] = 106
        nll = torch.empty((2,2,12),dtype=torch.float64)
        nll[0,0] = 4; nll[0,0,:2] = 2; nll[0,1] = 1
        nll[1,0] = 6; nll[1,1] = 3; nll[1,1,0] = 7
        correct_probability = torch.exp(-nll)
        probabilities = ((1-correct_probability)/106).unsqueeze(-1).expand(2,2,12,107).clone()
        probabilities.scatter_(-1,labels.unsqueeze(-1),correct_probability.unsqueeze(-1))
        logits = probabilities.log().requires_grad_(True)
        actual = loss(logits, labels)
        self.assertAlmostEqual(actual.item(), (3+1+6+5)/4, places=12)
        TEST_WORK["tensor_backwards"] += 1; actual.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()),0)
        uniform = loss(torch.zeros((2,3,12,107),dtype=torch.float64),torch.zeros((2,3,12),dtype=torch.long))
        self.assertAlmostEqual(uniform.item(),math.log(107),places=12)
        self.assertIs(state.auxiliary_loss,state.state_loss)

    def test_malformed_losses_rejected(self):
        logits, labels = torch.zeros((1,1,12,107)), torch.zeros((1,1,12),dtype=torch.long)
        cases = [(logits[:,:,:,:106],labels),(logits,labels.float()),(logits,labels[:,:,:11]),
            (logits,labels-1),(logits,labels+107),(logits+float("nan"),labels),
            (logits.to(torch.long),labels),(logits[:0],labels[:0])]
        for invalid_logits, invalid_labels in cases:
            with self.assertRaises(ValueError): loss(invalid_logits,invalid_labels)
            TEST_WORK["rejected_losses"] += 1


if __name__ == "__main__": unittest.main()
