"""Synthetic evaluator-boundary tests, not a learner or curriculum proof.

Rows deliberately use a mocked canonical validator. No canonical generation,
real learner, checkpoint, optimizer, backward or CUDA operation is performed.
Fake forward/decoder methods manufacture small CPU tensors; actual observation
packing and JSON metric arithmetic are exercised. Root owns the one test run.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_layout_evaluation as evaluation


MOCK_WORK = dict(synthetic_rows=0, fake_models=0, fake_sequence_calls=0,
    fake_decoder_calls=0, fake_recurrent_attempts=0, fake_recurrent_completions=0,
    admission_mock_calls=0, canonical_generation_calls=0, real_model_forwards=0,
    optimizer_updates=0, backwards=0, cuda_calls=0)


def _rows(turns=8, pairs=2):
    rows = []
    for pair in range(pairs):
        anchor = 1 if pair == 0 else turns-2
        for variant in (0, 1):
            observations = {**{key: [[0.0]*width] for key, width in
                (("visual", 32), ("auditory", 4), ("body", 4), ("feedback", 2))}, "tokens": [0]}
            entries = [dict(text="Visible "+("long " if (turn+variant) % 3 else "")+str(turn)+".",
                observations=deepcopy(observations), target=3, reply=evaluation.REPLIES[3], kind="statement")
                for turn in range(turns)]
            entries[anchor].update(target=variant, reply=evaluation.REPLIES[variant], kind="query")
            entries[3].update(target=0, reply=evaluation.REPLIES[0], kind="query")
            entries[-1].update(target=2, reply=evaluation.REPLIES[2], kind="query")
            rows.append(dict(id=f"mock-{turns}-{pair}-{variant}", family="color" if pair == 0 else "count",
                split="dev", depth=0, variant=variant, counterfactual_group=f"pair-{turns}-{pair}",
                recipe=dict(layout="varied", base_pair_sha256=f"basepair-{pair}",
                    base_ids=[f"base-{pair}-0", f"base-{pair}-1"]),
                anchor=dict(turn_index=anchor, query_id=f"anchor-{pair}"),
                queries=[dict(query_id=f"anchor-{pair}", original_turn_index=turns-1, turn_index=anchor),
                         dict(query_id=f"other-known-{pair}", original_turn_index=1, turn_index=3),
                         dict(query_id=f"unknown-{pair}", original_turn_index=0, turn_index=turns-1)], turns=entries))
            MOCK_WORK["synthetic_rows"] += 1
    return rows


def _admit(pair, **kwargs):
    MOCK_WORK["admission_mock_calls"] += 1
    if len(pair) != 2 or [row["variant"] for row in pair] != [0, 1]:
        raise ValueError("mock complete pairs required")
    return True


def _prepared(rows):
    with patch.object(evaluation.curriculum, "validate_pair", side_effect=_admit):
        return evaluation.PreparedLayoutBank(rows, role="dev", config=evaluation.SequenceConfig(
            width=8, layers=1, heads=2, feedforward=16, max_turns=12, max_input_bytes=32, max_positions=256))


def _records(rows):
    codec = evaluation.ByteCodec(max_bytes=32)
    records = []
    for row in rows:
        queries = []
        for query in sorted(row["queries"], key=lambda q: q["turn_index"]):
            turn = query["turn_index"]; target = row["turns"][turn]["target"]
            logits = [0.0]*4; logits[target] = 1.0
            queries.append(dict(query_id=query["query_id"], turn=turn, original_turn=query["original_turn_index"],
                target=target, action=target, logits=logits, **evaluation._reply(codec.encode(evaluation.REPLIES[target]), target)))
        records.append(dict(episode_id=row["id"], counterfactual_group=row["counterfactual_group"],
            base_pair_sha256=row["recipe"]["base_pair_sha256"], base_episode_id=row["recipe"]["base_ids"][row["variant"]],
            family=row["family"], depth=0, operator_group="direct", variant=row["variant"], turns=len(row["turns"]),
            layout="varied", anchor_query_id=row["anchor"]["query_id"], anchor_turn_index=row["anchor"]["turn_index"], queries=queries))
    return records


class _Hooks:
    def __init__(self):
        self.before, self.after = {}, {}
        self.serial = 0

    def _register(self, values, hook):
        self.serial += 1
        key = self.serial; values[key] = hook
        return SimpleNamespace(remove=lambda: values.pop(key, None))

    def register_forward_pre_hook(self, hook):
        return self._register(self.before, hook)

    def register_forward_hook(self, hook):
        return self._register(self.after, hook)

    def call(self, count, *, recurrent=False, interrupt=False, nonfinite=False):
        inputs = (torch.zeros(count, 1, 1),)
        if recurrent:
            MOCK_WORK["fake_recurrent_attempts"] += 1
        for hook in self.before.values():
            hook(self, inputs)
        if interrupt:
            raise KeyboardInterrupt("synthetic decoder interruption")
        value = torch.full((count, 1, 1), float("nan") if nonfinite else 0.)
        output = (value, value) if recurrent else value
        if recurrent:
            MOCK_WORK["fake_recurrent_completions"] += 1
        for hook in self.after.values():
            hook(self, inputs, output)


class _Model:
    def __init__(self, bank, rows, *, fault=None):
        MOCK_WORK["fake_models"] += 1
        self.config = bank._config
        self.weight = torch.tensor([0.0], requires_grad=True)
        self.weight.grad = torch.tensor([.25])
        self.training, self.child = True, SimpleNamespace(training=False)
        self.inferior_frontal = SimpleNamespace(recurrent=_Hooks(), readout=_Hooks())
        self.targets = [turn["target"] for row in rows for turn in row["turns"]]
        self.cursor, self.fault, self.seen = 0, fault, []

    def state_dict(self):
        return {"weight": self.weight}

    def parameters(self):
        return iter((self.weight,))

    def named_parameters(self):
        return iter((("weight", self.weight),))

    def modules(self):
        return iter((self, self.child))

    def eval(self):
        self.training = self.child.training = False
        return self

    def __call__(self, **inputs):
        MOCK_WORK["fake_sequence_calls"] += 1
        assert set(inputs) == {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"}
        assert inputs["decoder_input_ids"].shape[-1] == 1 and inputs["decoder_input_ids"].eq(evaluation.ByteCodec.BOS).all()
        assert not torch.is_grad_enabled() and not self.training and not self.child.training
        self.seen.append({key: value.clone() for key, value in inputs.items()})
        shape = inputs["eos_positions"].shape
        total = shape[0]*shape[1]
        self.inferior_frontal.recurrent.call(total, recurrent=True)
        self.inferior_frontal.readout.call(total)
        logits, contexts = torch.zeros((*shape, 4)), torch.zeros((*shape, 1))
        for offset in range(total):
            row, turn = divmod(offset, shape[1])
            logits[row, turn, self.targets[self.cursor+offset]] = 1.
            contexts[row, turn, 0] = self.cursor+offset
        self.cursor += total
        if self.fault == "mutation":
            self.weight.add_(1)
        return dict(logits=logits, production_context=contexts)


def _generate(model, contexts):
    MOCK_WORK["fake_decoder_calls"] += 1
    codec = evaluation.ByteCodec(max_bytes=32)
    rows = [codec.encode(evaluation.REPLIES[model.targets[int(index)]]) for index in contexts[:, 0].tolist()]
    width = max(map(len, rows)); count = len(rows)
    for step in range(width-1):
        model.inferior_frontal.recurrent.call(count, recurrent=True, interrupt=model.fault == "interrupt" and step == 1)
        model.inferior_frontal.readout.call(count, nonfinite=model.fault == "nonfinite" and step == 1)
    return torch.tensor([row+[0]*(width-len(row)) for row in rows], dtype=torch.long)


class FoundationLayoutEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = MOCK_WORK

    def test_explicit_anchor_metrics_and_independent_reply_truth(self):
        records = _records(_rows())
        # The first pair has perfect actions but one wrong exact reply.
        query = next(q for q in records[1]["queries"] if q["query_id"] == records[1]["anchor_query_id"])
        query.update(evaluation._reply(evaluation.ByteCodec(32).encode(evaluation.REPLIES[2]), query["target"]))
        # The second pair has perfect replies but one wrong action.
        query = next(q for q in records[3]["queries"] if q["query_id"] == records[3]["anchor_query_id"])
        query.update(action=2, logits=[0., 0., 1., 0.])
        counts = evaluation.score_records(records)["overall"]["counts"]
        self.assertEqual(counts["anchor_pair_action"], dict(count=1, total=2, rate=.5))
        self.assertEqual(counts["anchor_pair_reply"], dict(count=1, total=2, rate=.5))
        self.assertEqual(counts["anchor_pair_both"]["count"], 0)
        self.assertEqual(counts["query_action"], dict(count=11, total=12, rate=11/12))
        self.assertEqual(counts["other_known_action"], dict(count=4, total=4, rate=1.))
        self.assertEqual(counts["other_known_reply"], dict(count=4, total=4, rate=1.))
        self.assertEqual(counts["unknown_action"]["count"], 4)
        self.assertEqual(counts["known_unsupported_ask"]["count"], 1)
        self.assertEqual(counts["known_reply_unsupported_ask"]["count"], 1)

    def test_recount_rejects_corrupted_pair_coordinates_logits_and_bytes(self):
        original = _records(_rows())
        mutations = [lambda r: r.pop(), lambda r: r[1].update(variant=0),
            lambda r: r[0].update(anchor_turn_index=7),
            lambda r: r[0]["queries"][0].update(logits=[float("nan"), 0., 0., 0.]),
            lambda r: r[0]["queries"][0].update(reply_exact=False),
            lambda r: r[1]["queries"][0].update(query_id="different"),
            lambda r: r[0]["queries"][1].update(turn=1)]
        for mutate in mutations:
            with self.subTest(case=mutations.index(mutate)):
                records = deepcopy(original); mutate(records)
                with self.assertRaises(ValueError):
                    evaluation.score_records(records)
        expected = dict(query_inventory=[dict(episode_id=r["episode_id"], queries=[{k:q[k] for k in
            ("query_id", "turn", "original_turn", "target")} for q in r["queries"]]) for r in original])
        omitted = deepcopy(original)
        for row in omitted:
            row["queries"].pop()
        with self.assertRaisesRegex(ValueError, "complete query inventory"):
            evaluation.score_records(omitted, expected_bank=expected)

    def test_observation_only_score_immutable_bank_and_modes(self):
        rows = _rows(); original = deepcopy(rows); bank = _prepared(rows)
        identity = bank.identity; identity["anchors"][0]["turn_index"] = 7
        rows[0]["turns"][0]["text"] = "Mutated outside bank."
        model = _Model(bank, original)
        with patch.object(evaluation, "_generate_reply_tokens", side_effect=_generate):
            result = bank.score(model, batch_size=2)
        self.assertEqual(result["metrics"]["overall"]["counts"]["anchor_pair_both"]["count"], 2)
        self.assertEqual(result["metrics"], evaluation.score_records(result["raw_records"], expected_bank=bank.identity))
        self.assertEqual(result["work"]["sequence_forward_completions"], 2)
        self.assertEqual(result["work"]["encoder_episode_completions"], 4)
        self.assertEqual(result["work"]["bos_reply_context_completions"], 32)
        self.assertEqual(result["work"]["free_reply_context_completions"], 32)
        self.assertEqual(result["work"]["bos_decoder_row_step_completions"], 32)
        self.assertTrue(result["model_state_unchanged"] and result["training_modes_restored"])
        self.assertTrue(model.training); self.assertFalse(model.child.training)
        self.assertEqual(model.weight.grad.item(), .25)
        self.assertEqual(bank.identity["anchors"][0]["turn_index"], 1)
        self.assertFalse(model.inferior_frontal.recurrent.before or model.inferior_frontal.recurrent.after)
        self.assertTrue(any(not bool(inputs["valid_mask"].all()) for inputs in model.seen))

    def test_all_lengths_and_blank_reset_controls(self):
        for turns, control in ((8, "blank"), (8, "reset"), (10, "normal"), (12, "normal")):
            with self.subTest(turns=turns, control=control):
                rows = _rows(turns, pairs=1); bank = _prepared(rows); model = _Model(bank, rows)
                with patch.object(evaluation, "_generate_reply_tokens", side_effect=_generate):
                    result = bank.score(model, control=control)
                self.assertEqual(result["metrics"]["overall"]["counts"]["anchor_pair_both"]["count"], 1)
                self.assertEqual(result["work"]["encoder_episode_completions"], 2*turns if control == "reset" else 2)
                self.assertEqual(result["work"]["free_reply_context_completions"], 2*turns)
                if control == "blank":
                    self.assertEqual(set(model.seen[0]["token_ids"].flatten().tolist()), {1, 2})
                if control == "reset":
                    self.assertEqual(model.seen[0]["eos_positions"].shape, (2*turns, 1))

    def test_admission_rejects_later_bad_pair_before_packing(self):
        rows = _rows()
        with patch.object(evaluation.curriculum, "validate_pair", side_effect=[True, ValueError("bad pair")]) as validator, \
                patch.object(evaluation, "pack_composition_episodes") as pack:
            with self.assertRaisesRegex(ValueError, "bad pair"):
                evaluation.PreparedLayoutBank(rows, role="dev")
            self.assertEqual(validator.call_count, 2); pack.assert_not_called()
        for changed in (rows+deepcopy(rows), [*rows[:2], *_rows(10, pairs=1)]):
            with patch.object(evaluation.curriculum, "validate_pair", side_effect=_admit), \
                    patch.object(evaluation, "pack_composition_episodes") as pack:
                with self.assertRaises(ValueError):
                    evaluation.PreparedLayoutBank(changed, role="dev")
                pack.assert_not_called()

    def test_interrupt_nonfinite_and_mutation_leave_honest_failure_receipts(self):
        rows = _rows(pairs=1); bank = _prepared(rows)
        for fault, kind in (("interrupt", KeyboardInterrupt), ("nonfinite", FloatingPointError), ("mutation", RuntimeError)):
            with self.subTest(fault=fault):
                model = _Model(bank, rows, fault=fault); ledger = evaluation.EvaluationLedger()
                with patch.object(evaluation, "_generate_reply_tokens", side_effect=_generate), self.assertRaises(kind) as caught:
                    bank.score(model, work=ledger)
                report = caught.exception.layout_report
                self.assertEqual(report, bank.last_report)
                self.assertNotIn("metrics", report)
                self.assertEqual(report["work"]["sequence_forward_completions"], 1)
                self.assertTrue(model.training); self.assertFalse(model.child.training)
                self.assertFalse(model.inferior_frontal.recurrent.before or model.inferior_frontal.recurrent.after)
                self.assertFalse(model.inferior_frontal.readout.after)
                self.assertEqual(report["model_state_unchanged"], fault != "mutation")
                if fault == "interrupt":
                    self.assertEqual(report["status"], "interrupted")
                    self.assertEqual(report["work"]["unmatched_decoder_attempts"], 1)
                if fault == "nonfinite":
                    self.assertEqual(report["work"]["free_decoder_calls_completed"], 0)

    def test_expired_callback_and_compiled_input_refusals_before_forward(self):
        rows = _rows(pairs=1); bank = _prepared(rows); model = _Model(bank, rows)
        for options, error in ((dict(deadline=-1.), TimeoutError),
                               (dict(progress=lambda event: (_ for _ in ()).throw(RuntimeError("journal failure"))), RuntimeError)):
            with self.assertRaises(error) as caught:
                bank.score(model, **options)
            self.assertEqual(caught.exception.layout_report["work"]["sequence_forward_attempts"], 0)
            self.assertEqual(model.cursor, 0)
        malformed = evaluation.EvaluationLedger()
        malformed.counts["sequence_forward_attempts"] = "invalid"
        with self.assertRaisesRegex(ValueError, "fresh unmodified") as caught:
            bank.score(model, work=malformed)
        self.assertIsNone(caught.exception.layout_report["work"])
        self.assertEqual(model.cursor, 0)
        bank._inputs["normal"]["token_ids"][0, 1] += 1
        with self.assertRaisesRegex(ValueError, "compiled observations"):
            bank.score(model)
        self.assertEqual(model.cursor, 0)


if __name__ == "__main__":
    unittest.main()
