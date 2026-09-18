"""Mocked arithmetic/inference-boundary checks, never a neural execution proof.

No diagnostic episodes are generated. A deliberately miniature synthetic bank
is admitted only under a validator mock in these tests; production admission
would reject it. Fake forward/decoder methods manufacture CPU tensors and count
their calls. No real model, optimizer, backward, CUDA, or historical bank work.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_context_evaluation as evaluation


MOCK_WORK = dict(synthetic_query_records=0, synthetic_episode_descriptors=0,
    fake_model_instances=0, fake_sequence_calls=0, fake_decoder_driver_calls=0,
    fake_recurrent_attempts=0, fake_recurrent_completions=0, metadata_only_pack_calls=0,
    malformed_record_cases=0, diagnostic_generated_rows=0, real_model_forwards=0,
    optimizer_updates=0, backward_calls=0, cuda_calls=0)


def _query(turn, target, action=None, reply=None):
    action = target if action is None else action
    logits = [0.0] * 4
    logits[action] = 1.0
    tokens = evaluation.ByteCodec(max_bytes=32).encode(
        evaluation.diagnostic.oracle.REPLIES[target if reply is None else reply])
    return dict(turn=turn, target=target, action=action, logits=logits,
                **evaluation._reply(tokens, target))


def _records():
    result = []
    for gap in (0, 2, 4):
        for naming in (0, 1):
            for variant in (0, 1):
                known = _query(10, variant,
                    action=(1-variant if gap == 2 and variant == 1 else variant),
                    reply=(1-variant if naming == 1 and variant == 0 else variant))
                result.append(dict(episode_id=f"synthetic-{gap}-{naming}-{variant}",
                    comparison_group="synthetic-comparison", counterfactual_group=f"pair-{gap}-{naming}",
                    family="color", operator_group="direct", depth=0, pair_index=0,
                    gap=gap, naming_condition=naming, variant=variant,
                    queries=dict(known=known, unknown=_query(11, 2))))
                MOCK_WORK["synthetic_query_records"] += 1
    return result


def _bank(records):
    rows = []
    for record in records:
        turns = [dict(text=f"Visible observation {index}.", target=3, reply="Understood.") for index in range(12)]
        turns[10].update(target=record["variant"], reply=evaluation.diagnostic.oracle.REPLIES[record["variant"]])
        turns[11].update(target=2, reply=evaluation.diagnostic.oracle.REPLIES[2])
        rows.append(dict(id=record["episode_id"], comparison_group=record["comparison_group"],
            counterfactual_group=record["counterfactual_group"], family="color", operator_group="direct",
            depth=0, recipe=dict(pair_index=0), gap=record["gap"], naming_condition=record["naming_condition"],
            variant=record["variant"], turns=turns))
        MOCK_WORK["synthetic_episode_descriptors"] += 1
    return dict(schema=evaluation.diagnostic.BANK_SCHEMA, config=dict(seed=1, pairs_per_group=1),
                scope=evaluation.diagnostic.SCOPE, rows=rows, rows_sha256=evaluation._hash(rows))


class _Recurrent:
    def __init__(self):
        self.before, self.after = {}, {}
        self.serial = 0

    def _register(self, collection, hook):
        self.serial += 1
        key = self.serial
        collection[key] = hook
        return SimpleNamespace(remove=lambda: collection.pop(key, None))

    def register_forward_pre_hook(self, hook):
        return self._register(self.before, hook)

    def register_forward_hook(self, hook):
        return self._register(self.after, hook)

    def step(self, fail=False):
        MOCK_WORK["fake_recurrent_attempts"] += 1
        for hook in self.before.values():
            hook(self, ())
        if fail:
            raise RuntimeError("mock decoder failed after its intent")
        for hook in self.after.values():
            hook(self, (), None)
        MOCK_WORK["fake_recurrent_completions"] += 1


class _Model:
    def __init__(self, *, mutate=False):
        MOCK_WORK["fake_model_instances"] += 1
        self.weight = torch.tensor([0.0])
        self.training = True
        self.child = SimpleNamespace(training=False)
        self.config = SimpleNamespace(max_turns=12, max_input_bytes=128, max_positions=1024, max_output_bytes=32)
        self.inferior_frontal = SimpleNamespace(recurrent=_Recurrent())
        self.cursor, self.mutate, self.seen_inputs = 0, mutate, []

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

    def __call__(self, **kwargs):
        MOCK_WORK["fake_sequence_calls"] += 1
        assert set(kwargs) == {"token_ids", "valid_mask", "lengths", "eos_positions", "decoder_input_ids"}
        assert kwargs["decoder_input_ids"].eq(evaluation.ByteCodec.BOS).all()
        assert kwargs["decoder_input_ids"].shape[-1] == 1
        assert not self.training and not self.child.training
        assert not torch.is_grad_enabled()
        self.seen_inputs.append({k: v.clone() for k, v in kwargs.items()})
        batch = kwargs["eos_positions"].shape[0]
        logits = torch.zeros((batch, 12, 4))
        context = torch.empty((batch, 12, 1))
        for index in range(batch):
            absolute = self.cursor + index
            logits[index, :10, 3] = 1
            logits[index, 10, absolute % 2] = 1
            logits[index, 11, 2] = 1
            for turn in range(12):
                context[index, turn, 0] = absolute*12 + turn
        self.cursor += batch
        if self.mutate:
            self.weight.add_(1)
        return dict(logits=logits, production_context=context)


def _generate(model, context):
    MOCK_WORK["fake_decoder_driver_calls"] += 1
    codec = evaluation.ByteCodec(max_bytes=32)
    rows = []
    for value in context[:, 0].tolist():
        absolute, turn = divmod(int(value), 12)
        assert turn in (10, 11), "only the two selected query contexts may be decoded"
        target = absolute % 2 if turn == 10 else 2
        rows.append(codec.encode(evaluation.diagnostic.oracle.REPLIES[target]))
    width = max(map(len, rows))
    for _ in range(width-1):
        model.inferior_frontal.recurrent.step()
    return torch.tensor([row + [0]*(width-len(row)) for row in rows], dtype=torch.long)


class FoundationContextEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = _records()
        cls.bank = _bank(cls.records)

    def prepared(self):
        # This mock is local to tests. The deliberately tiny bank is NOT admitted
        # by the real validator and is not evidence of canonical bank validation.
        with patch.object(evaluation.diagnostic, "validate_bank", return_value=True) as admission:
            prepared = evaluation.PreparedContextBank(self.bank)
        admission.assert_called_once()
        return prepared

    def test_known_anchor_pairing_and_matched_transitions(self):
        metrics = evaluation.score_records(self.records)
        counts = metrics["overall"]["counts"]
        self.assertEqual(counts["paired_action"], dict(count=4, total=6, rate=4/6))
        self.assertEqual(counts["paired_reply"], dict(count=3, total=6, rate=.5))
        self.assertEqual(counts["paired_both"]["count"], 2)
        self.assertEqual(counts["unknown_action"]["count"], 12)
        placement = metrics["matched_contrasts"]["gap2-minus-gap0"]["overall"]
        self.assertEqual(placement["correctness_transitions"]["paired_action"]["reference_only"], 2)
        self.assertEqual(placement["correctness_transitions"]["known_action"]["delta_count"], -2)
        naming = metrics["matched_contrasts"]["naming1-minus-naming0"]["overall"]
        self.assertEqual(naming["correctness_transitions"]["paired_reply"]["reference_only"], 3)
        self.assertEqual(naming["correctness_transitions"]["unknown_reply"]["both_correct"], 6)
        self.assertEqual(metrics["by_family_group"]["color/direct"], metrics["overall"])

    def test_exact_reply_requires_eos_and_canonical_bytes(self):
        tokens = evaluation.ByteCodec(max_bytes=32).encode("No.")
        self.assertTrue(evaluation._reply(tokens + [0, 0], 0)["reply_exact"])
        self.assertFalse(evaluation._reply(tokens[:-1], 0)["reply_exact"])
        self.assertFalse(evaluation._reply(tokens + [2], 0)["reply_exact"])
        altered = [tokens[0], tokens[0], *tokens[1:]]
        self.assertFalse(evaluation._reply(altered, 0)["reply_exact"])
        self.assertEqual(evaluation._reply(altered, 0)["reply_action"], 0)

    def test_bad_record_pairing_anchor_and_targets_rejected(self):
        mutations = [lambda rs: rs[0]["queries"]["known"].update(turn=11),
                     lambda rs: rs[0].update(variant=True),
                     lambda rs: rs[0].update(episode_id=rs[1]["episode_id"]),
                     lambda rs: rs[0]["queries"]["known"].update(action=3),
                     lambda rs: rs.pop()]
        for mutate in mutations:
            rows = deepcopy(self.records)
            mutate(rows)
            MOCK_WORK["malformed_record_cases"] += 1
            with self.assertRaises(ValueError):
                evaluation.score_records(rows)

    def test_metadata_never_reaches_packing(self):
        rows = deepcopy(self.bank["rows"][:2])
        changed = deepcopy(rows)
        for row in changed:
            row.update(family="unused", hidden_answer="secret", gap=999)
            for turn in row["turns"]:
                turn.update(target=-1, reply="secret", observations={"answer": "secret"})
        config = dict(max_turns=12, max_input_bytes=128, max_positions=1024)
        seen = []
        def pack(text_rows, **kwargs):
            MOCK_WORK["metadata_only_pack_calls"] += 1
            seen.append((deepcopy(text_rows), deepcopy(kwargs)))
            return "packed"
        with patch.object(evaluation, "pack_observations", side_effect=pack):
            evaluation._pack_rows(rows, device="cpu", config=config)
            evaluation._pack_rows(changed, device="cpu", config=config)
        self.assertEqual(seen[0], seen[1])
        self.assertTrue(all(type(text) is str for row in seen[0][0] for text in row))

    def test_mocked_success_restores_modes_and_records_only_queries(self):
        prepared, model = self.prepared(), _Model()
        events, work = [], evaluation.EvaluationLedger()
        with patch.object(evaluation, "_generate_reply_tokens", side_effect=_generate):
            result = prepared.score(model, batch_size=4, progress=events.append, work=work)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertEqual(result["state_before"], result["state_after"])
        self.assertEqual(result["metrics"]["overall"]["counts"]["paired_both"]["count"], 6)
        self.assertEqual([e["event"] for e in events], ["batch_intent", "batch_complete"]*3)
        report = work.report()
        self.assertEqual(report["sequence_forward_attempts"], 3)
        self.assertEqual(report["sequence_forward_completions"], 3)
        self.assertEqual(report["encoder_episode_completions"], 12)
        self.assertEqual(report["bos_reply_context_attempts"], 144)
        self.assertEqual(report["free_query_context_completions"], 24)
        self.assertEqual(report["decoder_row_step_completions"], report["decoder_recurrent_completions"]*8)
        self.assertEqual(report["unmatched_decoder_attempts"], 0)
        self.assertFalse(model.inferior_frontal.recurrent.before)
        self.assertFalse(model.inferior_frontal.recurrent.after)

    def test_mocked_weight_mutation_rejected(self):
        prepared, model = self.prepared(), _Model(mutate=True)
        with patch.object(evaluation, "_generate_reply_tokens", side_effect=_generate):
            with self.assertRaisesRegex(RuntimeError, "changed model state"):
                prepared.score(model, batch_size=4)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def test_mocked_decoder_failure_retains_attempt_and_removes_hooks(self):
        prepared, model, work = self.prepared(), _Model(), evaluation.EvaluationLedger()
        def fail(model, context):
            MOCK_WORK["fake_decoder_driver_calls"] += 1
            model.inferior_frontal.recurrent.step(fail=True)
        with patch.object(evaluation, "_generate_reply_tokens", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "mock decoder failed"):
                prepared.score(model, batch_size=4, work=work)
        self.assertEqual(work.report()["sequence_forward_completions"], 1)
        self.assertEqual(work.report()["decoder_recurrent_attempts"], 1)
        self.assertEqual(work.report()["decoder_recurrent_completions"], 0)
        self.assertEqual(work.report()["free_query_context_attempts"], 8)
        self.assertEqual(work.report()["free_query_context_completions"], 0)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertFalse(model.inferior_frontal.recurrent.before)
        self.assertFalse(model.inferior_frontal.recurrent.after)

    def test_progress_failure_or_expiry_does_not_start_model_work(self):
        prepared, model = self.prepared(), _Model()
        def refuse(event):
            raise RuntimeError("mock journal cannot preserve intent")
        work = evaluation.EvaluationLedger()
        with self.assertRaisesRegex(RuntimeError, "cannot preserve intent"):
            prepared.score(model, batch_size=4, work=work, progress=refuse)
        self.assertEqual(work.report()["sequence_forward_attempts"], 0)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        work = evaluation.EvaluationLedger()
        with self.assertRaises(TimeoutError):
            prepared.score(model, deadline=0, work=work)
        self.assertEqual(work.report()["batch_intents"], 0)


if __name__ == "__main__":
    unittest.main()
