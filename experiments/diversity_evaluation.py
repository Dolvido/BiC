"""Authenticated v3 curriculum scoring with observation-only sequence inference.

Curriculum verification owns truth and provenance; shared pure prediction
metrics own scoring. Neither verifier state nor supervision enters the learner.
This evaluator is for the independent sequence baseline, not regional BiC.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from brain_in_computer.dialogue_student import (
    ByteCodec, _exact_replies, _generate_reply_tokens, checkpoint_digest,
)
from experiments.cognitive_curriculum import REPLIES
from experiments.sequence_data import pack_cognitive_episodes, pack_observations
from experiments.sequence_evaluation import prediction_metrics


def validate_diverse_evaluation_rows(rows, batch_size):
    """Verify the original complete bank before any controls or model calls."""
    from experiments.diverse_curriculum import VERSION, validate_diverse, validate_pair

    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError("evaluation batch size must preserve whole pairs")
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or not rows or len(rows) % 2):
        raise ValueError("evaluation requires a nonempty sequence of complete pairs")
    for row in rows:
        validate_diverse(row)
    seen = set()
    for index in range(0, len(rows), 2):
        validate_pair(rows[index:index + 2])
        group = rows[index]["counterfactual_group"]
        if rows[index + 1]["counterfactual_group"] != group or group in seen:
            raise ValueError("each adjacent pair must have its own counterfactual group")
        seen.add(group)
    return VERSION


def evaluate_diverse_sequence(model, rows, *, batch_size=64, score_replies=True,
                              blank_text=False, reset_history=False):
    """Score original-policy actions and separately generated, free replies.

    Reset history presents each unchanged utterance alone. Blank text retains
    utterance boundaries. Both controls keep the original authenticated truth.
    Decoder prefixes contain only BOS, never the canonical reply prefix.
    """
    for name, value in (("score_replies", score_replies), ("blank_text", blank_text),
                        ("reset_history", reset_history)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
    version = validate_diverse_evaluation_rows(rows, batch_size)
    before = checkpoint_digest(model)
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    logits, targets, reply_correct, reply_actions = [], [], [], []
    options = {"max_input_bytes": model.config.max_input_bytes,
               "max_context_tokens": model.config.max_positions}
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                episodes = rows[start:start + batch_size]
                # Structural packing is independent of v2/v3 canonical admission.
                packed = pack_cognitive_episodes(episodes, device=device, training=False,
                    blank_text=blank_text, max_reply_bytes=model.config.max_output_bytes, **options)
                observed = packed["inputs"]
                if reset_history:
                    observed = pack_observations(
                        [[turn["text"]] for episode in episodes for turn in episode["turns"]],
                        device=device, blank_text=blank_text, **options)
                batch, turns = observed["eos_positions"].shape
                decoder = torch.full((batch, turns, 1), ByteCodec.BOS, dtype=torch.long, device=device)
                output = model(**observed, decoder_input_ids=decoder)
                actions = output["logits"].reshape(len(episodes), 6, 4)
                if not torch.isfinite(actions).all():
                    raise ValueError("sequence learner produced nonfinite action logits")
                supervision = packed["supervision"]
                logits.append(actions.cpu())
                targets.append(supervision["action_targets"].cpu())
                if score_replies:
                    context = output["production_context"].reshape(len(episodes) * 6, -1)
                    if not torch.isfinite(context).all():
                        raise ValueError("sequence learner produced nonfinite reply context")
                    generated = _generate_reply_tokens(model, context)
                    expected = supervision["reply_targets"].reshape(len(episodes) * 6, -1)
                    reply_correct.append(_exact_replies(generated, expected).reshape(len(episodes), 6).cpu())
                    decoded = []
                    for tokens in generated:
                        try:
                            text = model.codec.decode(tokens.tolist())
                        except (ValueError, UnicodeError):
                            text = None
                        decoded.append(REPLIES.index(text) if text in REPLIES else -1)
                    reply_actions.append(torch.tensor(decoded, dtype=torch.long).reshape(len(episodes), 6))
        result = prediction_metrics(torch.cat(logits), torch.cat(targets), rows,
            reply_correct=torch.cat(reply_correct) if score_replies else None,
            reply_actions=torch.cat(reply_actions) if score_replies else None)
        result.update(curriculum_version=version, canonical_validation_before_controls=True,
            blank_text=blank_text, reset_history=reset_history, reset_each_turn=reset_history,
            free_running_replies=score_replies, teacher_used_for_policy=False,
            decoder_prefix="BOS only", evaluation_batch_size=batch_size,
            student_scope="independent causal sequence baseline, not regional BiC")
        return result
    finally:
        for module, training in modes:
            module.training = training
        if checkpoint_digest(model) != before:
            raise RuntimeError("diversity evaluation changed model parameters")


def evaluate_diverse_banks(model, banks, **options):
    """Equal-bank macro scores; absent denominators remain explicitly absent."""
    if not isinstance(banks, Mapping) or not banks:
        raise ValueError("at least one named evaluation bank is required")
    results = {name: evaluate_diverse_sequence(model, rows, **options) for name, rows in banks.items()}
    pairs = [row["counterfactual_accuracy"] for row in results.values()]
    later = [row["positions"]["last_known_later_query"]["accuracy"] for row in results.values()]
    return {"per_bank": results,
            "macro_query_accuracy": sum(row["query_accuracy"] for row in results.values()) / len(results),
            "macro_pair_accuracy": sum(pairs) / len(pairs) if None not in pairs else None,
            "macro_later_known_accuracy": sum(later) / len(later) if None not in later else None,
            "student_scope": "independent causal sequence baseline, not regional BiC"}
