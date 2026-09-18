"""Teacher-free evaluation for the independent causal sequence baseline.

Canonical rows are authenticated before any control. Only observed text and
BOS-only reply prefixes reach the model; targets are used after prediction.
This module neither trains nor claims regional BiC capability.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import (
    ACK, ByteCodec, _action_metrics, _exact_replies, _generate_reply_tokens,
    checkpoint_digest,
)
from experiments.audit_cognitive_credit import position_metrics
from experiments.cognitive_curriculum import REPLIES, validate_cognitive
from experiments.sequence_data import pack_cognitive_episodes, pack_observations


def validate_evaluation_rows(rows, batch_size):
    """Authenticate the source bank, including when its text will be blanked."""
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError("evaluation batch size must preserve whole pairs")
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or not rows or len(rows) % 2):
        raise ValueError("evaluation needs a nonempty sequence of complete pairs")
    for row in rows:
        validate_cognitive(row)
    seen = set()
    for index in range(0, len(rows), 2):
        group = rows[index]["counterfactual_group"]
        if rows[index + 1]["counterfactual_group"] != group or group in seen:
            raise ValueError("each adjacent pair must have its own counterfactual group")
        seen.add(group)


def prediction_metrics(logits, targets, rows, *, reply_correct=None, reply_actions=None):
    """Score recorded original-policy outputs, with [episode,turn,...] axes."""
    if logits.ndim != 3 or logits.shape[-1] != 4 or targets.shape != logits.shape[:2]:
        raise ValueError("expected [episode,turn,4] logits and matching targets")
    if logits.shape[0] != len(rows) or logits.shape[1] != 6 or targets.dtype != torch.long:
        raise ValueError("six-turn predictions must match the episode bank")
    if not torch.isfinite(logits).all():
        raise ValueError("sequence learner produced nonfinite action logits")
    if not ((targets >= 0) & (targets <= ACK)).all():
        raise ValueError("action targets outside the shared answer vocabulary")
    if (reply_correct is None) != (reply_actions is None):
        raise ValueError("reply correctness and decoded actions must be supplied together")
    if reply_correct is not None and (reply_correct.shape != targets.shape
            or reply_actions.shape != targets.shape or reply_correct.dtype != torch.bool):
        raise ValueError("reply scores must match [episode,turn] targets")
    predictions, query = logits.argmax(-1), targets.ne(ACK)
    query_logits, query_targets = logits[query], targets[query]
    query_predictions = predictions[query]
    result = _action_metrics(query_logits, query_targets)
    result.update(query_accuracy=result["accuracy"], query_total=result["total"],
                  query_correct=result["correct"], query_loss=result["loss"])
    confusion = torch.bincount(query_targets * 4 + query_predictions, minlength=16).reshape(4, 4)
    ask_true, ask_predicted, ask_correct = int(confusion[2].sum()), int(confusion[:, 2].sum()), int(confusion[2, 2])
    per_target = {str(target): _action_metrics(query_logits[query_targets == target], query_targets[query_targets == target])
                  for target in range(3)}
    known = [per_target[str(target)]["accuracy"] for target in (0, 1)]
    positions = position_metrics(predictions.T, targets.T, rows,
                                reply_correct.T if reply_correct is not None else None)
    pair_total = sum(row["slices"]["all"]["pairs"]["total"] for row in positions["by_turn"])
    pair_correct = sum(row["slices"]["all"]["pairs"]["correct"] for row in positions["by_turn"])
    probabilities = query_logits.softmax(-1)
    result.update(overall_accuracy=float(predictions.eq(targets).float().mean()),
        confusion_matrix_true_rows_predicted_columns=confusion.cpu().tolist(), per_target=per_target,
        ask_true=ask_true, ask_predicted=ask_predicted, ask_correct=ask_correct,
        ask_recall=ask_correct / ask_true if ask_true else None,
        ask_precision=ask_correct / ask_predicted if ask_predicted else None,
        allow_deny_macro_accuracy=sum(known) / 2 if None not in known else None,
        brier_score=float((probabilities - F.one_hot(query_targets, 4)).square().sum(-1).mean()),
        counterfactual_query_pairs=pair_total, counterfactual_correct=pair_correct,
        counterfactual_accuracy=pair_correct / pair_total if pair_total else None,
        positions=positions, query_reply_exact_accuracy=None, reply_exact_accuracy=None,
        action_reply_agreement=None, reply_parseable_query_count=None,
        teacher_forced_reply_loss=None)
    if reply_correct is not None:
        result.update(query_reply_exact_accuracy=float(reply_correct[query].float().mean()),
            reply_exact_accuracy=float(reply_correct.float().mean()),
            action_reply_agreement=float(reply_actions[query].eq(query_predictions).float().mean()),
            reply_parseable_query_count=int(reply_actions[query].ge(0).sum()))
    return result


def evaluate_sequence(model, episodes, *, batch_size=64, score_replies=True,
                      blank_text=False, reset_history=False):
    """Score canonical episodes in paired chunks without teacher reply prefixes.

    Reset-history presents each unchanged utterance as an independent one-turn
    sequence. Blank-text retains utterance boundaries but removes byte content.
    Both transformations occur only after authenticating the original examples.
    The input encoder never sees canonical replies, action labels or metadata.
    """
    for name, value in (("score_replies", score_replies), ("blank_text", blank_text),
                        ("reset_history", reset_history)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
    validate_evaluation_rows(episodes, batch_size)
    before = checkpoint_digest(model)
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    logits, targets, reply_correct, reply_actions = [], [], [], []
    options = {"max_input_bytes": model.config.max_input_bytes,
               "max_context_tokens": model.config.max_positions}
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(episodes), batch_size):
                rows = episodes[start:start + batch_size]
                packed = pack_cognitive_episodes(rows, device=device, training=False,
                    blank_text=blank_text, max_reply_bytes=model.config.max_output_bytes, **options)
                observed = packed["inputs"]
                if reset_history:
                    observed = pack_observations([[turn["text"]] for row in rows for turn in row["turns"]],
                        device=device, blank_text=blank_text, **options)
                batch, turns = observed["eos_positions"].shape
                decoder = torch.full((batch, turns, 1), ByteCodec.BOS, dtype=torch.long, device=device)
                output = model(**observed, decoder_input_ids=decoder)
                actions = output["logits"].reshape(len(rows), 6, 4)
                if not torch.isfinite(actions).all():
                    raise ValueError("sequence learner produced nonfinite action logits")
                supervision = packed["supervision"]
                logits.append(actions.cpu())
                targets.append(supervision["action_targets"].cpu())
                if score_replies:
                    context = output["production_context"].reshape(len(rows) * 6, -1)
                    if not torch.isfinite(context).all():
                        raise ValueError("sequence learner produced nonfinite reply context")
                    generated = _generate_reply_tokens(model, context)
                    expected = supervision["reply_targets"].reshape(len(rows) * 6, -1)
                    reply_correct.append(_exact_replies(generated, expected).reshape(len(rows), 6).cpu())
                    decoded = []
                    for tokens in generated:
                        try:
                            text = model.codec.decode(tokens.tolist())
                        except (ValueError, UnicodeError):
                            text = None
                        decoded.append(REPLIES.index(text) if text in REPLIES else -1)
                    reply_actions.append(torch.tensor(decoded, dtype=torch.long).reshape(len(rows), 6))
        result = prediction_metrics(torch.cat(logits), torch.cat(targets), episodes,
            reply_correct=torch.cat(reply_correct) if score_replies else None,
            reply_actions=torch.cat(reply_actions) if score_replies else None)
        result.update(blank_text=blank_text, reset_history=reset_history,
            reset_each_turn=reset_history, free_running_replies=score_replies,
            teacher_used_for_policy=False, decoder_prefix="BOS only",
            evaluation_batch_size=batch_size, canonical_validation_before_controls=True,
            student_scope="independent causal sequence baseline, not regional BiC")
        return result
    finally:
        for module, training in modes:
            module.training = training
        if checkpoint_digest(model) != before:
            raise RuntimeError("sequence evaluation changed model parameters")


def evaluate_banks(model, banks, **options):
    if not isinstance(banks, Mapping) or not banks:
        raise ValueError("at least one named evaluation bank is required")
    results = {name: evaluate_sequence(model, rows, **options) for name, rows in banks.items()}
    later = [row["positions"]["last_known_later_query"]["accuracy"] for row in results.values()]
    return {"per_bank": results,
            "macro_query_accuracy": sum(row["query_accuracy"] for row in results.values()) / len(results),
            "macro_pair_accuracy": sum(row["counterfactual_accuracy"] or 0 for row in results.values()) / len(results),
            "macro_later_known_accuracy": sum(later) / len(later) if None not in later else None,
            "student_scope": "independent causal sequence baseline, not regional BiC"}
