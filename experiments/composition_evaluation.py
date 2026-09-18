"""Prepared, authenticated variable-turn evaluation with teacher-free replies.

Canonical immutable JSON owns the evidence. Private CPU tensors are compiled
once and cloned for every model call; targets never enter model arguments.
The runner still binds checkpoint, source and bank identities at phase boundaries.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import io
import json
import time

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import (
    ByteCodec, _exact_replies, _generate_reply_tokens, checkpoint_digest,
)
from brain_in_computer.learning_student import _cpu_copy
from experiments.sequence_student import SequenceConfig, build_sequence_student


METRICS = ("query_accuracy", "known_accuracy", "final_accuracy", "final_pair_accuracy",
           "opposite_pair_accuracy", "query_reply_accuracy", "final_reply_pair_accuracy")


def prediction_metrics(logits, targets, *, reply_correct=None, reply_actions=None):
    if (logits.ndim != 3 or logits.shape[-1] != 4 or targets.shape != logits.shape[:2]
            or targets.dtype != torch.long or logits.shape[0] % 2 or logits.shape[0] < 2
            or not torch.isfinite(logits).all() or not ((targets >= 0) & (targets <= 3)).all()):
        raise ValueError("finite complete paired [episode,turn,4] predictions required")
    if (reply_correct is None) != (reply_actions is None):
        raise ValueError("reply scores must be supplied together")
    if reply_correct is not None and (reply_correct.dtype != torch.bool
            or reply_correct.shape != targets.shape or reply_actions.shape != targets.shape
            or reply_actions.dtype != torch.long):
        raise ValueError("reply score axes or types differ")
    prediction = logits.argmax(-1)
    correct = prediction.eq(targets)
    query, known = targets.ne(3), targets.lt(2)
    eligible = targets[::2].lt(2) & targets[1::2].lt(2) & targets[::2].ne(targets[1::2])
    pairs_correct = correct[::2] & correct[1::2] & eligible
    if not eligible[:, -1].all():
        raise ValueError("every final query must be a known opposite-answer pair")
    def score(mask):
        total = int(mask.sum())
        count = int(correct[mask].sum())
        return {"correct": count, "total": total, "accuracy": count / total if total else None}
    q, k = score(query), score(known)
    pair_count, pair_total = int(pairs_correct.sum()), int(eligible.sum())
    confusion = torch.bincount(targets[query] * 4 + prediction[query], minlength=16).reshape(4, 4)
    ask_true, ask_predicted, ask_correct = int(confusion[2].sum()), int(confusion[:, 2].sum()), int(confusion[2, 2])
    final_pairs = {"correct": int(pairs_correct[:, -1].sum()), "total": logits.shape[0] // 2}
    result = {"episodes": logits.shape[0], "turns_per_episode": logits.shape[1],
        "query_accuracy": q["accuracy"], "query_correct": q["correct"], "query_total": q["total"],
        "known_accuracy": k["accuracy"], "known_correct": k["correct"], "known_total": k["total"],
        "final_accuracy": float(correct[:, -1].float().mean()), "final_correct": int(correct[:, -1].sum()),
        "final_total": logits.shape[0], "final_pairs": final_pairs,
        "final_pair_accuracy": final_pairs["correct"] / final_pairs["total"],
        "opposite_pair_correct": pair_count, "opposite_pair_total": pair_total,
        "opposite_pair_accuracy": pair_count / pair_total,
        "confusion_matrix": confusion.tolist(),
        "per_target": {str(target): score(targets.eq(target)) for target in range(3)},
        "ask_true": ask_true, "ask_predicted": ask_predicted, "ask_correct": ask_correct,
        "ask_precision": ask_correct / ask_predicted if ask_predicted else None,
        "ask_recall": ask_correct / ask_true if ask_true else None,
        "query_loss": float(F.cross_entropy(logits[query], targets[query])),
        "brier_score": float((logits[query].softmax(-1) - F.one_hot(targets[query], 4)).square().sum(-1).mean()),
        "by_turn": [{"turn": index, **score(query & (torch.arange(targets.shape[1])[None] == index)),
            "opposite_pair_total": int(eligible[:, index].sum()),
            "opposite_pair_correct": int(pairs_correct[:, index].sum())} for index in range(targets.shape[1])],
        "query_reply_accuracy": None, "query_reply_correct": None,
        "final_reply_pair_accuracy": None, "final_reply_pair_correct": None,
        "action_reply_agreement": None, "reply_parseable_queries": None}
    if reply_correct is not None:
        reply_pairs = reply_correct[::2, -1] & reply_correct[1::2, -1]
        result.update(query_reply_accuracy=float(reply_correct[query].float().mean()),
            query_reply_correct=int(reply_correct[query].sum()),
            final_reply_pair_accuracy=float(reply_pairs.float().mean()),
            final_reply_pair_correct=int(reply_pairs.sum()),
            action_reply_agreement=float(reply_actions[query].eq(prediction[query]).float().mean()),
            reply_parseable_queries=int(reply_actions[query].ge(0).sum()))
    return result


class PreparedBank:
    """Private compiled view of canonical, uniform-length complete pairs."""

    def __init__(self, rows, *, role, config=None):
        from experiments.composition_curriculum import validate_pair, VERSION
        from experiments.composition_data import pack_composition_episodes, pack_observations

        start = time.monotonic()
        if role not in ("train_fit", "dev", "audit"):
            raise ValueError("explicit evaluation role required")
        self._canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        admitted = json.loads(self._canonical)
        if not admitted or len(admitted) % 2:
            raise ValueError("complete evaluation pairs required")
        seen = set()
        for index in range(0, len(admitted), 2):
            pair = admitted[index:index + 2]
            validate_pair(pair)
            group = pair[0]["counterfactual_group"]
            if group in seen or any(row["split"] != ("train" if role == "train_fit" else role) for row in pair):
                raise ValueError("evaluation bank role or group identity differs")
            seen.add(group)
        self._config = config or SequenceConfig(max_turns=12)
        c = self._config
        options = {"max_input_bytes": c.max_input_bytes, "max_context_tokens": c.max_positions,
                   "max_turns": c.max_turns}
        packed = pack_composition_episodes(admitted, training=False, max_reply_bytes=c.max_output_bytes, **options)
        self._targets = packed["supervision"]["action_targets"].clone()
        self._reply_targets = packed["supervision"]["reply_targets"].clone()
        texts = [[turn["text"] for turn in row["turns"]] for row in admitted]
        self._inputs = {"normal": packed["inputs"], "blank": pack_observations(texts, blank_text=True, **options),
            "reset": pack_observations([[text] for row in texts for text in row], **options)}
        self._identity = {"sha256": hashlib.sha256(self._canonical).hexdigest(), "version": VERSION,
            "episodes": len(admitted), "turns": len(texts[0]), "role": role, "config": asdict(c)}
        self.preparation_seconds = time.monotonic() - start

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def score(self, model, *, batch_size=32, score_replies=True, control="normal"):
        from experiments.composition_curriculum import REPLIES
        if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("evaluation batches must preserve pairs")
        if type(score_replies) is not bool or control not in self._inputs:
            raise ValueError("unknown evaluation options")
        if model.config != self._config:
            raise ValueError("prepared bank codec/model configuration differs")
        if hashlib.sha256(self._canonical).hexdigest() != self._identity["sha256"]:
            raise ValueError("prepared canonical evidence changed")
        before = checkpoint_digest(model)
        modes = [(module, module.training) for module in model.modules()]
        logits, replies, reply_actions = [], [], []
        device = next(model.parameters()).device
        count, turns = self._targets.shape
        start_time = time.monotonic()
        try:
            model.eval()
            with torch.inference_mode():
                for start in range(0, count, batch_size):
                    end = min(count, start + batch_size)
                    lo, hi = (start * turns, end * turns) if control == "reset" else (start, end)
                    inputs = {key: value[lo:hi].clone().to(device) for key, value in self._inputs[control].items()}
                    width = int(inputs["lengths"].max())
                    for key in ("token_ids", "valid_mask"):
                        inputs[key] = inputs[key][:, :width]
                    shape = inputs["eos_positions"].shape
                    output = model(**inputs, decoder_input_ids=torch.full((*shape, 1), ByteCodec.BOS,
                                                                         dtype=torch.long, device=device))
                    logits.append(output["logits"].reshape(end - start, turns, 4).cpu())
                    if score_replies:
                        generated = _generate_reply_tokens(model, output["production_context"].flatten(0, 1))
                        expected = self._reply_targets[start:end].flatten(0, 1).to(device)
                        replies.append(_exact_replies(generated, expected).reshape(end - start, turns).cpu())
                        decoded = []
                        # A single device transfer, then ordinary CPU text decoding.
                        for tokens in generated.cpu().tolist():
                            try:
                                text = model.codec.decode(tokens)
                            except (ValueError, UnicodeError):
                                text = None
                            decoded.append(REPLIES.index(text) if text in REPLIES else -1)
                        reply_actions.append(torch.tensor(decoded).reshape(end - start, turns))
            result = prediction_metrics(torch.cat(logits), self._targets,
                reply_correct=torch.cat(replies) if score_replies else None,
                reply_actions=torch.cat(reply_actions) if score_replies else None)
            result.update(bank=self.identity, control=control, free_running_replies=score_replies,
                teacher_used_for_policy=False, decoder_prefix="BOS only", seconds=time.monotonic() - start_time)
            return result
        finally:
            for module, mode in modes:
                module.training = mode
            if checkpoint_digest(model) != before:
                raise RuntimeError("evaluation changed model parameters")


def evaluate_banks(model, banks, **options):
    if not banks:
        raise ValueError("at least one prepared bank required")
    results = {name: bank.score(model, **options) for name, bank in banks.items()}
    return {"per_bank": results, **{f"macro_{key}": sum(row[key] for row in results.values()) / len(results)
        if all(row[key] is not None for row in results.values()) else None for key in METRICS}}


def verify_cpu_restart(model, row):
    from experiments.composition_data import pack_observations
    original = copy.deepcopy(model).cpu().eval()
    stream = io.BytesIO()
    torch.save({"config": asdict(model.config), "weights": _cpu_copy(model.state_dict())}, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    clone = build_sequence_student(0, config=SequenceConfig(**restored["config"])).eval()
    clone.load_state_dict(restored["weights"], strict=True)
    c = clone.config
    inputs = pack_observations([[turn["text"] for turn in row["turns"]]],
        max_turns=c.max_turns, max_input_bytes=c.max_input_bytes, max_context_tokens=c.max_positions)
    with torch.inference_mode():
        output = [item(**inputs, decoder_input_ids=torch.full((*inputs["eos_positions"].shape, 1),
                                ByteCodec.BOS, dtype=torch.long)) for item in (original, clone)]
        replies = [_generate_reply_tokens(item, value["production_context"].flatten(0, 1))
                   for item, value in zip((original, clone), output)]
    exact = torch.equal(output[0]["logits"], output[1]["logits"]) and torch.equal(replies[0], replies[1])
    if not exact:
        raise RuntimeError("CPU checkpoint restart is not exact")
    return {"exact": True, "device": "cpu", "weights_sha256": checkpoint_digest(clone),
            "utterances": inputs["eos_positions"].shape[1]}
