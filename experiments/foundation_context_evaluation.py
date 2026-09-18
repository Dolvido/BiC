"""Native, observation-only inference for the separate two-query context bank.

One sequence forward reads all twelve observations with BOS-only decoder inputs.
Only production_context[:, [10, 11], :] enters free reply generation: no labels,
answers, recipe, parser state, or condition tags are passed to the model. Bank
admission precedes inference and uses the diagnostic validator, not the older
canonical/final-turn scorers. Metrics always anchor known truth at index 10.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import _generate_reply_tokens, checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments.composition_data import pack_observations
from experiments import foundation_context_diagnostic as diagnostic


SCHEMA = "bic-context-sensitivity-evaluation-v1"
_ROW_KEYS = {"episode_id", "comparison_group", "counterfactual_group", "family",
             "operator_group", "depth", "pair_index", "gap", "naming_condition", "variant", "queries"}
_QUERY_KEYS = {"turn", "target", "action", "logits", "reply_tokens", "reply_text", "reply_action", "reply_exact"}
_COUNTERS = ("batch_intents", "completed_batches", "sequence_forward_attempts",
             "sequence_forward_completions", "encoder_episode_attempts", "encoder_episode_completions",
             "bos_reply_context_attempts", "free_query_context_attempts", "free_query_context_completions",
             "decoder_recurrent_attempts", "decoder_recurrent_completions",
             "decoder_row_step_attempts", "decoder_row_step_completions", "scored_episode_records")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = {"brain_in_computer/__init__.py", "brain_in_computer/model.py", "brain_in_computer/regions.py",
             "brain_in_computer/language.py", "brain_in_computer/learning_student.py",
             "brain_in_computer/dialogue_student.py", "experiments/sequence_data.py",
             "experiments/composition_data.py", "experiments/foundation_context_evaluation.py"}
    return {**diagnostic.source_hashes(), **{p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                           for p in sorted(names)}}


@dataclass
class EvaluationLedger:
    """Retain this object on failure; unmatched attempts are not zero work."""
    counts: dict = field(default_factory=lambda: dict.fromkeys(_COUNTERS, 0))

    def report(self):
        return {**deepcopy(self.counts),
                "unmatched_sequence_attempts": self.counts["sequence_forward_attempts"] - self.counts["sequence_forward_completions"],
                "unmatched_decoder_attempts": self.counts["decoder_recurrent_attempts"] - self.counts["decoder_recurrent_completions"],
                "scope": "Native sequence calls include twelve BOS decoder contexts/episode. Hooks count only subsequent free-query recurrent calls. Attempt/completion gaps represent uncertain partial work."}


def _tick(work, name, count=1):
    if type(work) is not EvaluationLedger or set(work.counts) != set(_COUNTERS):
        raise ValueError("unmodified EvaluationLedger required")
    work.counts[name] += count


def _ratio(count, total):
    return dict(count=count, total=total, rate=count / total if total else None)


def _trim(tokens):
    end = len(tokens)
    while end and tokens[end - 1] == ByteCodec.PAD:
        end -= 1
    return tokens[:end]


def _reply(tokens, target):
    if (type(tokens) is not list or not tokens or tokens[0] != ByteCodec.BOS
            or any(type(t) is not int or not 0 <= t < ByteCodec.VOCAB_SIZE for t in tokens)):
        raise ValueError("native generated BOS/byte tokens required")
    codec = ByteCodec(max_bytes=32)
    try:
        text = codec.decode(tokens)
    except (ValueError, UnicodeError):
        text = None
    action = diagnostic.oracle.REPLIES.index(text) if text in diagnostic.oracle.REPLIES else -1
    exact = _trim(tokens) == codec.encode(diagnostic.oracle.REPLIES[target])
    return dict(reply_tokens=tokens, reply_text=text, reply_action=action, reply_exact=exact)


def _validate_records(records):
    if type(records) is not list or not records or len(records) % 12:
        raise ValueError("complete diagnostic comparisons required")
    comparisons, ids, pair_ids = defaultdict(dict), set(), {}
    for row in records:
        if type(row) is not dict or set(row) != _ROW_KEYS:
            raise ValueError("exact raw query-record fields required")
        if any(type(row[k]) is not str or not row[k] for k in ("episode_id", "comparison_group", "counterfactual_group")):
            raise ValueError("nonempty diagnostic identities required")
        if row["episode_id"] in ids:
            raise ValueError("duplicate raw episode identity")
        ids.add(row["episode_id"])
        if row["family"] not in diagnostic.FAMILIES or row["operator_group"] not in diagnostic.GROUPS:
            raise ValueError("declared family/operator group required")
        depth = diagnostic._depth(row["operator_group"])
        if type(row["depth"]) is not int or row["depth"] != depth:
            raise ValueError("raw depth differs")
        for key, permitted in (("gap", diagnostic.GAPS), ("naming_condition", diagnostic.NAMINGS), ("variant", (0, 1))):
            if type(row[key]) is not int or row[key] not in permitted:
                raise ValueError("raw condition differs")
        if type(row["pair_index"]) is not int or not 0 <= row["pair_index"] < 64:
            raise ValueError("raw pair index differs")
        queries = row["queries"]
        if type(queries) is not dict or set(queries) != {"known", "unknown"}:
            raise ValueError("both fixed diagnostic queries required")
        for role, turn in (("known", 10), ("unknown", 11)):
            query = queries[role]
            targets = (0, 1) if role == "known" else (2,)
            if (type(query) is not dict or set(query) != _QUERY_KEYS
                    or type(query["turn"]) is not int or query["turn"] != turn
                    or type(query["target"]) is not int or query["target"] not in targets):
                raise ValueError("fixed query anchor or target differs")
            logits = query["logits"]
            if type(logits) is not list or len(logits) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits):
                raise ValueError("four finite native logits required")
            if type(query["action"]) is not int or query["action"] != max(range(4), key=logits.__getitem__):
                raise ValueError("action must be native first-argmax")
            detail = _reply(query["reply_tokens"], query["target"])
            if _json({k: query[k] for k in detail}) != _json(detail):
                raise ValueError("recorded reply interpretation differs from generated bytes")
        key = (row["gap"], row["naming_condition"], row["variant"])
        group = comparisons[row["comparison_group"]]
        if key in group:
            raise ValueError("duplicate condition/member in comparison")
        group[key] = row
        pair_key = (row["comparison_group"], *key[:2])
        previous = pair_ids.setdefault(row["counterfactual_group"], pair_key)
        if previous != pair_key:
            raise ValueError("pair identity crosses comparison/condition")
    expected = {(g, n, v) for g in diagnostic.GAPS for n in diagnostic.NAMINGS for v in (0, 1)}
    base_groups = set()
    for group in comparisons.values():
        if set(group) != expected:
            raise ValueError("missing comparison condition/member")
        reference = group[(0, 0, 0)]
        header = (reference["family"], reference["operator_group"], reference["depth"], reference["pair_index"])
        if header in base_groups:
            raise ValueError("base pair appears under multiple comparison identities")
        base_groups.add(header)
        for row in group.values():
            if (row["family"], row["operator_group"], row["depth"], row["pair_index"]) != header:
                raise ValueError("comparison crosses family/operator/base pair")
            variant = row["variant"]
            if row["queries"]["known"]["target"] != group[(0, 0, variant)]["queries"]["known"]["target"]:
                raise ValueError("matched conditions changed known truth")
        for gap in diagnostic.GAPS:
            for naming in diagnostic.NAMINGS:
                a, b = (group[(gap, naming, v)] for v in (0, 1))
                if (a["counterfactual_group"] != b["counterfactual_group"]
                        or {a["queries"]["known"]["target"], b["queries"]["known"]["target"]} != {0, 1}):
                    raise ValueError("known query pair must have opposite answers")
    return comparisons


def _flags(row):
    known, unknown = row["queries"]["known"], row["queries"]["unknown"]
    return {"known_action": known["action"] == known["target"], "known_reply": known["reply_exact"],
            "unknown_action": unknown["action"] == 2, "unknown_reply": unknown["reply_exact"]}


def _pair_flags(pair):
    flags = [_flags(row) for row in pair]
    action = all(f["known_action"] for f in flags)
    reply = all(f["known_reply"] for f in flags)
    return dict(paired_action=action, paired_reply=reply, paired_both=action and reply)


def _summary(rows):
    values, pairs = defaultdict(int), defaultdict(list)
    for row in rows:
        for name, value in _flags(row).items():
            values[name] += int(value)
        pairs[row["counterfactual_group"]].append(row)
        for query in row["queries"].values():
            values["query_action"] += query["action"] == query["target"]
            values["query_reply"] += query["reply_exact"]
            values["reply_parseable"] += query["reply_action"] >= 0
            values["action_reply_agreement"] += query["action"] == query["reply_action"]
        values["known_unsupported_ask"] += row["queries"]["known"]["action"] == 2
        values["known_reply_unsupported_ask"] += row["queries"]["known"]["reply_action"] == 2
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("summary split a counterfactual pair")
        for name, value in _pair_flags(pair).items():
            values[name] += int(value)
    denominators = {name: len(rows) for name in ("known_action", "known_reply", "unknown_action", "unknown_reply", "known_unsupported_ask", "known_reply_unsupported_ask")}
    denominators.update({name: 2 * len(rows) for name in ("query_action", "query_reply", "reply_parseable", "action_reply_agreement")})
    denominators.update({name: len(pairs) for name in ("paired_action", "paired_reply", "paired_both")})
    return dict(episodes=len(rows), query_turns=2*len(rows), pairs=len(pairs),
                counts={name: _ratio(values[name], total) for name, total in denominators.items()})


def _breakdown(items, summarize, header):
    result = {"overall": summarize(items)}
    for key, selector in (("by_family", lambda r: r["family"]),
                          ("by_group", lambda r: r["operator_group"]),
                          ("by_family_group", lambda r: r["family"] + "/" + r["operator_group"])):
        groups = defaultdict(list)
        for item in items:
            groups[selector(header(item))].append(item)
        result[key] = {name: summarize(group) for name, group in sorted(groups.items())}
    return result


def _matched_summary(matches):
    transitions, same = defaultdict(lambda: [0, 0, 0, 0]), defaultdict(int)
    episode_count = 0
    def add(name, reference, condition):
        transitions[name][2 * int(reference) + int(condition)] += 1
    for reference, condition in matches:
        for name, value in _pair_flags(reference).items():
            add(name, value, _pair_flags(condition)[name])
        for a, b in zip(reference, condition):
            episode_count += 1
            for name, value in _flags(a).items():
                add(name, value, _flags(b)[name])
            for query in ("known", "unknown"):
                qa, qb = a["queries"][query], b["queries"][query]
                same[query + "_action_unchanged"] += qa["action"] == qb["action"]
                same[query + "_reply_tokens_unchanged"] += _trim(qa["reply_tokens"]) == _trim(qb["reply_tokens"])
    tables = {}
    for name, (neither, condition_only, reference_only, both) in transitions.items():
        total = neither + condition_only + reference_only + both
        tables[name] = dict(neither_correct=neither, condition_only=condition_only,
            reference_only=reference_only, both_correct=both, total=total,
            delta_count=condition_only-reference_only, delta_rate=(condition_only-reference_only)/total)
    return dict(matched_pairs=len(matches), matched_episodes=episode_count,
                correctness_transitions=tables,
                prediction_consistency={name: _ratio(count, episode_count) for name, count in same.items()})


def score_records(records):
    """Pure arithmetic on complete raw query records; no model or bank generator."""
    groups = _validate_records(records)
    result = _breakdown(records, _summary, lambda row: row)
    result["by_condition"] = {f"gap{gap}/naming{naming}": _breakdown(
        [r for r in records if r["gap"] == gap and r["naming_condition"] == naming],
        _summary, lambda row: row) for gap in diagnostic.GAPS for naming in diagnostic.NAMINGS}
    contrasts = {}
    for gap in (2, 4):
        matches = [([group[(0, naming, v)] for v in (0, 1)],
                    [group[(gap, naming, v)] for v in (0, 1)])
                   for group in groups.values() for naming in diagnostic.NAMINGS]
        contrasts[f"gap{gap}-minus-gap0"] = _breakdown(matches, _matched_summary, lambda match: match[0][0])
    matches = [([group[(gap, 0, v)] for v in (0, 1)], [group[(gap, 1, v)] for v in (0, 1)])
               for group in groups.values() for gap in diagnostic.GAPS]
    contrasts["naming1-minus-naming0"] = _breakdown(matches, _matched_summary, lambda match: match[0][0])
    result["matched_contrasts"] = contrasts
    return result


def _configuration(model):
    config = asdict(model.config) if is_dataclass(model.config) else vars(model.config).copy()
    for name in ("max_turns", "max_input_bytes", "max_positions", "max_output_bytes"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError("positive configured inference bounds required")
    if config["max_turns"] < 12 or config["max_output_bytes"] != 32:
        raise ValueError("diagnostic requires twelve-turn capacity and the fixed32-byte reply limit")
    return config


def _state(model):
    return dict(weights_sha256=checkpoint_digest(model), config=_configuration(model),
                requires_grad=[(name, parameter.requires_grad) for name, parameter in model.named_parameters()])


def _boundary(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("context evaluation deadline reached before the next operation")


def _pack_rows(rows, *, device, config):
    """The only bank-to-model packing boundary; supervision is never selected."""
    return pack_observations([[turn["text"] for turn in row["turns"]] for row in rows],
        device=device, max_turns=config["max_turns"], max_input_bytes=config["max_input_bytes"],
        max_context_tokens=config["max_positions"])


def _free_queries(model, context, work):
    count = context.shape[0]
    def attempted(module, args):
        _tick(work, "decoder_recurrent_attempts")
        _tick(work, "decoder_row_step_attempts", count)
    def completed(module, args, output):
        _tick(work, "decoder_recurrent_completions")
        _tick(work, "decoder_row_step_completions", count)
    before = work.counts["decoder_recurrent_completions"]
    hook1 = model.inferior_frontal.recurrent.register_forward_pre_hook(attempted)
    hook2 = None
    try:
        hook2 = model.inferior_frontal.recurrent.register_forward_hook(completed)
        generated = _generate_reply_tokens(model, context)
    finally:
        hook1.remove()
        if hook2 is not None:
            hook2.remove()
    if (not isinstance(generated, torch.Tensor) or generated.dtype != torch.long or generated.ndim != 2
            or generated.shape[0] != count or not 2 <= generated.shape[1] <= model.config.max_output_bytes+2
            or work.counts["decoder_recurrent_completions"]-before != generated.shape[1]-1):
        raise ValueError("native free-generation shape or recurrent accounting differs")
    return generated


class PreparedContextBank:
    """Immutable JSON copy admitted once, with no teacher tensors in model inputs."""
    def __init__(self, bank, *, validation_work=None):
        started = time.monotonic()
        canonical = _json(bank).encode()
        admitted = json.loads(canonical)
        diagnostic.validate_bank(admitted, work=validation_work)
        self._canonical = canonical
        self._identity = dict(schema=diagnostic.BANK_SCHEMA, bank_sha256=hashlib.sha256(canonical).hexdigest(),
            rows_sha256=admitted["rows_sha256"], config=deepcopy(admitted["config"]),
            episodes=len(admitted["rows"]), known_query_turn=10, unknown_query_turn=11)
        self.preparation_seconds = time.monotonic()-started

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _check(self):
        if hashlib.sha256(self._canonical).hexdigest() != self._identity["bank_sha256"]:
            raise ValueError("prepared diagnostic bank changed")

    def score(self, model, *, batch_size=32, progress=None, deadline=None, work=None):
        """Nonpreemptive native batches; preserve a supplied ledger after any failure.

        Progress callback failures abort; they never authorize another batch.
        The caller's deadline is checked before packing, forward, and free
        decoding. A started model or recurrent operation may finish after it.
        """
        if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("positive even evaluation batch size required")
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("finite absolute monotonic deadline required")
        work = EvaluationLedger() if work is None else work
        if type(work) is not EvaluationLedger or any(work.counts.values()):
            raise ValueError("fresh EvaluationLedger required per score invocation")
        _boundary(deadline)
        self._check()
        admitted = json.loads(self._canonical)
        rows = admitted["rows"]
        before = _state(model)
        modules = list(model.modules())
        modes = [(module, module.training) for module in modules]
        device = next(model.parameters()).device
        config = before["config"]
        records, completed = [], False
        started, cpu_started = time.monotonic(), time.process_time()
        after = None
        try:
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(rows), batch_size):
                    _boundary(deadline)
                    chunk = rows[start:start+batch_size]
                    episodes = len(chunk)
                    event = dict(batch_index=start//batch_size, episode_start=start, episodes=episodes,
                        query_contexts=2*episodes, action_forward_calls=1,
                        query_turn_indices=[10, 11], maximum_decoder_recurrent_calls=config["max_output_bytes"]+1,
                        maximum_decoder_row_steps=2*episodes*(config["max_output_bytes"]+1))
                    _tick(work, "batch_intents")
                    if progress is not None:
                        progress({"event": "batch_intent", **event})
                    _boundary(deadline)
                    observed = _pack_rows(chunk, device=device, config=config)
                    _boundary(deadline)
                    _tick(work, "sequence_forward_attempts")
                    _tick(work, "encoder_episode_attempts", episodes)
                    _tick(work, "bos_reply_context_attempts", 12*episodes)
                    output = model(**observed, decoder_input_ids=torch.full((episodes, 12, 1), ByteCodec.BOS,
                                                                          dtype=torch.long, device=device))
                    actions, context = output["logits"], output["production_context"]
                    if (actions.shape != (episodes, 12, 4) or context.ndim != 3 or context.shape[:2] != (episodes, 12)
                            or not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(context).all())):
                        raise ValueError("finite native action logits and reply contexts required")
                    _tick(work, "sequence_forward_completions")
                    _tick(work, "encoder_episode_completions", episodes)
                    query_context = context[:, [10, 11], :].reshape(2*episodes, -1)
                    _boundary(deadline)
                    _tick(work, "free_query_context_attempts", 2*episodes)
                    before_steps = work.counts["decoder_recurrent_completions"]
                    generated = _free_queries(model, query_context, work)
                    native_logits = actions[:, [10, 11], :].detach().cpu().tolist()
                    generated_tokens = generated.detach().cpu().tolist()
                    _tick(work, "free_query_context_completions", 2*episodes)
                    for index, row in enumerate(chunk):
                        queries = {}
                        for offset, (role, turn_index) in enumerate((("known", 10), ("unknown", 11))):
                            logits = native_logits[index][offset]
                            target = row["turns"][turn_index]["target"]
                            queries[role] = dict(turn=turn_index, target=target,
                                action=max(range(4), key=logits.__getitem__), logits=logits,
                                **_reply(generated_tokens[2*index+offset], target))
                        records.append(dict(episode_id=row["id"], comparison_group=row["comparison_group"],
                            counterfactual_group=row["counterfactual_group"], family=row["family"],
                            operator_group=row["operator_group"], depth=row["depth"],
                            pair_index=row["recipe"]["pair_index"], gap=row["gap"],
                            naming_condition=row["naming_condition"], variant=row["variant"], queries=queries))
                        _tick(work, "scored_episode_records")
                    recurrent_calls = work.counts["decoder_recurrent_completions"]-before_steps
                    _tick(work, "completed_batches")
                    if progress is not None:
                        progress({"event": "batch_complete", **event,
                            "decoder_recurrent_calls": recurrent_calls,
                            "decoder_row_steps": recurrent_calls*2*episodes, "cumulative_work": work.report()})
                    del observed, output, actions, context, query_context, generated
            metrics = score_records(records)
            completed = True
        finally:
            for module, training in modes:
                module.training = training
            after = _state(model)
            self._check()
            if list(model.modules()) != modules or any(module.training != training for module, training in modes):
                raise RuntimeError("context evaluation changed model module/training-mode structure")
            if before != after:
                raise RuntimeError("context evaluation changed model state/configuration")
        if not completed:
            raise RuntimeError("incomplete context evaluation")
        return dict(schema=SCHEMA, status="completed", bank=self.identity, raw_records=records,
            metrics=metrics, state_before=before, state_after=after, training_modes_restored=True,
            evaluation_batch_size=batch_size, free_query_turn_indices=[10, 11],
            decoder_prefix="BOS only", teacher_used_for_policy=False,
            work=work.report(), wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
            timing_scope="Score interval starts after admission/JSON copy and the initial model-state guard; includes final guard and metrics. Caller must enclose preparation and both guards for complete cost.",
            scope=diagnostic.SCOPE)
