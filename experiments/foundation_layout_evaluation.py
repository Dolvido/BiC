"""Teacher-free evaluation with an explicit, variably placed known anchor.

Every pair is admitted before any model call. Immutable canonical JSON owns the
bank; only copied observation tensors and BOS enter inference. The caller owns
checkpoint/source/runtime authentication and durable invocation accounting.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from brain_in_computer.dialogue_student import _generate_reply_tokens, checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments.composition_data import pack_composition_episodes, pack_observations
from experiments.composition_curriculum import FAMILIES, REPLIES
from experiments.sequence_student import SequenceConfig
from experiments import foundation_layout_curriculum as curriculum


SCHEMA = "bic-foundation-layout-evaluation-v1"
_COUNTERS = ("batch_intents", "completed_batches", "sequence_forward_attempts",
    "sequence_forward_completions", "encoder_episode_attempts", "encoder_episode_completions",
    "encoder_actual_position_attempts", "encoder_actual_position_completions",
    "encoder_padded_position_attempts", "encoder_padded_position_completions",
    "bos_reply_context_attempts", "bos_reply_context_completions",
    "bos_decoder_recurrent_attempts", "bos_decoder_recurrent_completions",
    "bos_decoder_row_step_attempts", "bos_decoder_row_step_completions",
    "free_reply_context_attempts", "free_reply_context_completions",
    "free_decoder_calls_attempted", "free_decoder_calls_completed",
    "decoder_recurrent_attempts", "decoder_recurrent_completions",
    "decoder_row_step_attempts", "decoder_row_step_completions", "scored_episode_records")
_ROW_KEYS = {"episode_id", "counterfactual_group", "base_pair_sha256", "base_episode_id",
    "family", "depth", "operator_group", "variant", "turns", "layout",
    "anchor_query_id", "anchor_turn_index", "queries"}
_QUERY_KEYS = {"query_id", "turn", "original_turn", "target", "action", "logits",
               "reply_tokens", "reply_text", "reply_action", "reply_exact"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def source_hashes():
    """Relevant repository source identities; installed runtime is caller-bound."""
    root = Path(__file__).resolve().parents[1]
    names = {"experiments/foundation_layout_evaluation.py", "experiments/composition_data.py",
        "experiments/sequence_data.py", "experiments/sequence_student.py"}
    names.update(p.relative_to(root).as_posix() for p in (root / "brain_in_computer").glob("*.py"))
    return {**curriculum.source_hashes(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                           for name in sorted(names)}}


@dataclass
class EvaluationLedger:
    counts: dict = field(default_factory=lambda: dict.fromkeys(_COUNTERS, 0))

    def report(self):
        return {**deepcopy(self.counts),
            "unmatched_sequence_attempts": self.counts["sequence_forward_attempts"]-self.counts["sequence_forward_completions"],
            "unmatched_decoder_attempts": self.counts["decoder_recurrent_attempts"]-self.counts["decoder_recurrent_completions"],
            "scope": "Reset encodes each utterance as an independent episode pass. BOS work and all-turn free decoding are separate. Attempts without completions may contain partial physical work."}


def _tick(work, key, value=1):
    work.counts[key] += value


def _ratio(count, total):
    return dict(count=count, total=total, rate=count/total if total else None)


def _reply(tokens, target):
    if (type(tokens) is not list or len(tokens) < 2 or tokens[0] != ByteCodec.BOS
            or any(type(v) is not int or not 0 <= v < ByteCodec.VOCAB_SIZE for v in tokens)):
        raise ValueError("generated full-vocabulary BOS token row required")
    codec = ByteCodec(max_bytes=32)
    try:
        text = codec.decode(tokens)
    except (ValueError, UnicodeError):
        text = None
    end = len(tokens)
    while end and tokens[end-1] == ByteCodec.PAD:
        end -= 1
    return dict(reply_tokens=list(tokens), reply_text=text,
        reply_action=REPLIES.index(text) if text in REPLIES else -1,
        reply_exact=tokens[:end] == codec.encode(REPLIES[target]))


def _validate_records(records):
    if type(records) is not list or not records or len(records) % 2:
        raise ValueError("nonempty complete-pair record list required")
    seen, groups = set(), defaultdict(dict)
    for row in records:
        if type(row) is not dict or set(row) != _ROW_KEYS:
            raise ValueError("layout record fields differ")
        for key in ("episode_id", "counterfactual_group", "base_pair_sha256", "base_episode_id", "anchor_query_id"):
            if type(row[key]) is not str or not row[key]:
                raise ValueError("nonempty record identities required")
        if (row["episode_id"] in seen or row["family"] not in FAMILIES
                or type(row["variant"]) is not int or row["variant"] not in (0, 1)
                or type(row["depth"]) is not int or not 0 <= row["depth"] <= 5
                or type(row["turns"]) is not int or row["turns"] not in (8, 10, 12)
                or row["layout"] not in curriculum.LAYOUTS
                or row["operator_group"] not in ({"direct"} if row["depth"] == 0 else
                    {"copy", "advance"} if row["depth"] == 1 else {"composed_d"+str(row["depth"])})):
            raise ValueError("record identity or cell differs")
        seen.add(row["episode_id"])
        if (type(row["anchor_turn_index"]) is not int or not 0 <= row["anchor_turn_index"] < row["turns"]
                or type(row["queries"]) is not list or not row["queries"]):
            raise ValueError("explicit anchor and query list required")
        positions, identities, originals = set(), set(), set()
        anchor = None
        for query in row["queries"]:
            if type(query) is not dict or set(query) != _QUERY_KEYS:
                raise ValueError("query record fields differ")
            if (type(query["query_id"]) is not str or not query["query_id"] or query["query_id"] in identities
                    or any(type(query[k]) is not int or not 0 <= query[k] < row["turns"] for k in ("turn", "original_turn"))
                    or query["turn"] in positions or query["original_turn"] in originals
                    or type(query["target"]) is not int or query["target"] not in (0, 1, 2)
                    or type(query["action"]) is not int or query["action"] not in (0, 1, 2, 3)):
                raise ValueError("query coordinate, target or action differs")
            positions.add(query["turn"]); identities.add(query["query_id"]); originals.add(query["original_turn"])
            logits = query["logits"]
            if (type(logits) is not list or len(logits) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)
                    or query["action"] != max(range(4), key=logits.__getitem__)):
                raise ValueError("native logits and action disagree")
            if _json({k: query[k] for k in _reply(query["reply_tokens"], query["target"])}) != _json(_reply(query["reply_tokens"], query["target"])):
                raise ValueError("reply fields disagree with native bytes")
            if query["query_id"] == row["anchor_query_id"]:
                anchor = query
        if anchor is None or anchor["turn"] != row["anchor_turn_index"] or anchor["target"] not in (0, 1):
            raise ValueError("record anchor is not its declared known query")
        group = groups[row["counterfactual_group"]]
        if row["variant"] in group:
            raise ValueError("duplicate pair member")
        group[row["variant"]] = row
    for group in groups.values():
        if set(group) != {0, 1}:
            raise ValueError("missing pair member")
        left, right = group[0], group[1]
        keys = ("base_pair_sha256", "family", "depth", "operator_group", "turns", "layout", "anchor_query_id", "anchor_turn_index")
        if any(left[k] != right[k] for k in keys) or left["base_episode_id"] == right["base_episode_id"]:
            raise ValueError("counterfactual pair metadata differs")
        a, b = ({q["query_id"]: q for q in row["queries"]} for row in (left, right))
        if set(a) != set(b) or any((a[k]["turn"], a[k]["original_turn"]) != (b[k]["turn"], b[k]["original_turn"]) for k in a):
            raise ValueError("pair query coordinates differ")
        anchor = left["anchor_query_id"]
        if {a[anchor]["target"], b[anchor]["target"]} != {0, 1}:
            raise ValueError("anchor pair must have opposite known answers")
    return groups


def _summary(records):
    queries = [q for row in records for q in row["queries"]]
    other_known = [q for row in records for q in row["queries"]
                   if q["target"] < 2 and q["query_id"] != row["anchor_query_id"]]
    counts = {}
    for name, selected in (("query", queries), ("known", [q for q in queries if q["target"] < 2]),
                           ("other_known", other_known), ("unknown", [q for q in queries if q["target"] == 2])):
        counts[name+"_action"] = _ratio(sum(q["action"] == q["target"] for q in selected), len(selected))
        counts[name+"_reply"] = _ratio(sum(q["reply_exact"] for q in selected), len(selected))
    counts["action_reply_agreement"] = _ratio(sum(q["action"] == q["reply_action"] for q in queries), len(queries))
    counts["reply_parseable"] = _ratio(sum(q["reply_action"] >= 0 for q in queries), len(queries))
    known = [q for q in queries if q["target"] < 2]
    counts["known_unsupported_ask"] = _ratio(sum(q["action"] == 2 for q in known), len(known))
    counts["known_reply_unsupported_ask"] = _ratio(sum(q["reply_action"] == 2 for q in known), len(known))
    groups = defaultdict(list)
    for row in records:
        groups[row["counterfactual_group"]].append(next(q for q in row["queries"] if q["query_id"] == row["anchor_query_id"]))
    action = reply = both = 0
    for pair in groups.values():
        ac = all(q["action"] == q["target"] for q in pair)
        rc = all(q["reply_exact"] for q in pair)
        action += ac; reply += rc; both += ac and rc
    for name, value in (("action", action), ("reply", reply), ("both", both)):
        counts["anchor_pair_"+name] = _ratio(value, len(groups))
    return dict(episodes=len(records), pairs=len(groups), query_turns=len(queries), counts=counts)


def score_records(records, *, expected_bank=None):
    """Pure JSON arithmetic; authenticity of raw records remains caller-owned."""
    _validate_records(records)
    if expected_bank is not None:
        inventory = [dict(episode_id=row["episode_id"], queries=[{key: query[key] for key in
            ("query_id", "turn", "original_turn", "target")} for query in row["queries"]]) for row in records]
        if _json(inventory) != _json(expected_bank["query_inventory"]):
            raise ValueError("raw records differ from the admitted complete query inventory")
    result = dict(overall=_summary(records))
    for label, key in (("family", lambda r: r["family"]), ("group", lambda r: r["operator_group"]),
                       ("family_group", lambda r: r["family"]+"/"+r["operator_group"])):
        groups = defaultdict(list)
        for row in records:
            groups[key(row)].append(row)
        result["by_"+label] = {name: _summary(rows) for name, rows in sorted(groups.items())}
    return result


def _tensor_digest(value):
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def _state(model):
    tensors = model.state_dict()
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("finite learner state required")
    return dict(weights_sha256=checkpoint_digest(model), config=asdict(model.config),
        layout={name: [str(value.device), str(value.dtype), list(value.shape), list(value.stride()), value.storage_offset()]
                for name, value in tensors.items()},
        parameters=[dict(name=name, identity=id(value), requires_grad=value.requires_grad,
            grad=None if value.grad is None else dict(device=str(value.grad.device), dtype=str(value.grad.dtype),
                shape=list(value.grad.shape), sha256=_tensor_digest(value.grad))) for name, value in model.named_parameters()])


def _boundary(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("layout evaluation allowance expired before next operation")


def _operator(row):
    depth = row["depth"]
    if depth == 0:
        return "direct"
    if depth > 1:
        return "composed_d"+str(depth)
    query = next(q for q in row["queries"] if q["query_id"] == row["anchor"]["query_id"])
    operations = [v["event"]["op"] for v in query["version_signature"]["versions"] if v["event"]["op"] in ("copy", "advance")]
    if len(operations) != 1:
        raise ValueError("depth-one anchor must have one copy or advance")
    return operations[0]


class PreparedLayoutBank:
    def __init__(self, rows, *, role, config=None, validation_work=None):
        started = time.monotonic()
        if role not in ("train_fit", "dev", "audit"):
            raise ValueError("explicit train_fit/dev/audit role required")
        self._config = config or SequenceConfig(max_turns=12)
        if type(self._config) is not SequenceConfig:
            raise ValueError("exact SequenceConfig required")
        canonical = _json(rows).encode("utf-8")
        admitted = json.loads(canonical)
        if type(admitted) is not list or not admitted or len(admitted) % 2:
            raise ValueError("nonempty complete layout pairs required")
        groups, identifiers, turns, anchors = set(), set(), set(), []
        for offset in range(0, len(admitted), 2):
            pair = admitted[offset:offset+2]
            if curriculum.validate_pair(pair, work=validation_work) is False:
                raise ValueError("layout pair admission failed")
            group = pair[0]["counterfactual_group"]
            if group in groups or any(row["split"] != ("train" if role == "train_fit" else role) for row in pair):
                raise ValueError("duplicate layout group or role mismatch")
            groups.add(group)
            for row in pair:
                if row["id"] in identifiers:
                    raise ValueError("duplicate episode identity")
                identifiers.add(row["id"]); turns.add(len(row["turns"]))
                anchors.append(dict(episode_id=row["id"], query_id=row["anchor"]["query_id"], turn_index=row["anchor"]["turn_index"]))
                _operator(row)
        if len(turns) != 1 or next(iter(turns)) not in (8, 10, 12):
            raise ValueError("uniform eight/ten/twelve-turn layout bank required")
        c = self._config
        options = dict(max_turns=c.max_turns, max_input_bytes=c.max_input_bytes, max_context_tokens=c.max_positions)
        # All canonical pair checks precede this structural-only packing step.
        packed = pack_composition_episodes(admitted, training=False, max_reply_bytes=c.max_output_bytes, **options)
        texts = [[turn["text"] for turn in row["turns"]] for row in admitted]
        self._inputs = {"normal": packed["inputs"], "blank": pack_observations(texts, blank_text=True, **options),
                        "reset": pack_observations([[text] for row in texts for text in row], **options)}
        self._canonical = canonical
        self._identity = dict(schema=curriculum.VERSION, sha256=hashlib.sha256(canonical).hexdigest(),
            role=role, config=asdict(c), episodes=len(admitted), pairs=len(groups), turns=next(iter(turns)),
            anchors=anchors, anchors_sha256=_hash(anchors),
            query_inventory=[dict(episode_id=row["id"], queries=[dict(query_id=q["query_id"],
                turn=q["turn_index"], original_turn=q["original_turn_index"], target=row["turns"][q["turn_index"]]["target"])
                for q in sorted(row["queries"], key=lambda q: q["turn_index"])]) for row in admitted])
        self._identity_bytes = _json(self._identity).encode("utf-8")
        self._input_hash = self._compiled_digest()
        self.preparation_seconds = time.monotonic()-started
        self.last_report = None

    def _compiled_digest(self):
        return _hash({control: {name: [str(value.dtype), list(value.shape), _tensor_digest(value)]
                      for name, value in tensors.items()} for control, tensors in self._inputs.items()})

    def _check(self):
        if (hashlib.sha256(self._canonical).hexdigest() != self._identity["sha256"]
                or _json(self._identity).encode("utf-8") != self._identity_bytes
                or self._compiled_digest() != self._input_hash):
            raise ValueError("immutable bank identity or compiled observations changed")

    @property
    def identity(self):
        self._check()
        return json.loads(self._identity_bytes)

    def score(self, model, *, batch_size=32, control="normal", progress=None, deadline=None, work=None):
        started, cpu_started = time.monotonic(), time.process_time()
        report = dict(schema=SCHEMA, status="preflight", control=control, model_state_unchanged=None,
            training_modes_restored=False, optimizer_updates=0, backwards=0)
        self.last_report = deepcopy(report)
        handles, modes, modules, before, error, traceback, result = [], [], None, None, None, None, None
        ledger_valid = False
        try:
            if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
                raise ValueError("positive even batch size required")
            if control not in self._inputs or progress is not None and not callable(progress):
                raise ValueError("normal/blank/reset and optional progress callback required")
            if deadline is not None and (type(deadline) not in (float, int) or not math.isfinite(deadline)):
                raise ValueError("finite monotonic deadline required")
            work = EvaluationLedger() if work is None else work
            if type(work) is not EvaluationLedger or set(work.counts) != set(_COUNTERS) or any(type(v) is not int or v != 0 for v in work.counts.values()):
                raise ValueError("fresh unmodified EvaluationLedger required")
            ledger_valid = True
            self._check(); _boundary(deadline)
            if type(model.config) is not SequenceConfig or model.config != self._config:
                raise ValueError("model and bank configuration differ")
            rows = json.loads(self._canonical)
            before = _state(model)
            modules = list(model.modules()); modes = [(module, module.training) for module in modules]
            device = next(model.parameters()).device
            phase = "bos"

            def decoder_tick(suffix, inputs):
                value = inputs[0]
                prefix = "bos_" if phase == "bos" else ""
                _tick(work, prefix+"decoder_recurrent_"+suffix)
                _tick(work, prefix+"decoder_row_step_"+suffix, value.shape[0]*value.shape[1])

            def decoder_complete(module, inputs, output):
                decoder_tick("completions", inputs)
                if not isinstance(output, tuple) or any(not bool(torch.isfinite(v).all()) for v in output):
                    raise FloatingPointError("nonfinite decoder recurrent state")

            def readout_complete(module, inputs, output):
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError("nonfinite decoder logits before argmax")

            recurrent = model.inferior_frontal.recurrent
            handles.append(recurrent.register_forward_pre_hook(lambda module, inputs: decoder_tick("attempts", inputs)))
            handles.append(recurrent.register_forward_hook(decoder_complete))
            handles.append(model.inferior_frontal.readout.register_forward_hook(readout_complete))
            model.eval()
            records, turns = [], self._identity["turns"]
            report["status"] = "running"
            with torch.inference_mode():
                for start in range(0, len(rows), batch_size):
                    _boundary(deadline)
                    chunk = rows[start:start+batch_size]; count = len(chunk)
                    event = dict(batch_index=start//batch_size, episode_start=start, episodes=count,
                                 turns=turns, control=control, free_reply_contexts=count*turns)
                    _tick(work, "batch_intents")
                    if progress is not None:
                        progress(dict(event="batch_intent", **event))
                    _boundary(deadline)
                    lo, hi = (start*turns, (start+count)*turns) if control == "reset" else (start, start+count)
                    inputs = {key: value[lo:hi].clone().to(device) for key, value in self._inputs[control].items()}
                    width = int(inputs["lengths"].max())
                    for key in ("token_ids", "valid_mask"):
                        inputs[key] = inputs[key][:, :width]
                    shape = inputs["eos_positions"].shape
                    actual, padded = int(inputs["lengths"].sum()), inputs["token_ids"].numel()
                    _boundary(deadline); phase = "bos"
                    for key, amount in (("sequence_forward", 1), ("encoder_episode", shape[0]),
                        ("encoder_actual_position", actual), ("encoder_padded_position", padded), ("bos_reply_context", count*turns)):
                        _tick(work, key+"_attempts", amount)
                    output = model(**inputs, decoder_input_ids=torch.full((*shape, 1), ByteCodec.BOS, dtype=torch.long, device=device))
                    for key, amount in (("sequence_forward", 1), ("encoder_episode", shape[0]),
                        ("encoder_actual_position", actual), ("encoder_padded_position", padded), ("bos_reply_context", count*turns)):
                        _tick(work, key+"_completions", amount)
                    logits, context = output["logits"], output["production_context"]
                    if (not isinstance(logits, torch.Tensor) or logits.shape != (*shape, 4)
                            or not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[:2] != shape
                            or context.shape[2] < 1 or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(context).all())):
                        raise ValueError("finite native action logits and reply contexts required")
                    _boundary(deadline); phase = "free"
                    _tick(work, "free_reply_context_attempts", count*turns); _tick(work, "free_decoder_calls_attempted")
                    previous = work.counts["decoder_recurrent_completions"]
                    generated = _generate_reply_tokens(model, context.flatten(0, 1))
                    _tick(work, "free_decoder_calls_completed"); _tick(work, "free_reply_context_completions", count*turns)
                    if (not isinstance(generated, torch.Tensor) or generated.dtype != torch.long or generated.ndim != 2
                            or generated.shape[0] != count*turns or not 2 <= generated.shape[1] <= self._config.max_output_bytes+2
                            or not bool(generated[:, 0].eq(ByteCodec.BOS).all())
                            or not bool(((generated >= 0) & (generated < ByteCodec.VOCAB_SIZE)).all())
                            or work.counts["decoder_recurrent_completions"]-previous != generated.shape[1]-1):
                        raise ValueError("free decoder shape or recurrent accounting differs")
                    raw_logits = logits.reshape(count, turns, 4).detach().cpu().tolist()
                    raw_tokens = generated.detach().cpu().tolist()
                    for member, row in enumerate(chunk):
                        queries = []
                        for query in sorted(row["queries"], key=lambda q: q["turn_index"]):
                            turn = query["turn_index"]; target = row["turns"][turn]["target"]
                            values = raw_logits[member][turn]
                            queries.append(dict(query_id=query["query_id"], turn=turn, original_turn=query["original_turn_index"],
                                target=target, action=max(range(4), key=values.__getitem__), logits=values,
                                **_reply(raw_tokens[member*turns+turn], target)))
                        records.append(dict(episode_id=row["id"], counterfactual_group=row["counterfactual_group"],
                            base_pair_sha256=row["recipe"]["base_pair_sha256"], base_episode_id=row["recipe"]["base_ids"][row["variant"]],
                            family=row["family"], depth=row["depth"], operator_group=_operator(row), variant=row["variant"],
                            turns=turns, layout=row["recipe"]["layout"], anchor_query_id=row["anchor"]["query_id"],
                            anchor_turn_index=row["anchor"]["turn_index"], queries=queries))
                        _tick(work, "scored_episode_records")
                    _tick(work, "completed_batches")
                    if progress is not None:
                        progress(dict(event="batch_complete", **event, cumulative_work=work.report()))
                    del inputs, output, logits, context, generated
            result = dict(schema=SCHEMA, status="completed", bank=self.identity, control=control,
                metrics=score_records(records, expected_bank=self.identity), raw_records=records, evaluation_batch_size=batch_size,
                free_running_replies=True, teacher_used_for_policy=False, decoder_prefix="BOS only",
                decoded_turns="all", weights_sha256=before["weights_sha256"])
        except BaseException as caught:
            error, traceback = caught, caught.__traceback__
        finally:
            try:
                for handle in handles:
                    handle.remove()
                for module, training in modes:
                    module.training = training
                if before is not None:
                    report["training_modes_restored"] = (list(model.modules()) == modules
                        and all(module.training == training for module, training in modes))
                    report["model_state_unchanged"] = _state(model) == before
                    if not report["training_modes_restored"] or not report["model_state_unchanged"]:
                        raise RuntimeError("layout evaluation changed learner state or module structure")
                self._check()
                if before is not None and device.type == "cuda":
                    torch.cuda.synchronize(device)
            except BaseException as cleanup:
                if error is None:
                    error, traceback = cleanup, cleanup.__traceback__
                else:
                    report["cleanup_error"] = type(cleanup).__name__+": "+str(cleanup)
            report.update(status="completed" if error is None else "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
                work=work.report() if ledger_valid else None,
                timing_scope="Score entry through copies, recount, final state/bank guards and device synchronization; bank preparation and external serialization excluded.")
            if error is not None:
                report["error"] = type(error).__name__+": "+str(error)
            self.last_report = deepcopy(report)
        if error is not None:
            error.layout_report = deepcopy(report)
            raise error.with_traceback(traceback)
        result.update(report)
        return result
