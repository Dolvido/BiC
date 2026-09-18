"""Matched fixed/fresh realizations of canonical compact composition recipes.

The neural architecture and objective are unchanged. Structural draws use their
own generators; realization seeds are stateless and cannot alter those draws.
Consumed observations, not prefetched rows, own all counters and replay evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping

import torch

from brain_in_computer.learning_student import _check_finite_tree, _integer
from experiments.composition_curriculum import FAMILIES, generate_pair, parse_sentence, validate_pair
from experiments.composition_data import pack_composition_episodes
from experiments.composition_training import CompositionTrainer, COUNTS, _admit, _row_counts
from experiments.sequence_student import SequenceConfig
from experiments.train_cognitive import fingerprint_rows


VERSION = "bic-realization-stream-v1"
MAX_ATTEMPTS = 256
ZERO_DIGEST = "0" * 64
_BASE_FIELDS = {"recipe", "weights", "optimizer", "updates", "samplers", "family_microbatches",
                "bucket_microbatches", "exposures"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf8")).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _chain(previous, value):
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(value)).hexdigest()


def _transcript(row):
    from experiments.realization_banks import transcript_digest
    return transcript_digest(row)


def realization_seeds(family, turns, index, occurrence, attempt):
    """No model output or mutable RNG participates in names/value generation."""
    key = [VERSION, family, turns, index, occurrence, attempt]
    return tuple(int(_hash([*key, source]), 16) % (2**63) for source in ("names", "values"))


def stream_evidence(payload):
    """Comparable JSON evidence; runtime timings are deliberately excluded."""
    result = copy.deepcopy({key: value for key, value in payload.items()
                            if key not in ("seen_transcripts", "samplers", "timing")})
    result["seen_transcript_count"] = len(payload["seen_transcripts"])
    result["seen_transcripts_sha256"] = _hash(sorted(payload["seen_transcripts"]))
    result["sampler_sha256"] = {family: hashlib.sha256(bytes(state.tolist())).hexdigest()
                                for family, state in payload["samplers"].items()}
    return result


class RealizationStream:
    """Pure canonical stream, usable for training or one-pass evidence replay."""

    def __init__(self, banks, *, mode, protected_transcripts, sampler_seed=3901, micro_batch_size=32):
        if mode not in ("fixed", "fresh"):
            raise ValueError("realization mode must be fixed or fresh")
        _integer("sampler_seed", sampler_seed)
        _integer("micro_batch_size", micro_batch_size, 2)
        if sampler_seed >= 2**63 - len(FAMILIES) * 7919 or micro_batch_size % 2:
            raise ValueError("bounded seed and complete-pair microbatch required")
        if (not isinstance(protected_transcripts, (list, tuple, set, frozenset))
                or any(not _hex(value) for value in protected_transcripts)):
            raise ValueError("protected transcripts must be SHA256 strings")
        self.banks = _admit(banks)
        self.mode, self.micro_batch_size = mode, micro_batch_size
        self.protected = frozenset(protected_transcripts)
        initial = [_transcript(row) for buckets in self.banks.values() for rows in buckets.values() for row in rows]
        if len(set(initial)) != len(initial) or set(initial) & self.protected:
            raise ValueError("initial realizations must be unique and outside protected transcripts")
        self.reserved = frozenset(initial)
        self.generators = {family: torch.Generator().manual_seed(sampler_seed + sorted(FAMILIES).index(family) * 7919)
                           for family in self.banks}
        self.recipe = {"version": VERSION, "mode": mode, "sampler_seed": sampler_seed,
            "micro_batch_size": micro_batch_size, "max_attempts": MAX_ATTEMPTS,
            "banks": {family: {turns: fingerprint_rows(rows) for turns, rows in buckets.items()}
                      for family, buckets in self.banks.items()},
            "protected_count": len(self.protected), "protected_sha256": _hash(sorted(self.protected)),
            "initial_transcript_sha256": _hash(sorted(self.reserved)),
            "initial_policy": "occurrence zero uses original; refreshes exclude every reserved original transcript"}
        self.seen = set()
        self.state = {"recipe": copy.deepcopy(self.recipe),
            "family_microbatches": dict.fromkeys(self.banks, 0),
            "bucket_microbatches": {family: dict.fromkeys(buckets, 0) for family, buckets in self.banks.items()},
            "occurrences": {}, "latest": {}, "statistics": {},
            "exposures": {family: dict.fromkeys(COUNTS, 0) for family in self.banks},
            "canonical_stream_sha256": ZERO_DIGEST, "structural_stream_sha256": ZERO_DIGEST,
            "collisions": dict.fromkeys(("candidate_pairs", "rejected_candidates", "protected_pairs", "seen_pairs",
                "reserved_pairs", "intra_pair_pairs", "accepted_duplicate_transcripts"), 0),
            "timing": {"generation_validation_seconds": 0., "rejected_candidate_seconds": 0.}}
        for family, buckets in self.banks.items():
            self.state["occurrences"][family], self.state["latest"][family], self.state["statistics"][family] = {}, {}, {}
            for turns, rows in buckets.items():
                self.state["occurrences"][family][turns] = [0] * (len(rows) // 2)
                self.state["latest"][family][turns] = [None] * (len(rows) // 2)
                self.state["statistics"][family][turns] = {"target_counts_by_turn": [[0] * 4 for _ in range(turns)],
                    "opposite_pair_counts_by_turn": [0] * turns, "set_values": {}, "query_values": {}, "increments": {}}

    def _candidate(self, family, turns, index, occurrence, attempt):
        original = self.banks[family][turns][2 * index:2 * index + 2]
        if self.mode == "fixed" or occurrence == 0:
            pair = copy.deepcopy(original)
        else:
            names, values = realization_seeds(family, turns, index, occurrence, attempt)
            recipe = original[0]["recipe"]
            pair = generate_pair(family, recipe["seed"], split="train", turns=turns,
                naming_seed=names, value_seed=values, structure_split="train")
        validate_pair(pair)
        for actual, expected in zip(pair, original):
            if (any(actual[key] != expected[key] for key in
                    ("program_id", "structure_id", "structure_partition", "query_ancestries"))
                    or [turn["kind"] for turn in actual["turns"]] != [turn["kind"] for turn in expected["turns"]]):
                raise ValueError("realization changed admitted program or query structure")
        return pair

    def _consume_pair(self, family, turns, index):
        occurrence = self.state["occurrences"][family][turns][index]
        collisions = self.state["collisions"]
        for attempt in range(MAX_ATTEMPTS):
            tick = time.perf_counter()
            pair = self._candidate(family, turns, index, occurrence, attempt)
            digests = [_transcript(row) for row in pair]
            reasons = {"protected_pairs": bool(set(digests) & self.protected),
                "intra_pair_pairs": len(set(digests)) != 2,
                "seen_pairs": self.mode == "fresh" and bool(set(digests) & self.seen),
                "reserved_pairs": self.mode == "fresh" and occurrence > 0 and bool(set(digests) & self.reserved)}
            elapsed = time.perf_counter() - tick
            self.state["timing"]["generation_validation_seconds"] += elapsed
            collisions["candidate_pairs"] += 1
            if not any(reasons.values()):
                break
            collisions["rejected_candidates"] += 1
            self.state["timing"]["rejected_candidate_seconds"] += elapsed
            for name, rejected in reasons.items():
                collisions[name] += int(rejected)
            if self.mode == "fixed" or occurrence == 0:
                raise ValueError("reserved initial realization unexpectedly collides")
        else:
            raise ValueError("fresh realization retry budget exhausted")
        collisions["accepted_duplicate_transcripts"] += sum(digest in self.seen for digest in digests)
        self.seen.update(digests)
        self.state["occurrences"][family][turns][index] += 1
        pair_hash = fingerprint_rows(pair)
        self.state["latest"][family][turns][index] = {"occurrence": occurrence, "attempt": attempt,
                                                    "pair_sha256": pair_hash}
        self.state["canonical_stream_sha256"] = _chain(self.state["canonical_stream_sha256"], pair_hash)
        self.state["structural_stream_sha256"] = _chain(self.state["structural_stream_sha256"],
            _hash([family, turns, index, occurrence]))
        statistics = self.state["statistics"][family][turns]
        for turn_index, (first, second) in enumerate(zip(pair[0]["turns"], pair[1]["turns"])):
            statistics["opposite_pair_counts_by_turn"][turn_index] += int({first["target"], second["target"]} == {0, 1})
            for turn in (first, second):
                statistics["target_counts_by_turn"][turn_index][turn["target"]] += 1
                _, event = parse_sentence(turn["text"])
                field = {"set": "set_values", "query": "query_values", "advance": "increments"}.get(event["op"])
                if field:
                    value = event["amount"] if field == "increments" else event["value"]
                    value = str(value).lower()
                    statistics[field][value] = statistics[field].get(value, 0) + 1
        return pair

    def draw(self, family):
        if family not in self.banks:
            raise ValueError("unknown admitted stream family")
        generator = self.generators[family]
        buckets = sorted(self.banks[family])
        turns = buckets[int(torch.randint(len(buckets), (1,), generator=generator))]
        indices = torch.randint(len(self.banks[family][turns]) // 2,
            (self.micro_batch_size // 2,), generator=generator).tolist()
        rows = [row for index in indices for row in self._consume_pair(family, turns, index)]
        per_row = _row_counts(rows)
        counts = {key: sum(row[key] for row in per_row) for key in COUNTS}
        self.state["family_microbatches"][family] += 1
        self.state["bucket_microbatches"][family][turns] += 1
        for key, count in counts.items():
            self.state["exposures"][family][key] += count
        return rows, turns, indices, counts

    def snapshot(self):
        result = copy.deepcopy(self.state)
        result.update(seen_transcripts=sorted(self.seen), samplers={name: generator.get_state().clone()
            for name, generator in self.generators.items()})
        return result

    def restore(self, payload):
        expected = self.snapshot()
        if not isinstance(payload, Mapping) or set(payload) != set(expected) or payload["recipe"] != self.recipe:
            raise ValueError("realization snapshot fields or recipe differ")
        _check_finite_tree(payload, "realization snapshot")
        seen = payload["seen_transcripts"]
        if (not isinstance(seen, list) or any(not _hex(value) for value in seen)
                or seen != sorted(set(seen)) or set(seen) & self.protected):
            raise ValueError("invalid consumed transcript evidence")
        for digest in (payload["canonical_stream_sha256"], payload["structural_stream_sha256"]):
            if not _hex(digest):
                raise ValueError("invalid stream digest")
        for field in ("family_microbatches", "bucket_microbatches", "occurrences", "latest", "statistics", "exposures", "samplers"):
            if not isinstance(payload[field], Mapping) or set(payload[field]) != set(self.banks):
                raise ValueError("realization family fields differ")
        consumed_pairs = 0
        for family, buckets in self.banks.items():
            _integer("family microbatches", payload["family_microbatches"][family])
            for field in ("bucket_microbatches", "occurrences", "latest", "statistics"):
                if set(payload[field][family]) != set(buckets):
                    raise ValueError("realization bucket fields differ")
            for turns, rows in buckets.items():
                micro = payload["bucket_microbatches"][family][turns]
                _integer("bucket microbatches", micro)
                occurrences, latest = payload["occurrences"][family][turns], payload["latest"][family][turns]
                if not isinstance(occurrences, list) or len(occurrences) != len(rows) // 2 or len(latest) != len(occurrences):
                    raise ValueError("recipe occurrence axes differ")
                for count, last in zip(occurrences, latest):
                    _integer("recipe occurrence", count)
                    if count == 0:
                        if last is not None:
                            raise ValueError("unseen recipe has a latest realization")
                    elif (not isinstance(last, dict) or set(last) != {"occurrence", "attempt", "pair_sha256"}
                            or type(last["occurrence"]) is not int or last["occurrence"] != count - 1
                            or type(last["attempt"]) is not int or not 0 <= last["attempt"] < MAX_ATTEMPTS
                            or not _hex(last["pair_sha256"])):
                        raise ValueError("latest realization provenance differs")
                if sum(occurrences) != micro * self.micro_batch_size // 2:
                    raise ValueError("occurrences disagree with consumed microbatches")
                consumed_pairs += sum(occurrences)
                stats = payload["statistics"][family][turns]
                if set(stats) != set(expected["statistics"][family][turns]):
                    raise ValueError("realization statistics differ")
                targets, opposite = stats["target_counts_by_turn"], stats["opposite_pair_counts_by_turn"]
                if len(targets) != turns or len(opposite) != turns:
                    raise ValueError("per-turn statistics axes differ")
                for values, pairs in zip(targets, opposite):
                    if (len(values) != 4 or any(type(value) is not int or value < 0 for value in values)
                            or sum(values) != micro * self.micro_batch_size or type(pairs) is not int
                            or not 0 <= pairs <= micro * self.micro_batch_size // 2):
                        raise ValueError("per-turn exposure statistics differ")
                if opposite[-1] != micro * self.micro_batch_size // 2:
                    raise ValueError("final opposite-pair exposure differs")
                for field in ("set_values", "query_values", "increments"):
                    if not isinstance(stats[field], dict) or any(type(key) is not str or type(value) is not int
                            or value < 0 for key, value in stats[field].items()):
                        raise ValueError("invalid value/increment histogram")
            if sum(payload["bucket_microbatches"][family].values()) != payload["family_microbatches"][family]:
                raise ValueError("family and bucket microbatches differ")
            sampler = payload["samplers"][family]
            if not isinstance(sampler, torch.Tensor) or sampler.dtype != torch.uint8 or sampler.device.type != "cpu" or sampler.ndim != 1:
                raise ValueError("realization samplers require CPU byte state")
            try:
                torch.Generator().set_state(sampler)
            except RuntimeError as error:
                raise ValueError("invalid realization sampler") from error
        collisions = payload["collisions"]
        if set(collisions) != set(expected["collisions"]):
            raise ValueError("collision counter fields differ")
        for value in collisions.values():
            _integer("collision count", value)
        if (collisions["candidate_pairs"] - collisions["rejected_candidates"] != consumed_pairs
                or len(seen) + collisions["accepted_duplicate_transcripts"] != consumed_pairs * 2
                or self.mode == "fresh" and collisions["accepted_duplicate_transcripts"] != 0):
            raise ValueError("accepted/collision transcript counts differ")
        if set(payload["timing"]) != set(expected["timing"]) or any(type(value) not in (int, float) or value < 0
                for value in payload["timing"].values()):
            raise ValueError("invalid realization timing")
        self.state = copy.deepcopy({key: value for key, value in payload.items() if key not in ("seen_transcripts", "samplers")})
        self.seen = set(seen)
        for family, generator in self.generators.items():
            generator.set_state(payload["samplers"][family].clone())

    def latest_observed_banks(self):
        result, metadata = {}, {"selection": "most recently consumed occurrence per recipe, not an unbiased fresh-distribution sample",
                                "per_pair": [], "absent_recipes": [], "all_initial_realizations_encountered": True}
        for family, buckets in self.banks.items():
            for turns, originals in buckets.items():
                rows = []
                for index, last in enumerate(self.state["latest"][family][turns]):
                    identity = {"family": family, "turns": turns, "recipe_index": index}
                    if last is None:
                        metadata["absent_recipes"].append(identity)
                        metadata["all_initial_realizations_encountered"] = False
                        continue
                    pair = self._candidate(family, turns, index, last["occurrence"], last["attempt"])
                    if fingerprint_rows(pair) != last["pair_sha256"] or any(_transcript(row) not in self.seen for row in pair):
                        raise ValueError("latest consumed realization cannot be reconstructed")
                    rows.extend(pair)
                    metadata["per_pair"].append({**identity, **last})
                if rows:
                    result.setdefault(family, {})[turns] = rows
        return result, metadata


class RealizationTrainer(CompositionTrainer):
    def __init__(self, banks, *, mode, protected_transcripts, seed=2901, sampler_seed=3901,
                 device="cpu", micro_batch_size=32, learning_rate=.001, config=None, payload=None):
        super().__init__(banks, seed=seed, sampler_seed=sampler_seed, device=device,
            micro_batch_size=micro_batch_size, learning_rate=learning_rate,
            config=SequenceConfig(max_turns=12) if config is None else config)
        # Every microbatch is packed from its actual consumed realization;
        # inherited static GPU tensors and byte tables would be unused storage.
        self.encoded.clear()
        self.row_counts.clear()
        self.stream = RealizationStream(banks, mode=mode, protected_transcripts=protected_transcripts,
                                        sampler_seed=sampler_seed, micro_batch_size=micro_batch_size)
        self.generators = self.stream.generators
        self.recipe.update(model="bic-realization-sequence-v1", realization=copy.deepcopy(self.stream.recipe))
        self._failed = False
        if payload is not None:
            self._restore(payload)

    def _draw(self, family):
        rows, turns, indices, counts = self.stream.draw(family)
        batch = pack_composition_episodes(rows, device=next(self.model.parameters()).device, training=False,
            max_turns=self.model.config.max_turns, max_context_tokens=self.model.config.max_positions,
            max_input_bytes=self.model.config.max_input_bytes, max_reply_bytes=self.model.config.max_output_bytes)
        return batch, turns, indices, counts

    def step(self, families=None):
        if self._failed:
            raise RuntimeError("failed step requires checkpoint restoration")
        try:
            return super().step(tuple(FAMILIES) if families is None else families)
        except BaseException:
            self._failed = True
            raise

    def _synchronized(self):
        if self._failed:
            raise RuntimeError("cannot snapshot partially consumed failed optimizer step")
        for field in ("family_microbatches", "bucket_microbatches", "exposures"):
            if getattr(self, field) != self.stream.state[field]:
                raise ValueError("optimizer and realization stream counters differ")

    def snapshot(self):
        self._synchronized()
        return {**super().snapshot(), "realization": self.stream.snapshot()}

    def stream_evidence(self):
        self._synchronized()
        return stream_evidence(self.stream.snapshot())

    def latest_observed_banks(self):
        self._synchronized()
        return self.stream.latest_observed_banks()

    def _restore(self, payload):
        if not isinstance(payload, Mapping) or set(payload) != _BASE_FIELDS | {"realization"}:
            raise ValueError("invalid realization training checkpoint fields")
        self.stream.restore(payload["realization"])
        for family, state in payload["samplers"].items():
            if family not in self.generators or not torch.equal(state, self.generators[family].get_state()):
                raise ValueError("optimizer and realization sampler states differ")
        super()._restore({key: payload[key] for key in _BASE_FIELDS})
        self._failed = False
        self._synchronized()


def replay_evidence(banks, mode, protected_transcripts, steps, *, sampler_seed=3901, micro_batch_size=32):
    """Regenerate all consumed canonical samples once, capturing requested updates.

    This routine performs no neural forward or optimization. It follows the
    declared joint order, three family microbatches per update. Timing is absent
    from comparable evidence because it is not reproducible execution state.
    """
    if (not isinstance(steps, (list, tuple)) or not steps
            or any(type(step) is not int or step < 0 for step in steps) or len(set(steps)) != len(steps)):
        raise ValueError("unique nonnegative checkpoint steps required")
    if set(banks) != set(FAMILIES):
        raise ValueError("joint replay requires all three family banks")
    stream = RealizationStream(banks, mode=mode, protected_transcripts=protected_transcripts,
                              sampler_seed=sampler_seed, micro_batch_size=micro_batch_size)
    requested, result = set(steps), {}
    for step in range(max(steps) + 1):
        if step in requested:
            result[step] = stream_evidence(stream.snapshot())
        if step < max(steps):
            for family in FAMILIES:
                stream.draw(family)
    return result
