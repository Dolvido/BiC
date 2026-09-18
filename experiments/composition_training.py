"""Complete-pair, variable-turn learning with schedule-independent domain draws.

Every update averages three independent microbatch objectives, clips once and
updates AdamW once. The caller controls joint, sequential and rehearsal order;
the per-family bucket/pair streams depend only on the family and sampler seed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict

import torch

from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _finite_number, _integer
from experiments.composition_curriculum import FAMILIES, VERSION, validate_pair
from experiments.composition_data import pack_composition_episodes
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.sequence_training import sequence_objective
from experiments.train_cognitive import fingerprint_rows


COUNTS = ("episodes", "turns", "observation_tokens", "observation_bytes",
          "reply_target_tokens", "reply_target_bytes")
MICROBATCHES_PER_UPDATE = 3
FAMILY_SEED_STRIDE = 7919


def _admit(banks):
    if not isinstance(banks, Mapping) or not banks:
        raise ValueError("at least one named family bank is required")
    if any(type(name) is not str or name not in FAMILIES for name in banks):
        raise ValueError("unknown composition family")
    admitted = copy.deepcopy(dict(banks))
    authenticated = set()
    for family, buckets in admitted.items():
        if not isinstance(buckets, Mapping) or not buckets:
            raise ValueError("each family requires named turn buckets")
        for turns, rows in buckets.items():
            if type(turns) is not int or not 2 <= turns <= 12:
                raise ValueError("turn bucket keys must be integers from two to twelve")
            if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
                    or not rows or len(rows) % 2):
                raise ValueError("training banks require complete adjacent pairs")
            for offset in range(0, len(rows), 2):
                pair = rows[offset:offset + 2]
                try:
                    key = (family, turns, fingerprint_rows(pair))
                except (TypeError, ValueError) as error:
                    raise ValueError("training rows must be canonical JSON data") from error
                if key in authenticated:
                    continue
                validate_pair(pair)
                for row in pair:
                    if (row.get("family") != family or len(row.get("turns", [])) != turns
                            or row.get("split") != "train" or row.get("structure_partition") != "train"):
                        raise ValueError("training requires matching family/turn bucket and all train partitions")
                authenticated.add(key)
    return admitted


def _row_counts(rows):
    result = []
    for row in rows:
        turns = len(row["turns"])
        observed = sum(len(turn["text"].encode("utf-8")) for turn in row["turns"])
        replied = sum(len(turn["reply"].encode("utf-8")) for turn in row["turns"])
        result.append(dict(zip(COUNTS, (1, turns, observed + 2 * turns, observed,
                                       replied + turns, replied))))
    return result


class CompositionTrainer:
    """Train three equally weighted microbatches with fresh observation history.

    ``banks={family: {actual_turn_count: complete_pair_rows}}`` may contain a
    subset of the canonical families. Sampler offsets always use the sorted
    *global* family set, including absent families, so a fresh held-family
    control receives the same ordered samples as joint/sequential schedules.
    A family generator chooses a sorted turn bucket uniformly, then complete
    pairs with replacement. Each forward contains only that microbatch.
    """

    def __init__(self, banks, *, seed=2801, sampler_seed=3801, device="cpu",
                 micro_batch_size=32, learning_rate=.001, config=None, payload=None):
        for name, value in (("seed", seed), ("sampler_seed", sampler_seed)):
            _integer(name, value)
            if value >= 2**63 - len(FAMILIES) * FAMILY_SEED_STRIDE:
                raise ValueError("seed is too large for family offsets")
        _integer("micro_batch_size", micro_batch_size, 2)
        if micro_batch_size % 2:
            raise ValueError("microbatches must preserve complete pairs")
        _finite_number("learning_rate", learning_rate)
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        self.banks = _admit(banks)
        self.model = build_sequence_student(seed, device=device,
            config=SequenceConfig(max_turns=12) if config is None else config)
        self.micro_batch_size = micro_batch_size
        self.encoded, self.row_counts = {}, {}
        for family, buckets in self.banks.items():
            self.encoded[family], self.row_counts[family] = {}, {}
            for turns, rows in buckets.items():
                self.encoded[family][turns] = pack_composition_episodes(rows, device=device,
                    training=False, max_turns=self.model.config.max_turns,
                    max_input_bytes=self.model.config.max_input_bytes,
                    max_context_tokens=self.model.config.max_positions,
                    max_reply_bytes=self.model.config.max_output_bytes)
                self.row_counts[family][turns] = _row_counts(rows)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        family_order = sorted(FAMILIES)
        self.generators = {family: torch.Generator().manual_seed(
            sampler_seed + family_order.index(family) * FAMILY_SEED_STRIDE) for family in self.banks}
        options = {key: value for key, value in self.optimizer.state_dict()["param_groups"][0].items()
                   if key != "params"}
        self.recipe = {"model": "bic-composition-sequence-v1", "curriculum_version": VERSION,
            "seed": seed, "sampler_seed": sampler_seed, "config": asdict(self.model.config),
            "banks": {family: {turns: fingerprint_rows(rows) for turns, rows in buckets.items()}
                      for family, buckets in self.banks.items()},
            "micro_batch_size": micro_batch_size, "microbatches_per_update": MICROBATCHES_PER_UPDATE,
            "learning_rate": learning_rate, "optimizer": {"name": "AdamW", "options": options},
            "sampler": {"global_family_order": family_order, "family_seed_stride": FAMILY_SEED_STRIDE,
                "turn_bucket": "uniform over sorted available buckets; one draw per microbatch",
                "pairs": "adjacent complete pairs sampled with replacement"},
            "objective": "mean of three unchanged sequence_objective microbatch losses",
            "gradient_clip": 1.}
        self.updates = 0
        self.family_microbatches = {family: 0 for family in self.banks}
        self.bucket_microbatches = {family: dict.fromkeys(buckets, 0) for family, buckets in self.banks.items()}
        self.exposures = {family: dict.fromkeys(COUNTS, 0) for family in self.banks}
        if payload is not None:
            self._restore(payload)

    def _draw(self, family):
        generator = self.generators[family]
        buckets = sorted(self.banks[family])
        turns = buckets[int(torch.randint(len(buckets), (1,), generator=generator))]
        pairs = torch.randint(len(self.banks[family][turns]) // 2,
            (self.micro_batch_size // 2,), generator=generator)
        cpu_indices = (pairs[:, None] * 2 + torch.arange(2)[None]).flatten().tolist()
        counts = {name: sum(self.row_counts[family][turns][index][name] for index in cpu_indices)
                  for name in COUNTS}
        device = next(self.model.parameters()).device
        indices = torch.tensor(cpu_indices, dtype=torch.long, device=device)
        batch = {group: {name: value.index_select(0, indices) for name, value in fields.items()}
                 for group, fields in self.encoded[family][turns].items()}
        width = int(batch["inputs"]["lengths"].max())
        for name in ("token_ids", "valid_mask"):
            batch["inputs"][name] = batch["inputs"][name][:, :width]
        for name in ("observation_next_byte_targets", "observation_next_byte_mask"):
            batch["supervision"][name] = batch["supervision"][name][:, :width]
        return batch, turns, pairs.tolist(), counts

    def step(self, families):
        if (not isinstance(families, (tuple, list)) or len(families) != MICROBATCHES_PER_UPDATE
                or any(type(family) is not str or family not in self.banks for family in families)):
            raise ValueError("step requires exactly three admitted family names")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        reports = []
        for family in families:
            batch, turns, pairs, counts = self._draw(family)
            output = self.model(**batch["inputs"],
                decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
            losses = sequence_objective(output, batch)
            if not torch.isfinite(losses["loss"]):
                raise ValueError("nonfinite composition objective")
            (losses["loss"] / MICROBATCHES_PER_UPDATE).backward()
            reports.append({"family": family, "turns": turns, "pair_indices": pairs,
                **{name: float(value.detach()) for name, value in losses.items()}, "exposures": counts})
            # Release this graph before the next microbatch; gradients alone
            # accumulate until the single clip/update below.
            del output, losses, batch
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        for report in reports:
            family, turns = report["family"], report["turns"]
            self.family_microbatches[family] += 1
            self.bucket_microbatches[family][turns] += 1
            for name, count in report["exposures"].items():
                self.exposures[family][name] += count
        means = {name: sum(report[name] for report in reports) / MICROBATCHES_PER_UPDATE
                 for name in ("loss", "action_loss", "reply_loss", "observation_language_loss")}
        return {**means, "microbatches": reports}

    def snapshot(self):
        result = {"recipe": copy.deepcopy(self.recipe), "weights": _cpu_copy(self.model.state_dict()),
            "optimizer": _cpu_copy(self.optimizer.state_dict()), "updates": self.updates,
            "samplers": {family: generator.get_state().clone() for family, generator in self.generators.items()},
            "family_microbatches": copy.deepcopy(self.family_microbatches),
            "bucket_microbatches": copy.deepcopy(self.bucket_microbatches),
            "exposures": copy.deepcopy(self.exposures)}
        _check_finite_tree(result, "composition checkpoint")
        return result

    def _restore(self, payload):
        fields = {"recipe", "weights", "optimizer", "updates", "samplers",
                  "family_microbatches", "bucket_microbatches", "exposures"}
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("invalid composition checkpoint fields")
        _check_finite_tree(payload, "composition checkpoint")
        if payload["recipe"] != self.recipe:
            raise ValueError("checkpoint training recipe differs")
        updates = payload["updates"]
        _integer("checkpoint updates", updates)
        for name in ("samplers", "family_microbatches", "bucket_microbatches", "exposures"):
            if not isinstance(payload[name], Mapping) or set(payload[name]) != set(self.banks):
                raise ValueError("checkpoint family set differs")
        for family, buckets in self.banks.items():
            microbatches = payload["family_microbatches"][family]
            _integer("family microbatch count", microbatches)
            bucket_counts = payload["bucket_microbatches"][family]
            if not isinstance(bucket_counts, Mapping) or set(bucket_counts) != set(buckets):
                raise ValueError("checkpoint turn buckets differ")
            for value in bucket_counts.values():
                _integer("bucket microbatch count", value)
            if sum(bucket_counts.values()) != microbatches:
                raise ValueError("bucket counts disagree with family microbatches")
            counts = payload["exposures"][family]
            if not isinstance(counts, Mapping) or set(counts) != set(COUNTS):
                raise ValueError("checkpoint exposure fields differ")
            for value in counts.values():
                _integer("exposure count", value)
            turns = sum(turns * count for turns, count in bucket_counts.items()) * self.micro_batch_size
            if (counts["episodes"] != microbatches * self.micro_batch_size or counts["turns"] != turns
                    or counts["observation_tokens"] - counts["observation_bytes"] != 2 * turns
                    or counts["reply_target_tokens"] - counts["reply_target_bytes"] != turns
                    or counts["observation_bytes"] > turns * self.model.config.max_input_bytes
                    or counts["reply_target_bytes"] > turns * self.model.config.max_output_bytes):
                raise ValueError("exposure counts disagree with variable-turn boundaries")
            value = payload["samplers"][family]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.uint8 or value.ndim != 1):
                raise ValueError("checkpoint samplers must be CPU byte states")
            try:
                torch.Generator().set_state(value)
            except RuntimeError as error:
                raise ValueError("invalid checkpoint sampler state") from error
        if sum(payload["family_microbatches"].values()) != updates * MICROBATCHES_PER_UPDATE:
            raise ValueError("family microbatches disagree with optimizer updates")
        optimizer = payload["optimizer"]
        if (not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"}
                or optimizer["param_groups"] != self.optimizer.state_dict()["param_groups"]
                or not isinstance(optimizer["state"], Mapping)):
            raise ValueError("checkpoint optimizer configuration differs")
        parameters, states = list(self.model.parameters()), optimizer["state"]
        if set(states) != (set(range(len(parameters))) if updates else set()):
            raise ValueError("optimizer state disagrees with update count")
        for index, state in states.items():
            if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid AdamW state fields")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or not step.is_floating_point() or float(step) != updates):
                raise ValueError("optimizer step disagrees with update count")
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                        or value.dtype != parameters[index].dtype):
                    raise ValueError("optimizer moment shape or dtype differs")
            if bool(state["exp_avg_sq"].lt(0).any()):
                raise ValueError("optimizer second moment must be nonnegative")
        self.model.load_state_dict(payload["weights"], strict=True)
        self.optimizer.load_state_dict(_cpu_copy(optimizer))
        for family, generator in self.generators.items():
            generator.set_state(payload["samplers"][family].clone())
        self.updates = updates
        self.family_microbatches = copy.deepcopy(payload["family_microbatches"])
        self.bucket_microbatches = copy.deepcopy(payload["bucket_microbatches"])
        self.exposures = copy.deepcopy(payload["exposures"])
