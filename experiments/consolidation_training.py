"""General complete-pair replay through the unchanged sequence learner.

One support batch and optionally one old-family batch share an optimizer step.
Replay is extra exposure and compute, recorded separately; it does not replace
new lessons, inject targets into observations, or retain activity across steps.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict

import torch

from brain_in_computer.language import ByteCodec
from brain_in_computer.learning_student import _check_finite_tree, _finite_number, _integer
from experiments.diverse_curriculum import FAMILIES, VERSION, validate_pair
from experiments.diversity_training import DiversityTrainer
from experiments.sequence_data import pack_cognitive_episodes
from experiments.sequence_student import build_sequence_student
from experiments.sequence_training import sequence_objective
from experiments.train_cognitive import fingerprint_rows


REPLAY_SEED_OFFSET = 1_000_003
_BASE_FIELDS = {"recipe", "weights", "optimizer", "samplers", "updates", "family_updates"}
_COUNTS = ("episodes", "observation_tokens", "observation_bytes", "reply_target_tokens", "reply_target_bytes")


def _admit(banks):
    if not isinstance(banks, Mapping) or not banks:
        raise ValueError("at least one named training bank is required")
    if any(type(name) is not str or name not in FAMILIES for name in banks):
        raise ValueError("unknown training family")
    result = copy.deepcopy(dict(banks))
    hashes, authenticated = {}, set()
    for family, rows in result.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows or len(rows) % 2:
            raise ValueError("training banks require complete adjacent pairs")
        for offset in range(0, len(rows), 2):
            pair = rows[offset:offset + 2]
            identity = tuple(id(row) for row in pair)
            if identity not in hashes:
                try:
                    hashes[identity] = fingerprint_rows(pair)
                except (TypeError, ValueError) as error:
                    raise ValueError("training rows must be canonical JSON data") from error
            key = family, hashes[identity]
            if key in authenticated:
                continue
            validate_pair(pair)  # Authenticates every field of both v3 rows.
            for row in pair:
                if row["family"] != family or any(row[name] != "train"
                        for name in ("split", "world_partition", "name_split")):
                    raise ValueError("training requires matching families and all train partitions")
            authenticated.add(key)
    return result


def _draw(encoded, rows, generator, batch_size, device):
    pairs = torch.randint(len(rows) // 2, (batch_size // 2,), generator=generator)
    indices = (pairs[:, None] * 2 + torch.arange(2)[None]).flatten().to(device)
    batch = {group: {key: value.index_select(0, indices) for key, value in fields.items()}
             for group, fields in encoded.items()}
    width = int(batch["inputs"]["lengths"].max())
    for key in ("token_ids", "valid_mask"):
        batch["inputs"][key] = batch["inputs"][key][:, :width]
    for key in ("observation_next_byte_targets", "observation_next_byte_mask"):
        batch["supervision"][key] = batch["supervision"][key][:, :width]
    return batch


def _counts(batch):
    inputs, targets = batch["inputs"], batch["supervision"]
    return {"episodes": inputs["token_ids"].shape[0],
        "observation_tokens": int(inputs["valid_mask"].sum()),
        "observation_bytes": int(inputs["token_ids"].ge(ByteCodec.BYTE_OFFSET).sum()),
        "reply_target_tokens": int(targets["reply_target_mask"].sum()),
        "reply_target_bytes": int(targets["reply_targets"].ge(ByteCodec.BYTE_OFFSET).sum())}


class ConsolidationTrainer(DiversityTrainer):
    """Support plus weighted old-family rehearsal in one AdamW update.

    Support draws use the frozen seed + sorted-family-index*7919 scheme.
    Replay uses separate seed+1000003+index*7919 generators and sorted-family
    round robin. ``replay_banks=None`` disables replay entirely. If banks are
    supplied with weight zero, their forward computation/exposure still counts.
    Caller-supplied initial weights may be loaded before the first update;
    checkpoint provenance is the runner's responsibility. No optimizer or
    recurrent/teacher state is copied from a previous training phase.
    """

    def __init__(self, support_banks, *, replay_banks=None, seed=3701, device="cpu",
                 batch_size=64, replay_batch_size=32, replay_weight=.5,
                 learning_rate=.001, config=None, payload=None):
        _integer("seed", seed)
        if seed >= 2**63:
            raise ValueError("seed must be < 2**63")
        for name, value in (("batch_size", batch_size), ("replay_batch_size", replay_batch_size)):
            _integer(name, value, 2)
            if value % 2:
                raise ValueError("batch sizes must preserve complete pairs")
        _finite_number("learning_rate", learning_rate)
        _finite_number("replay_weight", replay_weight)
        if learning_rate <= 0 or replay_weight < 0:
            raise ValueError("learning rate must be positive and replay weight nonnegative")
        self.banks = _admit(support_banks)
        self.replay_banks = {} if replay_banks is None else _admit(replay_banks)
        if set(self.banks) & set(self.replay_banks):
            raise ValueError("support and replay family sets must be disjoint")
        self.model = build_sequence_student(seed, device=device, config=config)
        if self.model.config.max_turns < 6:
            raise ValueError("canonical training requires six turns")
        self.batch_size, self.replay_batch_size, self.replay_weight = batch_size, replay_batch_size, replay_weight
        def pack(banks):
            return {family: pack_cognitive_episodes(rows, device=device, training=False,
                max_input_bytes=self.model.config.max_input_bytes,
                max_context_tokens=self.model.config.max_positions,
                max_reply_bytes=self.model.config.max_output_bytes) for family, rows in banks.items()}
        self.encoded, self.replay_encoded = pack(self.banks), pack(self.replay_banks)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.generators = {family: torch.Generator().manual_seed(seed + index * 7919)
                           for index, family in enumerate(sorted(self.banks))}
        self.replay_generators = {family: torch.Generator().manual_seed(seed + REPLAY_SEED_OFFSET + index * 7919)
                                 for index, family in enumerate(sorted(self.replay_banks))}
        options = {key: value for key, value in self.optimizer.state_dict()["param_groups"][0].items() if key != "params"}
        self.recipe = {"model": "bic-sequence-consolidation-v1", "curriculum_version": VERSION,
            "seed": seed, "config": asdict(self.model.config),
            "banks": {name: fingerprint_rows(rows) for name, rows in self.banks.items()},
            "replay_banks": {name: fingerprint_rows(rows) for name, rows in self.replay_banks.items()},
            "batch_size": batch_size, "replay_batch_size": replay_batch_size, "replay_weight": replay_weight,
            "learning_rate": learning_rate, "optimizer": {"name": "AdamW", "options": options},
            "sampler": "complete adjacent pairs with replacement; sorted family seed offset 7919",
            "replay_sampler": {"seed_offset": REPLAY_SEED_OFFSET, "family_seed_stride": 7919,
                               "family_order": sorted(self.replay_banks)},
            "objective": "unchanged sequence_objective(support) + replay_weight * unchanged sequence_objective(replay)",
            "gradient_clip": 1.}
        self.updates = self.replay_updates = 0
        self.family_updates = {family: 0 for family in self.banks}
        self.replay_family_updates = {family: 0 for family in self.replay_banks}
        self.exposures = {f"{source}_{name}": 0 for source in ("support", "replay") for name in _COUNTS}
        if payload is not None:
            self._restore(payload)

    def step(self, family):
        if family not in self.banks:
            raise ValueError("family not admitted into the support phase")
        device = next(self.model.parameters()).device
        support = _draw(self.encoded[family], self.banks[family], self.generators[family], self.batch_size, device)
        replay_family = None
        if self.replay_banks:
            names = sorted(self.replay_banks)
            replay_family = names[self.replay_updates % len(names)]
            replay = _draw(self.replay_encoded[replay_family], self.replay_banks[replay_family],
                           self.replay_generators[replay_family], self.replay_batch_size, device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        def losses(batch):
            output = self.model(**batch["inputs"], decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
            return sequence_objective(output, batch)
        new = losses(support)
        old = losses(replay) if replay_family is not None else None
        loss = new["loss"] if old is None else new["loss"] + self.replay_weight * old["loss"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite consolidation objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.family_updates[family] += 1
        counts = {f"support_{name}": value for name, value in _counts(support).items()}
        counts.update({f"replay_{name}": value for name, value in
                       (_counts(replay) if old is not None else dict.fromkeys(_COUNTS, 0)).items()})
        if old is not None:
            self.replay_updates += 1
            self.replay_family_updates[replay_family] += 1
        for name, count in counts.items():
            self.exposures[name] += count
        return {"family": family, "replay_family": replay_family, "loss": float(loss.detach()),
            "support": {key: float(value.detach()) for key, value in new.items()},
            "replay": {key: float(value.detach()) for key, value in old.items()} if old is not None else None,
            "exposures": counts}

    def snapshot(self):
        return {**super().snapshot(), "replay_samplers": {name: generator.get_state().clone()
                    for name, generator in self.replay_generators.items()},
            "replay_updates": self.replay_updates, "replay_family_updates": dict(self.replay_family_updates),
            "exposures": dict(self.exposures)}

    def _restore(self, payload):
        extra = {"replay_samplers", "replay_updates", "replay_family_updates", "exposures"}
        if not isinstance(payload, Mapping) or set(payload) != _BASE_FIELDS | extra:
            raise ValueError("invalid consolidation checkpoint fields")
        _check_finite_tree(payload, "consolidation checkpoint")
        updates, replay_updates = payload["updates"], payload["replay_updates"]
        _integer("updates", updates)
        _integer("replay_updates", replay_updates)
        if replay_updates != (updates if self.replay_banks else 0):
            raise ValueError("replay update count differs")
        names = sorted(self.replay_banks)
        expected = {name: replay_updates // len(names) + int(index < replay_updates % len(names))
                    for index, name in enumerate(names)}
        if not isinstance(payload["replay_family_updates"], Mapping) or payload["replay_family_updates"] != expected:
            raise ValueError("replay round-robin counters differ")
        for count in payload["replay_family_updates"].values():
            _integer("replay family update count", count)
        states = payload["replay_samplers"]
        if not isinstance(states, Mapping) or set(states) != set(names):
            raise ValueError("replay sampler families differ")
        for value in states.values():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.uint8 or value.ndim != 1):
                raise ValueError("replay sampler must be a CPU byte state")
            try:
                torch.Generator().set_state(value)
            except RuntimeError as error:
                raise ValueError("invalid replay sampler state") from error
        exposures = payload["exposures"]
        if not isinstance(exposures, Mapping) or set(exposures) != set(self.exposures):
            raise ValueError("exposure counter fields differ")
        for value in exposures.values():
            _integer("exposure count", value)
        for source, episodes in (("support", updates * self.batch_size), ("replay", replay_updates * self.replay_batch_size)):
            if (exposures[f"{source}_episodes"] != episodes
                    or exposures[f"{source}_observation_tokens"] - exposures[f"{source}_observation_bytes"] != episodes * 12
                    or exposures[f"{source}_reply_target_tokens"] - exposures[f"{source}_reply_target_bytes"] != episodes * 6
                    or exposures[f"{source}_observation_bytes"] > episodes * 6 * self.model.config.max_input_bytes
                    or exposures[f"{source}_reply_target_bytes"] > episodes * 6 * self.model.config.max_output_bytes):
                raise ValueError("exposure counters disagree with episodes or byte boundaries")
        # Reuse the existing strict weights/optimizer/support-sampler validator.
        super()._restore({name: payload[name] for name in _BASE_FIELDS})
        for name, generator in self.replay_generators.items():
            generator.set_state(states[name].clone())
        self.replay_updates = replay_updates
        self.replay_family_updates = dict(expected)
        self.exposures = dict(exposures)
