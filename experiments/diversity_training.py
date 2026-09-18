"""Authenticated v3 curriculum admission for the frozen sequence objective.

The model, AdamW update, complete-pair sampling and losses reuse the preceding
sequence study. World/naming diversity is a property of caller-supplied banks;
neither provenance nor oracle metadata enters the neural policy. This trainer
does not introduce replay or change the matched curriculum comparison.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict
import math

import torch

from brain_in_computer.learning_student import _check_finite_tree, _cpu_copy, _integer
from experiments.diverse_curriculum import FAMILIES, VERSION, validate_pair
from experiments.sequence_data import pack_cognitive_episodes
from experiments.sequence_student import build_sequence_student
from experiments.sequence_training import SequenceTrainer
from experiments.train_cognitive import fingerprint_rows


class DiversityTrainer(SequenceTrainer):
    """Train complete authenticated pairs with independent family samplers.

    ``step(family)`` is inherited unchanged from SequenceTrainer: replacement
    pair sampling, the same query/ACK/reply/observation losses, AdamW defaults,
    and global gradient clipping at 1. Banks are copied before admission so a
    caller cannot mutate the encoded examples or their resume fingerprints.
    Saved activity contains no live computation graph; each update starts a
    new sequence forward. No development/audit partition is admitted here.
    """

    def __init__(self, banks, *, seed=2601, device="cpu", batch_size=64,
                 learning_rate=.001, config=None, payload=None):
        _integer("seed", seed)
        if seed >= 2**63:
            raise ValueError("seed must be < 2**63")
        if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
            raise ValueError("batch size must contain complete pairs")
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("learning rate must be finite and positive")
        if not isinstance(banks, Mapping) or not banks:
            raise ValueError("at least one training family is required")
        if any(type(family) is not str or family not in FAMILIES for family in banks):
            raise ValueError("unknown training family")
        admitted = copy.deepcopy(dict(banks))
        authenticated, content_by_identity = set(), {}
        for family, rows in admitted.items():
            if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
                    or not rows or len(rows) % 2):
                raise ValueError("training banks require complete adjacent pairs")
            for index in range(0, len(rows), 2):
                pair = rows[index:index + 2]
                identity = tuple(id(row) for row in pair)
                if identity not in content_by_identity:
                    try:
                        content_by_identity[identity] = fingerprint_rows(pair)
                    except (TypeError, ValueError) as error:
                        raise ValueError("training rows must be canonical JSON data") from error
                key = (family, content_by_identity[identity])
                if key in authenticated:
                    continue
                # Repeated logical slots are intentional in low-diversity
                # banks. Authenticate each full-content-identical pair once;
                # identity only caches hashes after the private deep copy.
                # validate_pair authenticates both complete rows before its
                # cross-row checks, avoiding duplicate oracle regeneration.
                validate_pair(pair)
                for row in pair:
                    if row["family"] != family:
                        raise ValueError("training row disagrees with its named family")
                    if any(row[name] != "train" for name in ("split", "world_partition", "name_split")):
                        raise ValueError("training requires train admission, world and naming partitions")
                authenticated.add(key)
        if len({len(rows) for rows in admitted.values()}) != 1:
            raise ValueError("matched training banks must have equal sizes")
        self.model = build_sequence_student(seed, device=device, config=config)
        if self.model.config.max_turns < 6:
            raise ValueError("canonical training requires six turns")
        self.banks, self.batch_size = admitted, batch_size
        # The old packer's v2 admission cannot authenticate v3 provenance.
        # Only after the strict v3 checks above do we use its structural path.
        self.encoded = {family: pack_cognitive_episodes(rows, device=device, training=False,
            max_input_bytes=self.model.config.max_input_bytes,
            max_context_tokens=self.model.config.max_positions,
            max_reply_bytes=self.model.config.max_output_bytes)
            for family, rows in admitted.items()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.generators = {family: torch.Generator().manual_seed(seed + index * 7919)
                           for index, family in enumerate(sorted(admitted))}
        options = {key: value for key, value in self.optimizer.state_dict()["param_groups"][0].items()
                   if key != "params"}
        self.recipe = {"model": "bic-diversity-sequence-v1", "curriculum_version": VERSION,
            "seed": seed, "config": asdict(self.model.config),
            "banks": {family: fingerprint_rows(rows) for family, rows in admitted.items()},
            "batch_size": batch_size, "learning_rate": learning_rate,
            "optimizer": {"name": "AdamW", "options": options},
            "sampler": "complete adjacent pairs with replacement; sorted family seed offset 7919",
            "objective": "sequence_objective: balanced query CE + .25 ACK CE + .1 reply + .1 observation",
            "gradient_clip": 1.}
        self.updates = 0
        self.family_updates = {family: 0 for family in admitted}
        if payload is not None:
            self._restore(payload)

    def _restore(self, payload):
        fields = {"recipe", "weights", "optimizer", "samplers", "updates", "family_updates"}
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("invalid diversity checkpoint fields")
        _check_finite_tree(payload, "diversity_checkpoint")
        if payload["recipe"] != self.recipe:
            raise ValueError("checkpoint training recipe differs")
        updates, family_updates = payload["updates"], payload["family_updates"]
        _integer("checkpoint updates", updates)
        if not isinstance(family_updates, Mapping) or set(family_updates) != set(self.generators):
            raise ValueError("checkpoint family update set differs")
        for count in family_updates.values():
            _integer("checkpoint family updates", count)
        if sum(family_updates.values()) != updates:
            raise ValueError("checkpoint family updates do not sum to total updates")
        samplers = payload["samplers"]
        if not isinstance(samplers, Mapping) or set(samplers) != set(self.generators):
            raise ValueError("checkpoint family sampler set differs")
        for value in samplers.values():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.uint8 or value.ndim != 1):
                raise ValueError("checkpoint samplers must be CPU byte states")
            try:
                torch.Generator().set_state(value)
            except RuntimeError as error:
                raise ValueError("invalid checkpoint sampler state") from error
        optimizer = payload["optimizer"]
        expected_groups = self.optimizer.state_dict()["param_groups"]
        if (not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"}
                or optimizer["param_groups"] != expected_groups
                or not isinstance(optimizer["state"], Mapping)):
            raise ValueError("checkpoint optimizer configuration differs")
        parameters = list(self.model.parameters())
        states = optimizer["state"]
        expected_ids = set(range(len(parameters))) if updates else set()
        if set(states) != expected_ids:
            raise ValueError("checkpoint optimizer state does not match update count")
        for index, state in states.items():
            if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid AdamW state fields")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or not step.is_floating_point() or float(step) != updates):
                raise ValueError("checkpoint optimizer step differs from update count")
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                        or value.dtype != parameters[index].dtype):
                    raise ValueError("checkpoint optimizer moment shape or dtype differs")
            if (state["exp_avg_sq"] < 0).any():
                raise ValueError("checkpoint optimizer second moment must be nonnegative")
        self.model.load_state_dict(payload["weights"], strict=True)
        self.optimizer.load_state_dict(_cpu_copy(optimizer))
        for family, generator in self.generators.items():
            generator.set_state(samplers[family].clone())
        self.updates = updates
        self.family_updates = dict(family_updates)

    def snapshot(self):
        payload = {"recipe": copy.deepcopy(self.recipe), "weights": _cpu_copy(self.model.state_dict()),
            "optimizer": _cpu_copy(self.optimizer.state_dict()),
            "samplers": {family: generator.get_state().clone() for family, generator in self.generators.items()},
            "updates": self.updates, "family_updates": dict(self.family_updates)}
        _check_finite_tree(payload, "diversity_checkpoint")
        return payload
