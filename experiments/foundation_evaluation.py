"""Authenticated foundation cells using the existing observation-only scorer.

Only admission changes. Causal input packing, BOS-only free replies, paired
metrics and blank/reset controls are inherited unchanged. Preparation may call
the canonical generator and independent oracles; model scoring never does.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import time

from experiments.composition_data import pack_composition_episodes, pack_observations
from experiments.composition_evaluation import PreparedBank
from experiments.sequence_student import SequenceConfig


class FoundationBank(PreparedBank):
    """One canonical family/depth/length cell, not a capability certificate."""

    def __init__(self, rows, *, role, config=None):
        from experiments.foundation_curriculum import VERSION, final_depth, validate_pair

        started = time.monotonic()
        if role not in ("train_fit", "dev", "audit"):
            raise ValueError("explicit foundation evaluation role required")
        if config is not None and not isinstance(config, SequenceConfig):
            raise ValueError("foundation evaluation requires a SequenceConfig")
        self._canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        admitted = json.loads(self._canonical)
        if not isinstance(admitted, list) or not admitted or len(admitted) % 2:
            raise ValueError("complete foundation evaluation pairs required")
        expected_split = "train" if role == "train_fit" else role
        groups, cells = set(), set()
        for offset in range(0, len(admitted), 2):
            pair = admitted[offset:offset + 2]
            validate_pair(pair)
            group = pair[0]["counterfactual_group"]
            if group in groups or any(row["split"] != expected_split for row in pair):
                raise ValueError("foundation evaluation role or duplicate group differs")
            groups.add(group)
            cells.update((row["family"], final_depth(row), len(row["turns"])) for row in pair)
        if len(cells) != 1:
            raise ValueError("foundation bank must contain one family, depth and turn count")
        family, depth, turns = next(iter(cells))
        self._cell = {"family": family, "depth": depth, "turns": turns,
                      "primitive_shared": depth <= 1}
        self._config = config or SequenceConfig(max_turns=12)
        c = self._config
        options = {"max_input_bytes": c.max_input_bytes, "max_context_tokens": c.max_positions,
                   "max_turns": c.max_turns}
        # Explicit canonical foundation admission above precedes structural
        # packing; never route these rows through the composition-v1 validator.
        packed = pack_composition_episodes(admitted, training=False,
                                           max_reply_bytes=c.max_output_bytes, **options)
        self._targets = packed["supervision"]["action_targets"].clone()
        self._reply_targets = packed["supervision"]["reply_targets"].clone()
        texts = [[turn["text"] for turn in row["turns"]] for row in admitted]
        self._inputs = {"normal": packed["inputs"],
            "blank": pack_observations(texts, blank_text=True, **options),
            "reset": pack_observations([[text] for row in texts for text in row], **options)}
        self._identity = {"sha256": hashlib.sha256(self._canonical).hexdigest(), "version": VERSION,
            "episodes": len(admitted), "turns": turns, "role": role, "config": asdict(c)}
        self.preparation_seconds = time.monotonic()-started

    @property
    def cell(self):
        return copy.deepcopy(self._cell)
