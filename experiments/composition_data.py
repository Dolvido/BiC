"""Variable-turn text-only packing with a separate canonical admission boundary.

The neural input contains only observed UTF-8 text and its causal boundaries.
Targets, replies, recipes, family names and oracle state remain supervision or
metadata. Evaluation and interactive packing never call a curriculum oracle.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from brain_in_computer.language import ByteCodec
from experiments.cognitive_curriculum import REPLIES
from experiments.sequence_data import _zero_observations


MAX_TURNS = 12


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def pack_observations(text_rows, device="cpu", *, blank_text=False,
                      max_turns=MAX_TURNS, max_input_bytes=128,
                      max_context_tokens=1560):
    """Pack uniform batches of 1..12 actual utterances, including reset controls.

    Byte/context overflow raises. Blank controls still validate original byte
    lengths. Outputs match SequenceStudent's token_ids, valid_mask, lengths and
    eos_positions arguments; right padding never creates extra turns.
    """
    for name, value in (("max_turns", max_turns), ("max_input_bytes", max_input_bytes),
                        ("max_context_tokens", max_context_tokens)):
        _positive_int(name, value)
    if max_turns > MAX_TURNS:
        raise ValueError("max_turns must not exceed twelve")
    if type(blank_text) is not bool:
        raise ValueError("blank_text must be boolean")
    if not _sequence(text_rows) or not text_rows or any(not _sequence(row) for row in text_rows):
        raise ValueError("observations require a nonempty batch of text sequences")
    turns = len(text_rows[0])
    if not 1 <= turns <= max_turns or any(len(row) != turns for row in text_rows):
        raise ValueError("batch rows require the same one-to-max_turns utterance count")
    codec = ByteCodec(max_bytes=max_input_bytes)
    encoded, positions = [], []
    for row in text_rows:
        tokens, ends = [], []
        for text in row:
            ids = codec.encode(text)
            tokens.extend((ByteCodec.BOS, ByteCodec.EOS) if blank_text else ids)
            ends.append(len(tokens) - 1)
        if len(tokens) > max_context_tokens:
            raise ValueError("observation context exceeds max_context_tokens; no truncation performed")
        encoded.append(tokens)
        positions.append(ends)
    lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long, device=device)
    width = max(map(len, encoded))
    token_ids = torch.full((len(encoded), width), ByteCodec.PAD, dtype=torch.long, device=device)
    for index, row in enumerate(encoded):
        token_ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
    return {"token_ids": token_ids, "valid_mask": valid, "lengths": lengths,
            "eos_positions": torch.tensor(positions, dtype=torch.long, device=device)}


def pack_composition_episodes(rows, device="cpu", *, training=False, pair_validator=None,
                              blank_text=False, max_turns=MAX_TURNS,
                              max_input_bytes=128, max_context_tokens=1560,
                              max_reply_bytes=32):
    """Pack uniform batches of 2..12 turns into inputs and supervision.

    Training requires adjacent canonical pairs from the train split and train
    structural partition. The default validator is imported lazily from the new
    curriculum; an explicit callable can provide another canonical admission
    implementation. Evaluation uses only structural checks and must be admitted
    by its caller. Decoder prefixes are never observation input.
    """
    if type(training) is not bool:
        raise ValueError("training must be boolean")
    _positive_int("max_reply_bytes", max_reply_bytes)
    if pair_validator is not None and not callable(pair_validator):
        raise ValueError("pair_validator must be callable")
    if not _sequence(rows) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("composition packing requires a nonempty episode sequence")
    if training:
        if len(rows) % 2:
            raise ValueError("training requires complete adjacent pairs")
        if pair_validator is None:
            from experiments.composition_curriculum import validate_pair
            pair_validator = validate_pair
        for row in rows:
            if row.get("split") != "train" or row.get("structure_partition") != "train":
                raise ValueError("training requires train admission and structural partitions")
        for offset in range(0, len(rows), 2):
            if pair_validator(rows[offset:offset + 2]) is False:
                raise ValueError("canonical pair admission failed")
    text_rows, actions, replies = [], [], []
    turn_count = None
    for row in rows:
        turns = row.get("turns")
        if not _sequence(turns) or not 2 <= len(turns) <= MAX_TURNS:
            raise ValueError("composition episodes require two to twelve turns")
        if turn_count is None:
            turn_count = len(turns)
        if len(turns) != turn_count:
            raise ValueError("episode batches require the same actual turn count")
        row_text, row_actions = [], []
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("each turn must be a mapping")
            target = turn.get("target")
            if type(target) is not int or not 0 <= target < len(REPLIES):
                raise ValueError("action targets must be integer labels in [0,3]")
            if turn.get("reply") != REPLIES[target]:
                raise ValueError("reply must agree with the declared canonical target")
            _zero_observations(turn.get("observations"))
            row_text.append(turn.get("text"))
            row_actions.append(target)
            replies.append(turn["reply"])
        text_rows.append(row_text)
        actions.append(row_actions)
    inputs = pack_observations(text_rows, device=device, blank_text=blank_text,
        max_turns=max_turns, max_input_bytes=max_input_bytes, max_context_tokens=max_context_tokens)
    reply_ids, _ = ByteCodec(max_bytes=max_reply_bytes).batch_encode(replies, device=device)
    shape = (len(rows), turn_count, -1)
    decoder, reply_targets = reply_ids[:, :-1].reshape(shape).contiguous(), reply_ids[:, 1:].reshape(shape).contiguous()
    observation_targets = torch.full_like(inputs["token_ids"], ByteCodec.PAD)
    observation_targets[:, :-1] = inputs["token_ids"][:, 1:]
    observation_targets.masked_fill_(inputs["token_ids"].eq(ByteCodec.EOS) | ~inputs["valid_mask"], ByteCodec.PAD)
    return {"inputs": inputs, "supervision": {
        "action_targets": torch.tensor(actions, dtype=torch.long, device=device),
        "reply_decoder_input_ids": decoder, "reply_targets": reply_targets,
        "reply_target_mask": reply_targets.ne(ByteCodec.PAD),
        "observation_next_byte_targets": observation_targets,
        "observation_next_byte_mask": observation_targets.ne(ByteCodec.PAD)}}
