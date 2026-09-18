"""Observation-only packing for a general sequence-learner comparison.

Each observed utterance contributes BOS, its UTF-8 bytes, then EOS. Actual
inputs and supervision are separate dictionaries: no answer, reply, family or
generator metadata enters the observation stream. Next-byte supervision stops
at every utterance EOS; EOS-to-next-BOS transitions and right padding are masked,
matching the existing learner's within-utterance observation objective.

This module neither trains nor calls a model. Canonical training admission is
explicit through ``training=True``. Evaluation packing is structural only, so
it does not call the curriculum parser/oracle. Evaluators must authenticate their
frozen banks independently. Interactive text uses ``pack_observations`` directly.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from brain_in_computer.language import ByteCodec
from experiments.cognitive_curriculum import REPLIES, validate_cognitive


MAX_TURNS = 6
_WIDTHS = {"visual": 32, "auditory": 4, "body": 4, "feedback": 2}


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def pack_observations(text_rows: Sequence[Sequence[str]], device="cpu", *,
                      blank_text: bool = False, max_input_bytes: int = 128,
                      max_context_tokens: int = 780) -> dict[str, torch.Tensor]:
    """Pack raw observed text without metadata, answer labels or an oracle.

    Every batch row must contain the same number of utterances, from one to six.
    Text lengths may differ. ``token_ids``/``valid_mask`` have shape [B,L],
    ``lengths`` is [B], and ``eos_positions`` is [B,T]. EOS positions are valid
    zero-based token indices; there are no sentinel or padded utterance indices.
    ``blank_text`` replaces utterance content with empty strings, preserving the
    number of turns and their boundaries. Original utterances must still satisfy
    the declared byte limit. Context overflow raises instead of truncating.
    """
    _positive_int("max_input_bytes", max_input_bytes)
    _positive_int("max_context_tokens", max_context_tokens)
    if type(blank_text) is not bool:
        raise ValueError("blank_text must be boolean")
    if not _sequence(text_rows) or not text_rows:
        raise ValueError("observations require a nonempty batch of text sequences")
    if any(not _sequence(row) for row in text_rows):
        raise ValueError("each observation row must be a sequence of utterances")
    turns = len(text_rows[0])
    if not 1 <= turns <= MAX_TURNS or any(len(row) != turns for row in text_rows):
        raise ValueError("batch rows require the same one-to-six utterance count")
    codec = ByteCodec(max_bytes=max_input_bytes)
    encoded, positions = [], []
    for row in text_rows:
        tokens, ends = [], []
        for text in row:
            # Validate the original text before applying an evaluation control.
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


def _zero_observations(observations):
    """Reject silently discarded sensor information in this text-only study."""
    if not isinstance(observations, Mapping) or set(observations) != {*_WIDTHS, "tokens"}:
        raise ValueError("cognitive observations require exactly the declared channels")
    tokens = observations["tokens"]
    if not isinstance(tokens, list) or len(tokens) != 1 or type(tokens[0]) is not int or tokens[0] != 0:
        raise ValueError("cognitive observation tokens must be integer zero")
    for name, width in _WIDTHS.items():
        rows = observations[name]
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) != width:
            raise ValueError("cognitive observation channels must have the fixed one-row shape")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value != 0 for value in rows[0]):
            raise ValueError("sequence comparison accepts only finite zero numeric observations")


def pack_cognitive_episodes(rows: Sequence[Mapping], device="cpu", *,
                             training: bool = False, blank_text: bool = False,
                             max_input_bytes: int = 128, max_context_tokens: int = 780,
                             max_reply_bytes: int = 32) -> dict:
    """Separate observation inputs from six-turn action/byte supervision.

    Set ``training=True`` when admitting training data: every row must have the
    train split and exact canonical provenance. Other splits are accepted for
    evaluation with ``training=False`` and no oracle call. Evaluation rows must
    already have been independently verified by the experiment's bank boundary.

    ``inputs`` is the dictionary from :func:`pack_observations`, with T=6.
    ``supervision`` contains action_targets [B,6], reply_decoder_input_ids and
    reply_targets [B,6,R], reply_target_mask [B,6,R], and observation_next_byte_
    targets/mask [B,L]. R is the batch's padded reply length after one-token
    shifting. Reply prefixes are teacher-forcing supervision, not encoder input.
    No caller-owned dictionaries, lists or tensors are modified.
    """
    if type(training) is not bool:
        raise ValueError("training must be boolean")
    _positive_int("max_reply_bytes", max_reply_bytes)
    if not _sequence(rows) or not rows:
        raise ValueError("cognitive packing requires a nonempty episode sequence")
    text_rows, actions, replies = [], [], []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each cognitive episode must be a mapping")
        if training:
            if row.get("split") != "train":
                raise ValueError("training admission accepts only canonical train episodes")
            validate_cognitive(row)
        turns = row.get("turns")
        if not isinstance(turns, list) or len(turns) != MAX_TURNS:
            raise ValueError("cognitive packing requires exactly six turns")
        row_text, row_actions = [], []
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("each cognitive turn must be a mapping")
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
        max_input_bytes=max_input_bytes, max_context_tokens=max_context_tokens)
    reply_ids, _ = ByteCodec(max_bytes=max_reply_bytes).batch_encode(replies, device=device)
    batch, turns = len(rows), MAX_TURNS
    reply_decoder = reply_ids[:, :-1].reshape(batch, turns, -1).contiguous()
    reply_targets = reply_ids[:, 1:].reshape(batch, turns, -1).contiguous()
    observation_targets = torch.full_like(inputs["token_ids"], ByteCodec.PAD)
    observation_targets[:, :-1] = inputs["token_ids"][:, 1:]
    # The final byte still predicts EOS. EOS-to-next-BOS and padding carry no
    # observation objective, including every boundary in blank-English controls.
    observation_targets.masked_fill_(inputs["token_ids"].eq(ByteCodec.EOS) | ~inputs["valid_mask"], ByteCodec.PAD)
    supervision = {
        "action_targets": torch.tensor(actions, dtype=torch.long, device=device),
        "reply_decoder_input_ids": reply_decoder,
        "reply_targets": reply_targets,
        "reply_target_mask": reply_targets.ne(ByteCodec.PAD),
        "observation_next_byte_targets": observation_targets,
        "observation_next_byte_mask": observation_targets.ne(ByteCodec.PAD),
    }
    return {"inputs": inputs, "supervision": supervision}
