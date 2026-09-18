"""Known-development diagnostics separating English wording from bindings.

The original dialogue split changes alias/color bindings and phrase families
together. This bank crosses their train/dev supports while changing no semantic
fact, ordering, response, or observation within each wording contrast. "Unseen"
means outside the original training support, not an untouched new audit: these
phrase families and benchmark rules have already been used in development.

The grammar is used only to construct and verify evaluation material. Returned
episodes carry a diagnostic-only split and cannot enter canonical training.
Student evaluation consumes their English, never the parser or provenance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json

from .dialogue_curriculum import (
    FOCI, _PHRASES, curriculum_digest, dialogue_oracle, generate_dialogues,
    parse_sentence,
)


SCHEMA = "bic-dialogue-diagnostic-matrix-v1"
DIAGNOSTIC_SPLIT = "diagnostic"
SLICE_SUPPORTS = {
    "familiar_bindings_familiar_phrases": ("train", "train"),
    "familiar_bindings_unseen_phrases": ("train", "dev"),
    "unseen_bindings_familiar_phrases": ("dev", "train"),
    "unseen_bindings_unseen_phrases": ("dev", "dev"),
}


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transcript_fingerprint(episodes: Mapping | Sequence[Mapping]) -> str:
    """Hash exact English turns, ignoring labels, observations and provenance.

    Accept one episode or an ordered collection; a one-item collection has the
    same fingerprint as that episode. Collection order matters. This detects
    repeated transcripts, not shared templates or semantic equivalence, and
    does not establish that a generated seed is independent held-out evidence.
    """
    if isinstance(episodes, Mapping):
        episodes = [episodes]
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise ValueError("transcripts require an episode or sequence of episodes")
    transcripts = []
    for episode in episodes:
        if not isinstance(episode, Mapping) or not isinstance(episode.get("turns"), list):
            raise ValueError("transcript episodes require a list of turns")
        turns = episode["turns"]
        if any(not isinstance(turn, Mapping) or not isinstance(turn.get("text"), str)
               for turn in turns):
            raise ValueError("transcript turns require English text")
        transcripts.append([turn["text"] for turn in turns])
    return _digest(transcripts)


def _rephrase(text: str, source_split: str, phrase_split: str) -> str:
    kind, facts = parse_sentence(text)
    originals = _PHRASES[source_split][kind]
    variant = next((index for index, template in enumerate(originals)
                    if template.format(**facts) == text), None)
    if variant is None:
        raise ValueError("sentence is not in its canonical source phrase family")
    result = _PHRASES[phrase_split][kind][variant].format(**facts)
    if parse_sentence(result) != (kind, facts):
        raise ValueError("diagnostic rephrasing changed the sentence meaning")
    return result


def diagnostic_bank(seed: int, count_per_focus: int = 64) -> dict[str, list[dict]]:
    """Return four development slices, each with all three dialogue focuses.

    ``seed`` is a nonnegative even canonical episode seed. A positive even
    ``count_per_focus`` preserves every counterfactual pair without unmatched
    boundary rows. Each focus uses a disjoint contiguous seed block. Within
    each binding support, both phrase conditions use precisely the same source
    episodes and phrase-variant indices. Binding conditions use the canonical
    train/dev generators and therefore are not matched semantic worlds.

    All source groups and supervision survive unchanged. Diagnostic metadata,
    identities and split deliberately prevent canonical training admission,
    including for the familiar/familiar slice. Generation does not reserve a
    final audit, prove transcript novelty, or train/evaluate a student.
    """
    if type(seed) is not int or seed < 0 or seed % 2:
        raise ValueError("diagnostic seed must be a nonnegative even integer")
    if type(count_per_focus) is not int or count_per_focus < 2 or count_per_focus % 2:
        raise ValueError("count_per_focus must be a positive even integer")
    sources = {
        split: [episode for index, focus in enumerate(FOCI)
                for episode in generate_dialogues(seed + index * count_per_focus,
                                                   count_per_focus, split, focus)]
        for split in ("train", "dev")
    }
    source_digest = curriculum_digest()
    bank = {}
    for name, (binding_split, phrase_split) in SLICE_SUPPORTS.items():
        episodes = []
        for original in sources[binding_split]:
            episode = copy.deepcopy(original)
            for turn in episode["turns"]:
                turn["text"] = _rephrase(turn["text"], binding_split, phrase_split)
            if dialogue_oracle(episode) != [turn["target"] for turn in episode["turns"]]:
                raise ValueError("diagnostic labels disagree with rephrased English")
            episode["split"] = DIAGNOSTIC_SPLIT
            episode["id"] = _digest([SCHEMA, original["id"], phrase_split])
            episode["diagnostic"] = {
                "schema": SCHEMA,
                "slice": name,
                "binding_support": binding_split,
                "phrase_support": phrase_split,
                "canonical_episode_id": original["id"],
                "curriculum_sha256": source_digest,
                "training_allowed": False,
                "evidence": "known_development_benchmark_not_pristine_audit",
            }
            episodes.append(episode)
        bank[name] = episodes
    return bank
