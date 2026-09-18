"""Opt-in validation against bounded, process-owned canonical JSON bytes.

Canonical values come only from the unchanged public foundation generator and
its independent oracles. Candidate labels/hashes never become expected truth.
This module changes neither public generation nor any live admission/training
path. It avoids copying expected pairs on repeated validation; it does not skip
candidate serialization, provenance comparison, or counterfactual invariants.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

from experiments import foundation_curriculum as curriculum


SCHEMA = "bic-foundation-canonical-validation-v1"
_ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    names = {"experiments/foundation_canonical_validation.py",
             "experiments/foundation_curriculum.py", "experiments/composition_curriculum.py",
             "experiments/cognitive_curriculum.py"}
    if (_ROOT/"experiments/__init__.py").exists():
        names.add("experiments/__init__.py")
    return {name: hashlib.sha256((_ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}


def _encoded(value):
    return curriculum._json(value).encode("utf8")


def _binding():
    return _encoded({"schema": SCHEMA, "curriculum_version": curriculum.VERSION,
        "composition_version": curriculum.legacy.VERSION,
        "python": sys.version, "implementation": sys.implementation.name,
        "source_sha256": source_hashes()})


_IMPORTED_BINDING = _binding()


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _recipe_key(row):
    """Mirror public request preflight BEFORE hash-key equality/cache lookup."""
    if (type(row) is not dict or set(row) != curriculum._FIELDS
            or type(row["variant"]) is not int or row["variant"] not in (0, 1)
            or type(row["primitive_shared"]) is not bool):
        raise ValueError("canonical foundation row fields/variant differ")
    recipe = row["recipe"]
    if type(recipe) is not dict or set(recipe) != curriculum._RECIPE:
        raise ValueError("compact foundation recipe fields differ")
    _integer(recipe["procedure_attempt"], "procedure_attempt", 0, curriculum.MAX_ATTEMPTS-1)
    family, split = row["family"], row["split"]
    if type(family) is not str or family not in curriculum.FAMILIES:
        raise ValueError("unknown typed family")
    if type(split) is not str or split not in curriculum.legacy.SPLITS:
        raise ValueError("admission split must be train, dev or audit")
    for name in ("seed", "naming_seed", "value_seed"):
        _integer(recipe[name], name, 0, 2**63-1)
    depth, turns, requested = recipe["depth"], recipe["turns"], recipe["structure_split"]
    if type(depth) is not int or depth not in curriculum.DEPTHS:
        raise ValueError("depth must be an integer from zero through five")
    if type(turns) is not int or turns not in curriculum.TURN_BUCKETS:
        raise ValueError("turn bucket must be eight, ten or twelve")
    if (type(requested) is not str
            or (requested != "shared" if depth < 2 else requested not in curriculum.legacy.SPLITS)):
        raise ValueError("primitive anchors require shared scope; compositions require a legacy partition")
    if depth >= 2 and (split == "train" and requested != "train" or split == "dev" and requested == "audit"):
        raise ValueError("requested composition violates admission partition")
    return (family, recipe["seed"], depth, turns, split,
            recipe["naming_seed"], recipe["value_seed"], requested)


class CanonicalValidator:
    """Finite LRU of immutable expected bytes, authenticated in this process.

    Not a security boundary against code modifying Python internals. No files
    are loaded as truth and no serialized validator may be restored. Cache byte
    accounting includes serialized keys and row bytes, not Python object overhead
    or the unchanged generator's own bounded caches. Calls are single-threaded.
    """
    __slots__ = ("_binding", "_identity", "_cache", "_bytes", "_limits", "_counts", "_poisoned")

    def __init__(self, *, max_entries=4096, max_cache_bytes=64*1024*1024,
                 max_batch_pairs=65536):
        _integer(max_entries, "max_entries", 1, 65536)
        _integer(max_cache_bytes, "max_cache_bytes", 1, 512*1024*1024)
        _integer(max_batch_pairs, "max_batch_pairs", 1, 1048576)
        self._binding = _binding()
        if self._binding != _IMPORTED_BINDING:
            raise RuntimeError("canonical validator sources changed after module import")
        self._limits = (max_entries, max_cache_bytes, max_batch_pairs)
        self._identity = _encoded({**json.loads(self._binding), "max_entries": max_entries,
            "max_cache_bytes": max_cache_bytes, "max_batch_pairs": max_batch_pairs})
        self._cache, self._bytes, self._poisoned = OrderedDict(), 0, False
        self._counts = dict.fromkeys(("batches_started", "batches_succeeded", "batches_failed",
            "candidate_rows_attempted", "candidate_rows_validated", "pairs_validated",
            "canonical_pairs_regenerated", "canonical_generation_attempts", "cache_hits",
            "cache_misses", "cache_evictions", "oversize_entries_not_cached",
            "peak_cache_entries", "peak_cache_bytes"), 0)
        self._check_sources()

    def _check_sources(self):
        if self._poisoned:
            raise RuntimeError("canonical validator source/version identity changed; create a new validator")
        try:
            current = _binding()
        except BaseException:
            self._poisoned = True
            self._cache.clear(); self._bytes = 0
            raise
        if current != self._binding:
            self._poisoned = True
            self._cache.clear(); self._bytes = 0
            raise RuntimeError("canonical validator source/version identity changed; create a new validator")

    @property
    def identity(self):
        self._check_sources()
        result = json.loads(self._identity)
        self._check_sources()
        return result

    @property
    def stats(self):
        # Accounting remains inspectable after failure or source drift.
        return {**self._counts, "cache_entries": len(self._cache), "cache_bytes": self._bytes,
            "max_entries": self._limits[0], "max_cache_bytes": self._limits[1],
            "max_batch_pairs": self._limits[2], "poisoned": self._poisoned,
            "scope": "Counts include failed attempts. Regenerated means successful public generator calls, which may use its existing cache. Byte counts cover retained serialized keys/rows only; generator caches and Python overhead are separate. No model work."}

    def _canonical(self, key):
        found = self._cache.get(key)
        if found is not None:
            self._counts["cache_hits"] += 1
            self._cache.move_to_end(key)
            return found[0]
        self._counts["cache_misses"] += 1
        self._counts["canonical_generation_attempts"] += 1
        family, seed, depth, turns, split, names, values, partition = key
        pair = curriculum.generate_pair(family, seed, depth=depth, turns=turns,
            split=split, naming_seed=names, value_seed=values, structure_split=partition)
        self._counts["canonical_pairs_regenerated"] += 1
        expected = tuple(_encoded(row) for row in pair)
        size = len(_encoded(key)) + sum(map(len, expected))
        if size > self._limits[1]:
            self._counts["oversize_entries_not_cached"] += 1
            return expected
        while len(self._cache) >= self._limits[0] or self._bytes + size > self._limits[1]:
            _, (_, old_size) = self._cache.popitem(last=False)
            self._bytes -= old_size; self._counts["cache_evictions"] += 1
        self._cache[key] = (expected, size); self._bytes += size
        self._counts["peak_cache_entries"] = max(self._counts["peak_cache_entries"], len(self._cache))
        self._counts["peak_cache_bytes"] = max(self._counts["peak_cache_bytes"], self._bytes)
        return expected

    def _validate_pair(self, rows):
        if type(rows) not in (list, tuple) or len(rows) != 2:
            raise ValueError("exactly two canonical pair members required")
        for row in rows:
            self._counts["candidate_rows_attempted"] += 1
            key = _recipe_key(row)
            if _encoded(row) != self._canonical(key)[row["variant"]]:
                raise ValueError("row differs from canonical foundation provenance")
            self._counts["candidate_rows_validated"] += 1
        # Preserve the original explicit pair conditions, beyond row membership.
        if [row["variant"] for row in rows] != [0, 1] or any(
                _encoded(rows[0][key]) != _encoded(rows[1][key])
                for key in curriculum._FIELDS - {"id", "variant", "turns"}):
            raise ValueError("matching canonical variants zero then one required")
        changed = [(left, right) for left, right in zip(rows[0]["turns"], rows[1]["turns"])
                   if left["text"] != right["text"]]
        if (len(changed) != 1 or any(turn["target"] != curriculum.legacy.ACK for turn in changed[0])
                or {row["turns"][-1]["target"] for row in rows} != {curriculum.legacy.DENY, curriculum.legacy.ALLOW}):
            raise ValueError("pair requires one changed statement and opposite known final answers")
        self._counts["pairs_validated"] += 1

    def validate_pairs(self, pairs):
        """Validate a finite batch, with source guards at both batch boundaries.

        Failures retain actual work counters; earlier successful rows/pairs in a
        failed batch are reported as checked, never as a successful whole batch.
        A source change poisons the instance and discards its cached bytes.
        """
        self._counts["batches_started"] += 1
        try:
            self._check_sources()
            try:
                if type(pairs) not in (list, tuple) or len(pairs) > self._limits[2]:
                    raise ValueError("bounded finite list or tuple of complete pairs required")
                for pair in pairs:
                    self._validate_pair(pair)
            finally:
                self._check_sources()
        except BaseException:
            self._counts["batches_failed"] += 1
            raise
        self._counts["batches_succeeded"] += 1
        return True

    def validate_pair(self, pair):
        return self.validate_pairs([pair])

    def __reduce_ex__(self, protocol):
        raise TypeError("canonical validators are process-owned and cannot be serialized")
