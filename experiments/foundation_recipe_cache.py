"""Opt-in bounded canonical recipe cache, separate from the live pilot.

Only Python recipe rows are cached. No tensors, learner state, model calls,
optimizer work or disk writes occur. Returned bundles and anchors are isolated
copies. Source drift is an error, never an invitation to reuse old entries.
The optional benchmark is not run on import and establishes no prior speedup.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

from experiments import foundation_plan as planning


SCHEMA = "bic-foundation-recipe-cache-v1"
_SOURCES = ("experiments/foundation_recipe_cache.py", "experiments/foundation_plan.py",
            "experiments/foundation_curriculum.py", "experiments/composition_curriculum.py",
            "experiments/cognitive_curriculum.py")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    """The materializer's local source closure, including vocabulary and cache."""
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _SOURCES}


class FoundationRecipeCache:
    """Own one admitted plan and at most ``max_bundles`` isolated LRU entries.

    ``admit`` may switch plans without conflating identities. Entries from old
    plans remain within the same capacity bound. The caller still authenticates
    historical/evaluation exclusions; shape-valid admission metadata alone is
    not an admission receipt or execution proof.
    """

    def __init__(self, plan, *, max_bundles=8):
        planning._integer(max_bundles, "max_bundles", 1, 4096)
        self._lock = threading.RLock()
        self._max_bundles = max_bundles
        self._sources = source_hashes()
        self._source_digest = _digest(self._sources)
        self._entries = OrderedDict()
        self._counts = dict.fromkeys(("hits", "misses", "materialized_bundles", "evictions",
                                     "failed_materializations", "source_rejections", "plan_admissions"), 0)
        self._materialization_seconds = 0.0
        self.admit(plan)

    def _check_sources(self):
        if source_hashes() != self._sources:
            self._counts["source_rejections"] += 1
            raise ValueError("recipe cache source identity changed; cached rows cannot be reused")

    @property
    def stats(self):
        with self._lock:
            return {"schema": SCHEMA, **self._counts, "entries": len(self._entries),
                    "max_bundles": self._max_bundles,
                    "materialization_seconds": self._materialization_seconds,
                    "scope": "Per-instance cumulative accesses; materialization time includes failed attempts. No learner counters are owned or advanced."}

    @property
    def plan_sha256(self):
        with self._lock:
            return self._plan_digest

    @property
    def source_sha256(self):
        with self._lock:
            return deepcopy(self._sources)

    def admit(self, plan):
        """Validate and isolate a plan before making it current; no generation."""
        with self._lock:
            self._check_sources()
            candidate = deepcopy(plan)
            planning.validate_plan(candidate)
            identity = _digest(candidate)
            self._check_sources()
            self._plan, self._plan_digest = candidate, identity
            self._counts["plan_admissions"] += 1
            return identity

    def _bundle(self, bundle_id):
        """Locked internal read; never expose the returned cache-owned object."""
        self._check_sources()
        planning._integer(bundle_id, "bundle_id", 0, len(self._plan["bundles"])-1)
        key = (self._source_digest, self._plan_digest, bundle_id)
        if key in self._entries:
            self._counts["hits"] += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        self._counts["misses"] += 1
        started = time.perf_counter()
        try:
            rows = planning._materialize_validated_bundle(self._plan, bundle_id)
            self._counts["materialized_bundles"] += 1
            self._check_sources()
        except BaseException:
            self._counts["failed_materializations"] += 1
            raise
        finally:
            self._materialization_seconds += time.perf_counter() - started
        self._entries[key] = rows
        if len(self._entries) > self._max_bundles:
            self._entries.popitem(last=False)
            self._counts["evictions"] += 1
        return rows

    def get_bundle(self, bundle_id):
        with self._lock:
            result = deepcopy(self._bundle(bundle_id))
            self._check_sources()
            return result

    def reconstruct_anchor(self, reference):
        """Equivalent pair/hash validation to foundation_evidence's helper."""
        from experiments.foundation_curriculum import FAMILIES
        with self._lock:
            self._check_sources()
            if (type(reference) is not dict or set(reference) != {
                    "bundle_id", "pair_index", "family", "pair_sha256"}
                    or type(reference["family"]) is not str or reference["family"] not in FAMILIES):
                raise ValueError("canonical anchor reference required")
            planning._integer(reference["pair_index"], "anchor pair index", 0,
                              self._plan["config"]["micro_batch_size"]//2-1)
            rows = self._bundle(reference["bundle_id"])[reference["family"]]
            start = 2 * reference["pair_index"]
            pair = rows[start:start+2]
            if _digest(pair) != reference["pair_sha256"]:
                raise ValueError("anchor differs from admitted training recipe")
            result = deepcopy(pair)
            self._check_sources()
            return result


def benchmark_anchors(plan, references, *, max_bundles=8, repetitions=1):
    """Optional read-only CPU measurement; callers explicitly choose when to run.

    Warm the same anchors explicitly, then counterbalance implementation order.
    Recipe caches are new for each measured trial; existing shared generator
    caches are never cleared. Individual timings include setup, source checks
    and copies where used. No speedup aggregate or training claim is returned.
    No automatic execution, file output, model scoring or optimization occurs.
    """
    from experiments.foundation_evidence import reconstruct_anchor
    planning._integer(repetitions, "repetitions", 1, 1000)
    planning._integer(max_bundles, "max_bundles", 1, 4096)
    if type(references) not in (list, tuple) or not references:
        raise ValueError("nonempty bounded caller-owned anchor sequence required")
    sources = source_hashes()
    plan = deepcopy(plan)
    planning.validate_plan(plan)
    references = deepcopy(references)
    started = time.perf_counter()
    expected = [_digest(reconstruct_anchor(plan, ref)) for ref in references]
    warmup = {"wall_seconds": time.perf_counter() - started,
              "anchor_calls": len(references), "bundle_materializer_calls": len(references)}
    trials = []
    for order in (("uncached", "cached"), ("cached", "uncached")):
        observations = {}
        for mode in order:
            started = time.perf_counter()
            cache = FoundationRecipeCache(plan, max_bundles=max_bundles) if mode == "cached" else None
            for _ in range(repetitions):
                for reference, identity in zip(references, expected):
                    pair = cache.reconstruct_anchor(reference) if cache else reconstruct_anchor(plan, reference)
                    if _digest(pair) != identity:
                        raise ValueError("cache benchmark reconstruction differs")
            observations[mode] = {"wall_seconds": time.perf_counter() - started,
                "anchor_calls": len(references) * repetitions,
                "bundle_materializer_calls": cache.stats["materialized_bundles"] if cache else len(references) * repetitions,
                "cache": cache.stats if cache else None}
            if source_hashes() != sources or (cache is not None and cache.source_sha256 != sources):
                raise ValueError("cache benchmark source identity differs")
        trials.append({"execution_order": list(order), "observations": observations})
    return {"schema": SCHEMA, "exact_pair_hashes": True, "warmup": warmup, "trials": trials,
            "total_anchor_calls": len(references) * (1 + 4 * repetitions),
            "plan_sha256": _digest(plan), "source_sha256": sources,
            "neural_training_or_inference": False, "speedup_aggregate": None,
            "scope": "Exploratory counterbalanced warm-start recipe/anchor timings. Warmup is recorded separately; preexisting global generator cache state and finite cache eviction can still affect timings. No shared cache clearing, tensor/gradient comparison, cross-process or training acceleration claim."}
