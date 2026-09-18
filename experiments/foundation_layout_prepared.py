"""Single-use CPU packing of already authenticated layout bundle evidence.

The caller must obtain the expected evidence and its digest from the admitted
frozen data record. Matching those bytes replaces canonical regeneration, not
protection admission. No tensor views are public, no cache is persisted, and
there is no producer thread. One owner holds at most one outstanding bundle.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import torch

from experiments import foundation_layout_training as training
from experiments.composition_data import pack_composition_episodes
from experiments.sequence_student import SequenceConfig

SCHEMA = "bic-foundation-layout-prepared-v1"
_MINT = object()


def source_hashes():
    result = training.source_hashes()
    name = "experiments/foundation_layout_prepared.py"
    result[name] = hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
    return result


def evidence_sha256(value):
    return training._hash(value)


def _tensor_digest(batches):
    records = []
    if tuple(batches) != tuple(training.FAMILIES):
        raise ValueError("canonical family order required")
    for family, batch in batches.items():
        if set(batch) != {"inputs", "supervision"}:
            raise ValueError("separate inputs and supervision required")
        for section, fields in batch.items():
            for name, value in fields.items():
                if (type(value) is not torch.Tensor or value.device.type != "cpu"
                        or value.dtype not in (torch.int64, torch.bool) or value.requires_grad
                        or value.layout != torch.strided):
                    raise ValueError("detached CPU integer/Boolean packed tensors required")
                records.append([family, section, name, str(value.dtype), list(value.shape), list(value.stride()),
                    value.storage_offset(), hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()])
    return training._hash(records)


class PreparedLayoutBundle:
    __slots__ = ("_owner", "_evidence", "_batches", "_tensor_sha256")

    def __init__(self, mint, owner, evidence, batches):
        if mint is not _MINT:
            raise TypeError("only a live owner may prepare a bundle")
        self._owner, self._evidence = owner, training._json(evidence)
        self._batches, self._tensor_sha256 = batches, _tensor_digest(batches)

    @property
    def evidence(self):
        import json
        return json.loads(self._evidence)

    def __reduce_ex__(self, protocol):
        raise TypeError("prepared bundles are transient and cannot be serialized")


class PreparedLayoutOwner:
    def __init__(self, *, config, layout, micro_batch_size):
        if (type(config) is not SequenceConfig or layout not in training.curriculum.LAYOUTS
                or type(micro_batch_size) is not int or micro_batch_size < 2 or micro_batch_size % 2):
            raise ValueError("exact config, layout and complete-pair microbatch size required")
        self.config, self.layout, self.micro_batch_size = config, layout, micro_batch_size
        self._contract = training._json([asdict(config), layout, micro_batch_size])
        self._sources, self._active, self.failed = source_hashes(), None, False
        self._active_evidence = self._active_tensor_sha256 = None
        self.work = dict(prepare_attempts=0, prepared_bundles=0, packed_microbatches=0,
            packed_episodes=0, exact_pair_matches=0, canonical_regenerations=0,
            consume_attempts=0, consumed_bundles=0, discarded_bundles=0,
            tensor_identity_checks=0, wall_seconds=0., cpu_seconds=0.)

    def _guard(self):
        if (self.failed or source_hashes() != self._sources
                or training._json([asdict(self.config), self.layout, self.micro_batch_size]) != self._contract):
            raise RuntimeError("prepared owner failed or its source/configuration changed")

    def prepare(self, bundle, *, expected_evidence, expected_evidence_sha256):
        started, cpu = time.monotonic(), time.process_time()
        self.work["prepare_attempts"] += 1
        try:
            self._guard()
            if self._active is not None:
                raise RuntimeError("consume or close the outstanding bundle first")
            expected = deepcopy(expected_evidence)
            if evidence_sha256(expected) != expected_evidence_sha256:
                raise ValueError("caller-pinned expected evidence differs")
            rows = deepcopy(bundle)
            if (type(rows) is not dict or set(rows) != {"schema", "bundle_id", "layout", "families"}
                    or rows["schema"] != training.BUNDLE_SCHEMA or rows["layout"] != self.layout
                    or rows["bundle_id"] != expected["bundle_id"] or expected["layout"] != self.layout
                    or type(rows["families"]) is not dict or set(rows["families"]) != set(training.FAMILIES)
                    or training._hash([[family, rows["families"][family]] for family in training.FAMILIES]) != expected["bundle_rows_sha256"]):
                raise ValueError("actual bundle differs from admitted immutable evidence")
            batches = {}
            for family in training.FAMILIES:
                examples = rows["families"][family]
                if (len(examples) != self.micro_batch_size or training._hash(examples) != expected["families"][family]["rows_sha256"]
                        or any(row["family"] != family or row["depth"] != expected["depth"]
                            or len(row["turns"]) != expected["turns"] or row["recipe"]["layout"] != self.layout for row in examples)):
                    raise ValueError("family size, layout or dimensions differ from admitted evidence")
                admitted_pairs = [training._json(examples[i:i+2]) for i in range(0, len(examples), 2)]
                matched = 0
                def admitted_pair(pair):
                    nonlocal matched
                    if matched >= len(admitted_pairs) or training._json(pair) != admitted_pairs[matched]:
                        raise ValueError("packer pair differs from the exact admitted pair")
                    matched += 1; self.work["exact_pair_matches"] += 1
                    return True
                batches[family] = pack_composition_episodes(examples, device="cpu", training=True,
                    pair_validator=admitted_pair, max_turns=self.config.max_turns,
                    max_input_bytes=self.config.max_input_bytes, max_context_tokens=self.config.max_positions,
                    max_reply_bytes=self.config.max_output_bytes)
                if matched != len(admitted_pairs):
                    raise ValueError("packer did not authenticate every admitted pair")
                self.work["packed_microbatches"] += 1
                self.work["packed_episodes"] += len(examples)
            self._guard()
            self._active = PreparedLayoutBundle(_MINT, self, expected, batches)
            self._active_evidence, self._active_tensor_sha256 = self._active._evidence, self._active._tensor_sha256
            self.work["tensor_identity_checks"] += 1
            self.work["prepared_bundles"] += 1
            return self._active
        except BaseException:
            self.failed = True
            raise
        finally:
            self.work["wall_seconds"] += time.monotonic()-started
            self.work["cpu_seconds"] += time.process_time()-cpu

    def _consume(self, token, *, cursor, config, layout, micro_batch_size):
        started, cpu = time.monotonic(), time.process_time()
        self.work["consume_attempts"] += 1
        try:
            self._guard()
            if (type(token) is not PreparedLayoutBundle or token is not self._active or token._owner is not self
                    or self._contract != training._json([asdict(config), layout, micro_batch_size])
                    or token._evidence != self._active_evidence or token._tensor_sha256 != self._active_tensor_sha256
                    or token.evidence["bundle_id"] != cursor):
                raise ValueError("live owner, unused bundle and exact consumer cursor/config required")
            self.work["tensor_identity_checks"] += 1
            if _tensor_digest(token._batches) != token._tensor_sha256:
                raise ValueError("prepared tensors were mutated")
            copies = {family: {section: {name: tensor.detach().clone() for name, tensor in values.items()}
                for section, values in batch.items()} for family, batch in token._batches.items()}
            self.work["tensor_identity_checks"] += 1
            if _tensor_digest(copies) != token._tensor_sha256:
                raise ValueError("detached consumer copies differ")
            evidence = token.evidence
            self._active = None
            token._batches = {}
            self.work["consumed_bundles"] += 1
            return copies, evidence
        except BaseException:
            self.failed = True
            raise
        finally:
            self.work["wall_seconds"] += time.monotonic()-started
            self.work["cpu_seconds"] += time.process_time()-cpu

    def close(self):
        if self._active is not None:
            self._active._batches = {}; self._active = None
            self.work["discarded_bundles"] += 1
        self.failed = True

    def report(self):
        return dict(**deepcopy(self.work), failed=self.failed, outstanding_bundles=int(self._active is not None))
