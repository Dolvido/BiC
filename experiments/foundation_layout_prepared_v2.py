"""Public immutable packed images and independent single-use consumer owners.

Rows are admitted and packed once. Tensor-only images can be authenticated from
bytes without decoding raw lesson metadata or repeating prefix interpretation.
All publication, protection admission, caching limits and scheduling are caller-owned.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import time

import torch

from experiments import foundation_layout_training as training
from experiments import foundation_layout_prepared as v1
from experiments import shared_state_targets as targets
from experiments.composition_data import pack_composition_episodes
from experiments.sequence_student import SequenceConfig

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-foundation-layout-prepared-v2"
IMAGE_SCHEMA = "bic-foundation-layout-packed-image-v2"
_MINT = object()
evidence_sha256 = training._hash


def source_hashes():
    return {**v1.source_hashes(), **targets.source_hashes(),
        "experiments/foundation_layout_prepared_v2.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _content(evidence):
    value = deepcopy(evidence)
    cursor = value.pop("bundle_id")
    if type(cursor) is not int or cursor < 0: raise ValueError("nonnegative consumed cursor required")
    return value


def _labels_digest(values, micro, turns):
    if type(values) is not dict or tuple(values) != tuple(training.FAMILIES):
        raise ValueError("ordered all-family prefix supervision required")
    records = []
    for family, value in values.items():
        if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.long
                or value.shape != (micro,turns,12) or value.requires_grad or value.layout != torch.strided
                or not bool(((value >= 0)&(value < 107)).all())):
            raise ValueError("detached CPU typed prefix labels required")
        records.append([family,list(value.shape),list(value.stride()),value.storage_offset(),
                        hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()])
    return training._hash(records)


def _copies(batches):
    return {f:{section:{name:value.detach().clone() for name,value in fields.items()}
        for section,fields in batch.items()} for f,batch in batches.items()}


class VerifiedPackedImage:
    """Public factory-only immutable image. No writable tensor views are exposed."""
    __slots__ = ("_identity", "_identity_sha256", "_batches", "_labels", "_work")

    def __init__(self, mint, identity, batches, labels, work):
        if mint is not _MINT: raise TypeError("use from_bundle or from_bytes")
        self._identity, self._identity_sha256 = deepcopy(identity), training._hash(identity)
        self._batches, self._labels, self._work = batches, labels, deepcopy(work)
        self._guard()

    @classmethod
    def from_bundle(cls, bundle, *, expected_evidence, expected_evidence_sha256,
                    config, layout, micro_batch_size):
        started,cpu = time.monotonic(),time.process_time()
        work = dict(pack_attempts=1,packed_images=0,packed_microbatches=0,packed_episodes=0,
                    exact_pair_matches=0,state_target_calls=0,state_target_episodes=0,
                    canonical_regenerations=0,archive_loads=0)
        try:
            if type(config) is not SequenceConfig or type(micro_batch_size) is not int or micro_batch_size < 2 or micro_batch_size%2:
                raise ValueError("explicit config and complete-pair microbatch required")
            sources, expected, rows = source_hashes(), deepcopy(expected_evidence), deepcopy(bundle)
            if (training._hash(expected) != expected_evidence_sha256
                    or type(rows) is not dict or set(rows) != {"schema","bundle_id","layout","families"}
                    or rows["schema"] != training.BUNDLE_SCHEMA or rows["layout"] != layout
                    or layout not in training.curriculum.LAYOUTS or rows["bundle_id"] != expected["bundle_id"]
                    or expected["layout"] != layout or type(rows["families"]) is not dict
                    or set(rows["families"]) != set(training.FAMILIES)
                    or training._hash([[f,rows["families"][f]] for f in training.FAMILIES]) != expected["bundle_rows_sha256"]):
                raise ValueError("actual rows differ from caller-admitted immutable evidence")
            batches, labels = {}, {}
            for family in training.FAMILIES:
                examples = rows["families"][family]
                if (len(examples) != micro_batch_size or training._hash(examples) != expected["families"][family]["rows_sha256"]
                        or any(r["family"] != family or r["depth"] != expected["depth"]
                               or len(r["turns"]) != expected["turns"] or r["recipe"]["layout"] != layout for r in examples)):
                    raise ValueError("family dimensions or exact evidence differ")
                admitted = [training._json(examples[i:i+2]) for i in range(0,len(examples),2)]
                matched = 0
                def validate(pair):
                    nonlocal matched
                    if matched >= len(admitted) or training._json(pair) != admitted[matched]:
                        raise ValueError("packer pair differs from admitted pair")
                    matched += 1; work["exact_pair_matches"] += 1
                    return True
                batches[family] = pack_composition_episodes(examples,device="cpu",training=True,pair_validator=validate,
                    max_turns=config.max_turns,max_input_bytes=config.max_input_bytes,
                    max_context_tokens=config.max_positions,max_reply_bytes=config.max_output_bytes)
                if matched != len(admitted): raise ValueError("every admitted pair must be checked")
                work["packed_microbatches"] += 1; work["packed_episodes"] += len(examples)
                labels[family] = targets.pack_state_targets(examples)
                work["state_target_calls"] += 1; work["state_target_episodes"] += len(examples)
            identity = dict(schema=IMAGE_SCHEMA,config=asdict(config),layout=layout,micro_batch_size=micro_batch_size,
                content_evidence=_content(expected),source_sha256=sources,tensor_sha256=v1._tensor_digest(batches),
                state_target_sha256=_labels_digest(labels,micro_batch_size,expected["turns"]))
            if source_hashes() != sources: raise ValueError("packing sources changed")
            work["packed_images"] = 1
            result=cls(_MINT,identity,batches,labels,work)
            result._work.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            return result
        except BaseException as error:
            work.update(failed=True,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            error.preparation_report=deepcopy(work); raise

    @classmethod
    def from_bytes(cls, raw, *, expected_sha256, expected_identity_sha256):
        started,cpu=time.monotonic(),time.process_time()
        work=dict(archive_load_attempts=0,archive_loads=0,archive_bytes=0,packed_images=0,
                  packed_microbatches=0,packed_episodes=0,state_target_calls=0,canonical_regenerations=0)
        try:
            if (type(raw) is not bytes or not 0<len(raw)<=1<<30
                    or hashlib.sha256(raw).hexdigest()!=expected_sha256):
                raise ValueError("caller-pinned immutable image bytes required before decode")
            work["archive_load_attempts"]+=1
            value=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
            work["archive_loads"]+=1;work["archive_bytes"]=len(raw)
            if (type(value) is not dict or set(value)!={"schema","identity","batches","state_targets"}
                    or value["schema"]!=IMAGE_SCHEMA or training._hash(value["identity"])!=expected_identity_sha256):
                raise ValueError("packed image identity differs")
            result=cls(_MINT,value["identity"],value["batches"],value["state_targets"],work)
            result._work.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            return result
        except BaseException as error:
            work.update(failed=True,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            error.preparation_report=deepcopy(work);raise

    @property
    def identity(self): return deepcopy(self._identity)
    @property
    def identity_sha256(self): return self._identity_sha256
    def report(self): return deepcopy(self._work)

    def _guard(self):
        value=self._identity
        if (type(value) is not dict or set(value)!={"schema","config","layout","micro_batch_size","content_evidence",
                "source_sha256","tensor_sha256","state_target_sha256"} or value["schema"]!=IMAGE_SCHEMA
                or training._hash(value)!=self._identity_sha256 or value["source_sha256"]!=source_hashes()
                or type(value["micro_batch_size"]) is not int or value["micro_batch_size"]<2 or value["micro_batch_size"]%2
                or value["layout"] not in training.curriculum.LAYOUTS
                or value["content_evidence"]["layout"]!=value["layout"]
                or v1._tensor_digest(self._batches)!=value["tensor_sha256"]
                or _labels_digest(self._labels,value["micro_batch_size"],value["content_evidence"]["turns"])!=value["state_target_sha256"]):
            raise ValueError("packed image source, identity or tensors changed")
        SequenceConfig(**value["config"])

    def snapshot(self):
        self._guard()
        return dict(schema=IMAGE_SCHEMA,identity=self.identity,batches=_copies(self._batches),
                    state_targets={f:v.detach().clone() for f,v in self._labels.items()})

    def __reduce_ex__(self, protocol): raise TypeError("serialize image.snapshot(), not a live image")


class PreparedLayoutBundle:
    __slots__=("_owner","_image","_evidence","_used")
    def __init__(self,mint,owner,image,evidence):
        if mint is not _MINT: raise TypeError("only the public owner may issue tokens")
        self._owner,self._image,self._evidence,self._used=owner,image,training._json(evidence),False
    @property
    def owner(self): return self._owner
    @property
    def evidence(self):
        import json
        return json.loads(self._evidence)
    def __reduce_ex__(self,protocol): raise TypeError("single-use tokens cannot be serialized")


class PreparedLayoutOwner:
    def __init__(self, *, config, layout, micro_batch_size):
        if (type(config) is not SequenceConfig or layout not in training.curriculum.LAYOUTS
                or type(micro_batch_size) is not int or micro_batch_size<2 or micro_batch_size%2):
            raise ValueError("explicit compatible config/layout/microbatch required")
        self.config,self.layout,self.micro_batch_size=config,layout,micro_batch_size
        self._contract=training._json([asdict(config),layout,micro_batch_size])
        self._sources=source_hashes();self._active=None;self.failed=False
        self.work=dict(prepare_attempts=0,prepared_bundles=0,packed_microbatches=0,packed_episodes=0,
            canonical_regenerations=0,consume_attempts=0,consumed_bundles=0,discarded_bundles=0,
            tensor_identity_checks=0,wall_seconds=0.,cpu_seconds=0.)

    def _guard(self):
        if self.failed: raise RuntimeError("poisoned owner cannot issue or consume")
        if source_hashes()!=self._sources or self._contract!=training._json([asdict(self.config),self.layout,self.micro_batch_size]):
            raise ValueError("packed owner source/config changed")

    def prepare_from_verified_image(self,image,*,expected_evidence,expected_evidence_sha256):
        started,cpu=time.monotonic(),time.process_time();self.work["prepare_attempts"]+=1
        try:
            self._guard()
            if type(image) is not VerifiedPackedImage or self._active is not None:
                raise ValueError("verified image and owner without outstanding token required")
            image._guard();self.work["tensor_identity_checks"]+=1
            expected=deepcopy(expected_evidence);identity=image.identity
            if (training._hash(expected)!=expected_evidence_sha256 or _content(expected)!=identity["content_evidence"]
                    or identity["config"]!=asdict(self.config) or identity["layout"]!=self.layout
                    or identity["micro_batch_size"]!=self.micro_batch_size):
                raise ValueError("packed content differs from caller-pinned evidence/configuration")
            token=PreparedLayoutBundle(_MINT,self,image,expected)
            self._active=(token,token._evidence,image.identity_sha256)
            self.work["prepared_bundles"]+=1
            return token
        except BaseException:
            self.failed=True;raise
        finally:
            self.work["wall_seconds"]+=time.monotonic()-started;self.work["cpu_seconds"]+=time.process_time()-cpu

    def consume(self,token,*,cursor,config,layout,micro_batch_size):
        started,cpu=time.monotonic(),time.process_time();self.work["consume_attempts"]+=1
        try:
            self._guard()
            if (type(token) is not PreparedLayoutBundle or self._active is None or token is not self._active[0]
                    or token._owner is not self or token._used or token._evidence!=self._active[1]
                    or token._image.identity_sha256!=self._active[2] or token.evidence["bundle_id"]!=cursor
                    or self._contract!=training._json([asdict(config),layout,micro_batch_size])):
                raise ValueError("exact live owner, unused token, cursor and configuration required")
            image=token._image;image._guard();self.work["tensor_identity_checks"]+=1
            batches=_copies(image._batches);labels={f:v.detach().clone() for f,v in image._labels.items()}
            if (v1._tensor_digest(batches)!=image._identity["tensor_sha256"]
                    or _labels_digest(labels,micro_batch_size,token.evidence["turns"])!=image._identity["state_target_sha256"]):
                raise ValueError("detached consumer copy differs")
            self.work["tensor_identity_checks"]+=1
            evidence=token.evidence;token._used=True;token._image=None;self._active=None
            self.work["consumed_bundles"]+=1
            return batches,evidence,labels
        except BaseException:
            self.failed=True;raise
        finally:
            self.work["wall_seconds"]+=time.monotonic()-started;self.work["cpu_seconds"]+=time.process_time()-cpu

    def close(self):
        if self._active is not None:
            self._active[0]._used=True;self._active[0]._image=None;self._active=None
            self.work["discarded_bundles"]+=1
        self.failed=True
    def report(self): return dict(**deepcopy(self.work),failed=self.failed,outstanding_bundles=int(self._active is not None))
