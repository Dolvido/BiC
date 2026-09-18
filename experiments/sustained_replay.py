"""Private bounded resident images for repeated unchanged whole-curriculum replay.

Archive bytes are authenticated before one weights-only CPU decode per residency.
Only a real PreparedLayoutOwner receives private rows; its public prepare copies
and validates them. No cached image/tensor reference is returned. Explicit source
checks are required at cycle boundaries; no recursive source hashing per lesson.
The byte cap measures serialized archive sizes, not Python/tensor resident RSS.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import time

from experiments import recurrent_read_data as original_data
from experiments import foundation_layout_prepared as preparation
from experiments import shared_state_targets as targets
from experiments.foundation_layout_study import native

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-sustained-resident-replay-v1"
MANIFEST_SHA256 = "c420a43a3a45f63347288dda7db48c71b3e7fa183b4a1681c2ec89eabeb08654"
UPDATES, MICRO = 648, 32


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _pin(value): return type(value) is str and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def source_hashes():
    return {**original_data.source_hashes(), **targets.source_hashes(), **preparation.source_hashes(),
        "experiments/sustained_replay.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _membership():
    return (tuple(sorted(p.name for p in (ROOT/"brain_in_computer").glob("*.py"))),
            (ROOT/"experiments/__init__.py").exists())


class ResidentReplay:
    def __init__(self, directory, expected_manifest_sha256, start_cursor=2160, max_archive_bytes=536870912):
        self.failed=True; self._cache=OrderedDict(); self._cached_bytes=0
        self.work=dict(prepare_attempts=0,prepares=0,cache_hits=0,cache_misses=0,archive_attempts=0,
            archive_loads=0,archive_bytes_read=0,decoded_rows=0,target_attempts=0,target_completions=0,
            target_rows=0,target_label_elements=0,evictions=0,evicted_archive_bytes=0,
            source_checks=0,source_file_reads=0,failures=0,peak_cached_archive_bytes=0,
            archive_wall_seconds=0.,archive_cpu_seconds=0.,targets_wall_seconds=0.,targets_cpu_seconds=0.,
            owner_preparation_wall_seconds=0.,owner_preparation_cpu_seconds=0.,copy_wall_seconds=0.,copy_cpu_seconds=0.,
            eviction_wall_seconds=0.,eviction_cpu_seconds=0.,source_wall_seconds=0.,source_cpu_seconds=0.,
            wall_seconds=0.,cpu_seconds=0.)
        started,cpu=time.monotonic(),time.process_time()
        try:
            if (expected_manifest_sha256!=MANIFEST_SHA256 or type(start_cursor) is not int or not 0<=start_cursor<2**63
                    or type(max_archive_bytes) is not int or not 1<=max_archive_bytes<=536870912):
                raise ValueError("fixed admitted manifest and bounded explicit replay start/cache required")
            self._directory=Path(directory).resolve();self.start_cursor=start_cursor;self.max_archive_bytes=max_archive_bytes
            path=self._directory/"manifest.json";raw=native(path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:raise ValueError("caller-pinned manifest differs")
            manifest=json.loads(raw)
            if (manifest.get("schema")!=original_data.SCHEMA or manifest.get("updates")!=UPDATES
                    or manifest.get("micro_batch_size")!=MICRO or manifest.get("layout")!="original"
                    or type(manifest.get("training")) is not list or len(manifest["training"])!=UPDATES
                    or manifest.get("source_sha256")!=original_data.source_hashes()):
                raise ValueError("original complete source-pinned admitted catalogue required")
            self._records=deepcopy(manifest["training"])
            for index,record in enumerate(self._records):
                if (set(record)!={"path","sha256","global_cursor","cycle_id","canonical_bundle_id"}
                        or record["path"]!=f"training/{index:04d}.pt" or not _pin(record["sha256"])
                        or type(record["global_cursor"]) is not int or record["global_cursor"]!=index
                        or type(record["cycle_id"]) is not int or record["cycle_id"]!=index//216
                        or type(record["canonical_bundle_id"]) is not int or record["canonical_bundle_id"]!=index%216):
                    raise ValueError("ordered canonical catalogue coordinates differ")
            receipt_path=self._directory/"preparation.json";receipt_raw=native(receipt_path).read_bytes();receipt=json.loads(receipt_raw)
            if receipt.get("status")!="completed" or receipt.get("manifest_sha256")!=expected_manifest_sha256:
                raise ValueError("completed original preparation required")
            self._input_files={path:expected_manifest_sha256,receipt_path:hashlib.sha256(receipt_raw).hexdigest()}
            self._sources=source_hashes();self._membership=_membership();self.failed=False
            self.authenticate_sources()
        except BaseException:
            self.failed=True;self.work["failures"]+=1;raise
        finally:
            self.work["wall_seconds"]+=time.monotonic()-started;self.work["cpu_seconds"]+=time.process_time()-cpu

    def authenticate_sources(self):
        """Fresh flat source/manifest check at caller-declared cycle boundaries."""
        started,cpu=time.monotonic(),time.process_time();self.work["source_checks"]+=1
        try:
            if self.failed:raise RuntimeError("poisoned resident reader")
            if _membership()!=self._membership:raise ValueError("source membership changed")
            for name,pin in self._sources.items():
                path=(ROOT/name).resolve()
                if not path.is_relative_to(ROOT):raise ValueError("source escapes repository")
                self.work["source_file_reads"]+=1
                if hashlib.sha256(native(path).read_bytes()).hexdigest()!=pin:raise ValueError("source bytes changed: "+name)
            for path,pin in self._input_files.items():
                if hashlib.sha256(native(path).read_bytes()).hexdigest()!=pin:raise ValueError("manifest or preparation changed")
        except BaseException:
            self.failed=True;raise
        finally:
            self.work["source_wall_seconds"]+=time.monotonic()-started;self.work["source_cpu_seconds"]+=time.process_time()-cpu

    def _load(self,index):
        if index in self._cache:
            self.work["cache_hits"]+=1;self._cache.move_to_end(index);return self._cache[index]
        self.work["cache_misses"]+=1;record=self._records[index];path=(self._directory/record["path"]).resolve()
        if not path.is_relative_to(self._directory):raise ValueError("archive escaped admitted directory")
        started,cpu=time.monotonic(),time.process_time();self.work["archive_attempts"]+=1
        try:
            if native(path).stat().st_size>self.max_archive_bytes:raise ValueError("one archive exceeds serialized-byte cache cap")
            raw=native(path).read_bytes();size=len(raw);self.work["archive_bytes_read"]+=size
            if size>self.max_archive_bytes or hashlib.sha256(raw).hexdigest()!=record["sha256"]:
                raise ValueError("authenticated archive bytes differ or exceed cap")
            import torch
            image=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True);self.work["archive_loads"]+=1
            if (type(image) is not dict or set(image)!={"bundle","expected_evidence","expected_evidence_sha256"}
                    or image["expected_evidence_sha256"]!=_hash(image["expected_evidence"])
                    or image["bundle"]["bundle_id"]!=index or image["expected_evidence"]["bundle_id"]!=index
                    or set(image["bundle"]["families"])!=set(targets.FAMILIES)
                    or any(len(rows)!=MICRO for rows in image["bundle"]["families"].values())):
                raise ValueError("archive evidence/coordinate/episode count differs")
            self.work["decoded_rows"]+=sum(len(rows) for rows in image["bundle"]["families"].values())
        finally:
            self.work["archive_wall_seconds"]+=time.monotonic()-started;self.work["archive_cpu_seconds"]+=time.process_time()-cpu
        del raw
        started,cpu=time.monotonic(),time.process_time();labels={}
        try:
            for family in targets.FAMILIES:
                self.work["target_attempts"]+=1
                labels[family]=targets.pack_state_targets(image["bundle"]["families"][family])
                self.work["target_completions"]+=1;self.work["target_rows"]+=len(image["bundle"]["families"][family])
                self.work["target_label_elements"]+=labels[family].numel()
        finally:
            self.work["targets_wall_seconds"]+=time.monotonic()-started;self.work["targets_cpu_seconds"]+=time.process_time()-cpu
        started,cpu=time.monotonic(),time.process_time()
        while self._cache and self._cached_bytes+size>self.max_archive_bytes:
            _,old=self._cache.popitem(last=False);self._cached_bytes-=old["archive_bytes"]
            self.work["evictions"]+=1;self.work["evicted_archive_bytes"]+=old["archive_bytes"];del old
        self.work["eviction_wall_seconds"]+=time.monotonic()-started;self.work["eviction_cpu_seconds"]+=time.process_time()-cpu
        entry=dict(image=image,labels=labels,archive_bytes=size)
        self._cache[index]=entry;self._cached_bytes+=size
        self.work["peak_cached_archive_bytes"]=max(self._cached_bytes,self.work["peak_cached_archive_bytes"])
        return entry

    def prepare(self,cursor,owner):
        started,cpu=time.monotonic(),time.process_time();self.work["prepare_attempts"]+=1
        try:
            if self.failed:raise RuntimeError("poisoned resident reader")
            if type(cursor) is not int or not self.start_cursor<=cursor<2**63:raise ValueError("monotonic-domain global cursor required")
            if type(owner) is not preparation.PreparedLayoutOwner or owner.layout!="original" or owner.micro_batch_size!=MICRO:
                raise ValueError("exact original prepared owner required")
            index=(cursor-self.start_cursor)%UPDATES;entry=self._load(index);source=entry["image"]
            bundle=dict(source["bundle"],bundle_id=cursor)
            evidence=deepcopy(source["expected_evidence"]);evidence["bundle_id"]=cursor
            evidence_pin=preparation.evidence_sha256(evidence)
            mark,ccpu=time.monotonic(),time.process_time()
            try:token=owner.prepare(bundle,expected_evidence=evidence,expected_evidence_sha256=evidence_pin)
            finally:
                self.work["owner_preparation_wall_seconds"]+=time.monotonic()-mark;self.work["owner_preparation_cpu_seconds"]+=time.process_time()-ccpu
            mark,ccpu=time.monotonic(),time.process_time()
            try:labels={family:value.detach().clone() for family,value in entry["labels"].items()}
            finally:
                self.work["copy_wall_seconds"]+=time.monotonic()-mark;self.work["copy_cpu_seconds"]+=time.process_time()-ccpu
            record=self._records[index]
            provenance=dict(schema=SCHEMA,source_manifest_sha256=MANIFEST_SHA256,source_index=index,
                replay_cycle=(cursor-self.start_cursor)//UPDATES,start_cursor=self.start_cursor,consumed_global_id=cursor,
                original_archive_path=str((self._directory/record["path"]).relative_to(ROOT).as_posix()) if self._directory.is_relative_to(ROOT) else str(self._directory/record["path"]),
                original_archive_sha256=record["sha256"],canonical_cycle_id=record["cycle_id"],canonical_bundle_id=record["canonical_bundle_id"],
                original_evidence_sha256=source["expected_evidence_sha256"],prepared_evidence_sha256=evidence_pin)
            self.work["prepares"]+=1
            return token,labels,provenance
        except BaseException:
            self.failed=True;self.work["failures"]+=1;raise
        finally:
            self.work["wall_seconds"]+=time.monotonic()-started;self.work["cpu_seconds"]+=time.process_time()-cpu

    def report(self):
        return dict(schema=SCHEMA,**deepcopy(self.work),failed=self.failed,cached_archives=len(self._cache),
            cached_archive_bytes=self._cached_bytes,max_archive_bytes=self.max_archive_bytes,
            scope="Private resident decoded images; cap is original serialized archive bytes, not RSS. Targets once per residency. Component times nest in constructor/prepare intervals; cycle source checks are separately timed. No learner or teacher work.")
