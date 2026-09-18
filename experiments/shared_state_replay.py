"""Authenticated in-memory replay IDs; original cached lesson bytes stay fixed."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-shared-state-replay-v1"
MANIFEST_SHA256 = "7c0d1f208f59a183cf5c052c991c842c2358107a3c40c84f4fc36ba42fa63612"
SOURCE_UPDATES, START, STOP = 648, 648, 1944
_MINT = object()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    from experiments import entity_retrieval_data as data, foundation_layout_prepared as prepared
    return {**data.source_hashes(), **prepared.source_hashes(),
        "experiments/shared_state_replay.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class ReplayReader:
    """One authenticated admitted schedule; loads only caller-selected training images."""
    def __init__(self, directory, *, expected_manifest_sha256):
        from experiments import entity_retrieval_data as data
        self.work = dict(load_attempts=0, archive_loads=0, remap_attempts=0, remaps=0,
            load_failures=0, remap_failures=0, archive_bytes_read=0, archive_bytes_loaded=0,
            wall_seconds=0., cpu_seconds=0.)
        started, cpu = time.monotonic(), time.process_time()
        self.failed = True
        try:
            if expected_manifest_sha256 != MANIFEST_SHA256:
                raise ValueError("the fixed already-admitted cache manifest is required")
            self._directory = Path(directory).resolve()
            if not self._directory.is_relative_to(ROOT): raise ValueError("local repository data required")
            self._sources = source_hashes()
            manifest = data.load_manifest(self._directory, expected_manifest_sha256=expected_manifest_sha256)
            if (manifest["updates"] != SOURCE_UPDATES or len(manifest["training"]) != SOURCE_UPDATES
                    or manifest["micro_batch_size"] != 32 or manifest["layout"] != "original"):
                raise ValueError("complete fixed original training cache required")
            self._manifest = deepcopy(manifest)
            self._manifest_identity = _hash(manifest)
            self._parent = (ROOT/manifest["parent_data_directory"]).resolve()
            if not self._parent.is_relative_to(ROOT): raise ValueError("local parent cache required")
            self.failed = False
            self._guard()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.work["wall_seconds"] += time.monotonic()-started
            self.work["cpu_seconds"] += time.process_time()-cpu

    def _guard(self):
        from experiments.foundation_layout_study import digest
        if (self.failed or source_hashes() != self._sources or _hash(self._manifest) != self._manifest_identity
                or digest(self._directory/"manifest.json") != MANIFEST_SHA256
                or digest(self._parent/"manifest.json") != self._manifest["parent_manifest_sha256"]):
            raise ValueError("replay admission/source changed or reader previously failed")

    def load(self, source_index):
        """Authenticate original file image before its single weights-only CPU load."""
        from experiments.foundation_layout_study import native
        started, cpu = time.monotonic(), time.process_time()
        self.work["load_attempts"] += 1
        try:
            self._guard()
            if type(source_index) is not int or not 0 <= source_index < SOURCE_UPDATES:
                raise ValueError("source index must address the original648 images")
            record = self._manifest["training"][source_index]
            if (record["global_cursor"] != source_index or record["path"] != f"training/{source_index:04d}.pt"
                    or type(record["cycle_id"]) is not int or not 0 <= record["cycle_id"] < 3
                    or type(record["canonical_bundle_id"]) is not int or not 0 <= record["canonical_bundle_id"] < 216):
                raise ValueError("original manifest source coordinates differ")
            path = (self._parent/record["path"]).resolve()
            if not path.is_relative_to(self._parent): raise ValueError("archive escapes admitted parent")
            raw = native(path).read_bytes()
            self.work["archive_bytes_read"] += len(raw)
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("original archive bytes differ before deserialization")
            import torch
            image = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            self.work["archive_loads"] += 1
            self.work["archive_bytes_loaded"] += len(raw)
            if (type(image) is not dict or set(image) != {"bundle", "expected_evidence", "expected_evidence_sha256"}
                    or image["expected_evidence_sha256"] != _hash(image["expected_evidence"])
                    or image["bundle"]["bundle_id"] != source_index
                    or image["expected_evidence"]["bundle_id"] != source_index):
                raise ValueError("original image and admitted evidence disagree")
            self._guard()
            return ReplaySource(_MINT, self, record, image)
        except BaseException:
            self.work["load_failures"] += 1
            self.failed = True
            raise
        finally:
            self.work["wall_seconds"] += time.monotonic()-started
            self.work["cpu_seconds"] += time.process_time()-cpu

    def report(self): return dict(**deepcopy(self.work), failed=self.failed)


class ReplaySource:
    """A loaded original image. Caller copies cannot alter its admitted identity."""
    def __init__(self, mint, reader, record, image):
        if mint is not _MINT: raise TypeError("ReplayReader must authenticate original bytes first")
        self._reader, self._record, self._image = reader, deepcopy(record), image
        self._identity = _hash([record, image])

    def _guard(self):
        self._reader._guard()
        if _hash([self._record, self._image]) != self._identity:
            raise ValueError("loaded source image or original provenance was mutated")

    @property
    def original_image(self):
        self._guard()
        return deepcopy(self._image)

    def remap(self, global_cursor):
        from experiments.foundation_layout_prepared import evidence_sha256
        started, cpu = time.monotonic(), time.process_time()
        self._reader.work["remap_attempts"] += 1
        try:
            self._guard()
            if (type(global_cursor) is not int or not START <= global_cursor < STOP
                    or global_cursor % SOURCE_UPDATES != self._record["global_cursor"]):
                raise ValueError("fixed replay pass/source index and lifetime cursor required")
            image = deepcopy(self._image)
            image["bundle"]["bundle_id"] = image["expected_evidence"]["bundle_id"] = global_cursor
            image["expected_evidence_sha256"] = _hash(image["expected_evidence"])
            provenance = dict(schema=SCHEMA, source_manifest_sha256=MANIFEST_SHA256,
                source_manifest_path=(self._reader._directory/"manifest.json").relative_to(ROOT).as_posix(),
                parent_manifest_sha256=self._reader._manifest["parent_manifest_sha256"],
                original_archive_path=(self._reader._parent/self._record["path"]).relative_to(ROOT).as_posix(),
                original_archive_sha256=self._record["sha256"], source_index=self._record["global_cursor"],
                canonical_cycle_id=self._record["cycle_id"], canonical_bundle_id=self._record["canonical_bundle_id"],
                replay_pass=global_cursor//SOURCE_UPDATES, consumed_global_id=global_cursor,
                original_evidence_sha256=self._image["expected_evidence_sha256"],
                replay_evidence_sha256=image["expected_evidence_sha256"],
                prepared_evidence_sha256=evidence_sha256(image["expected_evidence"]))
            self._reader.work["remaps"] += 1
            return image, provenance
        except BaseException:
            self._reader.work["remap_failures"] += 1
            raise
        finally:
            self._reader.work["wall_seconds"] += time.monotonic()-started
            self._reader.work["cpu_seconds"] += time.process_time()-cpu

    def __reduce_ex__(self, protocol):
        raise TypeError("authenticated replay sources are transient; persist original archive/provenance")
