"""Bounded, source-pinned JSON admission cache; no learner or provider imports.

The caller authenticates the parent manifest and supplies its exact JSON record
list and admission function. A miss authenticates bytes and admits once. A hit
returns the previously admitted immutable snapshot, not a fresh view of disk.
verify_sources() explicitly checks resident files at a caller-owned boundary.
No cursor changes, packing, semantic generation or tensor archive handling.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

SCHEMA = 'bic-immutable-lesson-cache-v1'


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def identity(value):
    """Canonical semantic JSON digest; file byte digests remain separate."""
    return hashlib.sha256(_encoded(value)).hexdigest()


def _pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')
    return value


def _native(path):
    value = str(path)
    return Path('\\\\?\\' + value if os.name == 'nt' and not value.startswith('\\\\?\\') else value)


class _Object(tuple):
    """Immutable JSON object, distinguished from an array tuple."""


def _freeze(value):
    if type(value) is dict:
        if any(type(k) is not str for k in value): raise ValueError('JSON string keys required')
        return _Object((k, _freeze(v)) for k, v in value.items())
    if type(value) is list: return tuple(_freeze(v) for v in value)
    if value is None or type(value) in (str, bool, int, float): return value
    raise ValueError('plain JSON values required')


def _thaw(value):
    if type(value) is _Object: return {k: _thaw(v) for k, v in value}
    if type(value) is tuple: return [_thaw(v) for v in value]
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError('duplicate JSON key')
        result[key] = value
    return result


class ImmutableLessonCache:
    """Private immutable entries with detached mutable results and honest costs.

    admission(image, record) must reject invalid content by raising. Its source
    file must be included in admission_source_sha256; callers supply the full
    dependency closure. Callback side effects cannot be sandboxed by this cache.
    Callback attempts/completions are counted; detailed oracle work, if any,
    remains in the callback's own ledger. Source code is authenticated at cache
    construction, on misses and at explicit boundaries, not on cache hits.
    """
    def __init__(self, root, *, records, records_sha256, admission,
                 admission_source_sha256, max_images=64, max_serialized_bytes=128*1024*1024):
        if type(max_images) is not int or not 1 <= max_images <= 4096:
            raise ValueError('cache image bound1..4096 required')
        if type(max_serialized_bytes) is not int or not 1 <= max_serialized_bytes <= 2**30:
            raise ValueError('cache serialized byte bound1..1GiB required')
        if type(records) is not list or not records or identity(records) != _pin(records_sha256):
            raise ValueError('exact caller-pinned admitted record list required')
        self.root = Path(root).resolve()
        self.max_images, self.max_serialized_bytes = max_images, max_serialized_bytes
        self.records_sha256 = records_sha256
        self._records = {}
        for record in records:
            if (type(record) is not dict or record.get('format') != 'json'
                    or type(record.get('provider')) is not str or not record['provider']
                    or type(record.get('index')) is not int or record['index'] < 0):
                raise ValueError('admitted indexed JSON provider record required')
            self._path(record['path']); _pin(record['sha256'])
            self._records[identity(record)] = _freeze(record)
        if not inspect.isfunction(admission) or admission.__closure__:
            raise ValueError('named source-defined admission function without closure required')
        if type(admission_source_sha256) is not dict or not admission_source_sha256:
            raise ValueError('explicit admission source closure required')
        self._sources = tuple(sorted((name, _pin(pin)) for name, pin in admission_source_sha256.items()))
        actual_file = Path(inspect.getsourcefile(admission)).resolve()
        if actual_file not in {self._path(name, suffix=None) for name, _ in self._sources}:
            raise ValueError('admission function source missing from closure')
        self._admission = admission
        self.admission_name = admission.__module__ + '.' + admission.__qualname__
        self._cache, self._bytes, self.failed = OrderedDict(), 0, False
        self._work = dict(get_attempts=0, gets=0, cache_hits=0, cache_misses=0,
            json_read_attempts=0, json_reads=0, json_bytes=0, decode_attempts=0, decodes=0,
            admission_attempts=0, admissions=0, admission_mutation_checks=0,
            source_file_hashes=0, source_bytes=0, verify_attempts=0, verifications=0,
            resident_file_checks=0, resident_check_bytes=0, detached_copies=0,
            evictions=0, peak_images=0, peak_serialized_bytes=0, failures=0,
            callback_partial_work_unknown=False, wall_seconds=0., cpu_seconds=0.)
        self._timed(self._guard)

    def _path(self, name, suffix='.json'):
        if type(name) is not str or not name or Path(name).is_absolute():
            raise ValueError('root-relative path required')
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or (suffix and path.suffix.lower() != suffix):
            raise ValueError('path must remain under root with required suffix')
        return path

    def _guard(self):
        if self.failed: raise ValueError('failed cache cannot be reused')
        for name, pin in self._sources:
            raw = _native(self._path(name, suffix=None)).read_bytes()
            self._work['source_file_hashes'] += 1; self._work['source_bytes'] += len(raw)
            if hashlib.sha256(raw).hexdigest() != pin: raise ValueError('admission source changed')

    def _timed(self, operation):
        wall, cpu = time.monotonic(), time.process_time()
        try: return operation()
        except BaseException:
            self.failed = True; self._work['failures'] += 1; raise
        finally:
            self._work['wall_seconds'] += time.monotonic() - wall
            self._work['cpu_seconds'] += time.process_time() - cpu

    def _read_bytes(self, record, *, resident=False):
        # A finite read also limits a source replaced by a much larger file.
        with _native(self._path(record['path'])).open('rb') as handle:
            raw = handle.read(self.max_serialized_bytes + 1)
        self._work['resident_check_bytes' if resident else 'json_bytes'] += len(raw)
        if len(raw) > self.max_serialized_bytes: raise ValueError('lesson exceeds cache byte bound')
        if hashlib.sha256(raw).hexdigest() != record['sha256']: raise ValueError('lesson file bytes changed')
        return raw

    def get(self, record):
        """Return exact original lesson coordinates; caller owns cursor remap."""
        self._work['get_attempts'] += 1
        def operation():
            if self.failed: raise ValueError('failed cache cannot be reused')
            key = identity(record)
            if key not in self._records: raise ValueError('record outside pinned catalogue')
            trusted = _thaw(self._records[key])
            if key in self._cache:
                self._work['cache_hits'] += 1; self._cache.move_to_end(key)
                frozen, size = self._cache[key]
            else:
                self._guard()
                self._work['cache_misses'] += 1; self._work['json_read_attempts'] += 1
                raw = self._read_bytes(trusted)
                self._work['json_reads'] += 1
                self._work['decode_attempts'] += 1
                image = json.loads(raw, object_pairs_hook=_pairs)
                if type(image) is not dict: raise ValueError('JSON lesson object required')
                original = identity(image)  # rejects nonfinite JSON numbers
                self._work['decodes'] += 1; self._work['admission_attempts'] += 1
                self._work['callback_partial_work_unknown'] = True
                self._admission(image, trusted)
                self._work['callback_partial_work_unknown'] = False
                self._work['admission_mutation_checks'] += 1
                if identity(image) != original or identity(trusted) != key:
                    raise ValueError('admission callback mutated lesson or record')
                self._work['admissions'] += 1; self._guard()
                frozen, size = _freeze(image), len(raw)
                while len(self._cache) >= self.max_images or self._bytes + size > self.max_serialized_bytes:
                    _, (_, old_size) = self._cache.popitem(last=False)
                    self._bytes -= old_size; self._work['evictions'] += 1
                self._cache[key] = (frozen, size); self._bytes += size
                self._work['peak_images'] = max(self._work['peak_images'], len(self._cache))
                self._work['peak_serialized_bytes'] = max(self._work['peak_serialized_bytes'], self._bytes)
            result = _thaw(frozen)
            self._work['detached_copies'] += 1; self._work['gets'] += 1
            return result
        return self._timed(operation)

    def verify_sources(self):
        """Explicitly rehash resident lesson files; no decode or readmission."""
        self._work['verify_attempts'] += 1
        def operation():
            self._guard()
            for key in self._cache:
                self._read_bytes(_thaw(self._records[key]), resident=True)
                self._work['resident_file_checks'] += 1
            self._work['verifications'] += 1
        return self._timed(operation)

    def report(self):
        return dict(schema=SCHEMA, **self._work, failed=self.failed,
            records_sha256=self.records_sha256, admission=self.admission_name,
            admission_source_sha256=dict(self._sources), cached_images=len(self._cache),
            cached_serialized_bytes=self._bytes, max_images=self.max_images,
            max_serialized_bytes=self.max_serialized_bytes,
            scope='Pinned byte snapshots, not live disk views. Serialized bytes are not RSS. '
                  'Callback work is counted by invocations; detailed semantic work remains caller-owned. '
                  'No cursor remapping, archive support, packing or performance claim.')


def source_hashes():
    return {'experiments/immutable_lesson_cache.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
