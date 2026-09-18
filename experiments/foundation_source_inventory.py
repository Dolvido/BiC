"""Fresh, once-per-file checks of an explicitly supplied source inventory.

This guard checks caller-declared bytes and membership, not their origin or any
learner provenance. It is deliberately not integrated into existing learners or
checkpoint schemas. Callers remain responsible for checks before and after an
operation, and for a trustworthy expected map and implementation pin.

Every check reads all expected files anew; only names and expected digests are
retained. Membership uses native, nonrecursive pathlib filename matching. Path
spelling aliases, symlinks, reparse points and nonregular files fail closed, a stricter
policy than the historical source closures. Checks detect drift at boundaries;
they are not an atomic filesystem snapshot or a defense against a compromised
interpreter, monkeypatched code, stale bytecode, or adversarial concurrent edits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading


SCHEMA = "bic-declared-source-inventory-v1"
_IMPLEMENTATION_PATH = Path(__file__).resolve()
_IMPORTED_SHA256 = hashlib.sha256(_IMPLEMENTATION_PATH.read_bytes()).hexdigest()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)


class SourceInventoryError(ValueError):
    """The declared source boundary could not be verified."""


class SourceInventoryPoisoned(SourceInventoryError):
    """A prior check failed; this instance can never become valid again."""


def _name(value, *, directory=False):
    if directory and value == ".":
        return value
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or any(ord(char) < 32 for char in value)):
        raise SourceInventoryError("canonical relative POSIX source name required")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise SourceInventoryError("source name is absolute or has an alias")
    if any(part in (".", "..") or part.endswith((".", " "))
           or any(char in part for char in '<>"|?*') or _RESERVED.fullmatch(part)
           for part in value.split("/")):
        raise SourceInventoryError("source name contains a noncanonical component")
    return value


def _digest(value):
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SourceInventoryError("lowercase SHA256 digest required")
    return value


def _plain(path, *, directory):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise SourceInventoryError("source paths must not contain symlinks or reparse points")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise SourceInventoryError("source path has the wrong file type")


def _absolute_directory(path):
    # Inspect ancestors before resolving, so a junction cannot be normalized away.
    if not path.is_absolute() or ".." in path.parts:
        raise SourceInventoryError("absolute canonical source root required")
    for ancestor in (*reversed(path.parents), path):
        _plain(ancestor, directory=True)
    if path.resolve(strict=True).as_posix() != path.as_posix():
        raise SourceInventoryError("source root spelling is not canonical")


def _relative_path(root, name, *, directory=False):
    path = root
    parts = () if name == "." else PurePosixPath(name).parts
    for index, part in enumerate(parts):
        path /= part
        _plain(path, directory=directory or index < len(parts) - 1)
    resolved = path.resolve(strict=True)
    if resolved.as_posix() != path.as_posix() or not resolved.is_relative_to(root):
        raise SourceInventoryError("source path escaped its root or has an alias")
    return path


@dataclass(frozen=True, slots=True)
class SourceMembership:
    """A directory and one nonrecursive native filename glob, possibly literal."""

    directory: str
    pattern: str

    def __post_init__(self):
        _name(self.directory, directory=True)
        if (type(self.pattern) is not str or not self.pattern
                or self.pattern in (".", "..") or "**" in self.pattern
                or any(char in self.pattern for char in '/\\:<>"|')
                or any(ord(char) < 32 for char in self.pattern)
                or self.pattern.endswith((".", " "))):
            raise SourceInventoryError("one nonrecursive filename pattern required")

    def includes(self, name):
        path = PurePosixPath(name)
        return path.parent.as_posix() == self.directory and Path(path.name).match(self.pattern)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SourceInventory:
    """Nonserializable process-owned expectations; failures poison this instance.

    Construction performs one complete check. ``expected_sha256`` must be a
    plain dict, copied before validation. Membership specs must be explicit;
    an empty tuple declares that the caller has no dynamic membership rules.
    A separately pinned guard is read once in addition to the logical closure,
    unless the exact implementation file is already among its expected files.
    """

    _root: Path
    _expected: tuple[tuple[str, str], ...]
    _memberships: tuple[SourceMembership, ...]
    _guard_sha256: str
    _guard_name: str | None
    _pid: int
    _poison: str | None
    _lock: object

    def __init__(self, root, expected_sha256, *, memberships, guard_sha256):
        if type(self) is not SourceInventory:
            raise SourceInventoryError("exact SourceInventory type required")
        if type(expected_sha256) is not dict or not expected_sha256:
            raise SourceInventoryError("explicit nonempty plain expected source dict required")
        expected = expected_sha256.copy()
        if type(memberships) not in (tuple, list):
            raise SourceInventoryError("explicit list or tuple of membership specifications required")
        membership = tuple(memberships)
        if any(type(spec) is not SourceMembership for spec in membership):
            raise SourceInventoryError("exact SourceMembership specifications required")
        if len(set(membership)) != len(membership):
            raise SourceInventoryError("duplicate membership specification")
        for name, digest in expected.items():
            _name(name)
            _digest(digest)
        if len({name.casefold() for name in expected}) != len(expected):
            raise SourceInventoryError("case-aliased source names are ambiguous")
        _digest(guard_sha256)
        if guard_sha256 != _IMPORTED_SHA256:
            raise SourceInventoryError("caller guard pin differs from imported implementation identity")
        root = Path(root)
        _absolute_directory(root)
        guard_name = (_IMPLEMENTATION_PATH.relative_to(root).as_posix()
                      if _IMPLEMENTATION_PATH.is_relative_to(root) else None)
        if guard_name in expected and expected[guard_name] != guard_sha256:
            raise SourceInventoryError("logical source map and guard pin disagree")
        for key, value in dict(_root=root, _expected=tuple(sorted(expected.items())),
                _memberships=membership, _guard_sha256=guard_sha256,
                _guard_name=guard_name if guard_name in expected else None,
                _pid=os.getpid(), _poison=None, _lock=threading.RLock()).items():
            object.__setattr__(self, key, value)
        self.check()

    @property
    def expected_sha256(self):
        """Detached expectations, not a claim that a check is current or passed."""
        return dict(self._expected)

    @property
    def memberships(self):
        return self._memberships

    @property
    def poisoned(self):
        return self._poison is not None

    @property
    def declaration(self):
        """Detached declaration, never a validation result or origin receipt."""
        return dict(schema=SCHEMA, expected_sha256=self.expected_sha256,
            memberships=[dict(directory=spec.directory, pattern=spec.pattern)
                         for spec in self._memberships], guard_sha256=self._guard_sha256,
            scope="Caller-declared bytes and membership only; not origin or learner provenance.")

    def __reduce__(self):
        raise TypeError("source inventories are process-owned and cannot be serialized")

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __copy__(self):
        raise TypeError("source inventories cannot be copied")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def _scan_membership(self):
        for spec in self._memberships:
            directory = _relative_path(self._root, spec.directory, directory=True)
            # scandir propagates unreadable-directory errors; glob can suppress
            # those errors. Path.match supplies native case/glob semantics and
            # intentionally includes hidden and nonregular matching entries.
            with os.scandir(directory) as entries:
                actual = {((PurePosixPath(spec.directory) / entry.name).as_posix())
                          for entry in entries if Path(entry.name).match(spec.pattern)}
            expected = {name for name, _ in self._expected if spec.includes(name)}
            if actual != expected:
                raise SourceInventoryError("declared source membership changed: "
                    + spec.directory + "/" + spec.pattern)

    def check(self):
        """Return a detached logical map after fresh checks, or poison on failure."""
        with self._lock:
            if self._poison is not None:
                raise SourceInventoryPoisoned("source inventory is poisoned: " + self._poison)
            try:
                if os.getpid() != self._pid:
                    raise SourceInventoryError("source inventory belongs to a different process")
                _absolute_directory(self._root)
                if self._guard_name is None:
                    _absolute_directory(_IMPLEMENTATION_PATH.parent)
                    _plain(_IMPLEMENTATION_PATH, directory=False)
                    if hashlib.sha256(_IMPLEMENTATION_PATH.read_bytes()).hexdigest() != self._guard_sha256:
                        raise SourceInventoryError("source guard implementation changed")
                self._scan_membership()
                actual = {}
                for name, expected in self._expected:
                    path = _relative_path(self._root, name)
                    actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual[name] != expected:
                        raise SourceInventoryError("declared source bytes changed: " + name)
                self._scan_membership()
                return actual
            except BaseException as error:
                # Interrupted/failed checks cannot be silently retried on this
                # object, even if the filesystem is repaired afterwards.
                object.__setattr__(self, "_poison", type(error).__name__ + ": " + str(error))
                raise
