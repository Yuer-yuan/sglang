"""Stable identities and ownership facts for durable session KV.

This module deliberately has no torch or CUDA dependency.  It models which
session *branch* owns each logical KV index; physical mapping and persistence
are supplied by kvcached and LocalKVStore in later layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_non_negative(**values: int) -> None:
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _require_range(name: str, value: tuple[int, int]) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise ValueError(f"{name} must be a non-empty half-open integer range")


def _require_shape(name: str, value: tuple[int, ...]) -> None:
    if not value or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value
    ):
        raise ValueError(f"{name} must contain only positive dimensions")


class KVRangeState(str, Enum):
    ABSENT = "ABSENT"
    ALLOCATING = "ALLOCATING"
    RESIDENT_RECLAIMABLE = "RESIDENT_RECLAIMABLE"
    PINNED = "PINNED"
    OFFLOADING = "OFFLOADING"
    ON_LOCAL_SSD = "ON_LOCAL_SSD"
    RESTORING = "RESTORING"
    PROVISIONAL = "PROVISIONAL"


class KVBranchNotFoundError(KeyError):
    pass


class PinnedKVRangeError(RuntimeError):
    pass


class StaleKVEpochError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, order=True)
class SessionBranchKey:
    instance_id: str
    stage_id: str
    session_id: str
    request_id: str

    def __post_init__(self) -> None:
        _require_text("instance_id", self.instance_id)
        _require_text("stage_id", self.stage_id)
        _require_text("session_id", self.session_id)
        _require_text("request_id", self.request_id)


@dataclass(frozen=True, slots=True)
class KVExtent:
    instance_id: str
    model_digest: str
    placement_version: int
    stage_id: str
    session_id: str
    request_id: str
    predecessor_request_id: str | None
    kv_epoch: int
    token_position: int
    token_range: tuple[int, int]
    local_layer_range: tuple[int, int]
    dtype: str
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    nbytes: int
    checksum: str

    def __post_init__(self) -> None:
        for name in (
            "instance_id",
            "model_digest",
            "stage_id",
            "session_id",
            "request_id",
            "dtype",
            "checksum",
        ):
            _require_text(name, getattr(self, name))
        if self.predecessor_request_id is not None:
            _require_text("predecessor_request_id", self.predecessor_request_id)
            if self.predecessor_request_id == self.request_id:
                raise ValueError("a request cannot be its own predecessor")
        _require_non_negative(
            placement_version=self.placement_version,
            kv_epoch=self.kv_epoch,
            token_position=self.token_position,
            nbytes=self.nbytes,
        )
        if self.nbytes == 0:
            raise ValueError("nbytes must be positive")
        _require_range("token_range", self.token_range)
        _require_range("local_layer_range", self.local_layer_range)
        if self.token_range[1] > self.token_position:
            raise ValueError("token_range extends beyond token_position")
        _require_shape("k_shape", self.k_shape)
        _require_shape("v_shape", self.v_shape)

    @property
    def branch_key(self) -> SessionBranchKey:
        return SessionBranchKey(
            self.instance_id,
            self.stage_id,
            self.session_id,
            self.request_id,
        )


@dataclass(frozen=True, slots=True)
class SessionKVDescriptor:
    key: SessionBranchKey
    predecessor_request_id: str | None
    instance_id: str
    model_digest: str
    placement_version: int
    stage_id: str
    kv_epoch: int
    token_ids: tuple[int, ...]
    kv_indices: tuple[int, ...]
    token_position: int
    logical_bytes: int

    def __post_init__(self) -> None:
        for name in ("instance_id", "model_digest", "stage_id"):
            _require_text(name, getattr(self, name))
        if self.key.instance_id != self.instance_id:
            raise ValueError("branch key instance_id does not match descriptor")
        if self.key.stage_id != self.stage_id:
            raise ValueError("branch key stage_id does not match descriptor")
        if self.predecessor_request_id is not None:
            _require_text("predecessor_request_id", self.predecessor_request_id)
            if self.predecessor_request_id == self.key.request_id:
                raise ValueError("a request cannot be its own predecessor")
        _require_non_negative(
            placement_version=self.placement_version,
            kv_epoch=self.kv_epoch,
            token_position=self.token_position,
            logical_bytes=self.logical_bytes,
        )
        if len(self.token_ids) != len(self.kv_indices):
            raise ValueError("token_ids and kv_indices must have equal length")
        if self.token_position < len(self.token_ids):
            raise ValueError("token_position precedes the cached token range")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in (*self.token_ids, *self.kv_indices)
        ):
            raise ValueError("token ids and KV indices must be non-negative integers")


@dataclass(frozen=True, slots=True)
class FrozenSessionKV:
    descriptor: SessionKVDescriptor
    shared_indices: tuple[int, ...]
    exclusive_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DetachReport:
    key: SessionBranchKey
    kv_epoch: int
    logical_bytes: int
    detached_indices: tuple[int, ...]
    actually_reclaimable_indices: tuple[int, ...]
    shared_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SessionKVSnapshot:
    descriptor: SessionKVDescriptor
    state: KVRangeState
    executing: bool
    lease_count: int


@dataclass(slots=True)
class _BranchRecord:
    descriptor: SessionKVDescriptor
    state: KVRangeState = KVRangeState.RESIDENT_RECLAIMABLE
    executing: bool = False
    lease_count: int = 0

    def snapshot(self) -> SessionKVSnapshot:
        return SessionKVSnapshot(
            descriptor=self.descriptor,
            state=self.state,
            executing=self.executing,
            lease_count=self.lease_count,
        )


class SessionKVRegistry:
    """Track logical branch ownership independently from physical KV pages."""

    def __init__(self) -> None:
        self._branches: dict[SessionBranchKey, _BranchRecord] = {}
        self._index_owners: dict[
            tuple[str, str, int], set[SessionBranchKey]
        ] = {}
        self._lock = threading.RLock()

    def attach(self, descriptor: SessionKVDescriptor) -> None:
        with self._lock:
            self._attach(descriptor)

    def attach_restored(
        self,
        descriptor: SessionKVDescriptor,
        *,
        expected_kv_epoch: int,
    ) -> None:
        with self._lock:
            if descriptor.kv_epoch != expected_kv_epoch:
                raise StaleKVEpochError(
                    f"restored KV epoch {descriptor.kv_epoch} does not match "
                    f"expected epoch {expected_kv_epoch}"
                )
            self._attach(descriptor)

    def _attach(self, descriptor: SessionKVDescriptor) -> None:
        key = descriptor.key
        if key in self._branches:
            raise ValueError(f"session branch is already resident: {key}")
        if len(set(descriptor.kv_indices)) != len(descriptor.kv_indices):
            raise ValueError("duplicate KV indices inside one branch")
        self._branches[key] = _BranchRecord(descriptor)
        for index in descriptor.kv_indices:
            index_key = self._index_key(descriptor, index)
            self._index_owners.setdefault(index_key, set()).add(key)

    def set_execution_state(
        self,
        key: SessionBranchKey,
        *,
        executing: bool,
        lease_count: int,
    ) -> None:
        _require_non_negative(lease_count=lease_count)
        with self._lock:
            record = self._require(key)
            record.executing = executing
            record.lease_count = lease_count
            record.state = (
                KVRangeState.PINNED
                if executing or lease_count
                else KVRangeState.RESIDENT_RECLAIMABLE
            )

    def freeze(
        self,
        key: SessionBranchKey,
        *,
        expected_kv_epoch: int | None = None,
    ) -> FrozenSessionKV:
        with self._lock:
            record = self._require(key)
            self._validate_epoch(record, expected_kv_epoch)
            self._require_reclaimable(record)
            shared: list[int] = []
            exclusive: list[int] = []
            for index in record.descriptor.kv_indices:
                index_key = self._index_key(record.descriptor, index)
                owners = self._index_owners.get(index_key)
                if owners is None or key not in owners:
                    raise RuntimeError(f"KV ownership is inconsistent for index {index}")
                (exclusive if len(owners) == 1 else shared).append(index)
            return FrozenSessionKV(
                descriptor=record.descriptor,
                shared_indices=tuple(shared),
                exclusive_indices=tuple(exclusive),
            )

    def detach(self, frozen: FrozenSessionKV) -> DetachReport:
        key = frozen.descriptor.key
        with self._lock:
            record = self._require(key)
            if record.descriptor != frozen.descriptor:
                raise StaleKVEpochError("resident branch changed after it was frozen")
            self._require_reclaimable(record)
            reclaimable: list[int] = []
            shared: list[int] = []
            for index in record.descriptor.kv_indices:
                index_key = self._index_key(record.descriptor, index)
                owners = self._index_owners.get(index_key)
                if owners is None or key not in owners:
                    raise RuntimeError(f"KV ownership is inconsistent for index {index}")
                owners.remove(key)
                if owners:
                    shared.append(index)
                else:
                    reclaimable.append(index)
                    del self._index_owners[index_key]
            del self._branches[key]
            return DetachReport(
                key=key,
                kv_epoch=record.descriptor.kv_epoch,
                logical_bytes=record.descriptor.logical_bytes,
                detached_indices=record.descriptor.kv_indices,
                actually_reclaimable_indices=tuple(reclaimable),
                shared_indices=tuple(shared),
            )

    def snapshot(self, key: SessionBranchKey) -> SessionKVSnapshot:
        with self._lock:
            return self._require(key).snapshot()

    def snapshots(self) -> tuple[SessionKVSnapshot, ...]:
        with self._lock:
            return tuple(
                self._branches[key].snapshot() for key in sorted(self._branches)
            )

    def _require(self, key: SessionBranchKey) -> _BranchRecord:
        try:
            return self._branches[key]
        except KeyError as exc:
            raise KVBranchNotFoundError(key) from exc

    @staticmethod
    def _index_key(
        descriptor: SessionKVDescriptor,
        index: int,
    ) -> tuple[str, str, int]:
        return descriptor.instance_id, descriptor.stage_id, index

    @staticmethod
    def _validate_epoch(
        record: _BranchRecord,
        expected_kv_epoch: int | None,
    ) -> None:
        if (
            expected_kv_epoch is not None
            and record.descriptor.kv_epoch != expected_kv_epoch
        ):
            raise StaleKVEpochError(
                f"resident KV epoch {record.descriptor.kv_epoch} does not match "
                f"expected epoch {expected_kv_epoch}"
            )

    @staticmethod
    def _require_reclaimable(record: _BranchRecord) -> None:
        if record.executing or record.lease_count:
            raise PinnedKVRangeError(
                f"session branch is pinned: executing={record.executing}, "
                f"lease_count={record.lease_count}"
            )
        if record.state is not KVRangeState.RESIDENT_RECLAIMABLE:
            raise PinnedKVRangeError(
                f"session branch is not reclaimable: {record.state.value}"
            )
