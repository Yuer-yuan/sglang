from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import threading
from typing import Any, Iterable, Iterator, Protocol, Sequence

from sglang.multi_model.uma.model_adapter import GroupValidation, ModelAdapter
from sglang.multi_model.uma.weight_plan import TensorExtent, WeightGroupSpec


class WeightRuntimeError(RuntimeError):
    pass


class GroupValidationError(WeightRuntimeError):
    def __init__(self, group_id: str, validation: GroupValidation) -> None:
        self.group_id = group_id
        self.validation = validation
        super().__init__(
            f"weight group {group_id} failed validation: "
            f"{', '.join(validation.errors)}"
        )


class PinnedWeightConflict(WeightRuntimeError):
    pass


class StaleWeightEpoch(WeightRuntimeError):
    pass


class WeightResidencyState(str, Enum):
    LOADING = "LOADING"
    RESIDENT_EVICTABLE = "RESIDENT_EVICTABLE"
    PINNED = "PINNED"
    EVICTING = "EVICTING"


class WeightReservation(Protocol):
    @property
    def active(self) -> bool: ...

    @property
    def final_bytes(self) -> int: ...

    @property
    def staging_bytes(self) -> int: ...

    def commit(self, resident_bytes: int) -> None: ...

    def release(self) -> None: ...


class LeaseHandle(Protocol):
    @property
    def active(self) -> bool: ...

    def release(self) -> None: ...


class LeaseBackend(Protocol):
    def acquire(self, resources: tuple[Any, ...], owner: str) -> LeaseHandle: ...

    def is_evictable(self, resource: Any) -> bool: ...


class AllocatorControl(Protocol):
    def synchronize(self) -> None: ...

    def allocated_bytes(self) -> int: ...

    def reserved_bytes(self) -> int: ...

    def trim(self) -> None: ...


class WeightGroupReader(Protocol):
    def iter_group(
        self,
        group: WeightGroupSpec,
        staging_bytes: int,
    ) -> Iterable[tuple[str, Any]]: ...


@dataclass(slots=True)
class LoadedWeightGroup:
    identity: Any
    group: WeightGroupSpec
    adapter: ModelAdapter
    module: Any
    state: WeightResidencyState
    parameter_bytes: int
    resident_bytes: int
    resource_epoch: int


@dataclass(frozen=True, slots=True)
class WeightRelease:
    identity: Any
    group_id: str
    parameter_bytes: int
    released_bytes: int
    allocator_reserved_released_bytes: int
    resource_epoch: int


class PinnedWeightLease:
    def __init__(
        self,
        runtime: "WeightRuntime",
        lease: LeaseHandle,
        identities: tuple[Any, ...],
    ) -> None:
        self._runtime = runtime
        self._lease = lease
        self.identities = identities

    @property
    def active(self) -> bool:
        return self._lease.active

    def release(self) -> None:
        if not self._lease.active:
            return
        self._lease.release()
        self._runtime._refresh_pin_states(self.identities)

    def __enter__(self) -> "PinnedWeightLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class SafetensorsExtentReader:
    """Read one immutable tensor extent at a time into bounded CPU staging."""

    def __init__(self) -> None:
        self.read_count = 0
        self.bytes_read = 0
        self._lock = threading.Lock()

    def iter_group(
        self,
        group: WeightGroupSpec,
        staging_bytes: int,
    ) -> Iterator[tuple[str, Any]]:
        if staging_bytes < 0:
            raise ValueError("staging_bytes must be non-negative")
        largest = max(extent.nbytes for extent in group.tensors)
        if largest > staging_bytes:
            raise WeightRuntimeError(
                f"weight group {group.group_id} needs {largest} staging bytes, "
                f"reservation provides {staging_bytes}"
            )
        for extent in group.tensors:
            buffer = self._read_extent(extent)
            yield extent.name, self._as_tensor(extent, buffer)

    def _read_extent(self, extent: TensorExtent) -> bytearray:
        buffer = bytearray(extent.nbytes)
        view = memoryview(buffer)
        total = 0
        with Path(extent.file).open("rb", buffering=0) as stream:
            stream.seek(extent.offset)
            while total < extent.nbytes:
                read = stream.readinto(view[total:])
                if not read:
                    raise WeightRuntimeError(
                        f"short read for {extent.name}: "
                        f"expected {extent.nbytes}, observed {total}"
                    )
                total += read
        checksum = "sha256:" + hashlib.sha256(buffer).hexdigest()
        if checksum != extent.checksum:
            raise WeightRuntimeError(
                f"checksum mismatch for {extent.name}: "
                f"expected {extent.checksum}, observed {checksum}"
            )
        with self._lock:
            self.read_count += 1
            self.bytes_read += extent.nbytes
        return buffer

    @staticmethod
    def _as_tensor(extent: TensorExtent, buffer: bytearray) -> Any:
        import torch

        dtype_names = {
            "BOOL": "bool",
            "U8": "uint8",
            "I8": "int8",
            "I16": "int16",
            "U16": "uint16",
            "F16": "float16",
            "BF16": "bfloat16",
            "I32": "int32",
            "U32": "uint32",
            "F32": "float32",
            "I64": "int64",
            "U64": "uint64",
            "F64": "float64",
            "C64": "complex64",
            "F8_E5M2": "float8_e5m2",
            "F8_E4M3": "float8_e4m3fn",
        }
        dtype_name = dtype_names.get(extent.dtype)
        dtype = getattr(torch, dtype_name, None) if dtype_name else None
        if dtype is None:
            raise WeightRuntimeError(
                f"unsupported staging dtype for {extent.name}: {extent.dtype}"
            )
        tensor = torch.frombuffer(buffer, dtype=dtype)
        return tensor.reshape(extent.shape)


class TorchCUDAAllocator:
    def __init__(self, device: Any) -> None:
        self.device = device

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize(self.device)

    def allocated_bytes(self) -> int:
        import torch

        return torch.cuda.memory_allocated(self.device)

    def reserved_bytes(self) -> int:
        import torch

        return torch.cuda.memory_reserved(self.device)

    def trim(self) -> None:
        import torch

        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()


class WeightRuntime:
    """Own materialized model storage and enforce data-plane safety leases."""

    def __init__(
        self,
        *,
        reader: WeightGroupReader,
        lease_table: LeaseBackend,
        device: Any,
        allocator: AllocatorControl | None = None,
    ) -> None:
        self.reader = reader
        self.lease_table = lease_table
        self.device = device
        self.allocator = allocator or TorchCUDAAllocator(device)
        self._records: dict[Any, LoadedWeightGroup] = {}
        self._loading: set[tuple[str, str, int]] = set()
        self._lock = threading.RLock()
        # CUDA allocator deltas are only attributable while materialization is
        # serialized.  SSD prefetch may run in parallel before this section.
        self._allocation_lock = threading.Lock()

    def load_group(
        self,
        identity: Any,
        adapter: ModelAdapter,
        module: Any,
        group: WeightGroupSpec,
        reservation: WeightReservation,
    ) -> LoadedWeightGroup:
        key: tuple[str, str, int] | None = None
        expected_bytes = 0
        materialized = False
        allocated_before = 0
        try:
            key = self._logical_key(identity, group.group_id)
            expected_bytes = adapter.group_storage_bytes(module, group)
            if reservation.final_bytes < expected_bytes:
                raise WeightRuntimeError(
                    f"weight group {group.group_id} needs {expected_bytes} "
                    f"resident bytes, reservation provides "
                    f"{reservation.final_bytes}"
                )
            largest_tensor = max(extent.nbytes for extent in group.tensors)
            if reservation.staging_bytes < largest_tensor:
                raise WeightRuntimeError(
                    f"weight group {group.group_id} needs {largest_tensor} "
                    f"staging bytes, reservation provides "
                    f"{reservation.staging_bytes}"
                )
            if not adapter.group_is_meta(module, group):
                raise WeightRuntimeError(
                    f"weight group {group.group_id} target storage is already "
                    "materialized"
                )
            with self._lock:
                if key in self._loading or any(
                    self._logical_key(record.identity, record.group.group_id)
                    == key
                    for record in self._records.values()
                ):
                    raise WeightRuntimeError(
                        f"weight group is already loading or resident: {key}"
                    )
                self._loading.add(key)
            with self._allocation_lock:
                self.allocator.synchronize()
                allocated_before = self.allocator.allocated_bytes()
                adapter.materialize_group(module, group, self.device)
                materialized = True
                adapter.load_group(
                    module,
                    group,
                    self.reader.iter_group(group, reservation.staging_bytes),
                )
                validation = adapter.validate_group(module, group)
                if not validation.ok:
                    raise GroupValidationError(group.group_id, validation)
                if validation.resident_bytes != expected_bytes:
                    raise WeightRuntimeError(
                        f"weight group {group.group_id} storage changed during "
                        f"load: expected {expected_bytes}, observed "
                        f"{validation.resident_bytes}"
                    )
                self.allocator.synchronize()
                allocated_after = self.allocator.allocated_bytes()
                allocator_bytes = allocated_after - allocated_before
                if allocator_bytes < validation.resident_bytes:
                    raise WeightRuntimeError(
                        f"weight group {group.group_id} allocator delta "
                        f"{allocator_bytes} is below parameter bytes "
                        f"{validation.resident_bytes}"
                    )
                if allocator_bytes > reservation.final_bytes:
                    raise WeightRuntimeError(
                        f"weight group {group.group_id} allocator uses "
                        f"{allocator_bytes} bytes, reservation provides "
                        f"{reservation.final_bytes}"
                    )
            record = LoadedWeightGroup(
                identity=identity,
                group=group,
                adapter=adapter,
                module=module,
                state=WeightResidencyState.RESIDENT_EVICTABLE,
                parameter_bytes=validation.resident_bytes,
                resident_bytes=allocator_bytes,
                resource_epoch=self._resource_epoch(identity),
            )
            with self._lock:
                self._records[identity] = record
                try:
                    reservation.commit(allocator_bytes)
                except BaseException:
                    del self._records[identity]
                    raise
            return record
        except BaseException:
            if materialized:
                with self._allocation_lock:
                    adapter.release_group(module, group)
                    self.allocator.synchronize()
                    self.allocator.trim()
            if reservation.active:
                reservation.release()
            raise
        finally:
            if key is not None:
                with self._lock:
                    self._loading.discard(key)

    def readiness(self, instance_id: str) -> frozenset[str]:
        with self._lock:
            return frozenset(
                record.group.group_id
                for record in self._records.values()
                if self._instance_id(record.identity) == instance_id
                and record.state
                in {
                    WeightResidencyState.RESIDENT_EVICTABLE,
                    WeightResidencyState.PINNED,
                }
            )

    def pin(
        self,
        group_ids: Sequence[str],
        batch_id: str,
        *,
        instance_id: str | None = None,
    ) -> PinnedWeightLease:
        if not batch_id.strip():
            raise ValueError("batch_id must not be empty")
        if not group_ids:
            raise ValueError("at least one weight group is required")
        with self._lock:
            records = tuple(
                self._resolve(group_id, instance_id=instance_id)
                for group_id in dict.fromkeys(group_ids)
            )
            identities = tuple(record.identity for record in records)
            lease = self.lease_table.acquire(identities, owner=batch_id)
            for record in records:
                record.state = WeightResidencyState.PINNED
        return PinnedWeightLease(self, lease, identities)

    def evict_group(
        self,
        group_id: str,
        expected_epoch: int,
        *,
        instance_id: str | None = None,
    ) -> WeightRelease:
        with self._lock:
            record = self._resolve(group_id, instance_id=instance_id)
            if record.resource_epoch != expected_epoch:
                raise StaleWeightEpoch(
                    f"weight group {group_id} epoch {record.resource_epoch}, "
                    f"expected {expected_epoch}"
                )
            if not self.lease_table.is_evictable(record.identity):
                raise PinnedWeightConflict(f"weight group is pinned: {group_id}")
            record.state = WeightResidencyState.EVICTING
            storage_released = False
            try:
                with self._allocation_lock:
                    self.allocator.synchronize()
                    allocated_before = self.allocator.allocated_bytes()
                    parameter_bytes = record.adapter.release_group(
                        record.module,
                        record.group,
                    )
                    storage_released = True
                    self.allocator.synchronize()
                    allocated_after = self.allocator.allocated_bytes()
                    released = allocated_before - allocated_after
                    reserved_before = self.allocator.reserved_bytes()
                    self.allocator.trim()
                    reserved_after = self.allocator.reserved_bytes()
            except BaseException:
                if storage_released:
                    del self._records[record.identity]
                else:
                    record.state = WeightResidencyState.RESIDENT_EVICTABLE
                raise
            del self._records[record.identity]
            if parameter_bytes != record.parameter_bytes:
                raise WeightRuntimeError(
                    f"weight group {group_id} released {parameter_bytes} parameter "
                    f"bytes, expected {record.parameter_bytes}"
                )
            if released != record.resident_bytes:
                raise WeightRuntimeError(
                    f"weight group {group_id} returned {released} allocator bytes, "
                    f"expected {record.resident_bytes}"
                )
            return WeightRelease(
                identity=record.identity,
                group_id=group_id,
                parameter_bytes=parameter_bytes,
                released_bytes=released,
                allocator_reserved_released_bytes=max(
                    0,
                    reserved_before - reserved_after,
                ),
                resource_epoch=record.resource_epoch,
            )

    def resident_groups(self) -> tuple[LoadedWeightGroup, ...]:
        with self._lock:
            return tuple(self._records.values())

    def _resolve(
        self,
        group_id: str,
        *,
        instance_id: str | None,
    ) -> LoadedWeightGroup:
        candidates = [
            record
            for record in self._records.values()
            if record.group.group_id == group_id
            and (
                instance_id is None
                or self._instance_id(record.identity) == instance_id
            )
        ]
        if not candidates:
            raise KeyError(f"weight group is not resident: {group_id}")
        if len(candidates) != 1:
            raise ValueError(
                f"weight group {group_id} is ambiguous; provide instance_id"
            )
        return candidates[0]

    def _refresh_pin_states(self, identities: tuple[Any, ...]) -> None:
        with self._lock:
            for identity in identities:
                record = self._records.get(identity)
                if record is None:
                    continue
                if self.lease_table.is_evictable(identity):
                    record.state = WeightResidencyState.RESIDENT_EVICTABLE

    @classmethod
    def _logical_key(
        cls,
        identity: Any,
        group_id: str,
    ) -> tuple[str, str, int]:
        return (
            cls._instance_id(identity),
            group_id,
            cls._resource_epoch(identity),
        )

    @staticmethod
    def _instance_id(identity: Any) -> str:
        value = getattr(identity, "instance_id", None)
        if not isinstance(value, str) or not value:
            raise ValueError("weight identity requires a non-empty instance_id")
        return value

    @staticmethod
    def _resource_epoch(identity: Any) -> int:
        value = getattr(identity, "resource_epoch", None)
        if not isinstance(value, int) or value < 0:
            raise ValueError("weight identity requires a non-negative resource_epoch")
        return value
