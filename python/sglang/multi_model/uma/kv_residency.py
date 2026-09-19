"""Worker-local transactional KV offload and no-recompute restore.

The runtime deliberately depends on a narrow adapter instead of SGLang's
concrete CUDA pools.  It owns transaction ordering and durable state; the
adapter owns framework-specific gather/scatter, RadixCache publication, and
allocator/VMM release.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import threading
from typing import Iterable, Protocol

from sglang.multi_model.uma.kv_domain import (
    FrozenSessionKV,
    KVExtent,
    KVRangeState,
    SessionBranchKey,
    SessionKVDescriptor,
    SessionKVRegistry,
)
from sglang.multi_model.uma.local_kv_store import (
    ChecksumMismatch,
    LocalKVStore,
    LocalKVStoreError,
    SessionManifest,
    content_checksum,
)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_non_negative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_layer_range(value: tuple[int, int]) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise ValueError("layer range must be a non-empty half-open integer range")


class RestorePolicy(str, Enum):
    FULL_BARRIER = "FULL_BARRIER"
    PROGRESSIVE_LAYER_GROUP = "PROGRESSIVE_LAYER_GROUP"
    ON_DEMAND_PAGE = "ON_DEMAND_PAGE"


@dataclass(frozen=True, slots=True)
class KVOffloadCommand:
    key: SessionBranchKey
    expected_kv_epoch: int
    operation_id: str
    layer_ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _require_non_negative("expected_kv_epoch", self.expected_kv_epoch)
        _require_text("operation_id", self.operation_id)
        if not self.layer_ranges:
            raise ValueError("offload requires at least one local layer range")
        for layer_range in self.layer_ranges:
            _require_layer_range(layer_range)
        ordered = tuple(sorted(self.layer_ranges))
        if ordered != self.layer_ranges:
            raise ValueError("local layer ranges must be ordered")
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] > current[0]:
                raise ValueError("local layer ranges must not overlap")


@dataclass(frozen=True, slots=True)
class KVRestoreCommand:
    key: SessionBranchKey
    expected_kv_epoch: int
    operation_id: str
    policy: RestorePolicy = RestorePolicy.FULL_BARRIER

    def __post_init__(self) -> None:
        _require_non_negative("expected_kv_epoch", self.expected_kv_epoch)
        _require_text("operation_id", self.operation_id)


@dataclass(frozen=True, slots=True)
class ExportedKVRange:
    local_layer_range: tuple[int, int]
    dtype: str
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    chunks: Iterable[bytes | bytearray | memoryview]

    def __post_init__(self) -> None:
        _require_layer_range(self.local_layer_range)
        _require_text("dtype", self.dtype)
        for name, shape in (("k_shape", self.k_shape), ("v_shape", self.v_shape)):
            if not shape or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0
                for item in shape
            ):
                raise ValueError(f"{name} must contain positive dimensions")


@dataclass(frozen=True, slots=True)
class RestoreAllocation:
    """Full logical mapping plus indices newly owned by this restore."""

    full_indices: tuple[int, ...]
    provisional_indices: tuple[int, ...]
    allocator_resident_bytes: int = 0

    def __post_init__(self) -> None:
        _require_non_negative("allocator_resident_bytes", self.allocator_resident_bytes)
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in (*self.full_indices, *self.provisional_indices)
        ):
            raise ValueError("restore indices must be non-negative integers")
        if not set(self.provisional_indices).issubset(self.full_indices):
            raise ValueError("provisional indices must belong to the full mapping")


@dataclass(frozen=True, slots=True)
class KVReleaseReceipt:
    logical_blocks: int
    allocator_released_bytes: int
    released_physical_bytes: int = 0
    post_release_ownership: str = "SLOT_REUSABLE"
    backing_left_slot_ownership: bool = False
    ownership_mechanism: str = ""

    def __post_init__(self) -> None:
        for name in (
            "logical_blocks",
            "allocator_released_bytes",
            "released_physical_bytes",
        ):
            _require_non_negative(name, getattr(self, name))
        if self.post_release_ownership not in {"SLOT_REUSABLE", "UNATTRIBUTED"}:
            raise ValueError("invalid post-release ownership")
        if self.backing_left_slot_ownership:
            if self.released_physical_bytes <= 0:
                raise ValueError("VMM ownership evidence requires released bytes")
            _require_text("ownership_mechanism", self.ownership_mechanism)

    @classmethod
    def empty(cls) -> "KVReleaseReceipt":
        return cls(0, 0)


class KVResidencyAdapter(Protocol):
    def export_kv_range(
        self,
        descriptor: SessionKVDescriptor,
        layer_range: tuple[int, int],
    ) -> ExportedKVRange: ...

    def detach_branch(self, frozen: FrozenSessionKV) -> None: ...

    def free_indices(self, indices: tuple[int, ...]) -> KVReleaseReceipt: ...

    def allocate_restore(
        self, descriptor: SessionKVDescriptor
    ) -> RestoreAllocation: ...

    def import_kv_range(
        self,
        indices: tuple[int, ...],
        extent: KVExtent,
        payload: bytes,
    ) -> None: ...

    def synchronize(self) -> None: ...

    def publish_restored(self, descriptor: SessionKVDescriptor) -> None: ...

    def rollback_restore(self, allocation: RestoreAllocation) -> None: ...


@dataclass(frozen=True, slots=True)
class KVResidencyResult:
    ok: bool
    code: str
    resource_epoch: int
    reason: str = ""
    committed_ssd_bytes: int = 0
    resident_bytes: int = 0
    allocator_resident_bytes: int = 0
    logical_released_bytes: int = 0
    allocator_released_bytes: int = 0
    released_physical_bytes: int = 0
    post_release_ownership: str = "UNATTRIBUTED"
    backing_left_slot_ownership: bool = False
    ownership_mechanism: str = ""
    new_indices: tuple[int, ...] = ()
    shared_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class KVRuntimeSnapshot:
    descriptor: SessionKVDescriptor
    state: str
    committed_ssd_bytes: int


@dataclass(slots=True)
class _DurableRecord:
    descriptor: SessionKVDescriptor
    state: KVRangeState
    manifest: SessionManifest | None = None

    @property
    def committed_ssd_bytes(self) -> int:
        if self.manifest is None:
            return 0
        return sum(item.extent.nbytes for item in self.manifest.extents)


class KVResidencyRuntime:
    """Serialize one branch's durable offload/restore state transitions."""

    def __init__(
        self,
        *,
        store: LocalKVStore,
        adapter: KVResidencyAdapter,
        registry: SessionKVRegistry | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.registry = registry or SessionKVRegistry()
        self._records: dict[tuple[SessionBranchKey, int], _DurableRecord] = {}
        self._lock = threading.RLock()
        # This counter is intentionally explicit: no error path in this
        # subsystem is permitted to turn a restore into prompt prefill.
        self.prefill_recompute_count = 0

    def attach_resident(self, descriptor: SessionKVDescriptor) -> None:
        with self._lock:
            catalog_key = (descriptor.key, descriptor.kv_epoch)
            existing = self._records.get(catalog_key)
            if existing is not None and existing.state is not KVRangeState.ON_LOCAL_SSD:
                raise ValueError("KV branch and epoch are already tracked")
            self.registry.attach(descriptor)
            self._records[catalog_key] = _DurableRecord(
                descriptor,
                KVRangeState.RESIDENT_RECLAIMABLE,
                existing.manifest if existing is not None else None,
            )

    def descriptor(self, key: SessionBranchKey) -> SessionKVDescriptor:
        return self.registry.snapshot(key).descriptor

    def snapshot(self, key: SessionBranchKey) -> KVRuntimeSnapshot:
        with self._lock:
            candidates = [
                record
                for (record_key, _), record in self._records.items()
                if record_key == key
            ]
            if not candidates:
                raise KeyError(key)
            record = max(candidates, key=lambda item: item.descriptor.kv_epoch)
            return KVRuntimeSnapshot(
                record.descriptor,
                record.state.value,
                record.committed_ssd_bytes,
            )

    def is_durably_offloaded(self, key: SessionBranchKey) -> bool:
        """Return whether the latest epoch has a committed SSD authority."""

        try:
            return self.snapshot(key).state == KVRangeState.ON_LOCAL_SSD.value
        except KeyError:
            return False

    def snapshots(self) -> tuple[KVRuntimeSnapshot, ...]:
        with self._lock:
            return tuple(
                KVRuntimeSnapshot(
                    record.descriptor,
                    record.state.value,
                    record.committed_ssd_bytes,
                )
                for _, record in sorted(
                    self._records.items(),
                    key=lambda item: (item[0][0], item[0][1]),
                )
            )

    def offload(self, command: KVOffloadCommand) -> KVResidencyResult:
        with self._lock:
            catalog_key = (command.key, command.expected_kv_epoch)
            record = self._records.get(catalog_key)
            if record is None:
                return self._failure(
                    command.expected_kv_epoch,
                    "STALE_RESOURCE_EPOCH",
                    "unknown KV branch or epoch",
                )
            if record.state is not KVRangeState.RESIDENT_RECLAIMABLE:
                return self._failure(
                    command.expected_kv_epoch,
                    "PINNED_RESOURCE_CONFLICT",
                    f"KV branch is {record.state.value}",
                )
            try:
                frozen = self.registry.freeze(
                    command.key,
                    expected_kv_epoch=command.expected_kv_epoch,
                )
            except Exception as exc:
                return self._failure(
                    command.expected_kv_epoch, "PINNED_RESOURCE_CONFLICT", str(exc)
                )

            record.state = KVRangeState.OFFLOADING
            try:
                extent_chunks = tuple(
                    self._export_extent(frozen.descriptor, layer_range)
                    for layer_range in command.layer_ranges
                )
                manifest = self.store.persist_session(
                    frozen.descriptor,
                    extent_chunks,
                    operation_id=command.operation_id,
                )
            except Exception as exc:
                record.state = KVRangeState.RESIDENT_RECLAIMABLE
                if isinstance(exc, ChecksumMismatch):
                    code = "CHECKSUM_MISMATCH"
                elif isinstance(exc, (LocalKVStoreError, OSError)):
                    code = "SSD_IO_FAILURE"
                else:
                    code = "KV_OFFLOAD_FAILURE"
                return self._failure(command.expected_kv_epoch, code, str(exc))

            # Durable manifest publication is the commit point.  No cache
            # mapping or allocator ownership changes before this line.
            try:
                self.adapter.detach_branch(frozen)
                detached = self.registry.detach(frozen)
                receipt = (
                    self.adapter.free_indices(detached.actually_reclaimable_indices)
                    if detached.actually_reclaimable_indices
                    else KVReleaseReceipt.empty()
                )
            except Exception as exc:
                # The durable copy remains authoritative.  Returning a
                # structured release failure prevents the control plane from
                # crediting memory whose allocator outcome is unknown.
                record.manifest = manifest
                record.state = KVRangeState.ON_LOCAL_SSD
                return self._failure(
                    command.expected_kv_epoch,
                    "KV_RELEASE_FAILURE",
                    str(exc),
                    committed_ssd_bytes=record.committed_ssd_bytes,
                )

            record.manifest = manifest
            record.state = KVRangeState.ON_LOCAL_SSD
            return KVResidencyResult(
                ok=True,
                code="OK",
                resource_epoch=command.expected_kv_epoch,
                committed_ssd_bytes=record.committed_ssd_bytes,
                logical_released_bytes=detached.logical_bytes,
                allocator_released_bytes=receipt.allocator_released_bytes,
                released_physical_bytes=receipt.released_physical_bytes,
                post_release_ownership=receipt.post_release_ownership,
                backing_left_slot_ownership=receipt.backing_left_slot_ownership,
                ownership_mechanism=receipt.ownership_mechanism,
                shared_indices=detached.shared_indices,
            )

    def restore(self, command: KVRestoreCommand) -> KVResidencyResult:
        with self._lock:
            if command.policy is not RestorePolicy.FULL_BARRIER:
                return self._failure(
                    command.expected_kv_epoch,
                    "UNSUPPORTED_RESTORE_POLICY",
                    "selected-layer restore is not enabled in the full-barrier milestone",
                )
            catalog_key = (command.key, command.expected_kv_epoch)
            record = self._records.get(catalog_key)
            if record is None:
                return self._failure(
                    command.expected_kv_epoch,
                    "STALE_RESOURCE_EPOCH",
                    "unknown KV branch or epoch",
                )
            if record.state is not KVRangeState.ON_LOCAL_SSD:
                return self._failure(
                    command.expected_kv_epoch,
                    "PINNED_RESOURCE_CONFLICT",
                    f"KV branch is {record.state.value}",
                )

            record.state = KVRangeState.RESTORING
            allocation: RestoreAllocation | None = None
            try:
                manifest = self.store.require_manifest(
                    command.key,
                    kv_epoch=command.expected_kv_epoch,
                )
                allocation = self.adapter.allocate_restore(manifest.descriptor)
                if len(allocation.full_indices) != len(manifest.descriptor.token_ids):
                    raise ValueError(
                        "restore allocation does not match the token count"
                    )
                for committed in manifest.extents:
                    payload = self.store.read_extent(committed)
                    self.adapter.import_kv_range(
                        allocation.full_indices,
                        committed.extent,
                        payload,
                    )
                # FULL_BARRIER means no branch is visible until every local
                # layer has passed checksum/scatter and CUDA completion.
                self.adapter.synchronize()
                restored = replace(
                    manifest.descriptor,
                    kv_indices=allocation.full_indices,
                )
                self.adapter.publish_restored(restored)
                self.registry.attach_restored(
                    restored,
                    expected_kv_epoch=command.expected_kv_epoch,
                )
            except Exception as exc:
                if allocation is not None:
                    try:
                        self.adapter.rollback_restore(allocation)
                    except Exception:
                        pass
                record.state = KVRangeState.ON_LOCAL_SSD
                code = (
                    "CHECKSUM_MISMATCH"
                    if isinstance(exc, ChecksumMismatch)
                    else "KV_RESTORE_FAILURE"
                )
                return self._failure(
                    command.expected_kv_epoch,
                    code,
                    str(exc),
                    committed_ssd_bytes=record.committed_ssd_bytes,
                )

            record.descriptor = restored
            record.state = KVRangeState.RESIDENT_RECLAIMABLE
            allocator_bytes = (
                allocation.allocator_resident_bytes
                if allocation.allocator_resident_bytes
                else restored.logical_bytes
            )
            return KVResidencyResult(
                ok=True,
                code="OK",
                resource_epoch=command.expected_kv_epoch,
                committed_ssd_bytes=record.committed_ssd_bytes,
                resident_bytes=restored.logical_bytes,
                allocator_resident_bytes=allocator_bytes,
                new_indices=restored.kv_indices,
            )

    @staticmethod
    def _failure(
        resource_epoch: int,
        code: str,
        reason: str,
        *,
        committed_ssd_bytes: int = 0,
    ) -> KVResidencyResult:
        return KVResidencyResult(
            ok=False,
            code=code,
            resource_epoch=resource_epoch,
            reason=reason,
            committed_ssd_bytes=committed_ssd_bytes,
        )

    def _export_extent(
        self,
        descriptor: SessionKVDescriptor,
        layer_range: tuple[int, int],
    ) -> tuple[KVExtent, tuple[bytes, ...]]:
        exported = self.adapter.export_kv_range(descriptor, layer_range)
        if exported.local_layer_range != layer_range:
            raise ValueError("adapter exported a different local layer range")
        chunks = tuple(bytes(memoryview(chunk).cast("B")) for chunk in exported.chunks)
        nbytes = sum(len(chunk) for chunk in chunks)
        if nbytes <= 0:
            raise ValueError("adapter exported an empty KV extent")
        token_start = descriptor.token_position - len(descriptor.token_ids)
        if token_start < 0 or token_start == descriptor.token_position:
            raise ValueError("cannot persist an empty or invalid token range")
        extent = KVExtent(
            instance_id=descriptor.instance_id,
            model_digest=descriptor.model_digest,
            placement_version=descriptor.placement_version,
            stage_id=descriptor.stage_id,
            session_id=descriptor.key.session_id,
            request_id=descriptor.key.request_id,
            predecessor_request_id=descriptor.predecessor_request_id,
            kv_epoch=descriptor.kv_epoch,
            token_position=descriptor.token_position,
            token_range=(token_start, descriptor.token_position),
            local_layer_range=layer_range,
            dtype=exported.dtype,
            k_shape=exported.k_shape,
            v_shape=exported.v_shape,
            nbytes=nbytes,
            checksum=content_checksum(chunks),
        )
        return extent, chunks


class SGLangKVResidencyAdapter:
    """Bridge the transaction runtime to one bound SGLang KV pool.

    Torch is imported lazily so the transaction and persistence layers remain
    CPU-testable.  The current full-barrier format stores one contiguous K
    tensor followed by one contiguous V tensor for each requested layer range.
    """

    def __init__(self, token_to_kv_pool_allocator, tree_cache) -> None:
        self.allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self._bytes_per_index = self._calculate_bytes_per_index()

    def _calculate_bytes_per_index(self) -> int:
        total = 0
        for layer_id in range(
            self.kv_cache.start_layer,
            self.kv_cache.start_layer + self.kv_cache.layer_num,
        ):
            key = self.kv_cache.get_key_buffer(layer_id)
            value = self.kv_cache.get_value_buffer(layer_id)
            total += key[0].numel() * key.element_size()
            total += value[0].numel() * value.element_size()
        return int(total)

    @property
    def bytes_per_index(self) -> int:
        return self._bytes_per_index

    def export_kv_range(
        self,
        descriptor: SessionKVDescriptor,
        layer_range: tuple[int, int],
    ) -> ExportedKVRange:
        import torch

        indices = torch.tensor(
            descriptor.kv_indices,
            dtype=torch.int64,
            device=self.kv_cache.device,
        )
        keys = torch.stack(
            [
                self.kv_cache.get_key_buffer(layer_id)[indices]
                for layer_id in range(*layer_range)
            ]
        ).contiguous()
        values = torch.stack(
            [
                self.kv_cache.get_value_buffer(layer_id)[indices]
                for layer_id in range(*layer_range)
            ]
        ).contiguous()
        keys_cpu = keys.to("cpu")
        values_cpu = values.to("cpu")
        self.synchronize()
        # NumPy has no native bfloat16 dtype.  Persist the exact storage bytes
        # through a uint8 view so every torch dtype follows one lossless path.
        payload = keys_cpu.view(torch.uint8).numpy().tobytes(
            order="C"
        ) + values_cpu.view(torch.uint8).numpy().tobytes(order="C")
        return ExportedKVRange(
            local_layer_range=layer_range,
            dtype=str(keys.dtype).removeprefix("torch."),
            k_shape=tuple(keys.shape),
            v_shape=tuple(values.shape),
            chunks=(payload,),
        )

    def detach_branch(self, frozen: FrozenSessionKV) -> None:
        self.tree_cache.detach_uma_branch(
            frozen.descriptor.token_ids,
            frozen.descriptor.kv_indices,
            frozen.exclusive_indices,
        )

    def free_indices(self, indices: tuple[int, ...]) -> KVReleaseReceipt:
        import torch

        if not indices:
            return KVReleaseReceipt.empty()
        tensor = torch.tensor(indices, dtype=torch.int64, device=self.allocator.device)
        raw = self.allocator.free(tensor)
        allocator_bytes = len(indices) * self._bytes_per_index
        if raw is None:
            return KVReleaseReceipt(
                logical_blocks=len(indices),
                allocator_released_bytes=allocator_bytes,
                post_release_ownership="SLOT_REUSABLE",
            )
        physical = int(getattr(raw, "released_physical_bytes", 0))
        left_slot = bool(getattr(raw, "backing_left_slot_ownership", False))
        logical_blocks = int(getattr(raw, "logical_blocks", len(indices)))
        allocator_bytes = (
            logical_blocks
            * int(getattr(self.allocator, "page_size", 1))
            * self._bytes_per_index
        )
        return KVReleaseReceipt(
            logical_blocks=logical_blocks,
            allocator_released_bytes=allocator_bytes,
            released_physical_bytes=physical,
            post_release_ownership=("UNATTRIBUTED" if left_slot else "SLOT_REUSABLE"),
            backing_left_slot_ownership=left_slot,
            ownership_mechanism=str(getattr(raw, "mechanism", "")),
        )

    def allocate_restore(self, descriptor: SessionKVDescriptor) -> RestoreAllocation:
        import torch

        match = self.tree_cache.match_prefix(list(descriptor.token_ids))
        prefix = match.device_indices
        prefix_count = len(prefix)
        if prefix_count > len(descriptor.token_ids):
            raise RuntimeError("RadixCache returned an overlong prefix")
        missing = len(descriptor.token_ids) - prefix_count
        provisional = self.allocator.alloc(missing) if missing else None
        if missing and provisional is None:
            raise MemoryError(f"cannot allocate {missing} KV indices")
        if provisional is None:
            provisional = torch.empty(
                (0,), dtype=torch.int64, device=self.allocator.device
            )
        full = torch.cat((prefix.to(self.allocator.device), provisional))
        full_indices = tuple(int(item) for item in full.to("cpu").tolist())
        provisional_indices = tuple(
            int(item) for item in provisional.to("cpu").tolist()
        )
        return RestoreAllocation(
            full_indices=full_indices,
            provisional_indices=provisional_indices,
            allocator_resident_bytes=len(provisional_indices) * self._bytes_per_index,
        )

    def import_kv_range(
        self,
        indices: tuple[int, ...],
        extent: KVExtent,
        payload: bytes,
    ) -> None:
        import math
        import torch

        dtype_name = extent.dtype.removeprefix("torch.")
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported KV dtype: {extent.dtype}") from exc
        key_elements = math.prod(extent.k_shape)
        value_elements = math.prod(extent.v_shape)
        expected_bytes = (key_elements + value_elements) * dtype.itemsize
        if len(payload) != expected_bytes:
            raise ValueError("KV extent payload length does not match its shapes")
        # bytearray gives torch a writable owner and avoids a non-writable
        # frombuffer warning; group-sized staging bounds this copy.
        flat = torch.frombuffer(bytearray(payload), dtype=dtype)
        keys = flat[:key_elements].reshape(extent.k_shape)
        values = flat[key_elements:].reshape(extent.v_shape)
        device_indices = torch.tensor(
            indices, dtype=torch.int64, device=self.kv_cache.device
        )
        for offset, layer_id in enumerate(range(*extent.local_layer_range)):
            self.kv_cache.get_key_buffer(layer_id)[device_indices] = keys[offset].to(
                self.kv_cache.device, non_blocking=True
            )
            self.kv_cache.get_value_buffer(layer_id)[device_indices] = values[
                offset
            ].to(self.kv_cache.device, non_blocking=True)

    def synchronize(self) -> None:
        import torch

        device_module = torch.get_device_module(self.kv_cache.device)
        device_module.synchronize()

    def publish_restored(self, descriptor: SessionKVDescriptor) -> None:
        self.tree_cache.publish_uma_branch(
            descriptor.token_ids,
            descriptor.kv_indices,
        )

    def rollback_restore(self, allocation: RestoreAllocation) -> None:
        if allocation.provisional_indices:
            self.free_indices(allocation.provisional_indices)
