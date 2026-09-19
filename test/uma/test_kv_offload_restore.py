from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"


def _package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_package("sglang", PYTHON_ROOT / "sglang")
_package("sglang.multi_model", PYTHON_ROOT / "sglang" / "multi_model")
_package("sglang.multi_model.uma", PYTHON_ROOT / "sglang" / "multi_model" / "uma")
kv_domain = _load(
    "sglang.multi_model.uma.kv_domain",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "kv_domain.py",
)
local_store = _load(
    "sglang.multi_model.uma.local_kv_store",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "local_kv_store.py",
)
kv_residency = _load(
    "sglang.multi_model.uma.kv_residency",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "kv_residency.py",
)

SessionBranchKey = kv_domain.SessionBranchKey
SessionKVDescriptor = kv_domain.SessionKVDescriptor
LocalKVStore = local_store.LocalKVStore
ExportedKVRange = kv_residency.ExportedKVRange
KVOffloadCommand = kv_residency.KVOffloadCommand
KVReleaseReceipt = kv_residency.KVReleaseReceipt
KVResidencyRuntime = kv_residency.KVResidencyRuntime
KVRestoreCommand = kv_residency.KVRestoreCommand
RestoreAllocation = kv_residency.RestoreAllocation
RestorePolicy = kv_residency.RestorePolicy


KEY = SessionBranchKey("instance-a", "stage-0", "session-a", "request-a")


def descriptor(*, indices: tuple[int, ...] = (1, 2, 3)) -> SessionKVDescriptor:
    return SessionKVDescriptor(
        key=KEY,
        predecessor_request_id=None,
        instance_id="instance-a",
        model_digest="sha256:model-a",
        placement_version=3,
        stage_id="stage-0",
        kv_epoch=7,
        token_ids=(11, 12, 13),
        kv_indices=indices,
        token_position=3,
        logical_bytes=24,
    )


class BlockingStore(LocalKVStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.before_commit = threading.Event()
        self.allow_commit = threading.Event()

    def commit_session(self, descriptor, extents, *, operation_id):
        self.before_commit.set()
        assert self.allow_commit.wait(timeout=5)
        return super().commit_session(
            descriptor,
            extents,
            operation_id=operation_id,
        )


class FakeKVAdapter:
    def __init__(self) -> None:
        self.payloads = {
            (0, 1): b"K000V000K001",
            (1, 2): b"K100V100K101",
        }
        self.free_calls: list[tuple[int, ...]] = []
        self.detached: list[SessionBranchKey] = []
        self.imported: list[tuple[tuple[int, ...], tuple[int, int], bytes]] = []
        self.published: list[SessionKVDescriptor] = []
        self.rollback_calls: list[tuple[int, ...]] = []
        self.synchronize_calls = 0
        self.next_indices = (101, 102, 103)
        self.fail_import_at: tuple[int, int] | None = None
        self.fail_export = False

    def export_kv_range(self, descriptor, layer_range):
        if self.fail_export:
            raise RuntimeError("injected gather failure")
        payload = self.payloads[layer_range]
        return ExportedKVRange(
            local_layer_range=layer_range,
            dtype="float16",
            k_shape=(1, len(descriptor.kv_indices), 1),
            v_shape=(1, len(descriptor.kv_indices), 1),
            chunks=(payload,),
        )

    def detach_branch(self, frozen):
        self.detached.append(frozen.descriptor.key)

    def allocate_restore(self, descriptor):
        return RestoreAllocation(
            full_indices=self.next_indices,
            provisional_indices=self.next_indices,
        )

    def import_kv_range(self, indices, extent, payload):
        if extent.local_layer_range == self.fail_import_at:
            raise RuntimeError("injected scatter failure")
        self.imported.append((tuple(indices), extent.local_layer_range, payload))

    def synchronize(self):
        self.synchronize_calls += 1

    def publish_restored(self, descriptor):
        self.published.append(descriptor)

    def rollback_restore(self, allocation):
        self.rollback_calls.append(allocation.provisional_indices)

    def free_indices(self, indices):
        indices = tuple(indices)
        self.free_calls.append(indices)
        return KVReleaseReceipt(
            logical_blocks=len(indices),
            allocator_released_bytes=len(indices) * 8,
            released_physical_bytes=len(indices) * 8,
            post_release_ownership="UNATTRIBUTED",
            backing_left_slot_ownership=True,
            ownership_mechanism="cuda-vmm-unmap-release",
        )


def runtime(tmp_path, *, store_type=LocalKVStore):
    adapter = FakeKVAdapter()
    store = store_type(tmp_path / "kv")
    result = KVResidencyRuntime(store=store, adapter=adapter)
    result.attach_resident(descriptor())
    return result, store, adapter


def offload_command() -> KVOffloadCommand:
    return KVOffloadCommand(
        key=KEY,
        expected_kv_epoch=7,
        operation_id="offload-1",
        layer_ranges=((0, 1), (1, 2)),
    )


def restore_command() -> KVRestoreCommand:
    return KVRestoreCommand(
        key=KEY,
        expected_kv_epoch=7,
        operation_id="restore-1",
        policy=RestorePolicy.FULL_BARRIER,
    )


def test_pages_release_only_after_manifest_commit(tmp_path) -> None:
    service, store, adapter = runtime(tmp_path, store_type=BlockingStore)
    observed: dict[str, object] = {}

    def run() -> None:
        observed["result"] = service.offload(offload_command())

    thread = threading.Thread(target=run)
    thread.start()
    assert store.before_commit.wait(timeout=5)
    assert adapter.detached == []
    assert adapter.free_calls == []

    store.allow_commit.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert observed["result"].committed_ssd_bytes == 24
    assert adapter.detached == [KEY]
    assert adapter.free_calls == [(1, 2, 3)]


def test_full_restore_publishes_new_indices_only_after_all_extents_validate(
    tmp_path,
) -> None:
    service, _, adapter = runtime(tmp_path)
    old = service.descriptor(KEY)
    service.offload(offload_command())

    restored = service.restore(restore_command())

    assert restored.ok
    assert restored.new_indices == (101, 102, 103)
    assert restored.new_indices != old.kv_indices
    assert service.descriptor(KEY).kv_indices == restored.new_indices
    assert [item[1] for item in adapter.imported] == [(0, 1), (1, 2)]
    assert adapter.synchronize_calls == 1
    assert adapter.published[-1].kv_indices == restored.new_indices
    assert service.prefill_recompute_count == 0


def test_restore_failure_rolls_back_provisional_indices_and_stays_on_ssd(
    tmp_path,
) -> None:
    service, store, adapter = runtime(tmp_path)
    service.offload(offload_command())
    adapter.fail_import_at = (1, 2)

    failed = service.restore(restore_command())

    assert not failed.ok
    assert failed.code == "KV_RESTORE_FAILURE"
    assert adapter.published == []
    assert adapter.rollback_calls == [(101, 102, 103)]
    assert service.snapshot(KEY).state == "ON_LOCAL_SSD"
    assert service.is_durably_offloaded(KEY)
    assert store.require_manifest(KEY, kv_epoch=7).descriptor.key == KEY
    assert service.prefill_recompute_count == 0


def test_shared_prefix_indices_are_not_returned_to_allocator(tmp_path) -> None:
    service, _, adapter = runtime(tmp_path)
    sibling_key = replace(KEY, request_id="request-b")
    service.attach_resident(
        replace(
            descriptor(indices=(1, 2, 4)),
            key=sibling_key,
            predecessor_request_id=KEY.request_id,
            token_ids=(11, 12, 14),
        )
    )

    result = service.offload(offload_command())

    assert result.ok
    assert result.shared_indices == (1, 2)
    assert adapter.free_calls == [(3,)]


def test_checksum_failure_never_publishes_partial_branch(tmp_path) -> None:
    service, store, adapter = runtime(tmp_path)
    service.offload(offload_command())
    manifest = store.require_manifest(KEY, kv_epoch=7)
    manifest.extents[-1].path.write_bytes(b"corrupt")

    failed = service.restore(restore_command())

    assert not failed.ok
    assert failed.code == "CHECKSUM_MISMATCH"
    assert adapter.published == []
    assert adapter.rollback_calls == [(101, 102, 103)]
    with pytest.raises(KeyError):
        service.descriptor(KEY)
    assert service.prefill_recompute_count == 0


def test_gather_failure_rolls_back_offloading_state(tmp_path) -> None:
    service, _, adapter = runtime(tmp_path)
    adapter.fail_export = True

    failed = service.offload(offload_command())

    assert not failed.ok
    assert failed.code == "KV_OFFLOAD_FAILURE"
    assert service.snapshot(KEY).state == "RESIDENT_RECLAIMABLE"
    assert adapter.detached == []
    assert adapter.free_calls == []
    assert not service.is_durably_offloaded(KEY)
