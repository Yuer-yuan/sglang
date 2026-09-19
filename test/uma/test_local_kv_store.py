from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
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
local_kv_store = _load(
    "sglang.multi_model.uma.local_kv_store",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "local_kv_store.py",
)

KVExtent = kv_domain.KVExtent
SessionBranchKey = kv_domain.SessionBranchKey
SessionKVDescriptor = kv_domain.SessionKVDescriptor
ChecksumMismatch = local_kv_store.ChecksumMismatch
InjectedIOError = local_kv_store.InjectedIOError
LocalKVStore = local_kv_store.LocalKVStore
ManifestConflict = local_kv_store.ManifestConflict
content_checksum = local_kv_store.content_checksum


DATA = b"abcdefgh"


def descriptor(
    *,
    instance_id: str = "model-a",
    request_id: str = "request-a",
    kv_epoch: int = 3,
) -> SessionKVDescriptor:
    stage_id = "stage-0"
    return SessionKVDescriptor(
        key=SessionBranchKey(
            instance_id,
            stage_id,
            "session-a",
            request_id,
        ),
        predecessor_request_id=None,
        instance_id=instance_id,
        model_digest=f"sha256:{instance_id}",
        placement_version=5,
        stage_id=stage_id,
        kv_epoch=kv_epoch,
        token_ids=(10, 11),
        kv_indices=(1, 2),
        token_position=2,
        logical_bytes=len(DATA),
    )


def extent(
    data: bytes = DATA,
    *,
    instance_id: str = "model-a",
    request_id: str = "request-a",
    kv_epoch: int = 3,
) -> KVExtent:
    return KVExtent(
        instance_id=instance_id,
        model_digest=f"sha256:{instance_id}",
        placement_version=5,
        stage_id="stage-0",
        session_id="session-a",
        request_id=request_id,
        predecessor_request_id=None,
        kv_epoch=kv_epoch,
        token_position=2,
        token_range=(0, 2),
        local_layer_range=(0, 4),
        dtype="bfloat16",
        k_shape=(4, 2, 1, 1),
        v_shape=(4, 2, 1, 1),
        nbytes=len(data),
        checksum=content_checksum((data,)),
    )


class RaiseAfter:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def checkpoint(self, phase: str) -> None:
        if phase == self.phase:
            raise InjectedIOError(phase)


def test_persist_and_recover_exact_session_manifest(tmp_path: Path) -> None:
    store = LocalKVStore(tmp_path)
    item = extent()

    manifest = store.persist_session(
        descriptor(),
        ((item, (DATA[:3], DATA[3:])),),
        operation_id="op-1",
    )

    assert manifest.descriptor == descriptor()
    assert len(manifest.extents) == 1
    assert store.read_extent(manifest.extents[0]) == DATA
    assert store.require_manifest(descriptor().key, kv_epoch=3) == manifest

    reopened = LocalKVStore(tmp_path)
    assert reopened.require_manifest(descriptor().key, kv_epoch=3).descriptor == descriptor()
    assert reopened.recover().committed_manifests == 1


def test_manifest_is_not_visible_before_data_is_durable(tmp_path: Path) -> None:
    store = LocalKVStore(tmp_path, fault_injector=RaiseAfter("data_fsync"))

    with pytest.raises(InjectedIOError):
        store.persist_session(
            descriptor(),
            ((extent(), (DATA,)),),
            operation_id="../../unsafe-operation",
        )

    assert store.list_manifests() == ()
    assert tuple((tmp_path / "tmp").iterdir()) == ()


def test_extent_committed_before_manifest_is_recoverable_but_not_visible(
    tmp_path: Path,
) -> None:
    store = LocalKVStore(
        tmp_path,
        fault_injector=RaiseAfter("manifest_fsync"),
    )

    with pytest.raises(InjectedIOError):
        store.persist_session(
            descriptor(),
            ((extent(), (DATA,)),),
            operation_id="op-orphan",
        )

    assert store.list_manifests() == ()
    report = LocalKVStore(tmp_path).recover()
    assert report.committed_manifests == 0
    assert report.orphan_extents == 1
    assert report.temporary_files_removed == 0


def test_corrupt_extent_is_never_returned(tmp_path: Path) -> None:
    store = LocalKVStore(tmp_path)
    manifest = store.persist_session(
        descriptor(),
        ((extent(), (DATA,)),),
        operation_id="op-corrupt",
    )
    manifest.extents[0].path.write_bytes(b"abcxefgh")

    with pytest.raises(ChecksumMismatch):
        store.read_extent(manifest.extents[0])


def test_wrong_expected_checksum_never_publishes_manifest(tmp_path: Path) -> None:
    store = LocalKVStore(tmp_path)
    wrong = replace(extent(), checksum=content_checksum((b"different",)))

    with pytest.raises(ChecksumMismatch):
        store.persist_session(
            descriptor(),
            ((wrong, (DATA,)),),
            operation_id="op-wrong-digest",
        )

    assert store.list_manifests() == ()


def test_manifest_rejects_extent_from_another_model_or_epoch(tmp_path: Path) -> None:
    store = LocalKVStore(tmp_path)

    with pytest.raises(ValueError, match="does not belong"):
        store.persist_session(
            descriptor(),
            ((extent(instance_id="model-b"), (DATA,)),),
            operation_id="op-other-model",
        )

    with pytest.raises(ValueError, match="does not belong"):
        store.persist_session(
            descriptor(),
            ((extent(kv_epoch=4), (DATA,)),),
            operation_id="op-other-epoch",
        )


def test_stale_epoch_lookup_fails_closed(tmp_path: Path) -> None:
    store = LocalKVStore(tmp_path)
    store.persist_session(
        descriptor(kv_epoch=3),
        ((extent(kv_epoch=3), (DATA,)),),
        operation_id="op-stale",
    )

    with pytest.raises(KeyError):
        store.require_manifest(descriptor().key, kv_epoch=2)


def test_manifest_is_immutable_and_identical_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    store = LocalKVStore(tmp_path)
    first = store.persist_session(
        descriptor(),
        ((extent(), (DATA,)),),
        operation_id="op-first",
    )
    retry = store.persist_session(
        descriptor(),
        ((extent(), (DATA,)),),
        operation_id="op-retry",
    )

    assert retry == first

    different = b"abcdwxyz"
    with pytest.raises(ManifestConflict):
        store.persist_session(
            descriptor(),
            ((extent(different), (different,)),),
            operation_id="op-conflict",
        )

    assert store.require_manifest(descriptor().key, kv_epoch=3) == first
