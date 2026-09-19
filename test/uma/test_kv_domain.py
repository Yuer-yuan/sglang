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

KVExtent = kv_domain.KVExtent
KVRangeState = kv_domain.KVRangeState
PinnedKVRangeError = kv_domain.PinnedKVRangeError
SessionBranchKey = kv_domain.SessionBranchKey
SessionKVDescriptor = kv_domain.SessionKVDescriptor
SessionKVRegistry = kv_domain.SessionKVRegistry
StaleKVEpochError = kv_domain.StaleKVEpochError


def descriptor(
    session_id: str = "session-a",
    request_id: str = "request-a",
    *,
    predecessor_request_id: str | None = None,
    indices: tuple[int, ...] = (10, 11, 12),
    kv_epoch: int = 3,
    instance_id: str = "model-a",
    stage_id: str = "stage-0",
) -> SessionKVDescriptor:
    return SessionKVDescriptor(
        key=SessionBranchKey(instance_id, stage_id, session_id, request_id),
        predecessor_request_id=predecessor_request_id,
        instance_id=instance_id,
        model_digest=f"sha256:{instance_id}",
        placement_version=5,
        stage_id=stage_id,
        kv_epoch=kv_epoch,
        token_ids=tuple(range(100, 100 + len(indices))),
        kv_indices=indices,
        token_position=len(indices),
        logical_bytes=len(indices) * 4096,
    )


def extent(**changes) -> KVExtent:
    base = KVExtent(
        instance_id="model-a",
        model_digest="sha256:model-a",
        placement_version=5,
        stage_id="stage-0",
        session_id="session-a",
        request_id="request-a",
        predecessor_request_id=None,
        kv_epoch=3,
        token_position=128,
        token_range=(0, 128),
        local_layer_range=(0, 4),
        dtype="bfloat16",
        k_shape=(4, 128, 2, 64),
        v_shape=(4, 128, 2, 64),
        nbytes=262_144,
        checksum="blake2b:test",
    )
    return replace(base, **changes)


def test_extent_identity_includes_branch_epoch_position_and_layer_range() -> None:
    base = extent()

    assert replace(base, request_id="request-b") != base
    assert replace(base, kv_epoch=4) != base
    assert replace(base, token_position=129) != base
    assert replace(base, local_layer_range=(4, 8)) != base


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("request_id", ""),
        ("placement_version", -1),
        ("token_range", (8, 8)),
        ("local_layer_range", (4, 3)),
        ("nbytes", -1),
        ("k_shape", (4, 0, 2, 64)),
    ),
)
def test_extent_rejects_invalid_identity_or_shape(field, value) -> None:
    with pytest.raises(ValueError):
        extent(**{field: value})


def test_session_id_alone_cannot_alias_two_request_branches() -> None:
    registry = SessionKVRegistry()
    first = descriptor(request_id="request-1", indices=(1, 2))
    second = descriptor(
        request_id="request-2",
        predecessor_request_id="request-1",
        indices=(1, 3),
    )
    registry.attach(first)
    registry.attach(second)

    frozen_first = registry.freeze(first.key)
    frozen_second = registry.freeze(second.key)

    assert frozen_first.descriptor.kv_indices == (1, 2)
    assert frozen_first.shared_indices == (1,)
    assert frozen_first.exclusive_indices == (2,)
    assert frozen_second.descriptor.kv_indices == (1, 3)
    assert frozen_second.shared_indices == (1,)
    assert frozen_second.exclusive_indices == (3,)


def test_detach_releases_only_indices_whose_final_owner_disappears() -> None:
    registry = SessionKVRegistry()
    first = descriptor(request_id="request-1", indices=(1, 2))
    second = descriptor(request_id="request-2", indices=(1, 3))
    registry.attach(first)
    registry.attach(second)

    report = registry.detach(registry.freeze(first.key))

    assert report.detached_indices == (1, 2)
    assert report.actually_reclaimable_indices == (2,)
    assert report.shared_indices == (1,)
    assert report.logical_bytes == first.logical_bytes
    assert registry.freeze(second.key).exclusive_indices == (1, 3)


def test_equal_numeric_indices_in_different_model_pools_are_not_shared() -> None:
    registry = SessionKVRegistry()
    first = descriptor(instance_id="model-a", indices=(1, 2))
    second = descriptor(instance_id="model-b", indices=(1, 2))
    registry.attach(first)
    registry.attach(second)

    assert registry.freeze(first.key).exclusive_indices == (1, 2)
    assert registry.freeze(second.key).exclusive_indices == (1, 2)


def test_executing_or_leased_branch_cannot_freeze_or_detach() -> None:
    registry = SessionKVRegistry()
    item = descriptor()
    registry.attach(item)
    registry.set_execution_state(item.key, executing=True, lease_count=1)

    with pytest.raises(PinnedKVRangeError):
        registry.freeze(item.key)

    registry.set_execution_state(item.key, executing=False, lease_count=0)
    frozen = registry.freeze(item.key)
    registry.set_execution_state(item.key, executing=False, lease_count=1)
    with pytest.raises(PinnedKVRangeError):
        registry.detach(frozen)


def test_expected_epoch_rejects_stale_freeze_and_restore() -> None:
    registry = SessionKVRegistry()
    item = descriptor(kv_epoch=7)
    registry.attach(item)

    with pytest.raises(StaleKVEpochError):
        registry.freeze(item.key, expected_kv_epoch=6)

    registry.detach(registry.freeze(item.key, expected_kv_epoch=7))
    with pytest.raises(StaleKVEpochError):
        registry.attach_restored(replace(item, kv_epoch=6), expected_kv_epoch=7)


def test_restored_branch_returns_to_reclaimable_residency() -> None:
    registry = SessionKVRegistry()
    original = descriptor(kv_epoch=3)
    registry.attach(original)
    registry.detach(registry.freeze(original.key))
    restored = replace(original, kv_epoch=4, kv_indices=(20, 21, 22))

    registry.attach_restored(restored, expected_kv_epoch=4)

    snapshot = registry.snapshot(original.key)
    assert snapshot.descriptor == restored
    assert snapshot.state is KVRangeState.RESIDENT_RECLAIMABLE
    assert snapshot.executing is False
    assert snapshot.lease_count == 0


def test_duplicate_indices_inside_one_branch_are_rejected() -> None:
    registry = SessionKVRegistry()

    with pytest.raises(ValueError, match="duplicate KV indices"):
        registry.attach(descriptor(indices=(1, 1)))
