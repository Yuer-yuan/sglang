from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"


def _package(name: str, path: Path) -> None:
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
weight_plan = _load(
    "sglang.multi_model.uma.weight_plan",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "weight_plan.py",
)
model_adapter = _load(
    "sglang.multi_model.uma.model_adapter",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "model_adapter.py",
)
weight_runtime = _load(
    "sglang.multi_model.uma.weight_runtime",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "weight_runtime.py",
)


GroupValidation = model_adapter.GroupValidation
TensorExtent = weight_plan.TensorExtent
WeightGroupSpec = weight_plan.WeightGroupSpec
WeightKind = weight_plan.WeightKind
PinnedWeightConflict = weight_runtime.PinnedWeightConflict
StaleWeightEpoch = weight_runtime.StaleWeightEpoch
BoundedWeightReservation = weight_runtime.BoundedWeightReservation
LocalLeaseTable = weight_runtime.LocalLeaseTable
WeightResidencyState = weight_runtime.WeightResidencyState
WeightRuntime = weight_runtime.WeightRuntime


@dataclass(frozen=True)
class Identity:
    instance_id: str = "model-a"
    resource_epoch: int = 1
    operation_id: str = "op-1"


class Reservation:
    def __init__(
        self,
        final_bytes: int = 16,
        staging_bytes: int = 16,
        *,
        fail_commit: bool = False,
    ):
        self.final_bytes = final_bytes
        self.staging_bytes = staging_bytes
        self.active = True
        self.committed = None
        self.fail_commit = fail_commit

    def commit(self, resident_bytes: int):
        assert self.active
        if self.fail_commit:
            raise RuntimeError("injected commit failure")
        self.committed = resident_bytes
        self.active = False

    def release(self):
        self.active = False


class Lease:
    def __init__(self, table, resources):
        self.table = table
        self.resources = resources
        self.active = True

    def release(self):
        if not self.active:
            return
        for resource in self.resources:
            self.table.holds[resource] -= 1
        self.active = False


class LeaseTable:
    def __init__(self):
        self.holds = {}

    def acquire(self, resources, owner):
        assert owner
        for resource in resources:
            self.holds[resource] = self.holds.get(resource, 0) + 1
        return Lease(self, resources)

    def is_evictable(self, resource):
        return self.holds.get(resource, 0) == 0


class Allocator:
    def __init__(self):
        self.allocated = 0
        self.reserved = 0

    def synchronize(self):
        pass

    def allocated_bytes(self):
        return self.allocated

    def reserved_bytes(self):
        return self.reserved

    def trim(self):
        self.reserved = self.allocated


class Reader:
    def __init__(self):
        self.calls = []

    def iter_group(self, group, staging_bytes):
        self.calls.append((group.group_id, staging_bytes))
        yield group.tensors[0].name, object()


class Module:
    def __init__(self):
        self.materialized = False


class Adapter:
    def __init__(
        self,
        allocator,
        *,
        valid=True,
        allocator_bytes=16,
        materialize_interference=0,
        release_interference=0,
    ):
        self.allocator = allocator
        self.valid = valid
        self.allocator_bytes = allocator_bytes
        self.materialize_interference = materialize_interference
        self.release_interference = release_interference
        self.releases = 0

    def group_storage_bytes(self, module, group):
        return 16

    def group_is_meta(self, module, group):
        return not module.materialized

    def materialize_group(self, module, group, device):
        assert device == "cuda"
        module.materialized = True
        self.allocator.allocated += (
            self.allocator_bytes + self.materialize_interference
        )
        self.allocator.reserved += 32
        return (object(),)

    def load_group(self, module, group, tensors):
        observed = list(tensors)
        assert len(observed) == 1
        assert observed[0][0] == group.tensors[0].name

    def validate_group(self, module, group):
        errors = () if self.valid else ("injected validation failure",)
        return GroupValidation(self.valid, errors, 16)

    def release_group(self, module, group):
        assert module.materialized
        module.materialized = False
        self.allocator.allocated -= (
            self.allocator_bytes + self.release_interference
        )
        self.releases += 1
        return 16


def group():
    extent = TensorExtent(
        name="model.layers.0.weight",
        file="/model/model.safetensors",
        offset=8,
        nbytes=16,
        shape=(2, 4),
        dtype="BF16",
        checksum="sha256:" + "a" * 64,
    )
    return WeightGroupSpec(
        group_id="layers-0-1",
        kind=WeightKind.TRANSFORMER_LAYERS,
        layer_range=(0, 1),
        tensors=(extent,),
        logical_bytes=16,
    )


def runtime():
    allocator = Allocator()
    return WeightRuntime(
        reader=Reader(),
        lease_table=LeaseTable(),
        device="cuda",
        allocator=allocator,
    )


def adapter(
    subject,
    *,
    valid=True,
    allocator_bytes=16,
    materialize_interference=0,
    release_interference=0,
):
    return Adapter(
        subject.allocator,
        valid=valid,
        allocator_bytes=allocator_bytes,
        materialize_interference=materialize_interference,
        release_interference=release_interference,
    )


def test_loading_uses_reserved_staging_and_publishes_after_validation():
    subject = runtime()
    reservation = Reservation()
    module = Module()
    loaded = subject.load_group(
        Identity(), adapter(subject), module, group(), reservation
    )

    assert loaded.state is WeightResidencyState.RESIDENT_EVICTABLE
    assert loaded.resident_bytes == 16
    assert reservation.committed == 16
    assert subject.reader.calls == [("layers-0-1", 16)]
    assert subject.readiness("model-a") == frozenset({"layers-0-1"})


def test_unrelated_free_does_not_reject_validated_group_load():
    subject = runtime()
    subject.allocator.allocated = 64
    reservation = Reservation()

    loaded = subject.load_group(
        Identity(),
        adapter(subject, materialize_interference=-32),
        Module(),
        group(),
        reservation,
    )

    assert loaded.parameter_bytes == 16
    assert loaded.resident_bytes == 16
    assert reservation.committed == 16
    assert subject.readiness("model-a") == frozenset({"layers-0-1"})


def test_failed_validation_releases_storage_and_reservation():
    subject = runtime()
    reservation = Reservation()
    module = Module()
    adapter_instance = adapter(subject, valid=False)

    with pytest.raises(weight_runtime.GroupValidationError):
        subject.load_group(
            Identity(), adapter_instance, module, group(), reservation
        )

    assert not module.materialized
    assert adapter_instance.releases == 1
    assert not reservation.active
    assert subject.readiness("model-a") == frozenset()


def test_pinned_group_cannot_be_evicted():
    subject = runtime()
    subject.load_group(
        Identity(), adapter(subject), Module(), group(), Reservation()
    )
    lease = subject.pin(["layers-0-1"], "batch-7", instance_id="model-a")

    assert subject.resident_groups()[0].state is WeightResidencyState.PINNED
    with pytest.raises(PinnedWeightConflict):
        subject.evict_group("layers-0-1", 1, instance_id="model-a")

    lease.release()
    assert (
        subject.resident_groups()[0].state
        is WeightResidencyState.RESIDENT_EVICTABLE
    )
    released = subject.evict_group("layers-0-1", 1, instance_id="model-a")
    assert released.released_bytes == 16
    assert released.logical_released_bytes == 16
    assert released.allocator_released_bytes == 16
    assert released.parameter_bytes == 16
    assert released.allocator_reserved_released_bytes == 32
    assert released.backing_left_slot_ownership is False
    assert released.ownership_mechanism == "torch-storage-release"
    assert subject.readiness("model-a") == frozenset()


def test_allocator_noise_after_release_does_not_turn_eviction_into_failure():
    subject = runtime()
    module = Module()
    loaded = subject.load_group(
        Identity(),
        adapter(subject, release_interference=5),
        module,
        group(),
        Reservation(),
    )
    subject.allocator.allocated += 5

    released = subject.evict_group(
        "layers-0-1",
        1,
        instance_id="model-a",
    )

    assert not module.materialized
    assert released.released_bytes == loaded.resident_bytes == 16
    assert released.logical_released_bytes == 16
    assert released.allocator_released_bytes == 21
    assert released.backing_left_slot_ownership is False
    assert subject.readiness("model-a") == frozenset()


def test_stale_epoch_does_not_release_resident_group():
    subject = runtime()
    module = Module()
    subject.load_group(
        Identity(), adapter(subject), module, group(), Reservation()
    )

    with pytest.raises(StaleWeightEpoch):
        subject.evict_group("layers-0-1", 2, instance_id="model-a")

    assert module.materialized
    assert subject.readiness("model-a") == frozenset({"layers-0-1"})


def test_reservation_must_cover_final_and_largest_staging_extent():
    subject = runtime()
    too_small_final = Reservation(final_bytes=15)
    with pytest.raises(weight_runtime.WeightRuntimeError, match="resident bytes"):
        subject.load_group(
            Identity(), adapter(subject), Module(), group(), too_small_final
        )
    assert not too_small_final.active

    too_small_staging = Reservation(staging_bytes=15)
    with pytest.raises(weight_runtime.WeightRuntimeError, match="staging bytes"):
        subject.load_group(
            Identity(), adapter(subject), Module(), group(), too_small_staging
        )
    assert not too_small_staging.active


def test_allocator_rounding_above_final_reservation_rolls_back():
    subject = runtime()
    module = Module()
    reservation = Reservation(final_bytes=16)

    with pytest.raises(weight_runtime.WeightRuntimeError, match="allocator uses"):
        subject.load_group(
            Identity(),
            adapter(subject, allocator_bytes=17),
            module,
            group(),
            reservation,
        )

    assert not module.materialized
    assert not reservation.active
    assert subject.allocator.allocated == 0
    assert subject.allocator.reserved == 0
    assert subject.readiness("model-a") == frozenset()


def test_commit_failure_rolls_back_unpublished_storage():
    subject = runtime()
    module = Module()
    reservation = Reservation(fail_commit=True)

    with pytest.raises(RuntimeError, match="injected commit failure"):
        subject.load_group(
            Identity(), adapter(subject), module, group(), reservation
        )

    assert not module.materialized
    assert not reservation.active
    assert subject.allocator.allocated == 0
    assert subject.allocator.reserved == 0
    assert subject.readiness("model-a") == frozenset()


def test_extent_reader_rejects_checksum_mismatch_and_short_read(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"12345678")
    reader = weight_runtime.SafetensorsExtentReader()
    bad_checksum = TensorExtent(
        name="bad-checksum",
        file=str(path),
        offset=0,
        nbytes=8,
        shape=(4,),
        dtype="BF16",
        checksum="sha256:" + "0" * 64,
    )
    with pytest.raises(weight_runtime.WeightRuntimeError, match="checksum mismatch"):
        reader._read_extent(bad_checksum)

    short_read = TensorExtent(
        name="short-read",
        file=str(path),
        offset=0,
        nbytes=16,
        shape=(8,),
        dtype="BF16",
        checksum="sha256:" + "0" * 64,
    )
    with pytest.raises(weight_runtime.WeightRuntimeError, match="short read"):
        reader._read_extent(short_read)

    assert reader.read_count == 0
    assert reader.bytes_read == 0


def test_adopt_bootstrap_group_without_reading_or_allocating_again():
    subject = runtime()
    module = Module()
    module.materialized = True
    subject.allocator.allocated = 16

    adopted = subject.adopt_group(
        Identity(),
        adapter(subject),
        module,
        group(),
    )

    assert adopted.parameter_bytes == 16
    assert adopted.resident_bytes == 16
    assert subject.reader.calls == []
    assert subject.allocator.allocated == 16
    assert subject.readiness("model-a") == frozenset({"layers-0-1"})


def test_local_lease_table_and_bounded_reservation_enforce_worker_limits():
    table = LocalLeaseTable()
    resource = Identity()
    lease = table.acquire((resource,), "batch-1")
    assert not table.is_evictable(resource)
    lease.release()
    assert table.is_evictable(resource)

    reservation = BoundedWeightReservation(16, 8)
    reservation.commit(12)
    assert reservation.committed_bytes == 12
    assert not reservation.active

    with pytest.raises(ValueError, match="exceed"):
        BoundedWeightReservation(16, 8).commit(17)
