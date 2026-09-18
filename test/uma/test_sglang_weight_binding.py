from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"


def _package(name: str, path: Path) -> None:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_package("sglang", PYTHON_ROOT / "sglang")
_package("sglang.multi_model", PYTHON_ROOT / "sglang" / "multi_model")
_package("sglang.multi_model.uma", PYTHON_ROOT / "sglang" / "multi_model" / "uma")
_load(
    "sglang.multi_model.uma.weight_plan",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "weight_plan.py",
)
model_adapter = _load(
    "sglang.multi_model.uma.model_adapter",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "model_adapter.py",
)
execution_slot = _load(
    "sglang.multi_model.uma.execution_slot",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "execution_slot.py",
)


BoundModelRuntime = model_adapter.BoundModelRuntime
ExecutionResourceBundle = model_adapter.ExecutionResourceBundle
UMAControlCode = execution_slot.UMAControlCode
UMAExecutionSlot = execution_slot.UMAExecutionSlot
assess_scheduler_safe_point = execution_slot.assess_scheduler_safe_point


class Module:
    def __init__(self) -> None:
        self.logits_processor = object()
        self.runtime_buffers = []

    def forward(self, input_ids, positions, forward_batch):
        return input_ids

    def named_buffers(self):
        return tuple(self.runtime_buffers)


class WeightLease:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def release(self) -> None:
        self.runtime.pins -= 1
        if self.runtime.fail_release:
            raise RuntimeError("injected lease release failure")


class WeightRuntime:
    def __init__(
        self,
        ready=frozenset({"layer-0"}),
        *,
        fail_release: bool = False,
    ) -> None:
        self.ready = ready
        self.pins = 0
        self.fail_release = fail_release

    def readiness(self, instance_id):
        assert instance_id
        return self.ready

    def pin(self, groups, owner, *, instance_id):
        assert groups and owner and instance_id
        self.pins += 1
        return WeightLease(self)


def runtime(
    instance_id: str = "model-b",
    *,
    placement_version: int = 3,
    resource_epoch: int = 7,
    required: frozenset[str] = frozenset({"layer-0"}),
    ready: frozenset[str] = frozenset({"layer-0"}),
    missing_resource: str | None = None,
    weight_runtime=None,
) -> BoundModelRuntime:
    module = Module()
    resources = {
        "attention_backend": object(),
        "req_to_token_pool": object(),
        "token_to_kv_pool": object(),
        "token_to_kv_pool_allocator": object(),
        "sampler": object(),
        "logits_processor": module.logits_processor,
        "kv_cache_dtype": object(),
    }
    if missing_resource is not None:
        resources[missing_resource] = None
    bundle = ExecutionResourceBundle(
        **resources,
        max_total_num_tokens=1024,
        max_running_requests=8,
        max_req_len=1023,
        max_req_input_len=1018,
        start_layer=0,
        end_layer=4,
        weight_runtime=weight_runtime or WeightRuntime(ready),
    )
    return BoundModelRuntime(
        deployment_id="deployment-1",
        placement_version=placement_version,
        instance_id=instance_id,
        stage_id="stage-0",
        resource_epoch=resource_epoch,
        model_config=SimpleNamespace(dtype="bf16"),
        module=module,
        kv_layout="mha:4x2x64",
        weight_epoch=11,
        required_weight_groups=required,
        ready_weight_groups=ready,
        execution_resources=bundle,
    )


def test_partial_weight_runtime_never_enters_registry():
    slot = UMAExecutionSlot()
    candidate = runtime(ready=frozenset())

    with pytest.raises(ValueError, match="missing weight groups"):
        slot.register(candidate)

    with pytest.raises(KeyError, match="not registered"):
        slot.get(candidate.instance_id)


def test_partial_execution_resource_bundle_is_rejected():
    slot = UMAExecutionSlot()

    with pytest.raises(ValueError, match="attention_backend"):
        slot.register(runtime(missing_resource="attention_backend"))


def test_runtime_with_meta_constructor_buffer_is_rejected_before_forward():
    slot = UMAExecutionSlot()
    candidate = runtime()
    candidate.module.runtime_buffers.append(
        (
            "model.layers.0.self_attn.rotary_emb.cos_sin_cache",
            SimpleNamespace(device=SimpleNamespace(type="meta")),
        )
    )

    with pytest.raises(ValueError, match="unmaterialized model buffers"):
        slot.register(candidate)


def test_bind_is_deferred_while_forward_lease_is_active():
    slot = UMAExecutionSlot()
    candidate = runtime()
    slot.register(candidate)
    entered = threading.Event()
    release = threading.Event()

    def forward() -> None:
        with slot.forward_lease():
            entered.set()
            assert release.wait(timeout=2)

    thread = threading.Thread(target=forward)
    thread.start()
    assert entered.wait(timeout=2)

    calls = []
    result = slot.bind(
        candidate.instance_id,
        placement_version=3,
        resource_epoch=7,
        binder=calls.append,
    )

    assert result.code is UMAControlCode.PINNED_RESOURCE_CONFLICT
    assert calls == []
    assert slot.active_instance_id is None
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()

    result = slot.bind(
        candidate.instance_id,
        placement_version=3,
        resource_epoch=7,
        binder=calls.append,
    )
    assert result.accepted
    assert calls == [candidate]
    assert slot.active_instance_id == candidate.instance_id


def test_forward_revalidates_and_pins_managed_weight_groups():
    weights = WeightRuntime()
    slot = UMAExecutionSlot()
    candidate = runtime(weight_runtime=weights)
    slot.register(candidate)
    assert slot.bind(
        candidate.instance_id,
        placement_version=3,
        resource_epoch=7,
        binder=lambda _: None,
    ).accepted

    with slot.forward_lease(owner="batch-1"):
        assert weights.pins == 1
    assert weights.pins == 0

    weights.ready = frozenset()
    with pytest.raises(execution_slot.IncompleteRuntimeError, match="lost ready"):
        with slot.forward_lease(owner="batch-2"):
            pytest.fail("forward entered with evicted weights")
    assert slot.in_flight_count == 0


def test_failed_binder_does_not_publish_new_active_instance():
    slot = UMAExecutionSlot()
    old = runtime("model-a")
    new = runtime("model-b")
    slot.register(old)
    slot.register(new)
    assert slot.bind(
        "model-a",
        placement_version=3,
        resource_epoch=7,
        binder=lambda _: None,
    ).accepted

    def fail(_: BoundModelRuntime) -> None:
        raise RuntimeError("injected runner bind failure")

    result = slot.bind(
        "model-b",
        placement_version=3,
        resource_epoch=7,
        binder=fail,
    )

    assert result.code is UMAControlCode.MODEL_BIND_FAILURE
    assert "injected runner bind failure" in result.reason
    assert slot.active_instance_id == "model-a"


def test_active_runtime_cannot_be_replaced_without_a_bind_transaction():
    slot = UMAExecutionSlot()
    candidate = runtime()
    slot.register(candidate)
    assert slot.bind(
        candidate.instance_id,
        placement_version=3,
        resource_epoch=7,
        binder=lambda _: None,
    ).accepted

    with pytest.raises(execution_slot.RuntimeBindError, match="active runtime"):
        slot.register(runtime(resource_epoch=8), replace=True)


def test_failed_weight_lease_release_still_retires_forward_count():
    weights = WeightRuntime(fail_release=True)
    slot = UMAExecutionSlot()
    candidate = runtime(weight_runtime=weights)
    slot.register(candidate)
    assert slot.bind(
        candidate.instance_id,
        placement_version=3,
        resource_epoch=7,
        binder=lambda _: None,
    ).accepted

    with pytest.raises(RuntimeError, match="lease release failure"):
        with slot.forward_lease(owner="batch-1"):
            pass
    assert slot.in_flight_count == 0


@pytest.mark.parametrize(
    ("placement_version", "resource_epoch", "expected"),
    [
        (2, 7, UMAControlCode.STALE_PLACEMENT_VERSION),
        (3, 6, UMAControlCode.STALE_RESOURCE_EPOCH),
    ],
)
def test_bind_rejects_stale_identity(
    placement_version: int,
    resource_epoch: int,
    expected: UMAControlCode,
):
    slot = UMAExecutionSlot()
    slot.register(runtime())

    result = slot.bind(
        "model-b",
        placement_version=placement_version,
        resource_epoch=resource_epoch,
        binder=lambda _: pytest.fail("stale bind reached the data plane"),
    )

    assert result.code is expected


def test_scheduler_safe_point_is_a_fact_not_a_drain_policy():
    blocked = assess_scheduler_safe_point(
        waiting_count=2,
        running_counts=(1, 3),
        grammar_count=1,
        session_count=2,
        has_chunked_request=True,
        overlap_enabled=False,
        overlap_result_count=0,
        worker_in_flight_count=1,
    )
    assert blocked.code is UMAControlCode.PINNED_RESOURCE_CONFLICT
    assert blocked.in_flight_count == 1
    assert blocked.reasons == (
        "2 queued request(s)",
        "4 running request(s)",
        "1 grammar request(s)",
        "2 session(s) still attached",
        "chunked prefill is active",
        "1 worker forward(s) in flight",
    )

    ready = assess_scheduler_safe_point(
        waiting_count=0,
        running_counts=(0,),
        grammar_count=0,
        session_count=0,
        has_chunked_request=False,
        overlap_enabled=False,
        overlap_result_count=0,
        worker_in_flight_count=0,
    )
    assert ready.accepted
