from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.multi_model.uma.model_adapter import (
    BoundModelRuntime,
    ExecutionResourceBundle,
    ModelAdapter,
)
from sglang.multi_model.uma.weight_plan import WeightGroupSpec, WeightPlan
from sglang.multi_model.uma.weight_runtime import WeightRuntime


@dataclass(slots=True)
class ManagedModelRuntime:
    deployment_id: str
    placement_version: int
    instance_id: str
    stage_id: str
    resource_epoch: int
    checkpoint_digest: str
    model_config: Any
    load_config: Any
    adapter: ModelAdapter
    module: Any
    plan: WeightPlan
    weight_runtime: WeightRuntime
    execution_resources: ExecutionResourceBundle
    runtime_buffer_bytes: int = 0
    published: bool = False

    def group(self, group_id: str) -> WeightGroupSpec:
        for group in self.plan.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(f"unknown weight group for {self.instance_id}: {group_id}")

    @property
    def required_groups(self) -> frozenset[str]:
        return frozenset(group.group_id for group in self.plan.groups)

    @property
    def ready_groups(self) -> frozenset[str]:
        return self.weight_runtime.readiness(self.instance_id)

    @property
    def complete(self) -> bool:
        return self.ready_groups == self.required_groups

    @property
    def resident_bytes(self) -> int:
        return self.runtime_buffer_bytes + sum(
            group.resident_bytes
            for group in self.weight_runtime.resident_groups()
            if getattr(group.identity, "instance_id", None) == self.instance_id
        )

    def bindable_runtime(self) -> BoundModelRuntime:
        if not self.complete:
            missing = sorted(self.required_groups - self.ready_groups)
            raise ValueError(f"cannot publish missing weight groups: {missing}")
        return self.adapter.bind_runtime(
            self.deployment_id,
            self.placement_version,
            self.instance_id,
            self.stage_id,
            self.resource_epoch,
            self.module,
            self.plan,
            sorted(self.ready_groups),
            kv_layout=(
                f"{type(self.execution_resources.token_to_kv_pool).__module__}."
                f"{type(self.execution_resources.token_to_kv_pool).__qualname__}"
            ),
            weight_epoch=self.resource_epoch,
            execution_resources=self.execution_resources,
        )


class ManagedRuntimeRegistry:
    def __init__(self) -> None:
        self._records: dict[str, ManagedModelRuntime] = {}

    def add(self, record: ManagedModelRuntime) -> None:
        if record.instance_id in self._records:
            raise ValueError(f"model runtime already prepared: {record.instance_id}")
        self._records[record.instance_id] = record

    def require(self, instance_id: str) -> ManagedModelRuntime:
        try:
            return self._records[instance_id]
        except KeyError as exc:
            raise KeyError(f"model runtime is not prepared: {instance_id}") from exc

    def all(self) -> tuple[ManagedModelRuntime, ...]:
        return tuple(self._records[key] for key in sorted(self._records))
