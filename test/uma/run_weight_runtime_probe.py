from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import time

import torch

from sglang.multi_model.uma.model_adapter import default_adapter_registry
from sglang.multi_model.uma.weight_plan import (
    StageWeightScope,
    build_safetensors_catalog,
    build_weight_plan,
)
from sglang.multi_model.uma.weight_runtime import (
    PinnedWeightConflict,
    SafetensorsExtentReader,
    WeightRuntime,
)
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@dataclass(frozen=True)
class Identity:
    instance_id: str
    resource_epoch: int
    operation_id: str


class Reservation:
    def __init__(self, final_bytes: int, staging_bytes: int) -> None:
        self.final_bytes = final_bytes
        self.staging_bytes = staging_bytes
        self.active = True
        self.committed_bytes: int | None = None

    def commit(self, resident_bytes: int) -> None:
        if not self.active:
            raise RuntimeError("reservation is inactive")
        self.committed_bytes = resident_bytes
        self.active = False

    def release(self) -> None:
        self.active = False


class Lease:
    def __init__(self, table: "LeaseTable", resources: tuple[object, ...]) -> None:
        self.table = table
        self.resources = resources
        self.active = True

    def release(self) -> None:
        if not self.active:
            return
        for resource in self.resources:
            self.table.holds[resource] -= 1
        self.active = False


class LeaseTable:
    def __init__(self) -> None:
        self.holds: dict[object, int] = {}

    def acquire(self, resources: tuple[object, ...], owner: str) -> Lease:
        if not owner:
            raise ValueError("owner must not be empty")
        for resource in resources:
            self.holds[resource] = self.holds.get(resource, 0) + 1
        return Lease(self, resources)

    def is_evictable(self, resource: object) -> bool:
        return self.holds.get(resource, 0) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--group-id", default="layers-0-1")
    args = parser.parse_args()

    model_config = ModelConfig(
        str(args.model_path),
        dtype="bfloat16",
        model_override_args="{}",
    )
    stage = StageWeightScope(
        stage_id="stage-0",
        layer_range=(0, model_config.hf_config.num_hidden_layers),
        owns_input_embedding=True,
        owns_final_norm=True,
        owns_output_head=True,
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
        backend="gloo",
    )
    initialize_model_parallel(1, 1, backend="gloo")
    initialize_dp_attention(False, 0, 1, 1, None, 1)
    try:
        adapter = default_adapter_registry().create(
            model_config,
            LoadConfig(load_format="safetensors"),
            stage,
        )
        catalog = build_safetensors_catalog(args.model_path)
        plan = build_weight_plan(
            catalog,
            adapter,
            model_digest=f"probe:{args.model_path.name}",
            stage=stage,
            group_layer_count=1,
        )
        group = next(group for group in plan.groups if group.group_id == args.group_id)
        module = adapter.build_meta_module()
        resident_bytes = adapter.group_storage_bytes(module, group)
        staging_bytes = max(extent.nbytes for extent in group.tensors)
        reader = SafetensorsExtentReader()
        runtime = WeightRuntime(
            reader=reader,
            lease_table=LeaseTable(),
            device=torch.device("cuda"),
        )
        identity = Identity("model-a", 1, "load-probe")
        reservation = Reservation(resident_bytes + (1 << 20), staging_bytes)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        start = time.perf_counter()
        loaded = runtime.load_group(
            identity,
            adapter,
            module,
            group,
            reservation,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        allocated_resident = torch.cuda.memory_allocated()
        reserved_resident = torch.cuda.memory_reserved()

        lease = runtime.pin([group.group_id], "batch-probe", instance_id="model-a")
        pinned_rejected = False
        try:
            runtime.evict_group(group.group_id, 1, instance_id="model-a")
        except PinnedWeightConflict:
            pinned_rejected = True
        lease.release()
        released = runtime.evict_group(group.group_id, 1, instance_id="model-a")
        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()

        result = {
            "allocated_after": allocated_after,
            "allocated_before": allocated_before,
            "allocated_resident": allocated_resident,
            "allocation_delta": allocated_resident - allocated_before,
            "architecture": model_config.hf_config.architectures[0],
            "committed_bytes": reservation.committed_bytes,
            "group_id": group.group_id,
            "load_seconds": load_seconds,
            "logical_bytes": group.logical_bytes,
            "pinned_eviction_rejected": pinned_rejected,
            "read_bytes": reader.bytes_read,
            "read_count": reader.read_count,
            "reserved_after": reserved_after,
            "reserved_before": reserved_before,
            "reserved_resident": reserved_resident,
            "allocator_reserved_released_bytes": (
                released.allocator_reserved_released_bytes
            ),
            "released_bytes": released.released_bytes,
            "resident_bytes": loaded.resident_bytes,
            "returned_to_baseline": allocated_after == allocated_before,
            "staging_bytes": staging_bytes,
        }
        if not (
            pinned_rejected
            and loaded.parameter_bytes == resident_bytes
            and reservation.committed_bytes == loaded.resident_bytes
            and reader.read_count == len(group.tensors)
            and reader.bytes_read == group.logical_bytes
            and released.parameter_bytes == resident_bytes
            and released.released_bytes == loaded.resident_bytes
            and allocated_after == allocated_before
            and reserved_after == reserved_before
            and adapter.group_is_meta(module, group)
        ):
            raise RuntimeError(json.dumps(result, sort_keys=True))
        print(json.dumps(result, sort_keys=True))
    finally:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
