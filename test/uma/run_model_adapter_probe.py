from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

import torch

from sglang.multi_model.uma.model_adapter import default_adapter_registry
from sglang.multi_model.uma.weight_plan import (
    StageWeightScope,
    build_safetensors_catalog,
    build_weight_plan,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
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
    initialize_dp_attention(
        enable_dp_attention=False,
        tp_rank=0,
        tp_size=1,
        dp_size=1,
        moe_dense_tp_size=None,
        pp_size=1,
    )
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
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        module = adapter.build_meta_module()
        physical_weight_bytes = sum(
            adapter.group_storage_bytes(module, group) for group in plan.groups
        )
        parameters = list(module.named_parameters(remove_duplicate=False))
        runtime_names = {name for name, _ in parameters}
        planned_runtime_names = {
            name
            for group in plan.groups
            for name in adapter.runtime_parameter_names(group)
        }
        missing_runtime_names = sorted(planned_runtime_names - runtime_names)
        non_meta = [
            name for name, parameter in parameters if parameter.device.type != "meta"
        ]
        result = {
            "adapter": type(adapter).__name__,
            "architecture": model_config.hf_config.architectures[0],
            "cuda_available": torch.cuda.is_available(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "cuda_storage_bytes": adapter.cuda_storage_bytes(module),
            "cuda_allocated_delta": torch.cuda.memory_allocated() - allocated_before,
            "cuda_reserved_delta": torch.cuda.memory_reserved() - reserved_before,
            "logical_weight_bytes": plan.total_bytes,
            "physical_weight_bytes": physical_weight_bytes,
            "weight_groups": len(plan.groups),
            "missing_runtime_names": missing_runtime_names,
            "non_meta_parameters": non_meta,
            "parameter_references": len(parameters),
            "unique_parameters": len({id(parameter) for _, parameter in parameters}),
        }
        if (
            non_meta
            or missing_runtime_names
            or result["cuda_storage_bytes"] != 0
            or result["cuda_allocated_delta"] != 0
            or result["cuda_reserved_delta"] != 0
        ):
            raise RuntimeError(json.dumps(result, sort_keys=True))
        print(json.dumps(result, sort_keys=True))
    finally:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
