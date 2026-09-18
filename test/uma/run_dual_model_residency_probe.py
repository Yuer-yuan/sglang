from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import socket

from sglang.srt.distributed import destroy_distributed_environment
from sglang.srt.managers.io_struct import (
    LoadWeightGroupReq,
    RegisterModelAdapterReq,
    UMAResourceIdentity,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _identity(instance_id: str, model_path: Path) -> UMAResourceIdentity:
    return UMAResourceIdentity(
        deployment_id=f"probe-{instance_id}",
        placement_version=1,
        instance_id=instance_id,
        stage_id="stage-0",
        resource_kind="MODEL",
        resource_group_id_or_extent_id="model",
        optional_layer_or_block_range=None,
        resource_epoch=1,
        operation_id=f"register-{instance_id}-{model_path.name}",
    )


def _register_request(
    instance_id: str,
    model_path: Path,
    *,
    adopt_bootstrap: bool,
) -> RegisterModelAdapterReq:
    return RegisterModelAdapterReq(
        identity=_identity(instance_id, model_path),
        model_path=str(model_path),
        checkpoint_digest=f"probe:{model_path.name}",
        architecture="probe",
        kv_layout="probe",
        owns_input_embedding=True,
        owns_final_norm=True,
        owns_output_head=True,
        group_layer_count=4,
        max_total_tokens=256,
        max_running_requests=2,
        adopt_bootstrap=adopt_bootstrap,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    args = parser.parse_args()

    server_args = ServerArgs(
        model_path=str(args.bootstrap),
        tokenizer_path=str(args.bootstrap),
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        pp_size=1,
        attention_backend="triton",
        page_size=16,
        max_total_tokens=512,
        max_running_requests=2,
        mem_fraction_static=0.65,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        disable_custom_all_reduce=True,
        disable_overlap_schedule=True,
        skip_tokenizer_init=True,
    )
    worker = None
    try:
        worker = TpModelWorker(
            server_args=server_args,
            gpu_id=0,
            tp_rank=0,
            pp_rank=0,
            dp_rank=None,
            nccl_port=_free_port(),
        )
        model_a = worker.prepare_uma_model(
            _register_request(
                "model-a",
                args.bootstrap,
                adopt_bootstrap=True,
            )
        )
        worker.register_uma_runtime(model_a.bindable_runtime())
        bind_a = worker.bind_uma_runtime(
            "model-a",
            placement_version=1,
            resource_epoch=1,
        )
        if not bind_a.accepted:
            raise RuntimeError(bind_a)

        model_b = worker.prepare_uma_model(
            _register_request(
                "model-b",
                args.second,
                adopt_bootstrap=False,
            )
        )
        loaded = []
        for index, group in enumerate(model_b.plan.groups):
            identity = replace(
                _identity("model-b", args.second),
                resource_kind="WEIGHT",
                resource_group_id_or_extent_id=group.group_id,
                optional_layer_or_block_range=group.layer_range,
                operation_id=f"load-model-b-{index}",
            )
            expected = model_b.adapter.group_storage_bytes(
                model_b.module,
                group,
            )
            record, result = worker.load_uma_weight_group(
                LoadWeightGroupReq(
                    identity=identity,
                    final_bytes=expected + 64 * 1024 * 1024,
                    staging_bytes=max(item.nbytes for item in group.tensors),
                )
            )
            loaded.append(
                {
                    "group_id": group.group_id,
                    "logical_bytes": group.logical_bytes,
                    "resident_bytes": result.resident_bytes,
                }
            )
        worker.register_uma_runtime(model_b.bindable_runtime())
        bind_b = worker.bind_uma_runtime(
            "model-b",
            placement_version=1,
            resource_epoch=1,
        )
        bind_a_again = worker.bind_uma_runtime(
            "model-a",
            placement_version=1,
            resource_epoch=1,
        )
        result = {
            "active_instance_id": worker.model_runner.active_instance_id,
            "bind_codes": [
                bind_a.code.value,
                bind_b.code.value,
                bind_a_again.code.value,
            ],
            "bootstrap_architecture": model_a.model_config.hf_config.architectures[0],
            "bootstrap_groups": len(model_a.plan.groups),
            "bootstrap_resident_bytes": model_a.resident_bytes,
            "bootstrap_runtime_buffer_bytes": model_a.runtime_buffer_bytes,
            "loaded": loaded,
            "second_architecture": model_b.model_config.hf_config.architectures[0],
            "second_groups": len(model_b.plan.groups),
            "second_resident_bytes": model_b.resident_bytes,
            "second_runtime_buffer_bytes": model_b.runtime_buffer_bytes,
            "second_meta_buffers": [
                name
                for name, buffer in model_b.module.named_buffers()
                if buffer.device.type == "meta"
            ],
            "weight_file_reads": worker.uma_weight_file_reads,
        }
        if (
            result["bind_codes"] != ["OK", "OK", "OK"]
            or not model_a.complete
            or not model_b.complete
            or result["second_meta_buffers"]
        ):
            raise RuntimeError(json.dumps(result, sort_keys=True))
        print(json.dumps(result, sort_keys=True))
    finally:
        if worker is not None:
            del worker
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
