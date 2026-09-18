from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from sglang.srt.distributed import destroy_distributed_environment
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    server_args = ServerArgs(
        model_path=str(args.model_path),
        tokenizer_path=str(args.model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        pp_size=1,
        attention_backend="triton",
        page_size=16,
        max_total_tokens=1024,
        max_running_requests=2,
        mem_fraction_static=0.55,
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
        runtime = worker.capture_current_uma_runtime(
            deployment_id="probe-deployment",
            placement_version=1,
            instance_id="probe-instance",
            stage_id="stage-0",
            resource_epoch=1,
            weight_epoch=1,
        )
        worker.register_uma_runtime(runtime)
        bind = worker.bind_uma_runtime(
            runtime.instance_id,
            placement_version=runtime.placement_version,
            resource_epoch=runtime.resource_epoch,
        )
        if not bind.accepted:
            raise RuntimeError(bind)
        stale = worker.bind_uma_runtime(
            runtime.instance_id,
            placement_version=runtime.placement_version + 1,
            resource_epoch=runtime.resource_epoch,
        )
        if stale.accepted:
            raise RuntimeError("stale placement version was accepted")
        result = {
            "active_instance_id": worker.model_runner.active_instance_id,
            "architecture": runtime.model_config.hf_config.architectures[0],
            "bind_code": bind.code.value,
            "cuda_graph_enabled": runtime.execution_resources.cuda_graph_runner
            is not None,
            "kv_layout": runtime.kv_layout,
            "max_req_input_len": worker.max_req_input_len,
            "max_req_len": worker.max_req_len,
            "max_running_requests": worker.max_running_requests,
            "max_total_num_tokens": worker.max_total_num_tokens,
            "resource_epoch": worker.model_runner.active_resource_epoch,
            "stale_bind_code": stale.code.value,
            "worker_in_flight_count": worker.uma_in_flight_count,
        }
        if (
            result["active_instance_id"] != runtime.instance_id
            or result["bind_code"] != "OK"
            or result["worker_in_flight_count"] != 0
        ):
            raise RuntimeError(json.dumps(result, sort_keys=True))
        print(json.dumps(result, sort_keys=True))
    finally:
        if worker is not None:
            del worker
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
