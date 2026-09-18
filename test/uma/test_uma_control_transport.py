from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


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


def _module(name: str, **attributes) -> None:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


_package("sglang", PYTHON_ROOT / "sglang")
_package("sglang.srt", PYTHON_ROOT / "sglang" / "srt")
_package("sglang.srt.managers", PYTHON_ROOT / "sglang" / "srt" / "managers")
_package("sglang.srt.multimodal", PYTHON_ROOT / "sglang" / "srt" / "multimodal")
_package("sglang.srt.sampling", PYTHON_ROOT / "sglang" / "srt" / "sampling")
_module(
    "sglang.srt.managers.schedule_batch",
    BaseFinishReason=object,
)
_module(
    "sglang.srt.multimodal.mm_utils",
    has_valid_data=lambda _: False,
)
_module(
    "sglang.srt.sampling.sampling_params",
    SamplingParams=object,
)
io_struct = _load(
    "sglang.srt.managers.io_struct",
    PYTHON_ROOT / "sglang" / "srt" / "managers" / "io_struct.py",
)


def _identity():
    return io_struct.UMAResourceIdentity(
        deployment_id="deployment-a",
        placement_version=3,
        instance_id="model-a",
        stage_id="stage-0",
        resource_kind="WEIGHT",
        resource_group_id_or_extent_id="layers-0-4",
        optional_layer_or_block_range=(0, 4),
        resource_epoch=7,
        operation_id="bind-a",
    )


def _async_method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in class_node.body
        if isinstance(node, ast.AsyncFunctionDef)
    }


def test_uma_control_request_keeps_full_versioned_identity():
    identity = _identity()
    request = io_struct.BindInstanceReq(identity, expected_weight_epoch=11)

    assert request.identity.instance_id == "model-a"
    assert request.identity.placement_version == 3
    assert request.identity.resource_epoch == 7
    assert request.expected_weight_epoch == 11


def test_tokenizer_transport_exposes_bind_and_quiesce_methods():
    methods = _async_method_names(
        PYTHON_ROOT / "sglang" / "srt" / "managers" / "tokenizer_manager.py",
        "TokenizerManager",
    )

    assert {"bind_uma_instance", "quiesce_uma_instance", "_send_uma_control"} <= methods


def test_http_server_declares_both_uma_control_routes():
    source = (
        PYTHON_ROOT / "sglang" / "srt" / "entrypoints" / "http_server.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    routes = {
        decorator.args[0].value
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "post"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
        )
    }

    assert "/uma/bind_instance" in routes
    assert "/uma/quiesce_instance" in routes
