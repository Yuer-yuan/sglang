from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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


# Import the UMA modules without importing sglang/__init__.py.  This keeps the
# ownership/registry tests runnable in a lightweight development environment.
_package("sglang", PYTHON_ROOT / "sglang")
_package("sglang.multi_model", PYTHON_ROOT / "sglang" / "multi_model")
_package("sglang.multi_model.uma", PYTHON_ROOT / "sglang" / "multi_model" / "uma")
_package(
    "sglang.multi_model.uma.adapters",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "adapters",
)
weight_plan = _load(
    "sglang.multi_model.uma.weight_plan",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "weight_plan.py",
)
model_adapter = _load(
    "sglang.multi_model.uma.model_adapter",
    PYTHON_ROOT / "sglang" / "multi_model" / "uma" / "model_adapter.py",
)
_load(
    "sglang.multi_model.uma.adapters.dense_decoder",
    PYTHON_ROOT
    / "sglang"
    / "multi_model"
    / "uma"
    / "adapters"
    / "dense_decoder.py",
)
qwen3 = _load(
    "sglang.multi_model.uma.adapters.qwen3",
    PYTHON_ROOT
    / "sglang"
    / "multi_model"
    / "uma"
    / "adapters"
    / "qwen3.py",
)
llama_family = _load(
    "sglang.multi_model.uma.adapters.llama_family",
    PYTHON_ROOT
    / "sglang"
    / "multi_model"
    / "uma"
    / "adapters"
    / "llama_family.py",
)


StageWeightScope = weight_plan.StageWeightScope
TensorExtent = weight_plan.TensorExtent
WeightKind = weight_plan.WeightKind
build_weight_plan = weight_plan.build_weight_plan
ModelAdapterRegistry = model_adapter.ModelAdapterRegistry
Qwen3Adapter = qwen3.Qwen3Adapter
LlamaFamilyAdapter = llama_family.LlamaFamilyAdapter


def _config(architecture: str, *, tied: bool = True):
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[architecture],
            tie_word_embeddings=tied,
        )
    )


def _extent(name: str, checksum: str | None = None) -> TensorExtent:
    return TensorExtent(
        name=name,
        file="/model/model.safetensors",
        offset=0,
        nbytes=16,
        shape=(2, 4),
        dtype="BF16",
        checksum=checksum or f"sha256:{name:0<64}"[:71],
    )


def _full_stage(layer_count: int = 2) -> StageWeightScope:
    return StageWeightScope(
        stage_id="stage-0",
        layer_range=(0, layer_count),
        owns_input_embedding=True,
        owns_final_norm=True,
        owns_output_head=True,
    )


def test_registry_selects_supported_heterogeneous_architectures():
    registry = ModelAdapterRegistry()
    registry.register(Qwen3Adapter)
    registry.register(LlamaFamilyAdapter)

    assert isinstance(
        registry.create(_config("Qwen3ForCausalLM"), None, _full_stage()),
        Qwen3Adapter,
    )
    assert isinstance(
        registry.create(_config("LlamaForCausalLM"), None, _full_stage()),
        LlamaFamilyAdapter,
    )
    assert isinstance(
        registry.create(_config("MistralForCausalLM"), None, _full_stage()),
        LlamaFamilyAdapter,
    )

    with pytest.raises(ValueError, match="unsupported dense decoder"):
        registry.create(_config("PhiForCausalLM"), None, _full_stage())


def test_qwen_plan_unifies_equal_tied_embedding_extents():
    tied_checksum = "sha256:" + "a" * 64
    catalog = {
        item.name: item
        for item in (
            _extent("model.embed_tokens.weight", tied_checksum),
            _extent("lm_head.weight", tied_checksum),
            _extent("model.layers.0.self_attn.q_proj.weight"),
            _extent("model.layers.0.self_attn.k_proj.weight"),
            _extent("model.layers.0.self_attn.v_proj.weight"),
            _extent("model.layers.1.mlp.gate_proj.weight"),
            _extent("model.layers.1.mlp.up_proj.weight"),
            _extent("model.norm.weight"),
        )
    }
    adapter = Qwen3Adapter(_config("Qwen3ForCausalLM"), None, _full_stage())

    plan = build_weight_plan(catalog, adapter, "qwen-digest", _full_stage(), 1)

    tied = plan.groups[0]
    assert tied.kind is WeightKind.TIED_EMBEDDING_HEAD
    assert tied.tensor_names == frozenset(
        {"model.embed_tokens.weight", "lm_head.weight"}
    )
    assert adapter.runtime_parameter_names(tied) == frozenset(
        {"model.embed_tokens.weight"}
    )
    assert [group.group_id for group in plan.groups] == [
        "tied-embedding-head",
        "layers-0-1",
        "layers-1-2",
        "final-norm",
    ]

    class FakeParameter:
        def numel(self):
            return 8

        def element_size(self):
            return 2

    tied_parameter = FakeParameter()

    class FakeModule:
        def named_parameters(self, remove_duplicate=False):
            assert remove_duplicate is False
            return [
                ("model.embed_tokens.weight", tied_parameter),
                ("lm_head.weight", tied_parameter),
            ]

    # The checkpoint contains two equal 16-byte extents, but the runtime has
    # one aliased 16-byte parameter.  Admission must use the latter.
    assert tied.logical_bytes == 32
    assert adapter.group_storage_bytes(FakeModule(), tied) == 16


def test_llama_tied_checkpoint_with_single_embedding_extent_is_complete():
    catalog = {
        item.name: item
        for item in (
            _extent("model.embed_tokens.weight"),
            _extent("model.layers.0.self_attn.q_proj.weight"),
            _extent("model.layers.0.self_attn.k_proj.weight"),
            _extent("model.layers.0.self_attn.v_proj.weight"),
            _extent("model.norm.weight"),
        )
    }
    adapter = LlamaFamilyAdapter(
        _config("LlamaForCausalLM"), None, _full_stage(layer_count=1)
    )

    plan = build_weight_plan(
        catalog,
        adapter,
        "llama-digest",
        _full_stage(layer_count=1),
        1,
    )

    assert plan.groups[0].kind is WeightKind.TIED_EMBEDDING_HEAD
    assert plan.groups[0].tensor_names == frozenset({"model.embed_tokens.weight"})
    assert sum(group.logical_bytes for group in plan.groups) == plan.total_bytes


def test_untied_embedding_and_head_are_independently_owned():
    catalog = {
        item.name: item
        for item in (
            _extent("model.embed_tokens.weight"),
            _extent("model.layers.0.self_attn.q_proj.weight"),
            _extent("model.norm.weight"),
            _extent("lm_head.weight"),
        )
    }
    adapter = Qwen3Adapter(
        _config("Qwen3ForCausalLM", tied=False),
        None,
        _full_stage(layer_count=1),
    )

    plan = build_weight_plan(
        catalog,
        adapter,
        "untied-digest",
        _full_stage(layer_count=1),
        1,
    )

    assert [group.kind for group in plan.groups] == [
        WeightKind.INPUT_EMBEDDING,
        WeightKind.TRANSFORMER_LAYERS,
        WeightKind.FINAL_NORM,
        WeightKind.LM_HEAD,
    ]
