from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


WEIGHT_PLAN_PATH = (
    Path(__file__).parents[2]
    / "python"
    / "sglang"
    / "multi_model"
    / "uma"
    / "weight_plan.py"
)
SPEC = importlib.util.spec_from_file_location(
    "dist_sglang_weight_plan", WEIGHT_PLAN_PATH
)
assert SPEC and SPEC.loader
weight_plan = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = weight_plan
SPEC.loader.exec_module(weight_plan)

DuplicateTensor = weight_plan.DuplicateTensor
StageWeightScope = weight_plan.StageWeightScope
TensorExtent = weight_plan.TensorExtent
UnownedTensor = weight_plan.UnownedTensor
WeightKind = weight_plan.WeightKind
WeightOwnership = weight_plan.WeightOwnership
build_safetensors_catalog = weight_plan.build_safetensors_catalog
build_weight_plan = weight_plan.build_weight_plan


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class FakeAdapter:
    def classify_tensor(
        self,
        name: str,
        stage: StageWeightScope,
        group_layer_count: int,
    ) -> WeightOwnership | None:
        if name == "model.embed_tokens.weight":
            if stage.owns_input_embedding and stage.owns_output_head:
                return WeightOwnership(
                    "tied-embedding-head",
                    WeightKind.TIED_EMBEDDING_HEAD,
                    tied_group_id="embedding-head",
                )
            if stage.owns_input_embedding:
                return WeightOwnership("input-embedding", WeightKind.INPUT_EMBEDDING)
        if name == "lm_head.weight" and stage.owns_output_head:
            return WeightOwnership("lm-head", WeightKind.LM_HEAD)
        if name == "model.norm.weight" and stage.owns_final_norm:
            return WeightOwnership("final-norm", WeightKind.FINAL_NORM)
        prefix = "model.layers."
        if name.startswith(prefix):
            layer = int(name[len(prefix) :].split(".", 1)[0])
            start, stop = stage.layer_range
            if start <= layer < stop:
                group_start = (
                    start
                    + ((layer - start) // group_layer_count) * group_layer_count
                )
                group_stop = min(group_start + group_layer_count, stop)
                return WeightOwnership(
                    f"layers-{group_start}-{group_stop}",
                    WeightKind.TRANSFORMER_LAYERS,
                    layer_range=(group_start, group_stop),
                )
        return None

    def ownership_sort_key(self, owner: WeightOwnership) -> tuple[object, ...]:
        order = {
            WeightKind.INPUT_EMBEDDING: 0,
            WeightKind.TIED_EMBEDDING_HEAD: 0,
            WeightKind.TRANSFORMER_LAYERS: 1,
            WeightKind.FINAL_NORM: 2,
            WeightKind.LM_HEAD: 3,
        }
        return (order[owner.kind], owner.layer_range or (-1, -1), owner.group_id)


def extent(name: str, size: int = 8) -> TensorExtent:
    return TensorExtent(
        name=name,
        file="model.safetensors",
        offset=128,
        nbytes=size,
        shape=(2, 2),
        dtype="BF16",
        checksum="sha256:" + hashlib.sha256(name.encode()).hexdigest(),
    )


def test_plan_covers_every_stage_tensor_once_and_preserves_tied_group() -> None:
    names = (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.norm.weight",
    )
    catalog = {name: extent(name) for name in names}
    stage = StageWeightScope(
        stage_id="s0",
        layer_range=(0, 4),
        owns_input_embedding=True,
        owns_final_norm=True,
        owns_output_head=True,
    )

    plan = build_weight_plan(
        catalog,
        FakeAdapter(),
        "sha256:qwen-test",
        stage,
        group_layer_count=2,
    )

    covered = [tensor.name for group in plan.groups for tensor in group.tensors]
    assert sorted(covered) == sorted(catalog)
    assert len(covered) == len(set(covered))
    assert plan.total_bytes == sum(item.nbytes for item in catalog.values())
    assert [group.group_id for group in plan.groups] == [
        "tied-embedding-head",
        "layers-0-2",
        "layers-2-4",
        "final-norm",
    ]
    tied = plan.groups[0]
    assert tied.kind is WeightKind.TIED_EMBEDDING_HEAD
    assert tied.tied_group_id == "embedding-head"


def test_plan_rejects_unowned_tensor() -> None:
    stage = StageWeightScope("s0", (0, 1), True, True, True)

    with pytest.raises(UnownedTensor, match="unknown.weight"):
        build_weight_plan(
            {"unknown.weight": extent("unknown.weight")},
            FakeAdapter(),
            "sha256:model",
            stage,
            group_layer_count=1,
        )


def test_catalog_reads_sharded_headers_offsets_and_checksums(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first_data = b"abcdefgh"
    second_data = b"ijklmnop"
    write_safetensors(first, {"model.layers.0.weight": ("BF16", (2, 2), first_data)})
    write_safetensors(second, {"model.layers.1.weight": ("BF16", (2, 2), second_data)})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.layers.0.weight": first.name,
                    "model.layers.1.weight": second.name,
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = build_safetensors_catalog(tmp_path)

    assert tuple(catalog) == (
        "model.layers.0.weight",
        "model.layers.1.weight",
    )
    assert catalog["model.layers.0.weight"].file == str(first.resolve())
    assert catalog["model.layers.0.weight"].nbytes == len(first_data)
    assert catalog["model.layers.0.weight"].checksum == (
        "sha256:" + hashlib.sha256(first_data).hexdigest()
    )
    with first.open("rb") as stream:
        stream.seek(catalog["model.layers.0.weight"].offset)
        assert stream.read(len(first_data)) == first_data


def test_catalog_rejects_duplicate_tensor_across_files(tmp_path: Path) -> None:
    write_safetensors(
        tmp_path / "first.safetensors",
        {"duplicate.weight": ("F32", (1,), b"abcd")},
    )
    write_safetensors(
        tmp_path / "second.safetensors",
        {"duplicate.weight": ("F32", (1,), b"efgh")},
    )

    with pytest.raises(DuplicateTensor, match="duplicate.weight"):
        build_safetensors_catalog(tmp_path)


def test_default_loader_opens_only_selected_shard(monkeypatch) -> None:
    pytest.importorskip("torch")
    from sglang.srt.model_loader import loader as loader_module

    opened: list[tuple[str, ...]] = []

    def fake_iterator(files, *, disable_mmap=False):
        opened.append(tuple(files))
        yield "model.layers.0.weight", object()

    monkeypatch.setattr(loader_module, "safetensors_weights_iterator", fake_iterator)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.schedule_batch",
        SimpleNamespace(global_server_args_dict={"weight_loader_disable_mmap": False}),
    )
    selected = extent("model.layers.0.weight")
    selected = TensorExtent(
        selected.name,
        "/weights/shard-0.safetensors",
        selected.offset,
        selected.nbytes,
        selected.shape,
        selected.dtype,
        selected.checksum,
    )
    source = loader_module.DefaultModelLoader.Source("/weights", None)
    instance = loader_module.DefaultModelLoader.__new__(
        loader_module.DefaultModelLoader
    )

    assert [name for name, _ in instance.iter_named_tensors(source, (selected,))] == [
        "model.layers.0.weight"
    ]
    assert opened == [("/weights/shard-0.safetensors",)]
