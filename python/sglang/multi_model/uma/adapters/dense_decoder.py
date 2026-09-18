from __future__ import annotations

import re

from sglang.multi_model.uma.model_adapter import ModelAdapter
from sglang.multi_model.uma.weight_plan import (
    StageWeightScope,
    WeightKind,
    WeightOwnership,
)


_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


class DenseDecoderAdapter(ModelAdapter):
    def classify_tensor(
        self,
        name: str,
        stage: StageWeightScope,
        group_layer_count: int,
    ) -> WeightOwnership | None:
        tied = bool(getattr(self.model_config.hf_config, "tie_word_embeddings", False))
        if name == "model.embed_tokens.weight":
            if tied and stage.owns_output_head:
                return WeightOwnership(
                    "tied-embedding-head",
                    WeightKind.TIED_EMBEDDING_HEAD,
                    tied_group_id="embedding-head",
                )
            if stage.owns_input_embedding:
                return WeightOwnership(
                    "input-embedding",
                    WeightKind.INPUT_EMBEDDING,
                )
        if name == "lm_head.weight" and stage.owns_output_head:
            if tied:
                return WeightOwnership(
                    "tied-embedding-head",
                    WeightKind.TIED_EMBEDDING_HEAD,
                    tied_group_id="embedding-head",
                )
            return WeightOwnership("lm-head", WeightKind.LM_HEAD)
        if name == "model.norm.weight" and stage.owns_final_norm:
            return WeightOwnership("final-norm", WeightKind.FINAL_NORM)
        match = _LAYER_PATTERN.match(name)
        if match is None:
            return None
        layer = int(match.group(1))
        start, stop = stage.layer_range
        if not start <= layer < stop:
            return None
        group_start = start + ((layer - start) // group_layer_count) * group_layer_count
        group_stop = min(group_start + group_layer_count, stop)
        return WeightOwnership(
            f"layers-{group_start}-{group_stop}",
            WeightKind.TRANSFORMER_LAYERS,
            layer_range=(group_start, group_stop),
        )

    def ownership_sort_key(
        self,
        owner: WeightOwnership,
    ) -> tuple[object, ...]:
        order = {
            WeightKind.INPUT_EMBEDDING: 0,
            WeightKind.TIED_EMBEDDING_HEAD: 0,
            WeightKind.TRANSFORMER_LAYERS: 1,
            WeightKind.FINAL_NORM: 2,
            WeightKind.LM_HEAD: 3,
        }
        return (
            order[owner.kind],
            owner.layer_range or (-1, -1),
            owner.group_id,
        )

    def runtime_parameter_name(self, checkpoint_name: str) -> str:
        replacements = (
            (".q_proj.", ".qkv_proj."),
            (".k_proj.", ".qkv_proj."),
            (".v_proj.", ".qkv_proj."),
            (".gate_proj.", ".gate_up_proj."),
            (".up_proj.", ".gate_up_proj."),
        )
        for source, target in replacements:
            if source in checkpoint_name:
                return checkpoint_name.replace(source, target)
        if (
            checkpoint_name == "lm_head.weight"
            and bool(
                getattr(
                    self.model_config.hf_config,
                    "tie_word_embeddings",
                    False,
                )
            )
        ):
            return "model.embed_tokens.weight"
        return checkpoint_name
