from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from sglang.multi_model.uma.weight_plan import (
    StageWeightScope,
    WeightGroupSpec,
    WeightOwnership,
    WeightPlan,
)


_METADATA_MODEL_BUILD = ContextVar("metadata_model_build", default=False)


@contextmanager
def metadata_model_build() -> Iterator[None]:
    token = _METADATA_MODEL_BUILD.set(True)
    try:
        yield
    finally:
        _METADATA_MODEL_BUILD.reset(token)


def is_metadata_model_build() -> bool:
    return _METADATA_MODEL_BUILD.get()


@dataclass(frozen=True, slots=True)
class GroupValidation:
    ok: bool
    errors: tuple[str, ...]
    resident_bytes: int


@dataclass(frozen=True, slots=True)
class BoundModelRuntime:
    instance_id: str
    model_config: Any
    module: Any
    attention_config: Any
    kv_layout: str
    weight_epoch: int
    required_weight_groups: frozenset[str]
    ready_weight_groups: frozenset[str]


class ModelAdapter(ABC):
    architectures: frozenset[str] = frozenset()

    def __init__(
        self,
        model_config: Any,
        load_config: Any,
        stage: StageWeightScope,
    ) -> None:
        self.model_config = model_config
        self.load_config = load_config
        self.stage = stage

    @abstractmethod
    def classify_tensor(
        self,
        name: str,
        stage: StageWeightScope,
        group_layer_count: int,
    ) -> WeightOwnership | None: ...

    @abstractmethod
    def ownership_sort_key(
        self,
        owner: WeightOwnership,
    ) -> tuple[object, ...]: ...

    @abstractmethod
    def runtime_parameter_name(self, checkpoint_name: str) -> str: ...

    def build_meta_module(self) -> Any:
        import torch

        from sglang.srt.model_loader.loader import _initialize_model
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        with metadata_model_build():
            with torch.device("meta"), set_default_torch_dtype(
                self.model_config.dtype
            ):
                module = _initialize_model(self.model_config, self.load_config)
        non_meta = [
            name
            for name, parameter in module.named_parameters()
            if parameter.device.type != "meta"
        ]
        if non_meta:
            raise RuntimeError(
                f"metadata model allocated non-meta parameters: {non_meta[:8]}"
            )
        return module

    def runtime_parameter_names(self, group: WeightGroupSpec) -> frozenset[str]:
        return frozenset(
            self.runtime_parameter_name(extent.name) for extent in group.tensors
        )

    def group_storage_bytes(self, module: Any, group: WeightGroupSpec) -> int:
        """Return the exact parameter storage required by a materialized group.

        Checkpoint bytes are not a safe admission estimate: fused QKV parameters
        combine several extents, while tied embedding/head extents can duplicate
        the same runtime parameter.  The metadata module already has the exact
        post-fusion shapes without owning physical storage.
        """
        named = dict(module.named_parameters(remove_duplicate=False))
        unique_parameters: dict[int, Any] = {}
        missing: list[str] = []
        for name in sorted(self.runtime_parameter_names(group)):
            parameter = named.get(name)
            if parameter is None:
                missing.append(name)
            else:
                unique_parameters[id(parameter)] = parameter
        if missing:
            raise KeyError(
                f"runtime parameters missing for {group.group_id}: {missing}"
            )
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in unique_parameters.values()
        )

    def materialize_group(
        self,
        module: Any,
        group: WeightGroupSpec,
        device: Any,
    ) -> tuple[Any, ...]:
        import torch

        named = dict(module.named_parameters(remove_duplicate=False))
        required_names = self.runtime_parameter_names(group)
        missing = sorted(required_names - set(named))
        if missing:
            raise KeyError(
                f"runtime parameters missing for {group.group_id}: {missing}"
            )
        selected_ids = {id(named[name]) for name in required_names}
        replacements: dict[int, Any] = {}
        for owner in module.modules():
            for attribute, parameter in tuple(owner._parameters.items()):
                if parameter is None or id(parameter) not in selected_ids:
                    continue
                replacement = replacements.get(id(parameter))
                if replacement is None:
                    if parameter.device.type == "meta":
                        tensor = torch.empty_like(parameter, device=device)
                        replacement = torch.nn.Parameter(
                            tensor,
                            requires_grad=parameter.requires_grad,
                        )
                        replacement.__dict__.update(parameter.__dict__)
                    else:
                        replacement = parameter
                    replacements[id(parameter)] = replacement
                owner._parameters[attribute] = replacement
        # ``replacements`` is already keyed by the identity of each original
        # parameter.  Tensor/Parameter objects are intentionally not used as
        # dictionary keys because their equality is elementwise and they are
        # not reliably hashable across PyTorch versions.
        return tuple(replacements.values())

    def load_group(
        self,
        module: Any,
        group: WeightGroupSpec,
        tensors: Iterable[tuple[str, Any]],
    ) -> None:
        if group.tied_group_id is not None:
            checksums = {extent.checksum for extent in group.tensors}
            if len(checksums) > 1:
                raise ValueError(
                    f"tied group {group.group_id} contains unequal tensors"
                )
        observed: set[str] = set()

        def recording_iterator() -> Iterator[tuple[str, Any]]:
            for name, tensor in tensors:
                if name in observed:
                    raise ValueError(f"duplicate loaded tensor: {name}")
                observed.add(name)
                yield name, tensor

        module.load_weights(recording_iterator())
        expected = group.tensor_names
        if observed != expected:
            raise ValueError(
                f"loaded tensor mismatch for {group.group_id}: "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )

    def validate_group(self, module: Any, group: WeightGroupSpec) -> GroupValidation:
        named = dict(module.named_parameters(remove_duplicate=False))
        errors: list[str] = []
        unique_parameters: dict[int, Any] = {}
        for name in sorted(self.runtime_parameter_names(group)):
            parameter = named.get(name)
            if parameter is None:
                errors.append(f"missing runtime parameter: {name}")
                continue
            if parameter.device.type == "meta":
                errors.append(f"parameter remains meta: {name}")
                continue
            unique_parameters[id(parameter)] = parameter
        resident_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in unique_parameters.values()
        )
        return GroupValidation(not errors, tuple(errors), resident_bytes)

    @staticmethod
    def cuda_storage_bytes(module: Any) -> int:
        unique: dict[int, Any] = {}
        for parameter in module.parameters():
            if parameter.device.type == "cuda":
                unique[id(parameter)] = parameter
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in unique.values()
        )

    def bind_runtime(
        self,
        instance_id: str,
        module: Any,
        plan: WeightPlan,
        ready_groups: Sequence[str],
        *,
        kv_layout: str,
        weight_epoch: int,
        attention_config: Any = None,
    ) -> BoundModelRuntime:
        required = frozenset(group.group_id for group in plan.groups)
        ready = frozenset(ready_groups)
        missing = required - ready
        if missing:
            raise ValueError(f"cannot bind missing weight groups: {sorted(missing)}")
        return BoundModelRuntime(
            instance_id=instance_id,
            model_config=self.model_config,
            module=module,
            attention_config=attention_config,
            kv_layout=kv_layout,
            weight_epoch=weight_epoch,
            required_weight_groups=required,
            ready_weight_groups=ready,
        )


class ModelAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, type[ModelAdapter]] = {}

    def register(self, adapter_type: type[ModelAdapter]) -> None:
        for architecture in adapter_type.architectures:
            existing = self._adapters.get(architecture)
            if existing is not None and existing is not adapter_type:
                raise ValueError(f"adapter already registered for {architecture}")
            self._adapters[architecture] = adapter_type

    def create(
        self,
        model_config: Any,
        load_config: Any,
        stage: StageWeightScope,
    ) -> ModelAdapter:
        architectures = tuple(model_config.hf_config.architectures or ())
        for architecture in architectures:
            adapter_type = self._adapters.get(architecture)
            if adapter_type is not None:
                return adapter_type(model_config, load_config, stage)
        raise ValueError(f"unsupported dense decoder architectures: {architectures}")

    @property
    def architectures(self) -> Mapping[str, type[ModelAdapter]]:
        return dict(self._adapters)


def default_adapter_registry() -> ModelAdapterRegistry:
    from sglang.multi_model.uma.adapters.llama_family import LlamaFamilyAdapter
    from sglang.multi_model.uma.adapters.qwen3 import Qwen3Adapter

    registry = ModelAdapterRegistry()
    registry.register(Qwen3Adapter)
    registry.register(LlamaFamilyAdapter)
    return registry
