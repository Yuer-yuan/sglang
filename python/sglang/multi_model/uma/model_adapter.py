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
class ExecutionResourceBundle:
    """All model-shaped objects that must switch as one execution unit."""

    attention_backend: Any
    req_to_token_pool: Any
    token_to_kv_pool: Any
    token_to_kv_pool_allocator: Any
    sampler: Any
    logits_processor: Any
    kv_cache_dtype: Any
    max_total_num_tokens: int
    max_running_requests: int
    max_req_len: int
    max_req_input_len: int
    start_layer: int
    end_layer: int
    weight_runtime: Any = None
    cuda_graph_runner: Any = None
    cuda_graph_mem_usage: int = 0

    def validate(self) -> None:
        required = {
            "attention_backend": self.attention_backend,
            "req_to_token_pool": self.req_to_token_pool,
            "token_to_kv_pool": self.token_to_kv_pool,
            "token_to_kv_pool_allocator": self.token_to_kv_pool_allocator,
            "sampler": self.sampler,
            "logits_processor": self.logits_processor,
            "kv_cache_dtype": self.kv_cache_dtype,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"incomplete execution resource bundle: {missing}")
        if self.max_total_num_tokens <= 0:
            raise ValueError("max_total_num_tokens must be positive")
        if self.max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")
        if self.max_req_len <= 0 or self.max_req_input_len <= 0:
            raise ValueError("request length limits must be positive")
        if self.start_layer < 0 or self.end_layer <= self.start_layer:
            raise ValueError("execution layer range must be non-empty")
        if self.cuda_graph_mem_usage < 0:
            raise ValueError("cuda_graph_mem_usage must be non-negative")


@dataclass(frozen=True, slots=True)
class BoundModelRuntime:
    deployment_id: str
    placement_version: int
    instance_id: str
    stage_id: str
    resource_epoch: int
    model_config: Any
    module: Any
    kv_layout: str
    weight_epoch: int
    required_weight_groups: frozenset[str]
    ready_weight_groups: frozenset[str]
    execution_resources: ExecutionResourceBundle

    def validate_complete(self) -> None:
        for name in ("deployment_id", "instance_id", "stage_id", "kv_layout"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.placement_version < 0 or self.resource_epoch < 0:
            raise ValueError("placement and resource epochs must be non-negative")
        if self.weight_epoch < 0:
            raise ValueError("weight_epoch must be non-negative")
        if self.model_config is None or self.module is None:
            raise ValueError("model config and module are required")
        missing = self.required_weight_groups - self.ready_weight_groups
        if missing:
            raise ValueError(f"cannot bind missing weight groups: {sorted(missing)}")
        unexpected = self.ready_weight_groups - self.required_weight_groups
        if unexpected:
            raise ValueError(
                f"runtime reports unknown ready weight groups: {sorted(unexpected)}"
            )
        if (
            self.required_weight_groups
            and self.execution_resources.weight_runtime is None
        ):
            raise ValueError("managed weight groups require a weight runtime")
        self.execution_resources.validate()
        module_logits = getattr(self.module, "logits_processor", None)
        if module_logits is not self.execution_resources.logits_processor:
            raise ValueError("logits processor does not belong to the bound module")
        named_buffers = getattr(self.module, "named_buffers", None)
        meta_buffers = (
            [
                name
                for name, buffer in named_buffers()
                if buffer.device.type == "meta"
            ]
            if callable(named_buffers)
            else []
        )
        if meta_buffers:
            raise ValueError(
                "runtime contains unmaterialized model buffers: "
                f"{meta_buffers[:8]}"
            )


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

    def materialize_runtime_buffers(self, module: Any, device: Any) -> int:
        """Materialize non-checkpoint state created by model constructors.

        A metadata model puts both parameters and buffers on the meta device.
        Weight groups replace checkpoint-backed parameters, but generated
        state such as RoPE caches never appears in safetensors.  Such buffers
        must be rebuilt explicitly before the runtime can be published.

        The first dense-decoder implementation deliberately supports only the
        zero-argument ``_compute_cos_sin_cache`` contract used by the Qwen3,
        Llama and Mistral rotary modules.  An unfamiliar meta buffer fails
        registration loudly instead of surviving until the first forward.
        """
        import torch

        materialized: dict[int, Any] = {}
        unsupported: list[str] = []
        for module_name, owner in module.named_modules():
            for buffer_name, buffer in tuple(owner._buffers.items()):
                if buffer is None or buffer.device.type != "meta":
                    continue
                qualified_name = (
                    f"{module_name}.{buffer_name}" if module_name else buffer_name
                )
                builder = getattr(owner, "_compute_cos_sin_cache", None)
                if buffer_name != "cos_sin_cache" or not callable(builder):
                    unsupported.append(qualified_name)
                    continue
                try:
                    with torch.device(device):
                        replacement = builder()
                    replacement = replacement.to(
                        device=device,
                        dtype=buffer.dtype,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "runtime buffer builder requires unsupported arguments: "
                        f"{qualified_name}"
                    ) from exc
                owner._buffers[buffer_name] = replacement
                materialized[id(replacement)] = replacement

        if unsupported:
            raise RuntimeError(
                "metadata model contains unsupported runtime buffers: "
                f"{unsupported[:8]}"
            )

        remaining = [
            name
            for name, buffer in module.named_buffers()
            if buffer.device.type == "meta"
        ]
        if remaining:
            raise RuntimeError(
                "runtime buffers remain meta after materialization: "
                f"{remaining[:8]}"
            )
        return sum(
            buffer.numel() * buffer.element_size()
            for buffer in materialized.values()
        )

    @staticmethod
    def runtime_buffer_storage_bytes(module: Any) -> int:
        unique: dict[int, Any] = {}
        for buffer in module.buffers():
            if buffer.device.type != "meta":
                unique[id(buffer)] = buffer
        return sum(
            buffer.numel() * buffer.element_size()
            for buffer in unique.values()
        )

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

    def release_group(self, module: Any, group: WeightGroupSpec) -> int:
        """Replace one group's physical parameters with aliased meta tensors."""
        import torch

        named = dict(module.named_parameters(remove_duplicate=False))
        required_names = self.runtime_parameter_names(group)
        missing = sorted(required_names - set(named))
        if missing:
            raise KeyError(
                f"runtime parameters missing for {group.group_id}: {missing}"
            )
        selected_ids = {id(named[name]) for name in required_names}
        resident_parameters = {
            id(named[name]): named[name]
            for name in required_names
            if named[name].device.type != "meta"
        }
        released_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in resident_parameters.values()
        )
        replacements: dict[int, Any] = {}
        for parameter_id, parameter in {
            id(parameter): parameter
            for parameter in named.values()
            if id(parameter) in selected_ids
        }.items():
            if parameter.device.type == "meta":
                replacements[parameter_id] = parameter
            else:
                tensor = torch.empty_like(parameter, device="meta")
                replacement = torch.nn.Parameter(
                    tensor,
                    requires_grad=parameter.requires_grad,
                )
                replacement.__dict__.update(parameter.__dict__)
                replacements[parameter_id] = replacement

        # Build every replacement before changing module bindings.  A failed
        # meta allocation therefore leaves the resident group intact.
        for owner in module.modules():
            for attribute, parameter in tuple(owner._parameters.items()):
                if parameter is None or id(parameter) not in selected_ids:
                    continue
                owner._parameters[attribute] = replacements[id(parameter)]
        return released_bytes

    def group_is_meta(self, module: Any, group: WeightGroupSpec) -> bool:
        named = dict(module.named_parameters(remove_duplicate=False))
        names = self.runtime_parameter_names(group)
        missing = sorted(names - set(named))
        if missing:
            raise KeyError(
                f"runtime parameters missing for {group.group_id}: {missing}"
            )
        return all(named[name].device.type == "meta" for name in names)

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
        deployment_id: str,
        placement_version: int,
        instance_id: str,
        stage_id: str,
        resource_epoch: int,
        module: Any,
        plan: WeightPlan,
        ready_groups: Sequence[str],
        *,
        kv_layout: str,
        weight_epoch: int,
        execution_resources: ExecutionResourceBundle,
    ) -> BoundModelRuntime:
        required = frozenset(group.group_id for group in plan.groups)
        ready = frozenset(ready_groups)
        missing = required - ready
        if missing:
            raise ValueError(f"cannot bind missing weight groups: {sorted(missing)}")
        runtime = BoundModelRuntime(
            deployment_id=deployment_id,
            placement_version=placement_version,
            instance_id=instance_id,
            stage_id=stage_id,
            resource_epoch=resource_epoch,
            model_config=self.model_config,
            module=module,
            kv_layout=kv_layout,
            weight_epoch=weight_epoch,
            required_weight_groups=required,
            ready_weight_groups=ready,
            execution_resources=execution_resources,
        )
        runtime.validate_complete()
        return runtime


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
