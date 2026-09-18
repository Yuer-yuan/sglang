from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import prod
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence


class WeightPlanError(ValueError):
    """Base class for immutable checkpoint-plan validation failures."""


class InvalidSafetensors(WeightPlanError):
    pass


class DuplicateTensor(WeightPlanError):
    pass


class UnownedTensor(WeightPlanError):
    pass


class TiedGroupSplit(WeightPlanError):
    pass


class WeightKind(str, Enum):
    INPUT_EMBEDDING = "INPUT_EMBEDDING"
    TRANSFORMER_LAYERS = "TRANSFORMER_LAYERS"
    FINAL_NORM = "FINAL_NORM"
    LM_HEAD = "LM_HEAD"
    TIED_EMBEDDING_HEAD = "TIED_EMBEDDING_HEAD"


_DTYPE_BITS: Mapping[str, int] = MappingProxyType(
    {
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "F8_E5M2": 8,
        "F8_E4M3": 8,
        "F8_E8M0": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "I64": 64,
        "U64": 64,
        "F64": 64,
        "C64": 64,
    }
)


@dataclass(frozen=True, slots=True)
class TensorExtent:
    name: str
    file: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    checksum: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if not self.file:
            raise ValueError("tensor file must not be empty")
        if self.offset < 0 or self.nbytes <= 0:
            raise ValueError("tensor extent must be non-empty and non-negative")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be non-negative")
        if not self.checksum.startswith("sha256:"):
            raise ValueError("tensor checksum must be a sha256 digest")


@dataclass(frozen=True, slots=True)
class StageWeightScope:
    stage_id: str
    layer_range: tuple[int, int]
    owns_input_embedding: bool
    owns_final_norm: bool
    owns_output_head: bool

    def __post_init__(self) -> None:
        start, stop = self.layer_range
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
        if start < 0 or stop <= start:
            raise ValueError("layer_range must be a non-empty half-open range")


@dataclass(frozen=True, slots=True)
class WeightOwnership:
    group_id: str
    kind: WeightKind
    layer_range: tuple[int, int] | None = None
    tied_group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if self.kind is WeightKind.TRANSFORMER_LAYERS:
            if self.layer_range is None:
                raise ValueError("transformer ownership requires a layer range")
        elif self.layer_range is not None:
            raise ValueError("only transformer ownership may carry a layer range")
        if self.layer_range is not None:
            start, stop = self.layer_range
            if start < 0 or stop <= start:
                raise ValueError("ownership layer range must be non-empty")


@dataclass(frozen=True, slots=True)
class WeightGroupSpec:
    group_id: str
    kind: WeightKind
    layer_range: tuple[int, int] | None
    tensors: tuple[TensorExtent, ...]
    logical_bytes: int
    tied_group_id: str | None = None

    @classmethod
    def from_extents(
        cls,
        owner: WeightOwnership,
        extents: Sequence[TensorExtent],
    ) -> "WeightGroupSpec":
        ordered = tuple(sorted(extents, key=lambda extent: extent.name))
        if not ordered:
            raise ValueError("weight group must own at least one tensor")
        return cls(
            group_id=owner.group_id,
            kind=owner.kind,
            layer_range=owner.layer_range,
            tensors=ordered,
            logical_bytes=sum(extent.nbytes for extent in ordered),
            tied_group_id=owner.tied_group_id,
        )

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(extent.name for extent in self.tensors)


@dataclass(frozen=True, slots=True)
class WeightPlan:
    model_digest: str
    stage_id: str
    groups: tuple[WeightGroupSpec, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        if not self.model_digest:
            raise ValueError("model_digest must not be empty")
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
        if self.total_bytes != sum(group.logical_bytes for group in self.groups):
            raise ValueError("weight plan total does not match its groups")


class TensorClassifier(Protocol):
    def classify_tensor(
        self,
        name: str,
        stage: StageWeightScope,
        group_layer_count: int,
    ) -> WeightOwnership | None: ...

    def ownership_sort_key(self, owner: WeightOwnership) -> tuple[object, ...]: ...


def _sha256_extent(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    remaining = nbytes
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise InvalidSafetensors(
                    f"tensor extent exceeds file size: {path}:{offset}+{nbytes}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return "sha256:" + digest.hexdigest()


def _read_safetensors_file(path: Path) -> dict[str, TensorExtent]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise InvalidSafetensors(f"missing safetensors header length: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length == 0 or header_length > file_size - 8:
            raise InvalidSafetensors(f"invalid safetensors header length: {path}")
        try:
            header = json.loads(stream.read(header_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidSafetensors(f"invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise InvalidSafetensors(f"safetensors header must be an object: {path}")
    data_start = 8 + header_length
    extents: dict[str, TensorExtent] = {}
    occupied: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise InvalidSafetensors(f"invalid metadata for tensor {name!r}: {path}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or dtype not in _DTYPE_BITS
            or not isinstance(shape, list)
            or not all(isinstance(value, int) for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise InvalidSafetensors(f"invalid metadata for tensor {name!r}: {path}")
        relative_start, relative_stop = offsets
        if relative_start < 0 or relative_stop <= relative_start:
            raise InvalidSafetensors(f"invalid extent for tensor {name!r}: {path}")
        nbytes = relative_stop - relative_start
        elements = prod(shape)
        expected_nbytes = (elements * _DTYPE_BITS[dtype] + 7) // 8
        if expected_nbytes != nbytes:
            raise InvalidSafetensors(
                f"shape/dtype size mismatch for tensor {name!r}: "
                f"expected {expected_nbytes}, observed {nbytes}"
            )
        absolute_start = data_start + relative_start
        absolute_stop = data_start + relative_stop
        if absolute_stop > file_size:
            raise InvalidSafetensors(f"extent exceeds file size for tensor {name!r}")
        occupied.append((absolute_start, absolute_stop, name))
        extents[name] = TensorExtent(
            name=name,
            file=str(path.resolve()),
            offset=absolute_start,
            nbytes=nbytes,
            shape=tuple(shape),
            dtype=dtype,
            checksum=_sha256_extent(path, absolute_start, nbytes),
        )
    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if current[0] < previous[1]:
            raise InvalidSafetensors(
                f"overlapping tensors {previous[2]!r} and {current[2]!r}: {path}"
            )
    return extents


def _indexed_files(model_path: Path) -> tuple[dict[str, str] | None, tuple[Path, ...]]:
    indexes = sorted(model_path.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise InvalidSafetensors(f"multiple safetensors indexes under {model_path}")
    if not indexes:
        return None, tuple(sorted(model_path.glob("*.safetensors")))
    try:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidSafetensors(f"invalid safetensors index: {indexes[0]}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise InvalidSafetensors(f"invalid weight_map: {indexes[0]}")
    filenames = sorted(set(weight_map.values()))
    files: list[Path] = []
    root = model_path.resolve()
    for filename in filenames:
        candidate = (model_path / filename).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise InvalidSafetensors(
                f"index path escapes model directory: {filename}"
            ) from exc
        if not candidate.is_file():
            raise InvalidSafetensors(f"indexed shard does not exist: {candidate}")
        files.append(candidate)
    return dict(weight_map), tuple(files)


def build_safetensors_catalog(model_path: Path) -> Mapping[str, TensorExtent]:
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    weight_map, files = _indexed_files(model_path)
    if not files:
        raise InvalidSafetensors(f"no safetensors files under {model_path}")
    catalog: dict[str, TensorExtent] = {}
    observed_file_by_name: dict[str, str] = {}
    for path in files:
        for name, extent in _read_safetensors_file(path).items():
            if name in catalog:
                raise DuplicateTensor(
                    f"duplicate tensor {name!r} in "
                    f"{observed_file_by_name[name]} and {path}"
                )
            catalog[name] = extent
            observed_file_by_name[name] = path.name
    if weight_map is not None:
        expected_names = set(weight_map)
        observed_names = set(catalog)
        if expected_names != observed_names:
            missing = sorted(expected_names - observed_names)
            extra = sorted(observed_names - expected_names)
            raise InvalidSafetensors(
                f"index/header mismatch: missing={missing}, extra={extra}"
            )
        for name, filename in weight_map.items():
            if observed_file_by_name[name] != filename:
                raise InvalidSafetensors(
                    f"index maps {name!r} to {filename!r}, observed "
                    f"{observed_file_by_name[name]!r}"
                )
    return MappingProxyType(dict(sorted(catalog.items())))


def _assert_exact_tensor_coverage(
    catalog: Mapping[str, TensorExtent],
    groups: Sequence[WeightGroupSpec],
) -> None:
    covered = [extent.name for group in groups for extent in group.tensors]
    duplicates = sorted(name for name in set(covered) if covered.count(name) > 1)
    if duplicates:
        raise DuplicateTensor(f"tensors assigned more than once: {duplicates}")
    if set(covered) != set(catalog):
        missing = sorted(set(catalog) - set(covered))
        extra = sorted(set(covered) - set(catalog))
        raise WeightPlanError(
            f"plan coverage mismatch: missing={missing}, extra={extra}"
        )


def build_weight_plan(
    catalog: Mapping[str, TensorExtent],
    adapter: TensorClassifier,
    model_digest: str,
    stage: StageWeightScope,
    group_layer_count: int,
) -> WeightPlan:
    if group_layer_count <= 0:
        raise ValueError("group_layer_count must be positive")
    grouped: dict[WeightOwnership, list[TensorExtent]] = defaultdict(list)
    tied_owners: dict[str, WeightOwnership] = {}
    for name, extent in sorted(catalog.items()):
        if extent.name != name:
            raise WeightPlanError(
                f"catalog key {name!r} does not match extent {extent.name!r}"
            )
        owner = adapter.classify_tensor(name, stage, group_layer_count)
        if owner is None:
            raise UnownedTensor(f"unowned tensor: {name}")
        if owner.tied_group_id is not None:
            previous = tied_owners.setdefault(owner.tied_group_id, owner)
            if previous != owner:
                raise TiedGroupSplit(
                    f"tied group {owner.tied_group_id!r} split between "
                    f"{previous.group_id!r} and {owner.group_id!r}"
                )
        grouped[owner].append(extent)
    groups = tuple(
        WeightGroupSpec.from_extents(owner, grouped[owner])
        for owner in sorted(grouped, key=adapter.ownership_sort_key)
    )
    _assert_exact_tensor_coverage(catalog, groups)
    return WeightPlan(
        model_digest=model_digest,
        stage_id=stage.stage_id,
        groups=groups,
        total_bytes=sum(group.logical_bytes for group in groups),
    )
