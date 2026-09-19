"""Transactional local-SSD storage for immutable session KV extents."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable, Protocol

from sglang.multi_model.uma.kv_domain import (
    KVExtent,
    SessionBranchKey,
    SessionKVDescriptor,
)


class LocalKVStoreError(RuntimeError):
    pass


class ChecksumMismatch(LocalKVStoreError):
    pass


class InjectedIOError(LocalKVStoreError):
    pass


class ManifestConflict(LocalKVStoreError):
    pass


class FaultInjector(Protocol):
    def checkpoint(self, phase: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CommittedExtent:
    extent: KVExtent
    path: Path
    operation_id: str


@dataclass(frozen=True, slots=True)
class SessionManifest:
    descriptor: SessionKVDescriptor
    extents: tuple[CommittedExtent, ...]
    operation_id: str
    created_at_ns: int
    path: Path


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    committed_manifests: int
    referenced_extents: int
    orphan_extents: int
    temporary_files_removed: int


def content_checksum(chunks: Iterable[bytes | bytearray | memoryview]) -> str:
    digest = hashlib.blake2b(digest_size=32)
    for chunk in chunks:
        digest.update(memoryview(chunk).cast("B"))
    return f"blake2b:{digest.hexdigest()}"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


class LocalKVStore:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.root = Path(root)
        self.extent_dir = self.root / "extents"
        self.manifest_dir = self.root / "manifests"
        self.temporary_dir = self.root / "tmp"
        self._fault_injector = fault_injector
        for directory in (
            self.root,
            self.extent_dir,
            self.manifest_dir,
            self.temporary_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def persist_session(
        self,
        descriptor: SessionKVDescriptor,
        extent_chunks: Iterable[
            tuple[KVExtent, Iterable[bytes | bytearray | memoryview]]
        ],
        *,
        operation_id: str,
    ) -> SessionManifest:
        self._require_operation_id(operation_id)
        items = tuple(extent_chunks)
        if not items:
            raise ValueError("a session manifest requires at least one KV extent")
        for extent, _ in items:
            self._validate_extent_owner(descriptor, extent)
        if sum(extent.nbytes for extent, _ in items) != descriptor.logical_bytes:
            raise ValueError("extent byte total does not match descriptor logical bytes")
        committed = tuple(
            self.write_extent(extent, chunks, operation_id=operation_id)
            for extent, chunks in items
        )
        return self.commit_session(
            descriptor,
            committed,
            operation_id=operation_id,
        )

    def write_extent(
        self,
        extent: KVExtent,
        chunks: Iterable[bytes | bytearray | memoryview],
        *,
        operation_id: str,
    ) -> CommittedExtent:
        self._require_operation_id(operation_id)
        temporary = self.temporary_dir / (
            f"{self._safe_id(operation_id)}-{self._extent_identity(extent)}.tmp"
        )
        final = self.extent_dir / f"{self._checksum_hex(extent.checksum)}.kv"
        digest = hashlib.blake2b(digest_size=32)
        total = 0
        try:
            with temporary.open("xb", buffering=0) as handle:
                for chunk in chunks:
                    view = memoryview(chunk).cast("B")
                    digest.update(view)
                    total += len(view)
                    while view:
                        written = handle.write(view)
                        if written is None or written <= 0:
                            raise OSError("short write while persisting KV extent")
                        view = view[written:]
                self._checkpoint("data_write")
                os.fsync(handle.fileno())
                self._checkpoint("data_fsync")
            observed = f"blake2b:{digest.hexdigest()}"
            if total != extent.nbytes:
                raise ChecksumMismatch(
                    f"KV extent length {total} does not match expected {extent.nbytes}"
                )
            if observed != extent.checksum:
                raise ChecksumMismatch(
                    f"KV extent checksum {observed} does not match {extent.checksum}"
                )
            if final.exists():
                self._verify_file(final, extent.nbytes, extent.checksum)
                temporary.unlink()
            else:
                os.replace(temporary, final)
            self._checkpoint("extent_rename")
            self._fsync_directory(self.extent_dir)
            self._checkpoint("extent_directory_fsync")
            return CommittedExtent(extent, final, operation_id)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def commit_session(
        self,
        descriptor: SessionKVDescriptor,
        extents: tuple[CommittedExtent, ...],
        *,
        operation_id: str,
    ) -> SessionManifest:
        self._require_operation_id(operation_id)
        if not extents:
            raise ValueError("a session manifest requires at least one KV extent")
        for committed in extents:
            self._validate_extent_owner(descriptor, committed.extent)
            self._verify_file(
                committed.path,
                committed.extent.nbytes,
                committed.extent.checksum,
            )
        manifest_path = self._manifest_path(descriptor.key, descriptor.kv_epoch)
        if manifest_path.exists():
            return self._require_identical_manifest(
                manifest_path,
                descriptor,
                extents,
            )
        created_at_ns = time.time_ns()
        temporary = self.temporary_dir / (
            f"{self._safe_id(operation_id)}-{manifest_path.stem}.manifest.tmp"
        )
        core = {
            "schema_version": self.SCHEMA_VERSION,
            "descriptor": self._descriptor_to_dict(descriptor),
            "extents": [self._committed_to_dict(item) for item in extents],
            "operation_id": operation_id,
            "created_at_ns": created_at_ns,
        }
        envelope = {
            "manifest": core,
            "checksum": content_checksum((_canonical_json(core),)),
        }
        try:
            with temporary.open("xb", buffering=0) as handle:
                payload = _canonical_json(envelope) + b"\n"
                view = memoryview(payload)
                while view:
                    written = handle.write(view)
                    if written is None or written <= 0:
                        raise OSError("short write while persisting KV manifest")
                    view = view[written:]
                self._checkpoint("manifest_write")
                os.fsync(handle.fileno())
                self._checkpoint("manifest_fsync")
            try:
                os.link(temporary, manifest_path)
            except FileExistsError:
                return self._require_identical_manifest(
                    manifest_path,
                    descriptor,
                    extents,
                )
            finally:
                temporary.unlink(missing_ok=True)
            self._checkpoint("manifest_rename")
            self._fsync_directory(self.manifest_dir)
            self._checkpoint("manifest_directory_fsync")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return SessionManifest(
            descriptor=descriptor,
            extents=extents,
            operation_id=operation_id,
            created_at_ns=created_at_ns,
            path=manifest_path,
        )

    def _require_identical_manifest(
        self,
        path: Path,
        descriptor: SessionKVDescriptor,
        extents: tuple[CommittedExtent, ...],
    ) -> SessionManifest:
        existing = self._read_manifest(path)
        expected_extents = tuple(
            (item.extent, item.path.name) for item in extents
        )
        observed_extents = tuple(
            (item.extent, item.path.name) for item in existing.extents
        )
        if existing.descriptor != descriptor or observed_extents != expected_extents:
            raise ManifestConflict(
                "a different manifest already exists for this branch and KV epoch"
            )
        return existing

    def read_extent(self, committed: CommittedExtent) -> bytes:
        self._require_extent_path(committed.path)
        data = committed.path.read_bytes()
        observed = content_checksum((data,))
        if len(data) != committed.extent.nbytes or observed != committed.extent.checksum:
            raise ChecksumMismatch(
                f"committed KV extent failed validation: {committed.path.name}"
            )
        return data

    def require_manifest(
        self,
        key: SessionBranchKey,
        *,
        kv_epoch: int,
    ) -> SessionManifest:
        path = self._manifest_path(key, kv_epoch)
        if not path.is_file():
            raise KeyError((key, kv_epoch))
        manifest = self._read_manifest(path)
        if manifest.descriptor.key != key or manifest.descriptor.kv_epoch != kv_epoch:
            raise ChecksumMismatch("manifest identity does not match its catalog key")
        return manifest

    def list_manifests(self) -> tuple[SessionManifest, ...]:
        return tuple(
            self._read_manifest(path)
            for path in sorted(self.manifest_dir.glob("*.json"))
        )

    def recover(self) -> RecoveryReport:
        removed = 0
        for path in self.temporary_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
        manifests = self.list_manifests()
        referenced = {
            extent.path.name for manifest in manifests for extent in manifest.extents
        }
        all_extents = {path.name for path in self.extent_dir.glob("*.kv")}
        return RecoveryReport(
            committed_manifests=len(manifests),
            referenced_extents=len(referenced),
            orphan_extents=len(all_extents - referenced),
            temporary_files_removed=removed,
        )

    def _read_manifest(self, path: Path) -> SessionManifest:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            core = envelope["manifest"]
            expected = envelope["checksum"]
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ChecksumMismatch(f"invalid KV manifest: {path.name}") from exc
        observed = content_checksum((_canonical_json(core),))
        if observed != expected:
            raise ChecksumMismatch(f"KV manifest checksum mismatch: {path.name}")
        if core.get("schema_version") != self.SCHEMA_VERSION:
            raise LocalKVStoreError(
                f"unsupported KV manifest schema: {core.get('schema_version')}"
            )
        try:
            descriptor = self._descriptor_from_dict(core["descriptor"])
            extents = tuple(
                self._committed_from_dict(item, core["operation_id"])
                for item in core["extents"]
            )
            manifest = SessionManifest(
                descriptor=descriptor,
                extents=extents,
                operation_id=core["operation_id"],
                created_at_ns=int(core["created_at_ns"]),
                path=path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChecksumMismatch(f"invalid KV manifest payload: {path.name}") from exc
        for committed in manifest.extents:
            self._validate_extent_owner(descriptor, committed.extent)
            self._require_extent_path(committed.path)
        return manifest

    def _committed_from_dict(
        self,
        payload: dict[str, object],
        operation_id: str,
    ) -> CommittedExtent:
        filename = payload["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("extent file must be a basename")
        return CommittedExtent(
            extent=self._extent_from_dict(payload["extent"]),
            path=self.extent_dir / filename,
            operation_id=operation_id,
        )

    @staticmethod
    def _committed_to_dict(committed: CommittedExtent) -> dict[str, object]:
        return {
            "file": committed.path.name,
            "extent": LocalKVStore._extent_to_dict(committed.extent),
        }

    @staticmethod
    def _extent_to_dict(extent: KVExtent) -> dict[str, object]:
        return {
            "instance_id": extent.instance_id,
            "model_digest": extent.model_digest,
            "placement_version": extent.placement_version,
            "stage_id": extent.stage_id,
            "session_id": extent.session_id,
            "request_id": extent.request_id,
            "predecessor_request_id": extent.predecessor_request_id,
            "kv_epoch": extent.kv_epoch,
            "token_position": extent.token_position,
            "token_range": list(extent.token_range),
            "local_layer_range": list(extent.local_layer_range),
            "dtype": extent.dtype,
            "k_shape": list(extent.k_shape),
            "v_shape": list(extent.v_shape),
            "nbytes": extent.nbytes,
            "checksum": extent.checksum,
        }

    @staticmethod
    def _extent_from_dict(payload: object) -> KVExtent:
        if not isinstance(payload, dict):
            raise ValueError("extent payload must be an object")
        values = dict(payload)
        for name in ("token_range", "local_layer_range", "k_shape", "v_shape"):
            values[name] = tuple(values[name])
        return KVExtent(**values)

    @staticmethod
    def _descriptor_to_dict(descriptor: SessionKVDescriptor) -> dict[str, object]:
        return {
            "key": {
                "instance_id": descriptor.key.instance_id,
                "stage_id": descriptor.key.stage_id,
                "session_id": descriptor.key.session_id,
                "request_id": descriptor.key.request_id,
            },
            "predecessor_request_id": descriptor.predecessor_request_id,
            "instance_id": descriptor.instance_id,
            "model_digest": descriptor.model_digest,
            "placement_version": descriptor.placement_version,
            "stage_id": descriptor.stage_id,
            "kv_epoch": descriptor.kv_epoch,
            "token_ids": list(descriptor.token_ids),
            "kv_indices": list(descriptor.kv_indices),
            "token_position": descriptor.token_position,
            "logical_bytes": descriptor.logical_bytes,
        }

    @staticmethod
    def _descriptor_from_dict(payload: object) -> SessionKVDescriptor:
        if not isinstance(payload, dict) or not isinstance(payload.get("key"), dict):
            raise ValueError("descriptor payload must contain a key object")
        values = dict(payload)
        key = dict(values.pop("key"))
        values["key"] = SessionBranchKey(**key)
        values["token_ids"] = tuple(values["token_ids"])
        values["kv_indices"] = tuple(values["kv_indices"])
        return SessionKVDescriptor(**values)

    @staticmethod
    def _validate_extent_owner(
        descriptor: SessionKVDescriptor,
        extent: KVExtent,
    ) -> None:
        expected = (
            descriptor.instance_id,
            descriptor.model_digest,
            descriptor.placement_version,
            descriptor.stage_id,
            descriptor.key.session_id,
            descriptor.key.request_id,
            descriptor.predecessor_request_id,
            descriptor.kv_epoch,
            descriptor.token_position,
        )
        observed = (
            extent.instance_id,
            extent.model_digest,
            extent.placement_version,
            extent.stage_id,
            extent.session_id,
            extent.request_id,
            extent.predecessor_request_id,
            extent.kv_epoch,
            extent.token_position,
        )
        if observed != expected:
            raise ValueError("KV extent does not belong to the session descriptor")

    def _manifest_path(self, key: SessionBranchKey, kv_epoch: int) -> Path:
        if kv_epoch < 0:
            raise ValueError("kv_epoch must be non-negative")
        payload = _canonical_json(
            {
                "instance_id": key.instance_id,
                "stage_id": key.stage_id,
                "session_id": key.session_id,
                "request_id": key.request_id,
                "kv_epoch": kv_epoch,
            }
        )
        name = hashlib.blake2b(payload, digest_size=20).hexdigest()
        return self.manifest_dir / f"{name}.json"

    def _verify_file(self, path: Path, nbytes: int, checksum: str) -> None:
        self._require_extent_path(path)
        if not path.is_file() or path.stat().st_size != nbytes:
            raise ChecksumMismatch(f"KV extent has the wrong length: {path.name}")
        observed = content_checksum((path.read_bytes(),))
        if observed != checksum:
            raise ChecksumMismatch(f"KV extent checksum mismatch: {path.name}")

    def _require_extent_path(self, path: Path) -> None:
        if path.parent.resolve() != self.extent_dir.resolve():
            raise LocalKVStoreError("KV extent path escapes the extent directory")

    @staticmethod
    def _checksum_hex(checksum: str) -> str:
        prefix, separator, value = checksum.partition(":")
        if prefix != "blake2b" or separator != ":" or len(value) != 64:
            raise ValueError("unsupported KV extent checksum")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("invalid KV extent checksum") from exc
        return value

    @staticmethod
    def _safe_id(value: str) -> str:
        return hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def _extent_identity(extent: KVExtent) -> str:
        payload = _canonical_json(LocalKVStore._extent_to_dict(extent))
        return hashlib.blake2b(payload, digest_size=12).hexdigest()

    @staticmethod
    def _require_operation_id(operation_id: str) -> None:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must not be empty")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _checkpoint(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector.checkpoint(phase)
