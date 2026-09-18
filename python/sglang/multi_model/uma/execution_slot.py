from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any, Callable, Iterator, Sequence

from sglang.multi_model.uma.model_adapter import BoundModelRuntime


class UMAControlCode(str, Enum):
    OK = "OK"
    PINNED_RESOURCE_CONFLICT = "PINNED_RESOURCE_CONFLICT"
    INSTANCE_NOT_REGISTERED = "INSTANCE_NOT_REGISTERED"
    STALE_PLACEMENT_VERSION = "STALE_PLACEMENT_VERSION"
    STALE_RESOURCE_EPOCH = "STALE_RESOURCE_EPOCH"
    MODEL_BIND_FAILURE = "MODEL_BIND_FAILURE"


@dataclass(frozen=True, slots=True)
class SafePointResult:
    code: UMAControlCode
    reasons: tuple[str, ...] = ()
    in_flight_count: int = 0

    @property
    def accepted(self) -> bool:
        return self.code is UMAControlCode.OK


@dataclass(frozen=True, slots=True)
class RuntimeBindResult:
    code: UMAControlCode
    instance_id: str
    placement_version: int
    resource_epoch: int
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.code is UMAControlCode.OK


class RuntimeBindError(RuntimeError):
    pass


class IncompleteRuntimeError(RuntimeBindError):
    pass


class StaleRuntimeError(RuntimeBindError):
    pass


def assess_scheduler_safe_point(
    *,
    waiting_count: int,
    running_counts: Sequence[int],
    grammar_count: int,
    session_count: int,
    has_chunked_request: bool,
    overlap_enabled: bool,
    overlap_result_count: int,
    worker_in_flight_count: int,
) -> SafePointResult:
    """Return facts about whether a model-neutral scheduler boundary exists.

    This function intentionally contains no waiting or preemption policy.  A
    controller may retry later, but the mechanism never drains or aborts work
    merely because a bind was requested.
    """

    reasons: list[str] = []
    if waiting_count:
        reasons.append(f"{waiting_count} queued request(s)")
    running = sum(running_counts)
    if running:
        reasons.append(f"{running} running request(s)")
    if grammar_count:
        reasons.append(f"{grammar_count} grammar request(s)")
    if session_count:
        reasons.append(f"{session_count} session(s) still attached")
    if has_chunked_request:
        reasons.append("chunked prefill is active")
    if overlap_enabled:
        # Dynamic model binding is deliberately conservative in the first
        # implementation: the overlap worker owns a background forward thread.
        reasons.append("overlap scheduling is enabled")
    if overlap_result_count:
        reasons.append(f"{overlap_result_count} overlap result(s) pending")
    if worker_in_flight_count:
        reasons.append(f"{worker_in_flight_count} worker forward(s) in flight")
    return SafePointResult(
        UMAControlCode.OK if not reasons else UMAControlCode.PINNED_RESOURCE_CONFLICT,
        tuple(reasons),
        worker_in_flight_count,
    )


class UMAExecutionSlot:
    """Thread-safe runtime registry and atomic binding guard for one worker.

    The slot owns no scheduling policy and performs no weight I/O.  It only
    records complete runtimes, prevents a bind during a forward pass, and
    commits a new active identity after the caller's binder has succeeded.
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, BoundModelRuntime] = {}
        self._active_instance_id: str | None = None
        self._active_weight_lease: Any | None = None
        self._in_flight_count = 0
        self._accepting_forwards = True
        self._lock = threading.RLock()

    @property
    def active_instance_id(self) -> str | None:
        with self._lock:
            return self._active_instance_id

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return self._in_flight_count

    @property
    def accepting_forwards(self) -> bool:
        with self._lock:
            return self._accepting_forwards

    def register(self, runtime: BoundModelRuntime, *, replace: bool = False) -> None:
        runtime.validate_complete()
        with self._lock:
            existing = self._runtimes.get(runtime.instance_id)
            if existing is not None and not replace:
                raise ValueError(f"runtime already registered: {runtime.instance_id}")
            if (
                existing is not None
                and self._active_instance_id == runtime.instance_id
            ):
                raise RuntimeBindError(
                    "cannot replace the active runtime through registration; "
                    "register a new epoch under an inactive identity and bind it"
                )
            self._runtimes[runtime.instance_id] = runtime

    def get(self, instance_id: str) -> BoundModelRuntime:
        with self._lock:
            try:
                return self._runtimes[instance_id]
            except KeyError as exc:
                raise KeyError(f"runtime is not registered: {instance_id}") from exc

    def unregister(self, instance_id: str) -> None:
        with self._lock:
            if self._active_instance_id == instance_id:
                raise RuntimeBindError("cannot unregister the active runtime")
            self._runtimes.pop(instance_id, None)

    @staticmethod
    def _acquire_weight_lease(runtime: BoundModelRuntime) -> Any | None:
        """Validate and pin a complete runtime for its whole bound lifetime."""

        runtime.validate_complete()
        if not runtime.required_weight_groups:
            return None
        weight_runtime = runtime.execution_resources.weight_runtime
        observed = weight_runtime.readiness(runtime.instance_id)
        missing = runtime.required_weight_groups - observed
        if missing:
            raise IncompleteRuntimeError(
                f"runtime lost ready weight groups: {sorted(missing)}"
            )
        return weight_runtime.pin(
            sorted(runtime.required_weight_groups),
            f"active:{runtime.instance_id}:{runtime.resource_epoch}",
            instance_id=runtime.instance_id,
        )

    def _release_active_weight_lease(self) -> None:
        lease = self._active_weight_lease
        if lease is None:
            return
        try:
            lease.release()
        except BaseException:
            if not getattr(lease, "active", True):
                self._active_weight_lease = None
            raise
        else:
            self._active_weight_lease = None

    def bind(
        self,
        instance_id: str,
        *,
        placement_version: int,
        resource_epoch: int,
        binder: Callable[[BoundModelRuntime], None],
    ) -> RuntimeBindResult:
        with self._lock:
            runtime = self._runtimes.get(instance_id)
            if runtime is None:
                return RuntimeBindResult(
                    UMAControlCode.INSTANCE_NOT_REGISTERED,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    "runtime is not registered",
                )
            if runtime.placement_version != placement_version:
                return RuntimeBindResult(
                    UMAControlCode.STALE_PLACEMENT_VERSION,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    f"registered placement version is {runtime.placement_version}",
                )
            if runtime.resource_epoch != resource_epoch:
                return RuntimeBindResult(
                    UMAControlCode.STALE_RESOURCE_EPOCH,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    f"registered resource epoch is {runtime.resource_epoch}",
                )
            if self._in_flight_count:
                return RuntimeBindResult(
                    UMAControlCode.PINNED_RESOURCE_CONFLICT,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    f"{self._in_flight_count} forward(s) in flight",
                )
            old_runtime = (
                self._runtimes.get(self._active_instance_id)
                if self._active_instance_id is not None
                else None
            )
            old_lease = self._active_weight_lease
            try:
                target_lease = self._acquire_weight_lease(runtime)
            except Exception as exc:
                return RuntimeBindResult(
                    UMAControlCode.MODEL_BIND_FAILURE,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    str(exc),
                )
            try:
                binder(runtime)
            except Exception as exc:
                if target_lease is not None:
                    target_lease.release()
                return RuntimeBindResult(
                    UMAControlCode.MODEL_BIND_FAILURE,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    str(exc),
                )
            try:
                if old_lease is not None:
                    old_lease.release()
            except Exception as exc:
                rollback_errors = []
                if old_runtime is not None:
                    try:
                        binder(old_runtime)
                    except Exception as rollback_exc:
                        rollback_errors.append(f"runner rollback: {rollback_exc}")
                if target_lease is not None:
                    try:
                        target_lease.release()
                    except Exception as rollback_exc:
                        rollback_errors.append(
                            f"target lease rollback: {rollback_exc}"
                        )
                if old_runtime is not None and not getattr(old_lease, "active", True):
                    try:
                        old_lease = self._acquire_weight_lease(old_runtime)
                    except Exception as rollback_exc:
                        rollback_errors.append(
                            f"old lease reacquire: {rollback_exc}"
                        )
                        old_lease = None
                self._active_weight_lease = old_lease
                detail = f"active lease release failed: {exc}"
                if rollback_errors:
                    detail += "; " + "; ".join(rollback_errors)
                return RuntimeBindResult(
                    UMAControlCode.MODEL_BIND_FAILURE,
                    instance_id,
                    placement_version,
                    resource_epoch,
                    detail,
                )
            self._active_instance_id = instance_id
            self._active_weight_lease = target_lease
            self._accepting_forwards = True
            return RuntimeBindResult(
                UMAControlCode.OK,
                instance_id,
                placement_version,
                resource_epoch,
            )

    def quiesce(self) -> SafePointResult:
        with self._lock:
            self._accepting_forwards = False
            if self._in_flight_count:
                return SafePointResult(
                    UMAControlCode.PINNED_RESOURCE_CONFLICT,
                    (f"{self._in_flight_count} worker forward(s) in flight",),
                    self._in_flight_count,
                )
            self._release_active_weight_lease()
            return SafePointResult(UMAControlCode.OK)

    def unbind(self, instance_id: str) -> SafePointResult:
        with self._lock:
            self._accepting_forwards = False
            if self._in_flight_count:
                return SafePointResult(
                    UMAControlCode.PINNED_RESOURCE_CONFLICT,
                    (f"{self._in_flight_count} worker forward(s) in flight",),
                    self._in_flight_count,
                )
            if self._active_instance_id not in {None, instance_id}:
                return SafePointResult(
                    UMAControlCode.MODEL_BIND_FAILURE,
                    (
                        f"active runtime is {self._active_instance_id}, "
                        f"not {instance_id}",
                    ),
                )
            self._release_active_weight_lease()
            self._active_instance_id = None
            return SafePointResult(UMAControlCode.OK)

    def resume(self) -> None:
        with self._lock:
            if self._active_instance_id is None:
                raise RuntimeBindError("cannot resume without an active runtime")
            if self._active_weight_lease is None:
                runtime = self._runtimes[self._active_instance_id]
                self._active_weight_lease = self._acquire_weight_lease(runtime)
            self._accepting_forwards = True

    @contextmanager
    def forward_lease(
        self,
        *,
        owner: str = "forward",
    ) -> Iterator[BoundModelRuntime | None]:
        with self._lock:
            if not self._accepting_forwards:
                raise RuntimeBindError("execution slot is quiesced")
            runtime = (
                self._runtimes.get(self._active_instance_id)
                if self._active_instance_id is not None
                else None
            )
            self._in_flight_count += 1
        try:
            yield runtime
        finally:
            with self._lock:
                self._in_flight_count -= 1
                if self._in_flight_count < 0:
                    self._in_flight_count = 0
                    raise RuntimeError("execution-slot forward count underflow")
