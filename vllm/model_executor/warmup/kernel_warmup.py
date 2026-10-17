# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_T = TypeVar("_T")


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    replay = worker.vllm_config.kernel_config.flashinfer_autotune_cache_mode == "replay"
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
        if replay:
            logger.warning_once(
                "FlashInfer autotune cache mode 'replay' has no effect, since "
                "FlashInfer autotuning is disabled."
            )
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)
    elif replay:
        logger.warning_once(
            "FlashInfer autotune cache mode 'replay' has no effect, since "
            "FlashInfer autotuning needs FlashInfer and a GPU of SM 9.0 or later."
        )

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def _autotune_cache_fingerprint(path: Path) -> tuple[str, int] | None:
    """Identify a saved autotune file by its FlashInfer metadata and size."""
    try:
        configs = json.loads(path.read_text())
        metadata = configs.pop("_metadata", None)
    except (OSError, ValueError, AttributeError):
        return None
    return json.dumps(metadata, sort_keys=True), len(configs)


def _all_ranks_have_matching_cache(path: Path, group) -> bool:
    """True iff every rank in ``group`` has its own, mutually consistent file."""
    fingerprint = _autotune_cache_fingerprint(path)
    if group.world_size == 1:
        return fingerprint is not None
    gathered: list[tuple[str, int] | None] = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, fingerprint, group=group.cpu_group)
    return fingerprint is not None and all(f == fingerprint for f in gathered)


class _ReplayCacheState(NamedTuple):
    """One rank's autotune cache file, as replay mode sees it."""

    status: str  # "missing", "malformed" or "present"
    fingerprint: tuple[str, int] | None


def _replay_cache_state(path: Path) -> _ReplayCacheState:
    if not path.exists():
        return _ReplayCacheState("missing", None)
    fingerprint = _autotune_cache_fingerprint(path)
    if fingerprint is None:
        return _ReplayCacheState("malformed", None)
    return _ReplayCacheState("present", fingerprint)


def _gather_from_group(obj: _T, group: "GroupCoordinator") -> list[_T]:
    """All-gather a picklable object over a group.

    Args:
        obj: This rank's object.
        group: The group to gather over.

    Returns:
        Every rank's object in rank order; ``[obj]`` for a group of one.
    """
    if group.world_size == 1:
        return [obj]
    gathered: list = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, obj, group=group.cpu_group)
    return gathered


def _all_ranks_have_matching_cache(path: Path, group) -> bool:
    """True iff every rank in ``group`` has its own, mutually consistent file."""
    fingerprint = _autotune_cache_fingerprint(path)
    gathered = _gather_from_group(fingerprint, group)
    return fingerprint is not None and all(f == fingerprint for f in gathered)


_CacheStatus = Literal["missing", "unreadable", "malformed", "present"]


class _ReplayCacheState(NamedTuple):
    """One rank's autotune cache file, as replay mode sees it."""

    status: _CacheStatus
    fingerprint: tuple[str, int] | None
    tuning_group: tuple[int, ...]


def _replay_cache_state(
    path: Path, tune_group: "GroupCoordinator"
) -> _ReplayCacheState:
    """Classify this rank's autotune cache file for replay mode.

    Args:
        path: This rank's cache file.
        tune_group: This rank's tuning group, whose ranks must hold
            matching files.

    Returns:
        The file's status, its fingerprint when it is a readable record, and
        the global ranks of the tuning group.
    """
    ranks = tuple(tune_group.ranks)
    try:
        if not path.exists():
            return _ReplayCacheState("missing", None, ranks)
        path.read_bytes()
    except OSError:
        return _ReplayCacheState("unreadable", None, ranks)
    fingerprint = _autotune_cache_fingerprint(path)
    if fingerprint is None:
        return _ReplayCacheState("malformed", None, ranks)
    return _ReplayCacheState("present", fingerprint, ranks)


def _check_replay_cache_states(
    states: list[_ReplayCacheState], cache_dir: Path, *, recorded: bool
) -> bool:
    """Decide what a replay-mode start does with the world's cache files.

    Every rank gets the same states and so the same decision, raising
    together rather than leaving the others in a collective. A record missing
    on some ranks is not completed by tuning those ranks: that would re-tune
    silently and rewrite the record.

    Args:
        states: Every rank's state, in world rank order.
        cache_dir: The directory to delete to record the tactics again.
        recorded: Whether this start just saved the record, which must then
            exist on every rank.

    Returns:
        False when no rank has a file and nothing was recorded yet, so this
        start records the tactics; True when every rank's file is present
        and the files agree within each tuning group.

    Raises:
        RuntimeError: Otherwise, with the same message on every rank.
    """
    what = "the record this start saved" if recorded else "the record"
    if not recorded and all(state.status == "missing" for state in states):
        return False
    problems = [
        f"rank {rank}: {state.status}"
        for rank, state in enumerate(states)
        if state.status != "present"
    ]
    if problems:
        raise RuntimeError(
            f"FlashInfer autotune cache mode is 'replay', but {what} is not "
            f"usable on every rank ({', '.join(problems)}). Delete {cache_dir} "
            "to record the tactics again."
        )
    fingerprints: dict[tuple[int, ...], set] = {}
    for state in states:
        fingerprints.setdefault(state.tuning_group, set()).add(state.fingerprint)
    split = sorted(group for group, found in fingerprints.items() if len(found) > 1)
    if split:
        raise RuntimeError(
            f"FlashInfer autotune cache mode is 'replay', but {what} differs "
            f"across the ranks of tuning groups {split}. Delete {cache_dir} to "
            "record the tactics again."
        )
    return True


def _load_replay_cache(tuner, path: Path, world: "GroupCoordinator") -> None:
    """Load this rank's record, raising on every rank if any rank fails.

    Args:
        tuner: FlashInfer's ``AutoTuner``.
        path: This rank's cache file.
        world: The world group, over which the results are gathered.

    Raises:
        RuntimeError: When FlashInfer rejects or cannot read the file on any
            rank, with the same message on every rank.
    """
    try:
        loaded = bool(tuner.load_configs(str(path)))
    except Exception:
        logger.exception("FlashInfer could not load the autotune cache %s.", path)
        loaded = False
    gathered = _gather_from_group(loaded, world)
    rejected = [rank for rank, ok in enumerate(gathered) if not ok]
    if rejected:
        raise RuntimeError(
            "FlashInfer autotune cache mode is 'replay', but FlashInfer "
            f"rejected or could not read the record on ranks {rejected}; its "
            "warning names what differs, such as the FlashInfer, CUDA, cuBLAS "
            f"or cuDNN version or the GPU. Delete {path.parent} to record the "
            "tactics again."
        )


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Results are persisted per rank under the configuration's hash, and a
    later start loads them when every rank has a matching file.

    With ``KernelConfig.flashinfer_autotune_cache_mode == "replay"`` the files
    are the record of the tactics, decided over the whole world group: when
    no rank has a file, every rank tunes and saves; when every rank has one,
    every rank loads it and this pass times nothing; anything else raises on
    every rank.
    """
    from flashinfer.autotuner import AutoTuner

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    is_leader = world.rank_in_group == 0
    tuner = AutoTuner.get()

    cache_path = resolve_flashinfer_autotune_file(runner)
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    cache_path = cache_path.with_name(
        f"{cache_path.stem}_dp{dp_rank}_rank{world.rank_in_group}{cache_path.suffix}"
    )
    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)
    replay = runner.vllm_config.kernel_config.flashinfer_autotune_cache_mode == "replay"

    tune_mode = True
    if replay:
        states = _gather_from_group(_replay_cache_state(cache_path, world), world)
        if _check_replay_cache_states(states, cache_path.parent, recorded=False):
            _load_replay_cache(tuner, cache_path, world)
            tune_mode = False
            if is_leader:
                logger.info(
                    "Loaded FlashInfer autotune cache file %s in replay mode; "
                    "no FlashInfer tactics were timed.",
                    cache_path,
                )
        elif is_leader:
            logger.info(
                "FlashInfer autotune cache mode is 'replay' and no cache file "
                "exists; autotuning to record the tactics in %s.",
                cache_path,
            )
    elif _all_ranks_have_matching_cache(cache_path, world):
        loaded = all(_gather_from_group(tuner.load_configs(str(cache_path)), world))
        if not loaded:
            tuner.clear_cache()

    with torch.inference_mode(), fi_utils.autotune(tune_mode=tune_mode):
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process.
        fi_utils._is_fi_autotuning = True

        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )

        fi_utils._is_fi_autotuning = False

    if world.world_size > 1:
        world.barrier()
    # Skip the rewrite when nothing was tuned this start (every entry came from
    # the file). A replay-mode start that records always saves, even with
    # nothing tuned, so that the next start replays; one that loaded the record
    # never does.
    if tune_mode and (tuner._dirty or replay):
        before = _file_signature(cache_path)
        tuner.save_configs(str(cache_path))
        # Only a changed file counts as saved.
        if is_leader and _file_signature(cache_path) not in (None, before):
            logger.info("Saved FlashInfer autotune cache file %s.", cache_path)
    if replay and tune_mode:
        states = _gather_from_group(_replay_cache_state(cache_path, world), world)
        _check_replay_cache_states(states, cache_path.parent, recorded=True)
