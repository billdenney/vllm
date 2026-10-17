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
    load_flashinfer_autotune_cache_only,
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
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
        if envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY:
            logger.warning_once(
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY has no effect because "
                "FlashInfer autotune is disabled; no cache is loaded."
            )
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

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
    """One rank's autotune cache file, as the cache-only mode sees it."""

    status: _CacheStatus
    fingerprint: tuple[str, int] | None
    tuning_group: tuple[int, ...]


def _replay_cache_state(
    path: Path, tune_group: "GroupCoordinator"
) -> _ReplayCacheState:
    """Classify this rank's autotune cache file for the cache-only mode.

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

    With ``VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY`` set, every rank loads its
    cache and the pass is skipped, and a cache that cannot be loaded on any
    rank fails every rank; a start where no rank has a cache tunes and records
    it, decided over the whole world group.
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
    record = False
    if envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY:
        states = _gather_from_group(_replay_cache_state(cache_path, world), world)
        if not all(state.status == "missing" for state in states):
            load_flashinfer_autotune_cache_only(cache_path, world)
            if is_leader:
                logger.info_once(
                    "Loaded the FlashInfer autotune cache; skipping autotuning "
                    "because VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY is set. Ops "
                    "without a cached entry use FlashInfer's default tactic."
                )
            return
        # No rank has a cache yet: this start records it.
        record = True
        if is_leader:
            logger.info(
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY is set and no rank has a "
                "FlashInfer autotune cache; autotuning to record it in %s.",
                cache_path,
            )
    elif _all_ranks_have_matching_cache(cache_path, world):
        loaded = all(_gather_from_group(tuner.load_configs(str(cache_path)), world))
        if not loaded:
            tuner.clear_cache()

    with torch.inference_mode(), fi_utils.autotune():
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
    # the file). A start that records the cache always saves, even with
    # nothing tuned, so that the next start loads it.
    if tuner._dirty or record:
        before = _file_signature(cache_path)
        tuner.save_configs(str(cache_path))
        # Only a changed file counts as saved.
        if is_leader and _file_signature(cache_path) not in (None, before):
            logger.info("Saved FlashInfer autotune cache file %s.", cache_path)
    if record:
        states = _gather_from_group(_replay_cache_state(cache_path, world), world)
        problems = [
            f"rank {rank}: {state.status}"
            for rank, state in enumerate(states)
            if state.status != "present"
        ]
        if problems:
            raise RuntimeError(
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY is set, but the FlashInfer "
                "autotune cache this start recorded is not usable on every rank "
                f"({', '.join(problems)}). Delete {cache_path.parent} to record "
                "it again."
            )
