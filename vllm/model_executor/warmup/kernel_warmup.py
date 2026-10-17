# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


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


def _gather_from_group(obj, group) -> list:
    if group.world_size == 1:
        return [obj]
    gathered: list = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, obj, group=group.cpu_group)
    return gathered


def _load_flashinfer_autotune_replay_cache(tuner, path: Path, group) -> bool:
    """Load the replay-mode cache on every rank of ``group``.

    Returns False when no rank has a file, which makes this start the one that
    records the tactics, and True when every rank loaded its file. Anything
    else raises on every rank, after the ranks have exchanged their states, so
    that no rank is left waiting in a collective: replay never falls back to
    tuning, since the tactics tuning picks depend on timing.
    """
    states = _gather_from_group(_replay_cache_state(path), group)
    if all(state.status == "missing" for state in states):
        return False
    problems = [
        f"rank {rank}: {state.status}"
        for rank, state in enumerate(states)
        if state.status != "present"
    ]
    if problems:
        raise RuntimeError(
            "FlashInfer autotune cache mode is 'replay', but the cache files "
            f"are not usable on every rank ({', '.join(problems)}; this rank's "
            f"file is {path}). Delete the cache directory to record the "
            "tactics again."
        )
    if len({state.fingerprint for state in states}) > 1:
        raise RuntimeError(
            "FlashInfer autotune cache mode is 'replay', but the cache files "
            f"differ across the ranks of the tuning group (this rank's file is "
            f"{path}). Delete the cache directory to record the tactics again."
        )
    loaded = _gather_from_group(tuner.load_configs(str(path)), group)
    rejected = [rank for rank, ok in enumerate(loaded) if not ok]
    if rejected:
        raise RuntimeError(
            "FlashInfer autotune cache mode is 'replay', but FlashInfer "
            f"rejected the cache file on ranks {rejected} (this rank's file is "
            f"{path}); its warning names the metadata that differs, such as the "
            "FlashInfer, CUDA, cuBLAS or cuDNN version or the GPU. Delete the "
            "cache directory to record the tactics again."
        )
    return True


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

    With ``KernelConfig.flashinfer_autotune_cache_mode == "replay"`` the
    loaded files are authoritative: nothing is timed, a file that cannot be
    loaded is an error, and only a start without any file tunes (and saves).
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
        if _load_flashinfer_autotune_replay_cache(tuner, cache_path, world):
            tune_mode = False
            if is_leader:
                logger.info(
                    "Loaded FlashInfer autotune cache file %s in replay mode; "
                    "no autotuning ran.",
                    cache_path,
                )
        elif is_leader:
            logger.info(
                "FlashInfer autotune cache mode is 'replay' and no cache file "
                "exists; autotuning to record the tactics in %s.",
                cache_path,
            )
    elif _all_ranks_have_matching_cache(cache_path, world):
        loaded = tuner.load_configs(str(cache_path))
        if world.world_size > 1:
            loaded_by_rank: list[bool | None] = [None] * world.world_size
            torch.distributed.all_gather_object(
                loaded_by_rank, loaded, group=world.cpu_group
            )
            loaded = all(loaded_by_rank)
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
    # nothing tuned, so that the next start replays instead of recording again.
    if tuner._dirty or (replay and tune_mode):
        tuner.save_configs(str(cache_path))
        if is_leader:
            logger.info("Saved FlashInfer autotune cache file %s.", cache_path)
