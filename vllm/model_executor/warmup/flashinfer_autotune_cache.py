# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig, replace
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _normalize_cache_config_for_hash(cache_config: CacheConfig) -> CacheConfig:
    """Normalize cache config for hashing"""
    if cache_config.mamba_block_size is None and cache_config.kv_cache_layout is None:
        return cache_config
    normalized = replace(cache_config, mamba_block_size=None)
    assert normalized.kv_cache_layout is None
    return normalized


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    vllm_config: VllmConfig = runner.vllm_config
    cache_config = _normalize_cache_config_for_hash(vllm_config.cache_config)
    if cache_config is not vllm_config.cache_config:
        vllm_config = replace(vllm_config, cache_config=cache_config)
    config_hash = vllm_config.compute_hash(include_version=False)
    return hashlib.sha256(config_hash.encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def use_flashinfer_autotune_v2(runner: "GPUModelRunner") -> bool:
    from vllm.distributed.parallel_state import get_node_count, get_world_group
    from vllm.utils.flashinfer import has_flashinfer_autotune_v2

    # Elastic EP transfers tuning state to joining workers with v1
    # save_configs/load_configs, which do not export managed-store winners.
    if runner.vllm_config.parallel_config.enable_elastic_ep:
        return False
    # PP stages profile independently. Keep their group-scoped v1 files until
    # managed-cache reads and publishes can be isolated across tuning groups.
    if runner.vllm_config.parallel_config.pipeline_parallel_size > 1:
        return False
    if not has_flashinfer_autotune_v2():
        return False
    # Use the initialized topology: Ray may span nodes with config.nnodes=1.
    # A cache hit skips profiling collectives, so node-local stores with
    # different contents cannot safely participate in synchronized tuning.
    if get_node_count() != 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-node deployments; managed-cache synchronization is "
            "currently supported only within one node."
        )
        return False
    # Lift this gate after pinning a release with the fix below and validating
    # distributed managed-cache tuning: https://github.com/flashinfer-ai/flashinfer/pull/6185
    if get_world_group().world_size > 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-rank deployments until FlashInfer coordinates cache hits."
        )
        return False
    return True


def resolve_flashinfer_autotune_v2_root() -> Path | None:
    """Placement root for FlashInfer's managed autotune store.

    Unlike :func:`resolve_flashinfer_autotune_file`, no vLLM-side identity
    (config hash, flashinfer version, arch) is folded into the path: the
    managed store keys entries per op and hashes the software/hardware
    environment itself. ``None`` selects FlashInfer's default root.
    """
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    return Path(override_dir).expanduser() if override_dir else None


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if (
        group.rank_in_group == 0
        and runner.vllm_config.kernel_config.enable_flashinfer_autotune
    ):
        try:
            from vllm.platforms import current_platform
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer() and current_platform.has_device_capability(90):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def load_flashinfer_autotune_cache_only(
    cache_path: Path, world: "GroupCoordinator"
) -> None:
    """Load this rank's autotune cache, or fail on every rank.

    Read and load errors are gathered over ``world`` and re-raised on all
    ranks, so every rank makes the same decision and none is left waiting in
    a later collective. A successful load only means FlashInfer accepted the
    file; ops without an entry use FlashInfer's default tactic.

    Raises:
        RuntimeError: If any rank's cache is missing, empty, unreadable or
            rejected by FlashInfer.

    """
    error: str | None = None
    try:
        from flashinfer.autotuner import AutoTuner

        if cache_path.stat().st_size == 0:
            error = f"{cache_path} is empty"
        elif not AutoTuner.get().load_configs(str(cache_path)):
            error = f"FlashInfer rejected {cache_path} (environment mismatch)"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    errors: list[str | None] = [error]
    if world.world_size > 1:
        errors = [None] * world.world_size
        torch.distributed.all_gather_object(errors, error, group=world.cpu_group)
    failures = "; ".join(
        f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
    )
    if failures:
        raise RuntimeError(
            "--flashinfer-autotune-cache-only is set but the FlashInfer "
            f"autotune cache could not be loaded: {failures}. Drop the flag "
            "to autotune, or prepare a cache for this configuration."
        )


def check_managed_flashinfer_autotune_cache(tuner: Any) -> int:
    """Check the managed store a cache-only start replays, or fail.

    FlashInfer treats a missing, malformed or misplaced entry as a cache miss
    and serves the default tactic, so a cache-only start refuses a store with
    no entry for this environment, or with an entry FlashInfer would ignore.
    The managed path runs on one rank only (``use_flashinfer_autotune_v2``),
    so the decision needs no collective. Call it inside the
    ``autotune_v2(mode="replay")`` context, which attaches the store.

    Returns:
        The number of entries in the store.

    Raises:
        RuntimeError: If no store is attached, or the store has no entries,
            or an entry is unreadable or not stored under its own key.

    """
    store = tuner._managed_cache
    if store is None:
        raise RuntimeError(
            "--flashinfer-autotune-cache-only is set but no FlashInfer managed "
            "autotune cache is attached."
        )
    entries_dir = Path(store.entries_dir)
    paths = sorted(entries_dir.glob("*.json")) if entries_dir.is_dir() else []
    if not paths:
        raise RuntimeError(
            "--flashinfer-autotune-cache-only is set but the FlashInfer managed "
            f"autotune cache has no entries for this environment ({entries_dir}). "
            "Drop the flag to autotune, or prepare the cache with a start "
            "without it."
        )
    invalid = []
    for path in paths:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            key = entry["key"]
            # Where ManagedAutotuneCache._entry_path files an entry.
            expected = hashlib.sha256(key.encode()).hexdigest()[:24]
            if path.stem != expected or "runner" not in entry or "tactic" not in entry:
                raise ValueError("not an entry stored under its own key")
        except Exception as exc:
            invalid.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if invalid:
        raise RuntimeError(
            "--flashinfer-autotune-cache-only is set but the FlashInfer managed "
            f"autotune cache {entries_dir} holds entries FlashInfer would ignore: "
            f"{'; '.join(invalid)}. Drop the flag to autotune, or prepare the "
            "cache again."
        )
    return len(paths)


@contextmanager
def count_flashinfer_autotune_misses(tuner: Any) -> Iterator[dict[str, set]]:
    """Collect the op shapes that fall back to FlashInfer's default tactic.

    Outside tuning, FlashInfer's ``AutoTuner.choose_one`` serves a lookup
    without a cached winner with the default tactic and records it nowhere,
    so the lookups are counted here: op name to the input shapes that missed.
    """
    misses: dict[str, set] = defaultdict(set)
    search_cache = tuner.search_cache

    def counted(custom_op, runners, input_shapes, *args, **kwargs):
        result = search_cache(custom_op, runners, input_shapes, *args, **kwargs)
        if not result[0]:
            misses[custom_op].add(tuple(tuple(shape) for shape in input_shapes))
        return result

    tuner.search_cache = counted
    try:
        yield misses
    finally:
        del tuner.search_cache


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
