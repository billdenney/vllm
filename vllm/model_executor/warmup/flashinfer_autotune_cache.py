# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    config_hash = runner.vllm_config.compute_hash()
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
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_ONLY is set but the FlashInfer "
            f"autotune cache could not be loaded: {failures}. Unset it to "
            "autotune, or prepare a cache for this configuration."
        )
