# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.attention_warmup import (
    mixed_batch_attention_warmup,
)
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    autotune_hisparse_flashinfer_attention,
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    kimi_k3_triton_warmup,
)
from vllm.model_executor.warmup.mamba_triton_warmup import mamba_triton_warmup
from vllm.model_executor.warmup.qwen4_exp_qsa_warmup import (
    qwen4_exp_qsa_triton_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.qwen_vl_triton_warmup import qwen_vl_triton_warmup
from vllm.model_executor.warmup.replayssm_warmup import (
    replayssm_autotune_warmup,
)
from vllm.model_executor.warmup.watermark_sample_warmup import (
    watermark_sample_warmup,
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

_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _ll_bf16_router_shapes_from_model(
    model: torch.nn.Module,
) -> tuple[tuple[int, int], ...]:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if not isinstance(module, GateLinear):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        n, k = weight.shape
        if k % 8 == 0:
            shapes.add((int(k), int(n)))
    return tuple(sorted(shapes))


def _warmup_ll_bf16_router_gemm(model: torch.nn.Module) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    shapes = _ll_bf16_router_shapes_from_model(model)
    if not shapes:
        logger.debug_once(
            "Skipping ll_bf16 router GEMM warmup: no bf16 GateLinear shapes found."
        )
        return

    logger.info_once("Warming up ll_bf16 router GEMM kernels for shapes: %s.", shapes)
    ll_bf16_gemm_kernel.warmup(
        shapes=shapes,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def _warmup_bf16x3_router_gemm(
    model: torch.nn.Module,
    max_num_tokens: int,
) -> None:
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
        warmup_bf16x3_router_gemm,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = next(
        (
            module
            for module in model.modules()
            if isinstance(module, GateLinear) and module.allow_bf16x3_router_gemm
        ),
        None,
    )
    if gate is None:
        logger.debug_once(
            "Skipping BF16x3 router GEMM warmup: no eligible GateLinear found."
        )
        return

    min_num_tokens = gate.FP32_MAX_TOKENS + 1 if gate.allow_fp32_router_gemm else 1
    logger.info_once(
        "Warming up BF16x3 router GEMM for K=%d, M=%d.",
        gate.input_size,
        gate.output_size,
    )
    configs = warmup_bf16x3_router_gemm(
        gate.input_size,
        gate.output_size,
        min_num_tokens,
        max_num_tokens,
    )
    logger.info_once("Warmed up BF16x3 router GEMM configs: %s.", configs)


def _warmup_gemm_rs_ar() -> None:
    # Model construction (Kimi-K3, DeepSeek-V4.1) imports this module only
    # when GEMM-RS/AR is enabled and initializes its singleton before
    # kernel_warmup runs. Avoid importing it here so other models do not
    # compile the RS/AR variants.
    module = sys.modules.get("vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar")
    if module is None:
        return
    compiled = module.warmup_gemm_rs_ar()
    if compiled:
        logger.info_once("Warmed up %d GEMM-RS/AR variants.", compiled)


def _autotune_kimi_k3_kda_qkvg(model: torch.nn.Module) -> None:
    module = sys.modules.get("vllm.models.kimi_k3.nvidia.low_latency_gemm")
    if module is not None:
        module.autotune_kda_qkvg(model)


def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    if not worker.use_v2_model_runner:
        # The KV-block zeroing kernel is driven by the scheduler's
        # `new_block_ids_to_zero`, so no dummy run ever reaches it.
        zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.warmup(worker.model_runner.kv_cache_config.num_blocks)

    enable_jit_warmup = worker.vllm_config.kernel_config.enable_jit_warmup
    if enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            registry = (
                worker.model_runner.jit_warmup_registry  # type: ignore[attr-defined]
            )
            registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)
    qwen_vl_triton_warmup(worker.model_runner)
    mamba_triton_warmup(worker.model_runner)

    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])

    # Run next so input-prep kernels JIT against pristine runner state.
    if enable_jit_warmup:
        kimi_k3_triton_warmup(worker)
        watermark_sample_warmup(worker)
        qwen4_exp_qsa_triton_warmup(worker)

    if enable_jit_warmup and current_platform.is_device_capability_family(100):
        _warmup_bf16x3_router_gemm(
            worker.get_model(),
            worker.scheduler_config.max_num_batched_tokens,
        )

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm(worker.get_model())

    _warmup_gemm_rs_ar()

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    if current_platform.is_rocm():
        from vllm.model_executor.warmup.rocm_segmented_attn_autotune_warmup import (
            rocm_segmented_attn_autotune_warmup,
        )

        rocm_segmented_attn_autotune_warmup(worker)

    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        is_deep_gemm_supported() and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    b12x_warmup(worker, cudagraph_capture_sizes)

    minimax_m3_msa_warmup(worker)

    # Allocate the exact decode-sized workspace, autotune cache misses, and
    # resolve every CUDA Graph bucket before capture begins.
    # Lazy import: flashinfer_pcie_ipc_all_reduce imports flashinfer.comm,
    # which initializes CUDA at import time and must not run at module scope.
    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:
        from vllm.distributed.device_communicators import (
            flashinfer_pcie_ipc_all_reduce,
        )

        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    replay = worker.vllm_config.kernel_config.flashinfer_autotune_cache_mode == "replay"
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
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
        logger.info_once("Warming up FlashInfer attention.")
        mixed_batch_attention_warmup(worker)


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


def _flashinfer_deferred_moe_token_counts(
    runner: "GPUModelRunner",
) -> tuple[int, ...]:
    """Return bounded token counts that exercise deferred MoE finalization."""
    from vllm.model_executor.layers.fused_moe import MoERunner

    max_tokens = runner.scheduler_config.max_num_batched_tokens
    token_counts: list[int] = []
    for module in runner.get_model().modules():
        if not isinstance(module, MoERunner):
            continue

        moe_config = module.moe_config
        max_deferred_tokens = moe_config.defer_moe_finalize_max_num_tokens
        if moe_config.use_deferred_moe_finalize and max_deferred_tokens > 0:
            token_counts.append(min(max_tokens, max_deferred_tokens))

    return tuple(dict.fromkeys(token_counts))


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    # Tune the widest bucket set first so bounded passes reuse its configs.
    token_counts = [max_tokens]
    token_counts.extend(_flashinfer_deferred_moe_token_counts(runner))
    return tuple(dict.fromkeys(token_counts))


def _run_flashinfer_autotune_dummy_runs(
    runner: "GPUModelRunner", *, skip_attn: bool = False, tune_mode: bool = True
) -> None:
    import vllm.utils.flashinfer as fi_utils

    dummy_run_kwargs = {"skip_attn": True} if skip_attn else {}
    for num_tokens in _flashinfer_autotune_token_counts(runner):
        # The drafter may use more tokens than this pass in the same dummy run.
        # Include its max M when building the autotune buckets.
        max_tuning_tokens = num_tokens
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            num_query_per_req = speculator.num_query_per_req
            num_draft_reqs = min(
                num_tokens,
                runner.max_num_reqs,
                runner.max_num_tokens // num_query_per_req,
            )
            max_tuning_tokens = max(
                max_tuning_tokens, num_draft_reqs * num_query_per_req
            )

        tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
            max_tuning_tokens
        )
        logger.info(
            "Running FlashInfer autotune with %d tokens and token buckets %s.",
            num_tokens,
            tuning_buckets,
        )
        # Round M up to match serving-time bucket selection for non-bucket M.
        # See https://github.com/flashinfer-ai/flashinfer/issues/5450
        # tune_mode is passed through: an inner autotune(True) would time
        # tactics even inside an outer autotune(False).
        with fi_utils.autotune(
            tune_mode=tune_mode, tuning_buckets=tuning_buckets, round_up=True
        ):
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
                randomize_inputs=True,
                **dummy_run_kwargs,
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


def flashinfer_autotune(runner: "GPUModelRunner", *, skip_attn: bool = False) -> None:
    """Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Args:
        runner: The model runner to tune.
        skip_attn: Additionally skip attention in the tuning dummy runs.

    With PP > 1, stages run different layers and may profile different ops,
    so each stage's TP group tunes separately with its own cache file;
    otherwise the world group tunes together. Per-tactic timings are
    averaged over the tuning group so all its ranks select the same tactic.

    Results are persisted per rank: FlashInfer keys MoE entries by tp/ep rank
    (``MoERunner.get_cache_key_extras``), so one rank's file only hits on that
    rank. A rank with a cache hit skips the per-tactic reduce the others block
    in, so ranks keep loaded configs only if every rank in the tuning group
    has a matching file and successfully loads it.

    With ``KernelConfig.flashinfer_autotune_cache_mode == "replay"`` the files
    are the record of the tactics, decided over the whole world group: when
    no rank has a file, every rank tunes and saves; when every rank has one,
    every rank loads it and this pass times nothing; anything else raises on
    every rank. Other warmup steps that tune FlashInfer kernels with their own
    cache (the SM120 sparse-MLA decode warmup, the PCIe all-reduce warmup) are
    not governed by the mode.

    """
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tp_group,
        get_world_group,
    )

    world = get_world_group()
    pp_size = get_pp_group().world_size
    tune_group = get_tp_group() if pp_size > 1 else world
    is_leader = tune_group.rank_in_group == 0
    tuner = AutoTuner.get()

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info_once(
            "Skipping FlashInfer autotuning for ops %s",
            tuple(sorted(skip_ops)),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    cache_path = resolve_flashinfer_autotune_file(runner)
    # The world group only spans DP ranks when vLLM folds DP into it.
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    cache_path = cache_path.with_name(
        f"{cache_path.stem}_dp{dp_rank}_rank{world.rank_in_group}{cache_path.suffix}"
    )
    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)
    replay = runner.vllm_config.kernel_config.flashinfer_autotune_cache_mode == "replay"

    tune_mode = True
    if replay:
        states = _gather_from_group(_replay_cache_state(cache_path, tune_group), world)
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
    elif _all_ranks_have_matching_cache(cache_path, tune_group):
        loaded = all(
            _gather_from_group(tuner.load_configs(str(cache_path)), tune_group)
        )
        if not loaded:
            tuner.clear_cache()

    # We skip EPLB here since we don't want to record dummy metrics.
    # Randomize inputs to avoid every token pick the same experts,
    # which lead to some EP ranks receiving no tokens and skipping their
    # MoE kernel entirely, and cause hang due to all-reduce collective
    # during synchronized autotuning.
    group = tune_group.cpu_group if tune_group.world_size > 1 else None
    set_autotune_process_group(group)
    try:
        with (
            torch.inference_mode(),
            fi_utils.autotune(tune_mode=tune_mode, **autotune_kwargs),
        ):
            hisparse_enabled = (
                runner.vllm_config.attention_config.hisparse_config is not None
            )
            skip_attn = skip_attn or hisparse_enabled
            if hisparse_enabled:
                # HiSparse hot-buffer attention is bounded by decode batch
                # size, not the prefill-sized batch used for the full model.
                autotune_hisparse_flashinfer_attention(runner)
            _run_flashinfer_autotune_dummy_runs(
                runner, skip_attn=skip_attn, tune_mode=tune_mode
            )
            replayssm_autotune_warmup(runner)
            _autotune_kimi_k3_kda_qkvg(runner.get_model())
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    # Skip the rewrite when nothing was tuned this start (every entry came from
    # the file). FlashInfer gates its own autotune(cache=...) save the same way.
    # A replay-mode start that records always saves, even with nothing tuned,
    # so that the next start replays; one that loaded the record never does,
    # even if an earlier warmup step left the tuner dirty.
    if tune_mode and (tuner._dirty or replay):
        before = _file_signature(cache_path)
        tuner.save_configs(str(cache_path))
        # FlashInfer skips the write when the file belongs to another
        # environment, so only a changed file counts as saved.
        if is_leader and _file_signature(cache_path) not in (None, before):
            logger.info("Saved FlashInfer autotune cache file %s.", cache_path)
    if replay and tune_mode:
        states = _gather_from_group(_replay_cache_state(cache_path, tune_group), world)
        _check_replay_cache_states(states, cache_path.parent, recorded=True)
