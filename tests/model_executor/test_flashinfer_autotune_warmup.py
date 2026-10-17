# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import re
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from vllm.model_executor.warmup import kernel_warmup as warmup
from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
    flashinfer_autotune,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    """Minimal MoERunner stand-in for token-count discovery tests."""

    moe_config: Any


def _make_moe(*, max_deferred_tokens: int = 128, enabled: bool = True):
    """Create a fake MoE layer with a deferred-finalize token limit."""
    moe = _FakeMoERunner()
    moe.moe_config = SimpleNamespace(
        use_deferred_moe_finalize=enabled,
        defer_moe_finalize_max_num_tokens=max_deferred_tokens,
    )
    return moe


def _make_runner(modules, *, max_tokens: int = 8192, cache_mode: str = "tune"):
    """Create a runner carrying only the state used by FlashInfer warmup."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None),
            parallel_config=SimpleNamespace(data_parallel_rank=0),
            kernel_config=SimpleNamespace(flashinfer_autotune_cache_mode=cache_mode),
        ),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
        _dummy_run=Mock(),
    )


def test_flashinfer_autotune_token_counts_include_deferred_moe_limits():
    runner = _make_runner(
        [
            object(),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=64, enabled=False),
            _make_moe(max_deferred_tokens=-1),
        ]
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (8192, 128)


def test_flashinfer_autotune_token_counts_are_bounded_and_deduplicated():
    runner = _make_runner(
        [_make_moe(max_deferred_tokens=4096)],
        max_tokens=32,
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (32,)


@pytest.mark.parametrize("tune_mode", [True, False])
@pytest.mark.parametrize("skip_attn", [False, True])
def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run(
    skip_attn, tune_mode
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1Runner

    runner = _make_runner([])
    dummy_run_signature = signature((V2Runner if skip_attn else V1Runner)._dummy_run)
    runner._dummy_run.side_effect = lambda **kwargs: dummy_run_signature.bind(
        None, **kwargs
    )
    max_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192)
    deferred_buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            side_effect=(max_buckets, deferred_buckets),
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(
            runner, skip_attn=skip_attn, tune_mode=tune_mode
        )

    assert get_buckets.call_args_list == [call(8192), call(128)]
    # The inner contexts carry the tune mode: autotune(True) nested in an outer
    # autotune(False) would time tactics again.
    assert autotune.call_args_list == [
        call(tune_mode=tune_mode, tuning_buckets=max_buckets, round_up=True),
        call(tune_mode=tune_mode, tuning_buckets=deferred_buckets, round_up=True),
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=8192,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
    ]


def test_flashinfer_autotune_buckets_cover_drafter_tokens():
    """The drafter's M can exceed the pass size; buckets must include it."""
    runner = _make_runner([])
    runner.max_num_reqs = 256
    runner.max_num_tokens = 8192
    runner.speculator = SimpleNamespace(num_query_per_req=6)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets"
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune"),
    ):
        _run_flashinfer_autotune_dummy_runs(runner)

    assert get_buckets.call_args_list == [call(8192), call(128 * 6)]
    assert [c.kwargs["num_tokens"] for c in runner._dummy_run.call_args_list] == [
        8192,
        128,
    ]


class _AutotuneGroup:
    def __init__(self, run, ranks):
        self.run = run
        self.ranks = tuple(ranks)
        self.world_size = len(self.ranks)
        self.rank_in_group = self.ranks.index(run.rank)
        self.cpu_group = self

    def record(self, operation):
        self.run.collectives[self.ranks][self.run.rank].append(operation)

    def all_gather_object(self, out, obj):
        # Ranks run sequentially: snapshot every rank's file before any saves.
        self.record(("all_gather",))
        if isinstance(obj, bool):
            out[:] = [rank not in self.run.rejected_load_ranks for rank in self.ranks]
            assert out[self.rank_in_group] == obj
            return
        out[:] = self.run.gathered.setdefault(
            self.ranks,
            [
                warmup._autotune_cache_fingerprint(self.run.cache_path(rank))
                for rank in self.ranks
            ],
        )

    def barrier(self):
        self.record(("barrier",))


class _AutotuneTuner:
    def __init__(self, run):
        self.run = run
        self.cache = {}
        self.loaded = None
        self._dirty = False

    def load_configs(self, path):
        if self.run.rank in self.run.rejected_load_ranks:
            return False
        self.loaded = json.loads(Path(path).read_text())
        self.cache.update(self.loaded)
        return True

    def clear_cache(self):
        self.cache.clear()
        self._dirty = False

    def save_configs(self, path):
        self.run.saves.append((self.run.rank, Path(path), dict(self.cache)))
        Path(path).write_text(json.dumps(self.cache))

    def profile(self, operation):
        # Outside tuning mode FlashInfer takes the fallback tactic of a miss.
        if operation in self.cache or not self.run.tune_mode:
            return
        group = self.run.tuning_group
        self.run.profile_groups[self.run.rank].append(
            None if group is None else group.ranks
        )
        for tactic in range(2):
            if group is not None:
                group.record(("all_reduce", operation, tactic))
        self.cache[operation] = self.run.rank // self.run.tp
        self._dirty = True


class _AutotuneRun:
    def __init__(self, pp, tp, cache_dir, cache_mode="tune"):
        self.pp, self.tp = pp, tp
        self.cache_dir = cache_dir
        self.cache_mode = cache_mode
        self.rank = 0
        self.tuning_group = None
        self.tune_mode = True
        self.autotune_modes: dict[int, list[bool]] = defaultdict(list)
        self.collectives: dict[tuple[int, ...], dict[int, list[tuple[Any, ...]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.gathered = {}
        self.rejected_load_ranks: set[int] = set()
        self.tuners = {}
        self.saves = []
        self.profile_groups = defaultdict(list)

    def cache_path(self, rank):
        return self.cache_dir / f"autotune_configs_dp0_rank{rank}.json"

    def world(self):
        return _AutotuneGroup(self, range(self.pp * self.tp))

    def tensor_group(self):
        start = self.rank // self.tp * self.tp
        return _AutotuneGroup(self, range(start, start + self.tp))

    def pipeline_group(self):
        return SimpleNamespace(world_size=self.pp)

    def set_group(self, group):
        self.tuning_group = group

    @contextmanager
    def autotune(self, *, tune_mode=True, **kwargs):
        self.autotune_modes[self.rank].append(tune_mode)
        self.tune_mode = tune_mode
        try:
            yield
        finally:
            self.tune_mode = True

    def dummy_runs(self, runner, *, skip_attn=False, tune_mode=True):
        self.tuners[self.rank].profile("shared_gemm")
        if self.pp > 1 and self.rank // self.tp == 0:
            self.tuners[self.rank].profile("pp0_extra_gemm")

    def execute(self):
        for rank in range(self.pp * self.tp):
            self.rank = rank
            self.tuners[rank] = _AutotuneTuner(self)
            flashinfer_autotune(_make_runner([], cache_mode=self.cache_mode))
        return self

    def assert_collectives_match(self):
        for ranks, traces in self.collectives.items():
            assert set(traces) == set(ranks)
            expected = traces[ranks[0]]
            assert all(trace == expected for trace in traces.values()), dict(traces)


@pytest.fixture
def autotune_run(monkeypatch, tmp_path):
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed import parallel_state

    def make_run(*, pp=2, tp=4, cache_mode="tune"):
        run = _AutotuneRun(pp, tp, tmp_path, cache_mode)
        autotuner = ModuleType("flashinfer.autotuner")
        monkeypatch.setattr(
            autotuner,
            "AutoTuner",
            SimpleNamespace(get=lambda: run.tuners[run.rank]),
            raising=False,
        )
        monkeypatch.setattr(
            autotuner, "set_autotune_process_group", run.set_group, raising=False
        )
        monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
        monkeypatch.setattr(parallel_state, "get_world_group", run.world)
        monkeypatch.setattr(parallel_state, "get_tp_group", run.tensor_group)
        monkeypatch.setattr(parallel_state, "get_pp_group", run.pipeline_group)
        monkeypatch.setattr(fi_utils, "autotune", run.autotune)
        monkeypatch.setattr(
            "torch.distributed.all_gather_object",
            lambda out, obj, group: group.all_gather_object(out, obj),
        )
        monkeypatch.setattr(
            warmup,
            "resolve_flashinfer_autotune_file",
            lambda runner: tmp_path / "autotune_configs.json",
        )
        monkeypatch.setattr(
            warmup, "_flashinfer_autotune_skip_ops", lambda runner: None
        )
        monkeypatch.setattr(
            warmup, "_run_flashinfer_autotune_dummy_runs", run.dummy_runs
        )
        monkeypatch.setattr(warmup, "replayssm_autotune_warmup", lambda runner: None)
        monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", lambda model: None)
        return run

    return make_run


@pytest.mark.parametrize("tp", [1, 4])
def test_heterogeneous_pp_stages_have_compatible_collectives(autotune_run, tp):
    run = autotune_run(tp=tp).execute()
    run.assert_collectives_match()
    assert dict(run.autotune_modes) == {rank: [True] for rank in range(2 * tp)}
    for rank, groups in run.profile_groups.items():
        expected = tuple(range(rank // tp * tp, (rank // tp + 1) * tp))
        assert groups and all(
            group == (expected if tp > 1 else None) for group in groups
        )


def test_pp_stage_cache_roundtrip_isolated_and_asymmetric_hits_safe(
    autotune_run, tmp_path
):
    legacy = tmp_path / "autotune_configs.json"
    legacy.write_text('{"legacy_world_cache": 99}')
    cold = autotune_run().execute()
    cold.assert_collectives_match()
    assert [rank for rank, _, _ in cold.saves] == list(range(8))
    assert [path for _, path, _ in cold.saves] == [cold.cache_path(r) for r in range(8)]
    assert json.loads(legacy.read_text()) == {"legacy_world_cache": 99}
    stage_caches = [{"shared_gemm": 0, "pp0_extra_gemm": 0}, {"shared_gemm": 1}]
    assert all(cache == stage_caches[r // 4] for r, _, cache in cold.saves)
    warm = autotune_run().execute()
    warm.assert_collectives_match()
    assert not warm.profile_groups and not warm.saves
    # Tune mode still enters autotune(tune_mode=True) with every entry loaded.
    assert dict(warm.autotune_modes) == {rank: [True] for rank in range(8)}
    for rank, tuner in warm.tuners.items():
        assert tuner.loaded == stage_caches[rank // 4]
    cold.cache_path(5).unlink()
    mixed = autotune_run().execute()
    mixed.assert_collectives_match()
    assert set(mixed.profile_groups) == {4, 5, 6, 7}
    assert all(mixed.tuners[rank].loaded is not None for rank in range(4))
    assert all(mixed.tuners[rank].loaded is None for rank in range(4, 8))


def test_pp1_tunes_world_group_and_saves_per_rank(autotune_run):
    run = autotune_run(pp=1, tp=4).execute()
    run.assert_collectives_match()
    assert dict(run.autotune_modes) == {rank: [True] for rank in range(4)}
    assert [(rank, path) for rank, path, _ in run.saves] == [
        (rank, run.cache_path(rank)) for rank in range(4)
    ]
    assert all(groups == [(0, 1, 2, 3)] for groups in run.profile_groups.values())


def test_mismatched_rank_caches_are_ignored_by_every_rank(autotune_run):
    """One rank loading while others tune deadlocks the per-tactic reduce."""
    cold = autotune_run(pp=1, tp=4).execute()
    stale = cold.cache_path(2)
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "old_gemm": 0}))
    rerun = autotune_run(pp=1, tp=4).execute()
    rerun.assert_collectives_match()
    assert all(tuner.loaded is None for tuner in rerun.tuners.values())
    assert set(rerun.profile_groups) == {0, 1, 2, 3}


@pytest.mark.parametrize("pp, tp", [(1, 1), (1, 4), (2, 4)])
def test_rejected_cache_retunes_every_rank_in_its_tuning_group(autotune_run, pp, tp):
    """Matching files can still be incompatible with one rank's runtime."""
    autotune_run(pp=pp, tp=tp).execute()
    rerun = autotune_run(pp=pp, tp=tp)
    rerun.rejected_load_ranks = {tp - 1}
    rerun.execute()
    rerun.assert_collectives_match()
    assert set(rerun.profile_groups) == set(range(tp))
    assert {rank for rank, _, _ in rerun.saves} == set(range(tp))


def test_autotune_cache_key_normalizes_kv_init_fields():
    """mamba_block_size/kv_cache_layout only resolve at engine KV-init time;
    they must not feed the autotune cache key, or nothing can compute the key
    before startup (the daemon tunes before any KV init). block_size must
    stay in the key: the daemon replays the engine's block-size resolution
    before tuning, and different block sizes may select different ops."""
    from vllm.config.cache import CacheConfig
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        _normalize_cache_config_for_hash,
    )

    fresh = CacheConfig()
    # Simulate the engine's KV-init write-backs (hybrid model values).
    mutated = CacheConfig()
    mutated.mamba_block_size = 512
    mutated.kv_cache_layout = "LBHNC"

    # Negative control: the write-backs are hashed factors.
    assert mutated.compute_hash() != fresh.compute_hash()
    normalized = _normalize_cache_config_for_hash(mutated)
    assert normalized.compute_hash() == fresh.compute_hash()
    # Factory state is returned untouched.
    assert _normalize_cache_config_for_hash(fresh) is fresh
    # block_size stays significant (the daemon keys on the resolved value).
    resolved = CacheConfig(block_size=64)
    assert (
        _normalize_cache_config_for_hash(resolved).compute_hash()
        != fresh.compute_hash()
    )


def test_cache_mode_defaults_to_tune_and_is_outside_the_cache_key():
    """Replay must find the file a tune-mode start wrote under the same key."""
    from pydantic import ValidationError

    from vllm.config.kernel import KernelConfig

    assert KernelConfig().flashinfer_autotune_cache_mode == "tune"
    replay = KernelConfig(flashinfer_autotune_cache_mode="replay")
    assert replay.compute_hash() == KernelConfig().compute_hash()
    # Negative control: a hashed kernel option does change the key.
    assert KernelConfig(moe_backend="triton").compute_hash() != (
        KernelConfig().compute_hash()
    )
    with pytest.raises(ValidationError):
        KernelConfig(flashinfer_autotune_cache_mode="record")


def test_flashinfer_saves_and_loads_an_empty_record(tmp_path):
    """The record of a model with nothing to tune is FlashInfer's own file."""
    autotuner = pytest.importorskip("flashinfer.autotuner")
    # Inner autotune() calls rely on tuning being the default.
    assert signature(autotuner.autotune).parameters["tune_mode"].default is True
    tuner = autotuner.AutoTuner()
    path = tmp_path / "autotune_configs_dp0_rank0.json"
    tuner.save_configs(str(path))
    assert "_metadata" in json.loads(path.read_text())
    assert tuner.load_configs(str(path)) is True
    assert warmup._autotune_cache_fingerprint(path) is not None
    # Negative control: a record from another GPU is rejected.
    record = json.loads(path.read_text())
    record["_metadata"]["gpu"] = "Another GPU"
    path.write_text(json.dumps(record))
    assert tuner.load_configs(str(path)) is False


class _ThreadedGroup:
    """A process group of rank threads: a collective waits for every member,
    so a rank that skips one breaks the barrier instead of passing."""

    def __init__(self, cluster, ranks):
        self.cluster = cluster
        self.ranks = tuple(ranks)
        self.world_size = len(self.ranks)
        self.cpu_group = self
        self._barrier = threading.Barrier(self.world_size, timeout=10)
        self._lock = threading.Lock()
        self._calls: dict[int, int] = defaultdict(int)
        self._slots: dict[int, list] = {}

    @property
    def rank_in_group(self):
        return self.ranks.index(self.cluster.rank)

    def all_gather_object(self, out, obj):
        rank = self.cluster.rank
        with self._lock:
            index = self._calls[rank]
            self._calls[rank] += 1
            slot = self._slots.setdefault(index, [None] * self.world_size)
            slot[self.ranks.index(rank)] = obj
        self.cluster.trace(self.ranks, ("all_gather", index))
        self._barrier.wait()
        out[:] = list(self._slots[index])

    def barrier(self):
        self.cluster.trace(self.ranks, ("barrier",))
        self._barrier.wait()


class _ThreadedTuner:
    """FlashInfer's AutoTuner for one rank: a record holds the environment it
    was written in, and loading it in another is refused, as FlashInfer does."""

    def __init__(self, cluster, rank):
        self.cluster = cluster
        self.rank = rank
        self.cache: dict[str, int] = {}
        self.loaded = None
        self._dirty = cluster.dirty_before

    def load_configs(self, path):
        if self.rank in self.cluster.vanishing_ranks:
            raise FileNotFoundError(path)
        configs = json.loads(Path(path).read_text())
        if configs.pop("_metadata", None) != self.cluster.environment:
            return False
        self.loaded = configs
        self.cache.update(configs)
        return True

    def save_configs(self, path):
        if self.rank in self.cluster.unwritten_ranks:
            return
        extra = {"odd_gemm": 0} if self.rank in self.cluster.odd_save_ranks else {}
        record = {"_metadata": self.cluster.environment, **self.cache, **extra}
        partial = Path(f"{path}.partial")
        partial.write_text(json.dumps(record))
        os.replace(partial, path)
        self.cluster.saves[self.rank] = dict(self.cache)
        self._dirty = False

    def profile(self, operation):
        # Outside tuning mode FlashInfer takes the fallback tactic of a miss.
        if operation in self.cache or not self.cluster.local.tune_mode:
            return
        self.cluster.profiled[self.rank].append(operation)
        self.cache[operation] = self.rank // self.cluster.tp
        self._dirty = True


class _ReplayCluster:
    """pp x tp ranks, each running flashinfer_autotune() in its own thread in
    replay mode, against the files in `cache_dir`."""

    def __init__(self, pp, tp, cache_dir):
        self.pp, self.tp = pp, tp
        self.cache_dir = cache_dir
        self.ranks = range(pp * tp)
        self.local = threading.local()
        self.world_group = _ThreadedGroup(self, self.ranks)
        self.tp_groups = [
            _ThreadedGroup(self, range(stage * tp, (stage + 1) * tp))
            for stage in range(pp)
        ]
        self.environment = {"flashinfer_version": "test", "gpu": "Test GPU"}
        self.operations = ["shared_gemm"]
        self.dirty_before = False
        self.vanishing_ranks: set[int] = set()
        self.unwritten_ranks: set[int] = set()
        self.odd_save_ranks: set[int] = set()
        self.tuners: dict[int, _ThreadedTuner] = {}
        self.autotune_modes: dict[int, list[bool]] = defaultdict(list)
        self.dummy_run_modes: dict[int, list[bool]] = defaultdict(list)
        self.profiled: dict[int, list[str]] = defaultdict(list)
        self.saves: dict[int, dict[str, int]] = {}
        self.errors: dict[int, str] = {}
        self.traces: dict[tuple[int, ...], dict[int, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._trace_lock = threading.Lock()

    @property
    def rank(self):
        return self.local.rank

    def cache_path(self, rank):
        return self.cache_dir / f"autotune_configs_dp0_rank{rank}.json"

    def tensor_group(self):
        return self.tp_groups[self.rank // self.tp]

    def trace(self, ranks, operation):
        with self._trace_lock:
            self.traces[ranks][self.rank].append(operation)

    @contextmanager
    def autotune(self, *, tune_mode=True, **kwargs):
        self.autotune_modes[self.rank].append(tune_mode)
        self.local.tune_mode = tune_mode
        try:
            yield
        finally:
            self.local.tune_mode = True

    def dummy_runs(self, runner, *, skip_attn=False, tune_mode=True):
        self.dummy_run_modes[self.rank].append(tune_mode)
        tuner = self.tuners[self.rank]
        for operation in self.operations:
            tuner.profile(operation)
        if self.pp > 1 and self.rank // self.tp == 0:
            tuner.profile("pp0_extra_gemm")

    def _run_rank(self, rank):
        self.local.rank = rank
        self.local.tune_mode = True
        try:
            flashinfer_autotune(_make_runner([], cache_mode="replay"))
        except Exception as error:
            self.errors[rank] = f"{type(error).__name__}: {error}"

    def execute(self):
        self.tuners = {rank: _ThreadedTuner(self, rank) for rank in self.ranks}
        threads = [
            threading.Thread(target=self._run_rank, args=(rank,)) for rank in self.ranks
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads)
        return self

    def assert_collectives_match(self):
        for ranks, traces in self.traces.items():
            assert set(traces) == set(ranks)
            expected = traces[ranks[0]]
            assert all(trace == expected for trace in traces.values()), dict(traces)

    def assert_every_rank_raised(self, match):
        """Every rank raised the same RuntimeError, after the same collectives."""
        assert set(self.errors) == set(self.ranks), self.errors
        messages = set(self.errors.values())
        assert len(messages) == 1, messages
        assert re.search(match, messages.pop())
        self.assert_collectives_match()
        assert not self.profiled


@pytest.fixture
def replay_cluster(monkeypatch, tmp_path):
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed import parallel_state

    clusters: list[_ReplayCluster] = []

    def current():
        return clusters[-1]

    autotuner = ModuleType("flashinfer.autotuner")
    monkeypatch.setattr(
        autotuner,
        "AutoTuner",
        SimpleNamespace(get=lambda: current().tuners[current().rank]),
        raising=False,
    )
    monkeypatch.setattr(
        autotuner, "set_autotune_process_group", lambda group: None, raising=False
    )
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: current().world_group
    )
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: current().tensor_group()
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=current().pp),
    )
    monkeypatch.setattr(
        fi_utils, "autotune", lambda **kwargs: current().autotune(**kwargs)
    )
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda out, obj, group: group.all_gather_object(out, obj),
    )
    monkeypatch.setattr(
        warmup,
        "resolve_flashinfer_autotune_file",
        lambda runner: tmp_path / "autotune_configs.json",
    )
    monkeypatch.setattr(warmup, "_flashinfer_autotune_skip_ops", lambda runner: None)
    monkeypatch.setattr(
        warmup,
        "_run_flashinfer_autotune_dummy_runs",
        lambda runner, **kwargs: current().dummy_runs(runner, **kwargs),
    )
    monkeypatch.setattr(warmup, "replayssm_autotune_warmup", lambda runner: None)
    monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", lambda model: None)

    def make(*, pp=1, tp=1):
        clusters.append(_ReplayCluster(pp, tp, tmp_path))
        return clusters[-1]

    return make


def _log_lines(caplog, prefix):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


@pytest.mark.parametrize("pp, tp", [(1, 1), (1, 4), (2, 4)])
def test_replay_records_once_then_loads_without_timing(
    replay_cluster, caplog_vllm, pp, tp
):
    ranks = range(pp * tp)
    leaders = [stage * tp for stage in range(pp)]
    record = replay_cluster(pp=pp, tp=tp).execute()
    assert record.errors == {}
    record.assert_collectives_match()
    assert set(record.saves) == set(ranks)
    assert dict(record.autotune_modes) == {rank: [True] for rank in ranks}
    assert dict(record.dummy_run_modes) == {rank: [True] for rank in ranks}
    paths = [record.cache_path(rank) for rank in leaders]
    assert sorted(_log_lines(caplog_vllm, "Saved FlashInfer")) == sorted(
        f"Saved FlashInfer autotune cache file {path}." for path in paths
    )
    assert sorted(_log_lines(caplog_vllm, "FlashInfer autotune cache mode")) == sorted(
        "FlashInfer autotune cache mode is 'replay' and no cache file exists; "
        f"autotuning to record the tactics in {path}."
        for path in paths
    )
    caplog_vllm.clear()

    replay = replay_cluster(pp=pp, tp=tp).execute()
    assert replay.errors == {}
    replay.assert_collectives_match()
    assert not replay.profiled
    assert not replay.saves
    assert dict(replay.autotune_modes) == {rank: [False] for rank in ranks}
    assert dict(replay.dummy_run_modes) == {rank: [False] for rank in ranks}
    assert {rank: tuner.loaded for rank, tuner in replay.tuners.items()} == (
        record.saves
    )
    assert sorted(_log_lines(caplog_vllm, "Loaded FlashInfer")) == sorted(
        f"Loaded FlashInfer autotune cache file {path} in replay mode; "
        "no FlashInfer tactics were timed."
        for path in paths
    )
    assert _log_lines(caplog_vllm, "Saved FlashInfer") == []


def test_replay_does_not_time_shapes_missing_from_the_record(replay_cluster):
    record = replay_cluster()
    record.operations = []
    record.execute()
    assert record.errors == {}
    assert record.saves == {0: {}}
    replay = replay_cluster()
    replay.operations = ["shared_gemm", "new_gemm"]
    replay.execute()
    assert replay.errors == {}
    assert not replay.profiled
    assert not replay.saves


@pytest.mark.parametrize("tp", [1, 4])
def test_replay_records_even_when_nothing_was_tuned(replay_cluster, tp):
    """An empty record still makes the next start replay, not record again."""
    record = replay_cluster(tp=tp)
    record.operations = []
    record.execute()
    assert record.errors == {}
    assert not record.profiled
    assert set(record.saves) == set(range(tp))
    replay = replay_cluster(tp=tp)
    replay.operations = []
    replay.execute()
    assert replay.errors == {}
    assert not replay.saves
    assert dict(replay.autotune_modes) == {rank: [False] for rank in range(tp)}


def test_replay_that_loaded_never_rewrites_the_record(replay_cluster):
    """An earlier warmup step that leaves the tuner dirty does not get its
    tactics written into a record replay loaded."""
    replay_cluster(tp=2).execute()
    replay = replay_cluster(tp=2)
    replay.dirty_before = True
    replay.execute()
    assert replay.errors == {}
    assert not replay.saves


def _break_one(cluster, rank, how):
    path = cluster.cache_path(rank)
    if how == "malformed":
        path.write_text("{not json")
    elif how == "missing":
        path.unlink()
    elif how == "unreadable":
        path.unlink()
        path.mkdir()
    elif how == "different":
        path.write_text(json.dumps({**json.loads(path.read_text()), "old_gemm": 0}))


@pytest.mark.parametrize(
    "how, match",
    [
        ("malformed", r"\(rank 3: malformed\)"),
        ("missing", r"\(rank 3: missing\)"),
        ("unreadable", r"\(rank 3: unreadable\)"),
        ("different", r"differs across the ranks of tuning groups \[\(0, 1, 2, 3\)\]"),
    ],
)
def test_replay_refuses_a_record_that_is_not_whole_on_every_rank(
    replay_cluster, tmp_path, how, match
):
    replay_cluster(tp=4).execute()
    replay = replay_cluster(tp=4)
    _break_one(replay, 3, how)
    replay.execute()
    replay.assert_every_rank_raised(match)
    assert f"Delete {tmp_path} to record the tactics again." in replay.errors[0]
    assert not replay.saves


def test_replay_decides_over_every_pipeline_stage(replay_cluster):
    """One stage whose files are gone does not record while the other replays."""
    replay_cluster(pp=2, tp=2).execute()
    replay = replay_cluster(pp=2, tp=2)
    for rank in (2, 3):
        replay.cache_path(rank).unlink()
    replay.execute()
    replay.assert_every_rank_raised(r"\(rank 2: missing, rank 3: missing\)")
    assert not replay.saves
    # The stages' records differ in what they hold, which is not a mismatch.
    for rank in (2, 3):
        replay.cache_path(rank).write_text(
            json.dumps({"_metadata": replay.environment, "shared_gemm": 1})
        )
    again = replay_cluster(pp=2, tp=2).execute()
    assert again.errors == {}


def test_replay_refuses_a_record_flashinfer_rejects_or_cannot_read(replay_cluster):
    replay_cluster(tp=4).execute()
    other_gpu = replay_cluster(tp=4)
    other_gpu.environment = {"flashinfer_version": "test", "gpu": "Another GPU"}
    other_gpu.execute()
    other_gpu.assert_every_rank_raised(
        r"rejected or could not read the record on ranks \[0, 1, 2, 3\]"
    )
    vanished = replay_cluster(tp=4)
    vanished.vanishing_ranks = {2}
    vanished.execute()
    vanished.assert_every_rank_raised(
        r"rejected or could not read the record on ranks \[2\]"
    )


def test_replay_checks_the_record_it_saved(replay_cluster):
    unwritten = replay_cluster(tp=2)
    unwritten.unwritten_ranks = {1}
    unwritten.execute()
    assert set(unwritten.errors) == {0, 1}
    assert len(set(unwritten.errors.values())) == 1
    assert (
        "the record this start saved is not usable on every rank (rank 1: missing)"
        in (unwritten.errors[0])
    )
    unwritten.assert_collectives_match()
    for path in unwritten.cache_dir.glob("autotune_configs_*.json"):
        path.unlink()
    odd = replay_cluster(tp=2)
    odd.odd_save_ranks = {1}
    odd.execute()
    assert set(odd.errors) == {0, 1}
    assert (
        "the record this start saved differs across the ranks of tuning groups"
        in (odd.errors[1])
    )
