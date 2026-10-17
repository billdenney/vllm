# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
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
        if isinstance(obj, warmup._ReplayCacheState):
            out[:] = self.run.gathered.setdefault(
                ("replay", self.ranks),
                [
                    warmup._replay_cache_state(self.run.cache_path(rank))
                    for rank in self.ranks
                ],
            )
            return
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
        self.operations = ["shared_gemm"]
        self.autotune_modes: dict[int, list[bool]] = defaultdict(list)
        self.dummy_run_modes: dict[int, list[bool]] = defaultdict(list)
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
        self.dummy_run_modes[self.rank].append(tune_mode)
        for operation in self.operations:
            self.tuners[self.rank].profile(operation)
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
    from vllm.config.kernel import KernelConfig

    assert KernelConfig().flashinfer_autotune_cache_mode == "tune"
    replay = KernelConfig(flashinfer_autotune_cache_mode="replay")
    assert replay.compute_hash() == KernelConfig().compute_hash()
    # Negative control: a hashed kernel option does change the key.
    assert KernelConfig(moe_backend="triton").compute_hash() != (
        KernelConfig().compute_hash()
    )


@pytest.mark.parametrize("pp, tp", [(1, 1), (1, 4), (2, 4)])
def test_replay_records_once_then_loads_without_timing(autotune_run, pp, tp):
    ranks = range(pp * tp)
    record = autotune_run(pp=pp, tp=tp, cache_mode="replay").execute()
    record.assert_collectives_match()
    saved = {rank: cache for rank, _, cache in record.saves}
    assert set(saved) == set(ranks)
    assert dict(record.autotune_modes) == {rank: [True] for rank in ranks}
    assert dict(record.dummy_run_modes) == {rank: [True] for rank in ranks}

    replay = autotune_run(pp=pp, tp=tp, cache_mode="replay").execute()
    replay.assert_collectives_match()
    assert not replay.profile_groups
    assert not replay.saves
    assert dict(replay.autotune_modes) == {rank: [False] for rank in ranks}
    assert dict(replay.dummy_run_modes) == {rank: [False] for rank in ranks}
    assert {rank: tuner.loaded for rank, tuner in replay.tuners.items()} == saved


def test_replay_does_not_time_shapes_missing_from_the_file(autotune_run):
    replay = autotune_run(pp=1, tp=1, cache_mode="replay")
    replay.cache_path(0).write_text("{}")
    replay.execute()
    assert replay.tuners[0].loaded == {}
    assert not replay.profile_groups
    assert not replay.saves
    # Negative control: tune mode times the same miss and saves it.
    tune = autotune_run(pp=1, tp=1).execute()
    assert dict(tune.profile_groups) == {0: [None]}
    assert [rank for rank, _, _ in tune.saves] == [0]


@pytest.mark.parametrize("tp", [1, 4])
def test_replay_refuses_a_malformed_file(autotune_run, tp):
    autotune_run(pp=1, tp=tp, cache_mode="replay").execute()
    replay = autotune_run(pp=1, tp=tp, cache_mode="replay")
    replay.cache_path(tp - 1).write_text("{not json")
    with pytest.raises(RuntimeError, match=rf"rank {tp - 1}: malformed"):
        replay.execute()
    assert not replay.profile_groups
    assert not replay.saves


def test_replay_refuses_files_missing_on_some_ranks(autotune_run):
    autotune_run(pp=1, tp=4, cache_mode="replay").execute()
    replay = autotune_run(pp=1, tp=4, cache_mode="replay")
    replay.cache_path(2).unlink()
    with pytest.raises(RuntimeError, match=r"\(rank 2: missing;"):
        replay.execute()
    assert not replay.profile_groups
    assert not replay.saves


def test_replay_refuses_files_that_differ_across_ranks(autotune_run):
    autotune_run(pp=1, tp=4, cache_mode="replay").execute()
    replay = autotune_run(pp=1, tp=4, cache_mode="replay")
    stale = replay.cache_path(1)
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "old_gemm": 0}))
    with pytest.raises(RuntimeError, match="differ across the ranks"):
        replay.execute()
    assert not replay.profile_groups
    assert not replay.saves


@pytest.mark.parametrize("tp", [1, 4])
def test_replay_refuses_a_file_flashinfer_rejects(autotune_run, tp):
    autotune_run(pp=1, tp=tp, cache_mode="replay").execute()
    replay = autotune_run(pp=1, tp=tp, cache_mode="replay")
    replay.rejected_load_ranks = {tp - 1}
    with pytest.raises(
        RuntimeError, match=rf"rejected the cache file on ranks \[{tp - 1}\]"
    ):
        replay.execute()
    assert not replay.profile_groups
    assert not replay.saves


@pytest.mark.parametrize("tp", [1, 4])
def test_replay_records_even_when_nothing_was_tuned(autotune_run, tp):
    """An empty record still makes the next start replay, not record again."""
    record = autotune_run(pp=1, tp=tp, cache_mode="replay")
    record.operations = []
    record.execute()
    assert not record.profile_groups
    assert [rank for rank, _, _ in record.saves] == list(range(tp))
    replay = autotune_run(pp=1, tp=tp, cache_mode="replay")
    replay.operations = []
    replay.execute()
    assert not replay.saves
    assert dict(replay.autotune_modes) == {rank: [False] for rank in range(tp)}
    # Negative control: tune mode saves nothing when nothing was tuned.
    tune = autotune_run(pp=1, tp=tp)
    tune.operations = []
    for rank in range(tp):
        tune.cache_path(rank).unlink()
    tune.execute()
    assert not tune.saves
