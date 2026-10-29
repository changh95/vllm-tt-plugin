# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Which models keep token-chunked prefill."""

from types import SimpleNamespace

import pytest

# vLLM's own bootstrap resolves the platform plugin, which imports this module.
# Letting the plugin module trigger that bootstrap deadlocks the cycle on a
# half-built module, so let vLLM finish importing itself first.
import vllm  # noqa: F401

from vllm_tt_plugin.config import (
    TTPrefillChunkPolicy,
    get_tt_prefill_chunk_policy,
    get_tt_prefill_chunk_unit,
    store_tt_lane_count,
)
from vllm_tt_plugin.platform import _apply_chunked_prefill_policy


class _FakeModel:
    """Stand-in for the resolved TT model class; the policy only reads its name."""


def _vllm_config(
    *,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 2048,
    max_model_len: int = 16384,
    long_prefill_token_threshold: int = 512,
    max_num_seqs: int = 4,
    max_num_scheduled_tokens: int | None = None,
    async_scheduling: bool = False,
    data_parallel_size: int = 1,
    kv_transfer_config=None,
    tt: dict | None = None,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
            disable_chunked_mm_input=False,
            max_num_seqs=max_num_seqs,
            async_scheduling=async_scheduling,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size),
        kv_transfer_config=kv_transfer_config,
        additional_config={"tt": dict(tt or {})},
    )


def _apply(config, capabilities):
    _apply_chunked_prefill_policy(config, capabilities, _FakeModel)


def test_declared_support_keeps_chunked_prefill():
    config = _vllm_config(max_num_batched_tokens=3000)

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.disable_chunked_mm_input is True


def test_declared_support_leaves_the_scheduler_budget_alone():
    # Resume offsets are floored by the tt-metal generator, so the plugin
    # passes max_num_batched_tokens through. That includes vLLM's unset
    # 2048/8192 defaults: chunked prefill is on, so those *are* the split size.
    config = _vllm_config(
        max_num_batched_tokens=3000, long_prefill_token_threshold=1000
    )

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.max_num_batched_tokens == 3000
    assert config.scheduler_config.long_prefill_token_threshold == 1000


def test_vllm_serve_default_budget_is_kept():
    config = _vllm_config(max_num_batched_tokens=2048, max_model_len=16384)

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_llm_class_default_budget_is_kept():
    config = _vllm_config(max_num_batched_tokens=8192, max_model_len=16384)

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.max_num_batched_tokens == 8192


def test_undeclared_model_loses_chunked_prefill_and_gets_a_full_prompt_budget():
    config = _vllm_config()

    _apply(config, {"supports_prefix_caching": True})

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 16384
    assert config.scheduler_config.long_prefill_token_threshold == 0


def test_model_without_any_capabilities_loses_chunked_prefill():
    config = _vllm_config()

    _apply(config, None)

    assert config.scheduler_config.enable_chunked_prefill is False


def test_unsplit_prefill_leaves_chunked_mm_input_enabled():
    # vLLM raises outright when this is set and one mm item is larger than
    # max_num_batched_tokens, e.g. a VL model pinned to a short max_model_len.
    # With prefill never split the flag is inert, so it must stay off.
    config = _vllm_config(max_num_batched_tokens=2048, max_model_len=2048)

    _apply(config, {"supports_prefix_caching": True})

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.disable_chunked_mm_input is False


def test_undeclared_model_zeroes_the_long_prefill_threshold():
    # The base scheduler applies this cap before it consults
    # enable_chunked_prefill, so leaving it set would still split a prefill.
    config = _vllm_config(enable_chunked_prefill=False)

    _apply(config, None)

    assert config.scheduler_config.long_prefill_token_threshold == 0
    # Chunked prefill was already off, so the token budget is left alone.
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_token_budget_is_left_alone_when_it_already_covers_the_model_len():
    config = _vllm_config(max_num_batched_tokens=32768)

    _apply(config, None)

    assert config.scheduler_config.max_num_batched_tokens == 32768


def test_declared_support_with_the_flag_off_falls_back_to_the_unsplit_policy():
    config = _vllm_config(enable_chunked_prefill=False)

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.disable_chunked_mm_input is False


def test_block_output_model_loses_chunked_prefill_even_when_declared():
    config = _vllm_config(max_num_batched_tokens=3000, max_model_len=16384)

    _apply(
        config,
        {"supports_chunked_prefill": True, "output_tokens_per_step": 256},
    )

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 16384
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.disable_chunked_mm_input is False


# ---- TT chunk core: models declaring tt_prefill_chunk_tokens ----------------

_UNIT = 512


def _chunk_capabilities(**extra):
    return {
        "supports_chunked_prefill": True,
        "tt_prefill_chunk_tokens": _UNIT,
        **extra,
    }


def test_declared_unit_resolves_the_chunk_core():
    config = _vllm_config(max_num_batched_tokens=2048, max_model_len=16384)

    _apply(config, _chunk_capabilities())

    scheduler_config = config.scheduler_config
    assert get_tt_prefill_chunk_policy(config) == TTPrefillChunkPolicy(_UNIT, False)
    assert get_tt_prefill_chunk_unit(config) == _UNIT
    assert scheduler_config.enable_chunked_prefill is True
    assert scheduler_config.disable_chunked_mm_input is True
    assert scheduler_config.long_prefill_token_threshold == _UNIT
    # max_model_len + max_num_seqs * chunk: the budget never splits a prompt.
    assert scheduler_config.max_num_batched_tokens == 16384 + 4 * _UNIT
    assert scheduler_config.max_num_scheduled_tokens is None


def test_chunk_core_raises_a_set_scheduled_token_cap_and_keeps_a_larger_budget():
    config = _vllm_config(max_num_batched_tokens=65536, max_num_scheduled_tokens=4096)

    _apply(config, _chunk_capabilities())

    assert config.scheduler_config.max_num_batched_tokens == 65536
    assert config.scheduler_config.max_num_scheduled_tokens == 16384 + 4 * _UNIT


def test_exclusive_declaration_reaches_the_policy():
    config = _vllm_config()

    _apply(config, _chunk_capabilities(tt_prefill_exclusive_chunks=True))

    assert get_tt_prefill_chunk_policy(config) == TTPrefillChunkPolicy(_UNIT, True)


@pytest.mark.parametrize("chunk", [_UNIT, 2 * _UNIT, 8 * _UNIT])
def test_chunk_knob_accepts_positive_multiples_of_the_unit(chunk):
    config = _vllm_config(tt={"prefill_chunk_tokens": chunk})

    _apply(config, _chunk_capabilities())

    assert get_tt_prefill_chunk_policy(config).chunk_tokens == chunk
    assert config.scheduler_config.long_prefill_token_threshold == chunk
    assert config.scheduler_config.max_num_batched_tokens == 16384 + 4 * chunk


@pytest.mark.parametrize(
    "chunk", [0, -_UNIT, _UNIT // 2, _UNIT + 1, True, 512.0, "512", None]
)
def test_chunk_knob_rejects_anything_else(chunk):
    config = _vllm_config(tt={"prefill_chunk_tokens": chunk})

    with pytest.raises(ValueError, match="positive multiple"):
        _apply(config, _chunk_capabilities())


def test_chunk_knob_without_a_declared_unit_is_refused():
    config = _vllm_config(tt={"prefill_chunk_tokens": _UNIT})

    with pytest.raises(ValueError, match="requires a model that declares"):
        _apply(config, {"supports_chunked_prefill": True})


@pytest.mark.parametrize("unit", [0, -1, True, 64.0, "64"])
def test_invalid_unit_is_refused(unit):
    with pytest.raises(ValueError, match="tt_prefill_chunk_tokens must be"):
        _apply(_vllm_config(), _chunk_capabilities(tt_prefill_chunk_tokens=unit))


def test_unit_without_supports_chunked_prefill_is_refused():
    with pytest.raises(ValueError, match="requires supports_chunked_prefill"):
        _apply(_vllm_config(), {"tt_prefill_chunk_tokens": _UNIT})


@pytest.mark.parametrize(
    ("capabilities", "match"),
    [
        (
            {"supports_chunked_prefill": True, "tt_prefill_exclusive_chunks": True},
            "requires tt_prefill_chunk_tokens",
        ),
        (_chunk_capabilities(tt_prefill_exclusive_chunks=1), "must be a boolean"),
    ],
)
def test_invalid_exclusive_declaration_is_refused(capabilities, match):
    with pytest.raises(ValueError, match=match):
        _apply(_vllm_config(), capabilities)


def test_one_seat_disables_chunked_prefill_like_an_undeclared_model():
    declared = _vllm_config(max_num_seqs=1)
    undeclared = _vllm_config(max_num_seqs=1)

    _apply(declared, _chunk_capabilities())
    _apply(undeclared, None)

    assert declared.scheduler_config == undeclared.scheduler_config
    assert declared.scheduler_config.enable_chunked_prefill is False
    assert get_tt_prefill_chunk_policy(declared) is None
    # The runner still passes prefill_final_mask to a declaring model.
    assert get_tt_prefill_chunk_unit(declared) == _UNIT


def test_block_output_model_gets_no_chunk_core():
    config = _vllm_config()

    _apply(config, _chunk_capabilities(output_tokens_per_step=8))

    assert config.scheduler_config.enable_chunked_prefill is False
    assert get_tt_prefill_chunk_policy(config) is None


@pytest.mark.parametrize(
    "launch",
    [
        pytest.param({"kv_transfer_config": object()}, id="kv-transfer"),
        pytest.param({"lanes": 2}, id="lane-dp-resolved"),
        pytest.param({"data_parallel_size": 2, "folds": True}, id="lane-dp-pending"),
    ],
)
def test_launches_the_core_cannot_serve_disable_chunked_prefill(monkeypatch, launch):
    monkeypatch.setattr(
        "vllm_tt_plugin.platform._model_folds_dp_into_lanes",
        lambda _model_class: launch.get("folds", False),
    )
    config = _vllm_config(
        data_parallel_size=launch.get("data_parallel_size", 1),
        kv_transfer_config=launch.get("kv_transfer_config"),
    )
    if "lanes" in launch:
        store_tt_lane_count(config, launch["lanes"])

    _apply(config, _chunk_capabilities())

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert get_tt_prefill_chunk_policy(config) is None


def test_standard_dp_keeps_the_core(monkeypatch):
    """Multi-process DP runs one independent scheduler per rank."""
    monkeypatch.setattr(
        "vllm_tt_plugin.platform._model_folds_dp_into_lanes", lambda _cls: False
    )
    config = _vllm_config(data_parallel_size=2)

    _apply(config, _chunk_capabilities())

    assert get_tt_prefill_chunk_policy(config) == TTPrefillChunkPolicy(_UNIT, False)


def test_async_scheduling_keeps_the_core():
    config = _vllm_config(async_scheduling=True)

    _apply(config, _chunk_capabilities())

    assert config.scheduler_config.async_scheduling is True
    assert get_tt_prefill_chunk_policy(config) == TTPrefillChunkPolicy(_UNIT, False)


def test_a_rerun_after_disabling_clears_a_stale_policy():
    """check_and_update_config re-runs in the worker; a policy resolved before
    must not survive a resolution that turns the core off."""
    config = _vllm_config()
    _apply(config, _chunk_capabilities())
    assert get_tt_prefill_chunk_policy(config) is not None

    config.scheduler_config.max_num_seqs = 1
    _apply(config, _chunk_capabilities())

    assert get_tt_prefill_chunk_policy(config) is None


def test_undeclared_model_records_no_chunk_state():
    config = _vllm_config()

    _apply(config, {"supports_chunked_prefill": True})

    assert config.additional_config == {"tt": {}}
