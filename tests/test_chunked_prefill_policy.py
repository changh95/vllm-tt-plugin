# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Which models keep token-chunked prefill."""

from types import SimpleNamespace

import pytest

# vLLM's own bootstrap resolves the platform plugin, which imports this module.
# Letting the plugin module trigger that bootstrap deadlocks the cycle on a
# half-built module, so let vLLM finish importing itself first.
import vllm  # noqa: F401

from vllm_tt_plugin.config import get_tt_prefill_chunk_policy
from vllm_tt_plugin.platform import (
    _apply_chunked_prefill_policy,
    _finalize_tt_prefill_chunk_policy,
)


class _FakeModel:
    """Stand-in for the resolved TT model class; the policy only reads its name."""


def _vllm_config(
    *,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 2048,
    max_model_len: int = 16384,
    long_prefill_token_threshold: int = 512,
    max_num_seqs: int = 8,
    tt: dict | None = None,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
            disable_chunked_mm_input=False,
            max_num_seqs=max_num_seqs,
            async_scheduling=False,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        cache_config=SimpleNamespace(block_size=64),
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


# ---- TT chunk policy: a model that declares its chunk unit ------------------

_CHUNK_CAPS = {"supports_chunked_prefill": True, "tt_prefill_chunk_tokens": 2048}


def test_declared_chunk_unit_resolves_the_tt_chunk_policy():
    config = _vllm_config(max_num_batched_tokens=65536, max_model_len=65536)

    _apply(config, _CHUNK_CAPS)

    sched = config.scheduler_config
    assert get_tt_prefill_chunk_policy(config) == (2048, 4)
    assert sched.enable_chunked_prefill is True
    assert sched.long_prefill_token_threshold == 2048
    # one long prompt + max_num_seqs short ones never exceed the budget
    assert sched.max_num_batched_tokens == 65536 + 8 * 2048


def test_chunk_policy_knobs():
    config = _vllm_config(
        tt={"prefill_chunk_tokens": 4096, "chunked_prefill_decode_steps": 0}
    )
    _apply(config, _CHUNK_CAPS)
    assert get_tt_prefill_chunk_policy(config) == (4096, 0)

    for tt in (
        {"prefill_chunk_tokens": 3072},
        {"prefill_chunk_tokens": 0},
        {"chunked_prefill_decode_steps": -1},
    ):
        with pytest.raises(ValueError):
            _apply(_vllm_config(tt=tt), _CHUNK_CAPS)


def test_chunk_unit_without_the_flag_or_on_block_output_resolves_no_policy():
    config = _vllm_config(enable_chunked_prefill=False)
    _apply(config, _CHUNK_CAPS)
    assert get_tt_prefill_chunk_policy(config) is None

    config = _vllm_config()
    _apply(config, {**_CHUNK_CAPS, "output_tokens_per_step": 16})
    assert get_tt_prefill_chunk_policy(config) is None
    assert config.scheduler_config.long_prefill_token_threshold == 0


def test_supports_chunked_prefill_without_a_unit_keeps_plain_chunking():
    config = _vllm_config(max_num_batched_tokens=3000)
    _apply(config, {"supports_chunked_prefill": True})
    assert get_tt_prefill_chunk_policy(config) is None
    assert config.scheduler_config.max_num_batched_tokens == 3000


@pytest.mark.parametrize("case", ["async", "lanes", "kv_transfer"])
def test_chunk_policy_is_refused_where_phase_1_does_not_support_it(case):
    config = _vllm_config()
    _apply(config, _CHUNK_CAPS)
    if case == "async":
        config.scheduler_config.async_scheduling = True
    if case == "kv_transfer":
        config.kv_transfer_config = object()

    _finalize_tt_prefill_chunk_policy(config, is_lane_mode=case == "lanes")

    assert get_tt_prefill_chunk_policy(config) is None
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0


def test_chunk_policy_survives_finalize_for_plain_sync_serving():
    config = _vllm_config()
    _apply(config, _CHUNK_CAPS)
    _finalize_tt_prefill_chunk_policy(config, is_lane_mode=False)
    assert get_tt_prefill_chunk_policy(config) == (2048, 4)
