# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for ``TTWorker._try_get_spec_from_model_hook``'s arch lookup.

The hook must resolve the ``TT``-prefixed class that carries ``get_kv_cache_spec``.
Since vLLM 0.25 ``model_config.architectures`` is a validated copy of
``hf_config.architectures`` (``model_arch_config``), so the platform's in-place
``TT`` prefix only shows in ``hf_config.architectures``; for an arch unknown
upstream ``model_config.architecture`` holds the Transformers-backend name.
"""

from types import SimpleNamespace
from unittest.mock import patch

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.worker import TTWorker


class _Hooked:
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        return None  # "fall back to the default spec" -- the lookup is what we test


class _Unhooked:
    pass


def _worker(architectures, hf_architectures, architecture):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=list(architectures),
            hf_config=SimpleNamespace(architectures=list(hf_architectures)),
            architecture=architecture,
        ),
        vllm_config=SimpleNamespace(),
    )


def _resolve_recorder(model_cls):
    requested: list[str] = []

    def resolve_model_cls(arch, model_config=None):
        requested.append(arch)
        return model_cls, arch

    return requested, resolve_model_cls


def _run_hook(worker, model_cls):
    requested, resolver = _resolve_recorder(model_cls)
    with patch(
        "vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls",
        side_effect=resolver,
    ):
        spec = TTWorker._try_get_spec_from_model_hook(worker)
    return requested, spec


def test_prefixed_entry_in_hf_config_wins_over_transformers_fallback_name():
    """vLLM 0.25 + an arch unknown upstream (the live Solar-Open case)."""
    worker = _worker(
        architectures=["SolarOpenForCausalLM"],  # model_arch_config copy: unprefixed
        hf_architectures=["TTSolarOpenForCausalLM"],  # what the platform prefixed
        architecture="TransformersMoEForCausalLM",  # upstream's fallback resolution
    )
    requested, spec = _run_hook(worker, _Hooked)
    assert requested == ["TTSolarOpenForCausalLM"]
    assert spec is None  # hook present, returned None -> default spec


def test_prefixed_entry_in_model_config_architectures_still_wins():
    """vLLM < 0.25 behaviour (in-place list) keeps working."""
    worker = _worker(
        architectures=["TTLlamaForCausalLM"],
        hf_architectures=["TTLlamaForCausalLM"],
        architecture="LlamaForCausalLM",
    )
    requested, _ = _run_hook(worker, _Hooked)
    assert requested == ["TTLlamaForCausalLM"]


def test_falls_back_to_prefixing_the_resolved_name():
    worker = _worker(
        architectures=["LlamaForCausalLM"],
        hf_architectures=["LlamaForCausalLM"],
        architecture="LlamaForCausalLM",
    )
    requested, _ = _run_hook(worker, _Hooked)
    assert requested == ["TTLlamaForCausalLM"]


def test_missing_hf_architectures_attribute_is_tolerated():
    worker = _worker(
        architectures=["LlamaForCausalLM"],
        hf_architectures=[],
        architecture="LlamaForCausalLM",
    )
    del worker.model_config.hf_config.architectures
    requested, _ = _run_hook(worker, _Hooked)
    assert requested == ["TTLlamaForCausalLM"]


def test_class_without_hook_returns_none():
    worker = _worker(
        architectures=["SolarOpenForCausalLM"],
        hf_architectures=["TTSolarOpenForCausalLM"],
        architecture="TransformersMoEForCausalLM",
    )
    requested, spec = _run_hook(worker, _Unhooked)
    assert requested == ["TTSolarOpenForCausalLM"]
    assert spec is None
