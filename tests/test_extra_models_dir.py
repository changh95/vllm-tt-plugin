# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for ``EXTRA_MODELS_DIR`` bundle registration.

A bundle's arch must be registered under the plugin's ``TT`` prefix AND under
its plain HF name when upstream vLLM has no class of that name: since vLLM
0.25 ``ModelConfig`` resolves ``hf_config.architectures`` before the platform
prefixes them and keeps a validated copy (``model_arch_config``), so an arch
unknown upstream would otherwise resolve to the Transformers backend and the
worker's KV-spec hook would look up ``TTTransformers...ForCausalLM``.
"""

import json
import sys

import pytest

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.platform import _register_models_from_extra_dir


class _FakeRegistry:
    """Stand-in for vLLM's ``ModelRegistry``.

    Only ``register_model`` / ``get_supported_archs`` are used.
    """

    def __init__(self, known: tuple[str, ...] = ()):
        self.models: dict[str, str] = {name: f"upstream:{name}" for name in known}
        self.calls: list[tuple[str, str]] = []

    def get_supported_archs(self):
        return self.models.keys()

    def register_model(self, model_arch: str, model_cls: str) -> None:
        self.calls.append((model_arch, model_cls))
        self.models[model_arch] = model_cls


def _write_bundle(root, folder: str, arch: str, main_class: str) -> None:
    bundle = root / folder
    bundle.mkdir()
    (bundle / "vllm_metadata.json").write_text(
        json.dumps({"arch": arch, "main_class": main_class})
    )


MAIN_CLASS = "models.tt_transformers.tt.generator_vllm:SolarOpenForCausalLM"


def test_bundle_registers_tt_and_plain_arch(tmp_path, monkeypatch):
    """An arch unknown upstream is registered under both names, once each."""
    _write_bundle(tmp_path, "solar_open", "SolarOpenForCausalLM", MAIN_CLASS)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    registry = _FakeRegistry()

    count = _register_models_from_extra_dir(registry)

    assert count == 1
    assert registry.models == {
        "TTSolarOpenForCausalLM": MAIN_CLASS,
        "SolarOpenForCausalLM": MAIN_CLASS,
    }
    assert str(tmp_path / "solar_open") in sys.path


def test_plain_arch_known_upstream_is_left_alone(tmp_path, monkeypatch):
    """A bundle for a natively supported arch keeps upstream's class for the
    plain name."""
    _write_bundle(
        tmp_path,
        "llama",
        "LlamaForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    registry = _FakeRegistry(known=("LlamaForCausalLM",))

    assert _register_models_from_extra_dir(registry) == 1
    assert registry.models["LlamaForCausalLM"] == "upstream:LlamaForCausalLM"
    assert (
        registry.models["TTLlamaForCausalLM"]
        == "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    )
    assert [arch for arch, _ in registry.calls] == ["TTLlamaForCausalLM"]


def test_tt_prefixed_bundle_arch_registers_once(tmp_path, monkeypatch):
    """A bundle that already names the ``TT`` arch gets exactly that one entry."""
    _write_bundle(tmp_path, "dummy", "TTDummyModel", "pkg.mod:DummyModel")
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    registry = _FakeRegistry()

    assert _register_models_from_extra_dir(registry) == 1
    assert registry.models == {"TTDummyModel": "pkg.mod:DummyModel"}


def test_registration_is_idempotent(tmp_path, monkeypatch):
    """The three call sites (general plugin, worker import, config hook) never
    re-register."""
    _write_bundle(tmp_path, "solar_open", "SolarOpenForCausalLM", MAIN_CLASS)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    registry = _FakeRegistry()

    _register_models_from_extra_dir(registry)
    _register_models_from_extra_dir(registry)

    assert len(registry.calls) == 2
    assert sys.path.count(str(tmp_path / "solar_open")) == 1


@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_dir_registers_nothing(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("EXTRA_MODELS_DIR", raising=False)
    else:
        monkeypatch.setenv("EXTRA_MODELS_DIR", value)
    registry = _FakeRegistry()

    assert _register_models_from_extra_dir(registry) == 0
    assert registry.models == {}
