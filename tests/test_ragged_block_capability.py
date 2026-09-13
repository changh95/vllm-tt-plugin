# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Platform parse of the ``tt_adaptive_block_ragged`` capability."""

import pytest

from tests.test_block_request_validation import (
    BlockModel,
    _config,
    _patch_model_resolution,
)
from vllm_tt_plugin.config import (
    is_tt_adaptive_block_batched,
    is_tt_adaptive_block_ragged,
)
from vllm_tt_plugin.platform import TTPlatform


class _BatchedBlockModel(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_adaptive_block_output": True,
        "tt_adaptive_block_batched": True,
    }


class _RaggedBlockModel(_BatchedBlockModel):
    model_capabilities = {
        **_BatchedBlockModel.model_capabilities,
        "tt_adaptive_block_ragged": True,
    }


class _RaggedWithoutBatched(BlockModel):
    model_capabilities = {
        **BlockModel.model_capabilities,
        "tt_adaptive_block_output": True,
        "tt_adaptive_block_ragged": True,
    }


def test_startup_stores_the_ragged_capability(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _RaggedBlockModel)

    TTPlatform.check_and_update_config(config)

    assert is_tt_adaptive_block_batched(config) is True
    assert is_tt_adaptive_block_ragged(config) is True


def test_startup_defaults_ragged_off(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _BatchedBlockModel)

    TTPlatform.check_and_update_config(config)

    assert is_tt_adaptive_block_batched(config) is True
    assert is_tt_adaptive_block_ragged(config) is False


def test_startup_rejects_ragged_without_batched(monkeypatch):
    config = _config(max_num_seqs=4)
    _patch_model_resolution(monkeypatch, _RaggedWithoutBatched)

    with pytest.raises(
        ValueError, match="tt_adaptive_block_ragged requires tt_adaptive_block_batched"
    ):
        TTPlatform.check_and_update_config(config)
