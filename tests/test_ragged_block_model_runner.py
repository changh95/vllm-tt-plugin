# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Runner side of ``tt_adaptive_block_ragged``.

Contract: a decode-only block step returns a rectangular int32 ``[num_reqs, W]``
tensor; row ``i`` holds ``1 <= n_i <= W`` real token ids followed by ``-1``
padding. The runner counts ``n_i`` as the non-negative ids of the row, strips
the padding and appends only those ``n_i`` tokens to the request state and the
published ``ModelRunnerOutput``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_tt_plugin.config import TT_RAGGED_BLOCK_PAD_TOKEN_ID
from vllm_tt_plugin.model_runner import (
    TTModelRunner,
    _committed_row_widths,
    _ragged_row_widths,
)

PAD = TT_RAGGED_BLOCK_PAD_TOKEN_ID
W = 8


def _row(ids, width: int = W) -> list[int]:
    ids = list(ids)
    assert len(ids) <= width
    return [*ids, *([PAD] * (width - len(ids)))]


def _bind(runner: SimpleNamespace, *, ragged: bool) -> SimpleNamespace:
    """Attach the width helpers the commit methods resolve through ``self``."""
    runner._is_adaptive_block_output = True
    runner._is_adaptive_block_ragged = ragged
    runner._tt_committed_width = lambda toks: TTModelRunner._tt_committed_width(
        runner, toks
    )
    return runner


def _runner(*, ragged: bool, num_tokens=(0, 0), max_model_len: int = 32):
    outputs: list[list[int]] = [[], []]
    runner = _bind(
        SimpleNamespace(
            _output_tokens_per_step=W,
            input_batch=SimpleNamespace(
                num_reqs=2,
                req_ids=["a", "b"],
                num_tokens=np.array(num_tokens, dtype=np.int32),
                token_ids_cpu=np.zeros((2, max_model_len), dtype=np.int32),
                req_output_token_ids=outputs,
            ),
            model_config=SimpleNamespace(max_model_len=max_model_len),
        ),
        ragged=ragged,
    )
    return runner, outputs


# ── _ragged_row_widths ────────────────────────────────────────────────────────


def test_ragged_row_widths_count_the_real_ids():
    rows = np.array([_row([1, 2, 3]), _row([4]), _row(range(10, 18))], dtype=np.int32)
    assert _ragged_row_widths(rows).tolist() == [3, 1, W]


def test_ragged_row_widths_empty_batch():
    assert _ragged_row_widths(np.zeros((0, W), dtype=np.int32)).tolist() == []


def test_ragged_row_widths_reject_a_row_with_no_real_id():
    rows = np.array([_row([1, 2]), _row([])], dtype=np.int32)
    with pytest.raises(ValueError, match="commits no token"):
        _ragged_row_widths(rows)


def test_ragged_row_widths_reject_padding_before_a_real_id():
    rows = np.array([_row([1, 2, 3]), [1, PAD, 3, PAD, PAD, PAD, PAD, PAD]])
    with pytest.raises(ValueError, match=r"before a real token id in row\(s\) \[1\]"):
        _ragged_row_widths(rows)


# ── Commit paths ──────────────────────────────────────────────────────────────


def test_ragged_rows_commit_only_their_real_ids():
    runner, outputs = _runner(ragged=True)
    block = torch.tensor([_row([11, 12, 13, 14, 15]), _row([21])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [5, 1]
    assert runner.input_batch.token_ids_cpu[0, :5].tolist() == [11, 12, 13, 14, 15]
    assert runner.input_batch.token_ids_cpu[1, :1].tolist() == [21]
    assert not (runner.input_batch.token_ids_cpu < 0).any()  # no pad leaked
    assert outputs == [[11, 12, 13, 14, 15], [21]]
    assert output.sampled_token_ids == [[11, 12, 13, 14, 15], [21]]
    assert output.req_ids == ["a", "b"]


def test_ragged_full_rows_commit_the_whole_width():
    runner, outputs = _runner(ragged=True)
    block = torch.arange(2 * W, dtype=torch.int32).reshape(2, W)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [W, W]
    assert outputs == [list(range(W)), list(range(W, 2 * W))]
    assert output.sampled_token_ids == [list(range(W)), list(range(W, 2 * W))]


def test_ragged_width_one_rows_are_plain_anchors():
    """Prefill anchors (width-1 rows) carry no padding: the adaptive width-1
    contract applies unchanged with the ragged flag on."""
    runner, outputs = _runner(ragged=True)
    anchors = torch.tensor([[5], [6]], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, anchors)
    output = TTModelRunner._build_runner_output(runner, anchors)

    assert runner.input_batch.num_tokens.tolist() == [1, 1]
    assert outputs == [[5], [6]]
    assert output.sampled_token_ids == [[5], [6]]


def test_ragged_off_keeps_fixed_width_rows_verbatim():
    """Without the flag nothing strips: a padded row is committed as W ids
    exactly as before (the batched contract fills every row to W)."""
    runner, outputs = _runner(ragged=False)
    block = torch.tensor([_row([11, 12, 13, 14, 15]), _row([21])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [W, W]
    assert outputs == [_row([11, 12, 13, 14, 15]), _row([21])]
    assert output.sampled_token_ids == [_row([11, 12, 13, 14, 15]), _row([21])]


def test_ragged_rows_keep_the_fixed_contract_shape_check():
    runner, _ = _runner(ragged=True)
    with pytest.raises(ValueError, match="violates output_tokens_per_step"):
        TTModelRunner._build_runner_output(
            runner, torch.zeros((2, W - 1), dtype=torch.int32)
        )


def test_ragged_row_is_clipped_at_max_model_len():
    """The max_model_len clip applies to the row's real ids only."""
    runner, outputs = _runner(ragged=True, num_tokens=(30, 0), max_model_len=32)
    block = torch.tensor(
        [_row([11, 12, 13, 14, 15]), _row([21, 22, 23])], dtype=torch.int32
    )

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [32, 3]
    assert runner.input_batch.token_ids_cpu[0, 30:32].tolist() == [11, 12]
    assert outputs == [[11, 12], [21, 22, 23]]


def _captured_runner(*, ragged: bool, num_tokens=(0, 0), in_batch=("a", "b")):
    states = {
        "a": SimpleNamespace(output_token_ids=[]),
        "b": SimpleNamespace(output_token_ids=[]),
    }
    runner = _bind(
        SimpleNamespace(
            _output_tokens_per_step=W,
            requests=states,
            input_batch=SimpleNamespace(
                req_id_to_index={
                    req_id: idx
                    for idx, req_id in enumerate(("a", "b"))
                    if req_id in in_batch
                },
                num_tokens=np.array(num_tokens, dtype=np.int32),
                token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
            ),
            model_config=SimpleNamespace(max_model_len=32),
        ),
        ragged=ragged,
    )
    return runner, states


def test_ragged_captured_rows_commit_only_their_real_ids():
    """The deferred-apply path (explicit req_ids) strips the same way."""
    runner, states = _captured_runner(ragged=True)
    block = torch.tensor([_row([11, 12, 13]), _row([21])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block, req_ids=["a", "b"])

    assert runner.input_batch.num_tokens.tolist() == [3, 1]
    assert runner.input_batch.token_ids_cpu[0, :3].tolist() == [11, 12, 13]
    assert runner.input_batch.token_ids_cpu[1, :1].tolist() == [21]
    assert states["a"].output_token_ids == [11, 12, 13]
    assert states["b"].output_token_ids == [21]


def test_ragged_captured_row_without_a_live_batch_row_is_still_stripped():
    """A request already gone from the persistent batch still receives only
    its real ids in the runner-side request state."""
    runner, states = _captured_runner(ragged=True, in_batch=("a",))
    block = torch.tensor([_row([11, 12, 13]), _row([21, 22])], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block, req_ids=["a", "b"])

    assert states["a"].output_token_ids == [11, 12, 13]
    assert states["b"].output_token_ids == [21, 22]
    assert runner.input_batch.num_tokens.tolist() == [3, 0]


def test_row_widths_default_to_the_fixed_width_without_the_runner_flag():
    """A runner (or stub) that never set the flag keeps the fixed contract:
    every row commits the step width, padding and all."""
    runner, outputs = _runner(ragged=False)
    del runner._is_adaptive_block_ragged
    TTModelRunner._apply_sampled_tokens_to_state(
        runner, torch.tensor([_row([1]), _row([2, 3])], dtype=torch.int32)
    )
    assert runner.input_batch.num_tokens.tolist() == [W, W]
    assert outputs == [_row([1]), _row([2, 3])]
    rows = np.array([_row([1]), _row([2, 3])], dtype=np.int32)
    assert _committed_row_widths(rows, W, ragged=False).tolist() == [W, W]
    assert _committed_row_widths(rows, W, ragged=True).tolist() == [1, 2]


def test_ragged_width_one_step_never_strips():
    """Width-1 steps (anchors) resolve to one token per row even with the flag:
    the ragged parse only applies to block-width rows."""
    anchors = np.array([[5], [6]], dtype=np.int32)
    assert _committed_row_widths(anchors, 1, ragged=True).tolist() == [1, 1]
