# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler, topk_first_filter


def _full_probs(logits, k, p):
    return apply_top_k_top_p(logits.clone(), k, p).softmax(dim=-1, dtype=torch.float32)


def _fast_probs(logits, k, p):
    filtered = topk_first_filter(logits, k, p)
    assert filtered is not None
    values, indices = filtered
    probs = torch.zeros(logits.shape, dtype=torch.float32)
    return probs.scatter_(1, indices, values.softmax(dim=-1, dtype=torch.float32))


@pytest.mark.parametrize(
    "k_rows, p_rows",
    [
        ([1, 1, 1], None),
        ([20, 20, 20], [0.95, 0.95, 0.95]),
        ([20, 5, 80], [0.95, 1.0, 0.5]),
        ([32, 1, 1024], [0.8, 0.99, 0.95]),
        ([7, 300, 64], None),
    ],
)
def test_filter_matches_full_vocab_masks_and_probs(k_rows, p_rows):
    torch.manual_seed(5)
    logits = torch.randn(len(k_rows), 4101) * 4
    k = torch.tensor(k_rows, dtype=torch.int32)
    p = torch.tensor(p_rows) if p_rows is not None else None
    full = _full_probs(logits, k, p)
    fast = _fast_probs(logits, k, p)
    # the same tokens survive, with the same probabilities
    assert torch.equal(full > 0, fast > 0)
    torch.testing.assert_close(fast, full, rtol=0, atol=1e-6)


@pytest.mark.parametrize("p_value", [None, 0.95])
def test_bf16_logits_keep_boundary_ties_on_the_fast_path(p_value):
    # device logits are bf16: values tied with the k-th largest are common
    torch.manual_seed(13)
    logits = (torch.randn(16, 248320) * 3).to(torch.bfloat16).float()
    k = torch.full((16,), 20, dtype=torch.int32)
    p = torch.full((16,), p_value) if p_value is not None else None
    kth = logits.topk(20, dim=-1).values[:, -1:]
    assert bool(((logits >= kth).sum(dim=1) > 20).any()), "no boundary ties to cover"
    full = _full_probs(logits, k, p)
    fast = _fast_probs(logits, k, p)
    # which of several equal logits a top-p cut drops is up to vLLM's unstable
    # sort; how many survive, and with what probabilities, is not
    assert torch.equal((full > 0).sum(dim=1), (fast > 0).sum(dim=1))
    for row in range(logits.shape[0]):
        kept_full = full[row][full[row] > 0].sort().values
        kept_fast = fast[row][fast[row] > 0].sort().values
        torch.testing.assert_close(kept_fast, kept_full, rtol=0, atol=1e-6)
    if p is None:  # top-k alone keeps every tie: the same tokens
        assert torch.equal(full > 0, fast > 0)


def test_ties_at_kth_value_beyond_candidates_take_the_full_path():
    logits = torch.full((2, 5000), -3.0)
    logits[:, :4] = torch.tensor([5.0, 4.0, 3.0, 2.0])
    logits[0, 100:200] = 2.0  # row 0: 100 more logits tied with its 4th largest
    k = torch.tensor([4, 4], dtype=torch.int32)
    assert topk_first_filter(logits, k, None) is None

    torch.manual_seed(11)
    expected, _ = TopKTopPSampler().forward_native(logits.clone(), {}, k, None)
    torch.manual_seed(11)
    actual, _ = TTTopKTopPSampler()(logits.clone(), {}, k, None)
    assert torch.equal(actual, expected)


def test_draws_follow_the_full_vocab_distribution():
    torch.manual_seed(3)
    row = torch.randn(1, 6000) * 3
    rows = 40000
    logits = row.expand(rows, -1).contiguous()
    k = torch.full((rows,), 20, dtype=torch.int32)
    p = torch.full((rows,), 0.9)
    target = _full_probs(row, k[:1], p[:1])[0]
    tokens, scores = TTTopKTopPSampler()(logits.clone(), {}, k, p)
    assert scores is None
    assert bool((target[tokens] > 0).all()), "a draw outside the top-k / top-p set"
    freq = torch.bincount(tokens, minlength=row.shape[1]).float() / rows
    assert (freq - target).abs().max().item() < 0.01


@pytest.mark.parametrize("k_value", [None, 6000])
def test_no_top_k_or_disabled_top_k_takes_the_full_path(k_value):
    torch.manual_seed(2)
    logits = torch.randn(3, 6000)
    k = None if k_value is None else torch.full((3,), k_value, dtype=torch.int32)
    p = torch.full((3,), 0.9)
    torch.manual_seed(8)
    expected, _ = TopKTopPSampler().forward_native(logits.clone(), {}, k, p)
    torch.manual_seed(8)
    actual, _ = TTTopKTopPSampler()(logits.clone(), {}, k, p)
    assert torch.equal(actual, expected)
