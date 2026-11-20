# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.sample.sampler import Sampler

from vllm_tt_plugin.host_sampler import TTHostSampler

VOCAB = 5003


def _metadata(presence, frequency, repetition, prompts):
    longest = max(len(p) for p in prompts)
    prompt_ids = torch.full((len(prompts), longest), VOCAB, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        prompt_ids[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
    return SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=prompt_ids,
        presence_penalties=torch.tensor(presence, dtype=torch.float32),
        frequency_penalties=torch.tensor(frequency, dtype=torch.float32),
        repetition_penalties=torch.tensor(repetition, dtype=torch.float32),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_penalties_are_bit_identical_to_vllm(dtype):
    g = torch.Generator().manual_seed(4)
    rows = 6
    logits = (torch.randn(rows, VOCAB, generator=g) * 5).to(dtype)
    logits[2, 7] = -float("inf")

    def ids(n):
        return torch.randint(0, VOCAB, (n,), generator=g).tolist()

    outputs = [ids(40) + ids(5) * 3, [], ids(30), ids(12) + [-1, -1], ids(60), ids(9)]
    outputs[2] += [7, 7]  # a -inf logit that is also penalized
    prompts = [ids(100), ids(3), ids(50), ids(1), ids(70), ids(20)]
    metadata = _metadata(
        presence=[1.5, 1.5, 0.0, -0.5, 0.0, 0.0],
        frequency=[0.0, 0.3, 0.7, 0.0, 0.0, 0.0],
        repetition=[1.0, 1.0, 1.0, 1.3, 0.8, 1.0],  # row 5: no penalty at all
        prompts=prompts,
    )
    expected = Sampler.apply_penalties(logits.clone(), metadata, outputs)
    actual = TTHostSampler.apply_penalties(logits.clone(), metadata, outputs)
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual, expected)


def test_cpu_penalties_never_build_full_vocab_bin_counts(monkeypatch):
    import vllm.model_executor.layers.utils as layer_utils

    def full_vocab(*args, **kwargs):
        raise AssertionError("full-vocab bin counts on the CPU path")

    monkeypatch.setattr(layer_utils, "get_token_bin_counts_and_mask", full_vocab)
    metadata = _metadata([1.5], [0.2], [1.1], [[1, 2, 3]])
    logits = torch.randn(1, VOCAB)
    TTHostSampler.apply_penalties(logits, metadata, [[3, 4, 4]])


def test_no_penalties_returns_the_logits_untouched():
    logits = torch.randn(2, VOCAB)
    metadata = SimpleNamespace(no_penalties=True)
    assert TTHostSampler.apply_penalties(logits, metadata, [[1], [2]]) is logits
