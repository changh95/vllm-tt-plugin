"""Host tests for the compact top-k/top-p sampler (vllm_tt_plugin.host_sampler)."""

import os

import pytest
import torch

os.environ.setdefault("VLLM_TARGET_DEVICE", "tt")

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p  # noqa: E402

from vllm_tt_plugin.host_sampler import (  # noqa: E402
    TTTopKTopPSampler,
    make_host_sampler,
)

V = 248320


def _logits(B, seed, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    # bf16-quantized like the device output: ties are common
    return (torch.randn(B, V, generator=g) * scale).to(torch.bfloat16).float()


def _upstream_probs(logits, k, p):
    masked = apply_top_k_top_p(
        logits.clone(), k.clone(), p.clone() if p is not None else None
    )
    return masked.softmax(dim=-1)


def _compact_probs(sampler, logits, k, p):
    """Re-run the compact masking; scatter survivor probabilities to vocab width."""
    k_long = k.to(torch.long)
    width = int(k_long.max()) + 32
    top_vals, top_idx = logits.topk(width, dim=-1)
    kth = top_vals.gather(1, (k_long - 1).unsqueeze(1))
    comp = top_vals.masked_fill(~(top_vals >= kth), -float("inf"))
    if p is not None:
        asc = comp.flip(1)
        ps = asc.softmax(dim=-1)
        cs = torch.cumsum(ps, dim=-1)
        m = cs <= 1 - p.unsqueeze(1)
        m[:, -1] = False
        comp = asc.masked_fill(m, -float("inf")).flip(1)
    probs = comp.softmax(dim=-1)
    out = torch.zeros_like(logits)
    out.scatter_(1, top_idx, probs)
    return out


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize(
    "kp", [(20, 0.95), (1, 1.0), (50, 0.5), (200, 0.99), (20, None)]
)
def test_compact_matches_upstream_distribution(B, kp):
    k_val, p_val = kp
    logits = _logits(B, 1234 + B)
    k = torch.full((B,), k_val, dtype=torch.int64)
    p = None if p_val is None else torch.full((B,), p_val)
    sampler = TTTopKTopPSampler()
    up = _upstream_probs(logits, k, p)
    comp = _compact_probs(sampler, logits, k, p)
    # Same survivor count and the same multiset of surviving logits / probabilities
    # per row. Token ids may differ only among tokens that TIE on the logit value at
    # a top-k / top-p boundary (upstream's own choice among ties is torch.sort order,
    # i.e. arbitrary); bf16-quantized logits make such ties common.
    for r in range(B):
        up_ids = torch.nonzero(up[r] > 0).flatten()
        comp_ids = torch.nonzero(comp[r] > 0).flatten()
        assert up_ids.numel() == comp_ids.numel(), f"row {r}: survivor counts differ"
        assert torch.equal(
            logits[r, up_ids].sort().values, logits[r, comp_ids].sort().values
        ), f"row {r}: logits"
        assert torch.allclose(
            up[r, up_ids].sort().values,
            comp[r, comp_ids].sort().values,
            atol=1e-6,
            rtol=1e-4,
        )
        only_up = set(up_ids.tolist()) - set(comp_ids.tolist())
        only_comp = set(comp_ids.tolist()) - set(up_ids.tolist())
        for t in only_up | only_comp:
            tied = [u for u in only_up | only_comp if logits[r, u] == logits[r, t]]
            assert len(tied) >= 2, f"row {r}: token {t} differs without a tie"
    # sampled ids are survivors
    ids, _ = sampler(
        logits.clone(), {}, k.clone(), p.clone() if p is not None else None
    )
    assert ids.shape == (B,)
    assert bool((up[torch.arange(B), ids] > 0).all())
    assert sampler.compact_steps == 1 and sampler.fallback_steps == 0


def test_fallback_paths():
    B = 2
    logits = _logits(B, 7)
    sampler = TTTopKTopPSampler(kmax=64)
    # k above kmax -> upstream path
    ids, _ = sampler(
        logits.clone(), {}, torch.tensor([20, 100]), torch.tensor([0.9, 0.9])
    )
    assert ids.shape == (B,) and sampler.fallback_steps == 1
    # no top-k (k == vocab) -> upstream path
    ids, _ = sampler(logits.clone(), {}, torch.tensor([V, V]), torch.tensor([0.9, 0.9]))
    assert sampler.fallback_steps == 2
    # massive ties at the boundary (constant row) -> upstream path
    flat = torch.zeros(B, V)
    ids, _ = sampler(flat, {}, torch.tensor([20, 20]), None)
    assert sampler.fallback_steps == 3
    # k is None -> upstream
    ids, _ = sampler(logits.clone(), {}, None, torch.tensor([0.9, 0.9]))
    assert sampler.fallback_steps == 4


def test_seeded_is_deterministic():
    logits = _logits(3, 99)
    k = torch.full((3,), 20, dtype=torch.int64)
    p = torch.full((3,), 0.95)
    outs = []
    for _ in range(3):
        gens = {i: torch.Generator().manual_seed(9472 + i) for i in range(3)}
        ids, _ = TTTopKTopPSampler()(logits.clone(), gens, k.clone(), p.clone())
        outs.append(ids.tolist())
    assert outs[0] == outs[1] == outs[2]


def test_sampling_frequencies_match():
    """Empirical frequencies over many draws match the upstream probabilities."""
    logits = _logits(1, 5, scale=1.5)
    k = torch.tensor([5])
    p = torch.tensor([0.95])
    sampler = TTTopKTopPSampler()
    target = _upstream_probs(logits, k, p)[0]
    n = 20000
    counts = torch.zeros(V)
    torch.manual_seed(0)
    for _ in range(n):
        ids, _ = sampler(logits.clone(), {}, k.clone(), p.clone())
        counts[ids[0]] += 1
    freq = counts / n
    support = target > 0
    assert bool((counts[~support] == 0).all())
    assert torch.allclose(freq[support], target[support], atol=0.02)


def test_make_host_sampler_env(monkeypatch):
    monkeypatch.setenv("TT_HOST_SAMPLER_FAST", "0")
    assert not isinstance(make_host_sampler().topk_topp_sampler, TTTopKTopPSampler)
    monkeypatch.setenv("TT_HOST_SAMPLER_FAST", "1")
    assert isinstance(make_host_sampler().topk_topp_sampler, TTTopKTopPSampler)
