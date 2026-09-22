"""Host tests for the compact top-k/top-p sampler (vllm_tt_plugin.host_sampler).

Every random draw here is seeded (torch.manual_seed for the global RNG, explicit
torch.Generator objects for per-request seeds): the tests are deterministic run to run.
"""

import os

import pytest
import torch

os.environ.setdefault("VLLM_TARGET_DEVICE", "tt")

from vllm.v1.sample.ops.topk_topp_sampler import (  # noqa: E402
    TopKTopPSampler,
    apply_top_k_top_p,
)

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


def _compact_probs(logits, k, p):
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


def _gens(B, base):
    return {i: torch.Generator().manual_seed(base + i) for i in range(B)}


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
    comp = _compact_probs(logits, k, p)
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
    # Sampled ids are survivors. Value-based check (a sampled token's logit is at
    # least the smallest surviving logit), so it cannot depend on tie order; the
    # draw itself is seeded.
    torch.manual_seed(0)
    ids, _ = sampler(
        logits.clone(), {}, k.clone(), p.clone() if p is not None else None
    )
    assert ids.shape == (B,)
    for r in range(B):
        min_surv = logits[r, torch.nonzero(up[r] > 0).flatten()].min()
        assert logits[r, ids[r]] >= min_surv, f"row {r}: sampled a non-survivor"
    assert sampler.compact_steps == 1 and sampler.fallback_steps == 0


def test_fallback_paths():
    B = 2
    logits = _logits(B, 7)
    torch.manual_seed(1)
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
        ids, _ = TTTopKTopPSampler()(
            logits.clone(), _gens(3, 9472), k.clone(), p.clone()
        )
        outs.append(ids.tolist())
    assert outs[0] == outs[1] == outs[2]


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("kp", [(20, 0.95), (50, 0.5), (20, None), (5, 0.99)])
def test_seeded_rows_reproduce_upstream_token_for_token(B, kp):
    """A seeded request draws its noise exactly as upstream (a full-vocab row from its
    generator), so over a sequence of steps the compact sampler emits the same token
    as upstream's TopKTopPSampler.forward_native from the same seed -- except when
    the top-p boundary falls inside a group of equal (bf16) logits, where upstream's
    own survivor choice is torch.sort order (checked separately below)."""
    k_val, p_val = kp
    k = torch.full((B,), k_val, dtype=torch.int64)
    p = None if p_val is None else torch.full((B,), p_val)
    upstream = TopKTopPSampler()
    compact = TTTopKTopPSampler()
    gens_up = _gens(B, 31337)
    gens_c = _gens(B, 31337)
    n_rows = n_equal = 0
    for step in range(6):
        logits = _logits(B, 5000 + 17 * step + B, scale=2.5)
        ref, _ = upstream.forward_native(
            logits.clone(), gens_up, k.clone(), p.clone() if p is not None else None
        )
        got, _ = compact(
            logits.clone(), gens_c, k.clone(), p.clone() if p is not None else None
        )
        up = _upstream_probs(logits, k, p)
        for r in range(B):
            n_rows += 1
            if int(ref[r]) == int(got[r]):
                n_equal += 1
                continue
            # The only admissible difference: the top-p boundary cuts a group of
            # equal logits, upstream kept an arbitrary (torch.sort order) subset of
            # that group and the compact path another. Then the minimum surviving
            # logit is shared by more tokens than survive at that value, and both
            # sampled tokens are survivors of their respective sets.
            surv = torch.nonzero(up[r] > 0).flatten()
            min_surv = logits[r, surv].min()
            tokens_at_min = int((logits[r] == min_surv).sum())
            surv_at_min = int((logits[r, surv] == min_surv).sum())
            assert tokens_at_min > surv_at_min, (
                f"step {step} row {r}: {int(ref[r])} vs {int(got[r])} without a "
                "boundary tie"
            )
            assert logits[r, ref[r]] >= min_surv and logits[r, got[r]] >= min_surv
    # ties at the boundary are the exception, not the rule
    assert n_equal >= 0.8 * n_rows, f"only {n_equal}/{n_rows} rows equal"
    assert compact.compact_steps == 6 and compact.fallback_steps == 0


def test_mixed_seeded_rows_match_upstream():
    """Only some rows carry a seed: those rows still match upstream; the others are
    sampled from the compact noise (global RNG, seeded here for determinism)."""
    B = 6
    k = torch.full((B,), 20, dtype=torch.int64)
    p = torch.full((B,), 0.95)
    seeded_rows = [1, 4]
    upstream = TopKTopPSampler()
    compact = TTTopKTopPSampler()
    gens_up = {i: torch.Generator().manual_seed(777 + i) for i in seeded_rows}
    gens_c = {i: torch.Generator().manual_seed(777 + i) for i in seeded_rows}
    for step in range(4):
        logits = _logits(B, 9000 + step)
        torch.manual_seed(step)
        ref, _ = upstream.forward_native(logits.clone(), gens_up, k.clone(), p.clone())
        torch.manual_seed(step)
        got, _ = compact(logits.clone(), gens_c, k.clone(), p.clone())
        up = _upstream_probs(logits, k, p)
        for r in seeded_rows:
            if int(ref[r]) != int(got[r]):
                surv = torch.nonzero(up[r] > 0).flatten()
                min_surv = logits[r, surv].min()
                assert int((logits[r] == min_surv).sum()) > int(
                    (logits[r, surv] == min_surv).sum()
                ), f"step {step} row {r}: differs without a boundary tie"
                assert logits[r, got[r]] >= min_surv
        for r in range(B):
            min_surv = logits[r, torch.nonzero(up[r] > 0).flatten()].min()
            assert logits[r, got[r]] >= min_surv


def test_sampling_frequencies_match():
    """Empirical frequencies over many seeded draws match the upstream probabilities."""
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
    # value-based support check (tie order independent): every sampled token has a
    # logit at least the smallest surviving one
    min_surv = logits[0, support].min()
    assert bool((logits[0, counts > 0] >= min_surv).all())
    # With k=5 the survivors are the same tokens on both sides (no boundary tie in this
    # seeded row), so the frequencies can be compared token by token.
    assert torch.allclose(freq[support], target[support], atol=0.02)
    assert abs(float(freq.sum()) - 1.0) < 1e-6


def test_make_host_sampler_env(monkeypatch):
    monkeypatch.setenv("TT_HOST_SAMPLER_FAST", "0")
    assert not isinstance(make_host_sampler().topk_topp_sampler, TTTopKTopPSampler)
    monkeypatch.setenv("TT_HOST_SAMPLER_FAST", "1")
    assert isinstance(make_host_sampler().topk_topp_sampler, TTTopKTopPSampler)
