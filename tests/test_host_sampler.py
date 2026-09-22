"""Host tests for the compact top-k/top-p sampler (vllm_tt_plugin.host_sampler).

Every random draw is seeded (torch.manual_seed for the global RNG, explicit
torch.Generator objects for per-request seeds), so the tests are deterministic.
The compact masking is checked through the sampler's own ``compact_candidates``
(never a re-implementation) against upstream ``apply_top_k_top_p``.
"""

import os

import pytest
import torch

os.environ.setdefault("VLLM_TARGET_DEVICE", "tt")

from vllm.v1.sample.logits_processor import LogitsProcessors  # noqa: E402
from vllm.v1.sample.metadata import SamplingMetadata  # noqa: E402
from vllm.v1.sample.ops.topk_topp_sampler import (  # noqa: E402
    TopKTopPSampler,
    apply_top_k_top_p,
)
from vllm.v1.sample.sampler import Sampler  # noqa: E402

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
    """The sampler's own compact probabilities scattered to vocab width."""
    cand = sampler.compact_candidates(logits, k, p)
    assert cand is not None
    top_idx, probs = cand
    out = torch.zeros_like(logits)
    out.scatter_(1, top_idx, probs)
    return out


def _gens(B, base, rows=None):
    rows = range(B) if rows is None else rows
    return {i: torch.Generator().manual_seed(base + i) for i in rows}


def _boundary_tie(logits_row, up_row):
    """True when the smallest surviving logit is shared by more tokens than survive at
    that value: upstream's survivor choice inside that group is torch.sort order."""
    surv = torch.nonzero(up_row > 0).flatten()
    min_surv = logits_row[surv].min()
    return int((logits_row == min_surv).sum()) > int(
        (logits_row[surv] == min_surv).sum()
    )


def _assert_same_distribution(logits, up, comp):
    """Same survivor count and multiset of surviving logits / probabilities per row;
    token ids may differ only among tokens tied on the logit value."""
    for r in range(logits.shape[0]):
        up_ids = torch.nonzero(up[r] > 0).flatten()
        comp_ids = torch.nonzero(comp[r] > 0).flatten()
        assert up_ids.numel() == comp_ids.numel(), f"row {r}: survivor counts differ"
        assert torch.equal(
            logits[r, up_ids].sort().values, logits[r, comp_ids].sort().values
        ), f"row {r}: surviving logits differ"
        assert torch.allclose(
            up[r, up_ids].sort().values,
            comp[r, comp_ids].sort().values,
            atol=1e-6,
            rtol=1e-4,
        ), f"row {r}: probabilities differ"
        only_up = set(up_ids.tolist()) - set(comp_ids.tolist())
        only_comp = set(comp_ids.tolist()) - set(up_ids.tolist())
        for t in only_up | only_comp:
            tied = [u for u in only_up | only_comp if logits[r, u] == logits[r, t]]
            assert len(tied) >= 2, f"row {r}: token {t} differs without a tie"


def _assert_survivor(logits, up, row, token):
    """Value-based survivor check (tie-order independent)."""
    surv = torch.nonzero(up[row] > 0).flatten()
    assert logits[row, token] >= logits[row, surv].min(), f"row {row}: non-survivor"


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
    _assert_same_distribution(logits, up, comp)
    torch.manual_seed(0)
    ids, _ = sampler(
        logits.clone(), {}, k.clone(), p.clone() if p is not None else None
    )
    assert ids.shape == (B,)
    for r in range(B):
        _assert_survivor(logits, up, r, ids[r])
    assert sampler.compact_steps == 1 and sampler.fallback_steps == 0


def test_mixed_k_batch():
    """Per-row differing k: the window is max(k) + 32 wide and each row's own k-th
    value is the mask threshold."""
    k = torch.tensor([1, 20, 50, 200, 7, 33], dtype=torch.int64)
    p = torch.tensor([1.0, 0.95, 0.5, 0.99, 0.8, 1.0])
    logits = _logits(6, 4242)
    sampler = TTTopKTopPSampler()
    up = _upstream_probs(logits, k, p)
    comp = _compact_probs(sampler, logits, k, p)
    _assert_same_distribution(logits, up, comp)
    torch.manual_seed(3)
    ids, _ = sampler(logits.clone(), {}, k.clone(), p.clone())
    for r in range(6):
        _assert_survivor(logits, up, r, ids[r])
    assert sampler.compact_steps == 1 and sampler.fallback_steps == 0


def test_masked_row_stays_compact():
    """A row with fewer than k finite logits (grammar bitmask / allowed_token_ids /
    bad_words leave -inf elsewhere) must not trip the boundary-tie fallback."""
    B = 4
    logits = _logits(B, 77)
    allowed = torch.tensor([5, 1000, 200000])
    masked = torch.full((V,), -float("inf"))
    masked[allowed] = logits[1, allowed]
    logits[1] = masked  # 3 finite logits, top_k 20
    logits[2, :] = -float("inf")
    logits[2, 12345] = 1.0  # a single allowed token
    k = torch.full((B,), 20, dtype=torch.int64)
    p = torch.full((B,), 0.95)
    sampler = TTTopKTopPSampler()
    up = _upstream_probs(logits, k, p)
    comp = _compact_probs(sampler, logits, k, p)
    _assert_same_distribution(logits, up, comp)
    torch.manual_seed(5)
    ids, _ = sampler(logits.clone(), {}, k.clone(), p.clone())
    assert sampler.compact_steps == 1 and sampler.fallback_steps == 0
    assert int(ids[1]) in allowed.tolist()
    assert int(ids[2]) == 12345
    for r in (0, 3):
        _assert_survivor(logits, up, r, ids[r])


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
    # more than k + 32 FINITE ties at the boundary (constant row) -> upstream path
    flat = torch.zeros(B, V)
    ids, _ = sampler(flat, {}, torch.tensor([20, 20]), None)
    assert sampler.fallback_steps == 3
    # k is None -> upstream
    ids, _ = sampler(logits.clone(), {}, None, torch.tensor([0.9, 0.9]))
    assert sampler.fallback_steps == 4
    # processed logprobs modes need the processed full-vocab logits -> upstream
    for mode in ("processed_logits", "processed_logprobs"):
        s2 = TTTopKTopPSampler(logprobs_mode=mode)
        ids, extra = s2(
            logits.clone(), {}, torch.tensor([20, 20]), torch.tensor([0.9, 0.9])
        )
        assert s2.fallback_steps == 1 and extra is not None


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


def _check_seeded_equality(logits, k, p, ref, got, rows):
    """Seeded rows equal upstream token for token, except at a top-p boundary tie
    (upstream's own choice there is torch.sort order); returns the exact-row count."""
    up = _upstream_probs(logits, k, p)
    n_equal = 0
    for r in rows:
        if int(ref[r]) == int(got[r]):
            n_equal += 1
            continue
        assert _boundary_tie(logits[r], up[r]), (
            f"row {r}: {int(ref[r])} vs {int(got[r])} differ without a boundary tie"
        )
        _assert_survivor(logits, up, r, got[r])
        _assert_survivor(logits, up, r, ref[r])
    return n_equal


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("kp", [(20, 0.95), (50, 0.5), (20, None), (5, 0.99)])
@pytest.mark.parametrize("fp64", [False, True])
def test_seeded_rows_reproduce_upstream_token_for_token(B, kp, fp64):
    """A seeded request draws its noise exactly as upstream (a full-vocab row from its
    generator), so over a sequence of steps the compact sampler emits the same token
    as upstream's TopKTopPSampler.forward_native from the same seed, in fp32 and in
    the fp64 Gumbel mode."""
    k_val, p_val = kp
    k = torch.full((B,), k_val, dtype=torch.int64)
    p = None if p_val is None else torch.full((B,), p_val)
    upstream = TopKTopPSampler(use_fp64_gumbel=fp64)
    compact = TTTopKTopPSampler(use_fp64_gumbel=fp64)
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
        n_rows += B
        n_equal += _check_seeded_equality(logits, k, p, ref, got, range(B))
    assert n_equal >= 0.8 * n_rows, f"only {n_equal}/{n_rows} rows equal"
    assert compact.compact_steps == 6 and compact.fallback_steps == 0


def test_mixed_seeded_rows_and_mixed_k_match_upstream():
    """Only some rows carry a seed and rows differ in k: seeded rows still match
    upstream; unseeded rows are compact-noise survivors (global RNG seeded here)."""
    B = 6
    k = torch.tensor([20, 200, 20, 1, 50, 20], dtype=torch.int64)
    p = torch.tensor([0.95, 0.99, 0.9, 1.0, 0.5, 0.95])
    seeded_rows = [1, 2, 4]
    upstream = TopKTopPSampler()
    compact = TTTopKTopPSampler()
    gens_up = _gens(B, 777, seeded_rows)
    gens_c = _gens(B, 777, seeded_rows)
    for step in range(4):
        logits = _logits(B, 9000 + step)
        torch.manual_seed(step)
        ref, _ = upstream.forward_native(logits.clone(), gens_up, k.clone(), p.clone())
        torch.manual_seed(step)
        got, _ = compact(logits.clone(), gens_c, k.clone(), p.clone())
        _check_seeded_equality(logits, k, p, ref, got, seeded_rows)
        up = _upstream_probs(logits, k, p)
        for r in range(B):
            _assert_survivor(logits, up, r, got[r])
    assert compact.compact_steps == 4


def _metadata(B, temperature, k, p, gens):
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=bool((temperature == 0.0).all()),
        all_random=bool((temperature != 0.0).all()),
        top_p=p,
        top_k=k,
        generators=gens,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(B),
        presence_penalties=torch.zeros(B),
        repetition_penalties=torch.ones(B),
        output_token_ids=[[] for _ in range(B)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_end_to_end_sampler_matches_upstream_mixed_batch():
    """The path model_runner calls: make_host_sampler() (compact) vs a plain Sampler()
    on the same logits and SamplingMetadata -- greedy and random rows mixed, per-row
    differing k / p / temperature, one masked row, all random rows seeded."""
    B = 8
    temperature = torch.tensor([0.0, 1.0, 0.7, 0.0, 1.2, 1.0, 1.0, 0.5])
    k = torch.tensor([1, 20, 50, 1, 200, 20, 20, 5], dtype=torch.int64)
    p = torch.tensor([1.0, 0.95, 0.5, 1.0, 0.99, 0.95, 0.9, 1.0])
    random_rows = [i for i in range(B) if temperature[i] > 0]
    ours = make_host_sampler()
    assert isinstance(ours.topk_topp_sampler, TTTopKTopPSampler)
    theirs = Sampler()
    gens_a = _gens(B, 2024, random_rows)
    gens_b = _gens(B, 2024, random_rows)
    for step in range(5):
        logits = _logits(B, 600 + step)
        logits[6, :] = -float("inf")
        logits[6, [11, 22, 33]] = torch.tensor(
            [3.0, 2.5, 2.0]
        )  # masked row (3 allowed)
        ref = theirs(logits.clone(), _metadata(B, temperature, k, p, gens_a))
        got = ours(logits.clone(), _metadata(B, temperature, k, p, gens_b))
        ref_ids = ref.sampled_token_ids.flatten()
        got_ids = got.sampled_token_ids.flatten()
        # greedy rows: argmax, exact
        for r in (0, 3):
            assert int(ref_ids[r]) == int(got_ids[r]) == int(logits[r].argmax())
        scaled = logits.clone()
        scaled[random_rows] = scaled[random_rows] / temperature[random_rows].unsqueeze(
            1
        )
        _check_seeded_equality(scaled, k, p, ref_ids, got_ids, random_rows)
    assert ours.topk_topp_sampler.compact_steps == 5
    assert ours.topk_topp_sampler.fallback_steps == 0


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
    assert not _boundary_tie(logits[0], target), "pick another seed: boundary tie"
    assert bool((counts[~support] == 0).all())
    assert torch.allclose(freq[support], target[support], atol=0.02)


def test_make_host_sampler_env_and_logprobs_mode(monkeypatch):
    monkeypatch.setenv("TT_HOST_SAMPLER_FAST", "0")
    assert not isinstance(make_host_sampler().topk_topp_sampler, TTTopKTopPSampler)
    monkeypatch.setenv("TT_HOST_SAMPLER_FAST", "1")
    s = make_host_sampler(logprobs_mode="processed_logprobs", use_fp64_gumbel=True)
    assert isinstance(s.topk_topp_sampler, TTTopKTopPSampler)
    assert s.logprobs_mode == "processed_logprobs"
    assert s.topk_topp_sampler.logprobs_mode == "processed_logprobs"
    assert s.topk_topp_sampler.use_fp64_gumbel is True
