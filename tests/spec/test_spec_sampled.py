# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative sampling through the ``sampled_ids`` accept mode.

A verify in ``sampled_ids`` mode returns, per candidate column, a draw from the
row's own sampling distribution with noise independent of the drafts. The host
walks it by id equality, exactly as it walks ``argmax_ids``. For a drafter whose
proposal is a point mass (a greedy draft ``d``, ``q = onehot(d)``) that walk is
the standard rejection rule:

* accept iff ``x == d``, which has probability ``p(d) = min(1, p(d) / q(d))``;
* on a reject ``x`` is distributed as ``p`` restricted to ``v != d`` and
  renormalised, i.e. ``norm(max(0, p - q))``;
* the draft row's own draw is the bonus.

These tests cover the four parts of that contract:

1. the gate: a sampled request is speculable only on a launch the runner
   latched into ``sampled_ids`` (plan offers it, decode steps sample on
   device); penalties and host filters keep the PS-1 hold-back;
2. the wire: ``VerifyOutput`` pairs the mode with its field, and the runner
   refuses an answer in the other id mode;
3. the math: with injected logits (temperature, top-k, top-p and a host
   min-p), the id walk over sampled ids commits the target's exact law,
   matching the plugin's reference rejection sampler
   (``spec_accept.accept_speculated_tokens``), within multinomial noise;
4. end to end through the runner: with counter-keyed (seed, position) noise,
   a speculating launch commits token for token what a launch without
   speculation commits, for seeded sampled requests.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin import model_runner as model_runner_module
from vllm_tt_plugin.async_decode import TTAsyncDecodeController, _verify_output_tensor
from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_accept import _constrained_probs, accept_speculated_tokens
from vllm_tt_plugin.spec_admission import _RUNNABLE_ACCEPT_MODES
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_SAMPLED_IDS,
    ACCEPT_MODES,
    DRAFTER_STATE_INTERNAL,
    MODE_REQUIRED_FIELDS,
    PLACEHOLDER_TOKEN_ID,
    DraftOutput,
    SpecPlan,
    VerifyOutput,
    accept_greedy_drafts,
    check_spec_side_tensors,
)

from . import test_spec_uniform as uniform

# region The contract surface


def test_the_mode_is_declared_with_its_field():
    assert ACCEPT_MODE_SAMPLED_IDS in ACCEPT_MODES
    assert MODE_REQUIRED_FIELDS[ACCEPT_MODE_SAMPLED_IDS] == ("sampled_ids",)
    assert ACCEPT_MODE_SAMPLED_IDS in _RUNNABLE_ACCEPT_MODES


def test_a_sampled_ids_return_must_carry_its_ids():
    with pytest.raises(ValueError, match="missing \\['sampled_ids'\\]"):
        VerifyOutput(spec_mode=ACCEPT_MODE_SAMPLED_IDS, argmax_ids=torch.zeros(2, 2))
    out = VerifyOutput(
        spec_mode=ACCEPT_MODE_SAMPLED_IDS,
        sampled_ids=torch.zeros(2, 2, dtype=torch.int32),
    )
    assert out.sampled_ids.shape == (2, 2)


def test_the_runner_unwraps_the_mode_it_asked_for():
    ids = torch.tensor([[3, 4], [5, -1]], dtype=torch.int32)
    out = VerifyOutput(spec_mode=ACCEPT_MODE_SAMPLED_IDS, sampled_ids=ids)
    assert torch.equal(_verify_output_tensor(out, "m", ACCEPT_MODE_SAMPLED_IDS), ids)
    greedy = VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids)
    assert torch.equal(_verify_output_tensor(greedy, "m", ACCEPT_MODE_ARGMAX_IDS), ids)


@pytest.mark.parametrize(
    "asked, answered",
    [
        (ACCEPT_MODE_SAMPLED_IDS, ACCEPT_MODE_ARGMAX_IDS),
        (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_SAMPLED_IDS),
    ],
)
def test_an_answer_in_the_other_id_mode_is_refused(asked, answered):
    """An argmax answer to a sampled question would commit argmax for sampled rows."""
    ids = torch.zeros(2, 2, dtype=torch.int32)
    field = "sampled_ids" if answered == ACCEPT_MODE_SAMPLED_IDS else "argmax_ids"
    out = VerifyOutput(spec_mode=answered, **{field: ids})
    with pytest.raises(ValueError, match="answers in the mode it was asked for"):
        _verify_output_tensor(out, "m", asked)


def test_a_plan_offering_only_sampled_ids_is_admitted():
    plan = SpecPlan(
        effective_k=1,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_SAMPLED_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
    )
    assert plan.accept_modes == (ACCEPT_MODE_SAMPLED_IDS,)


# endregion The contract surface

# region The latch


@pytest.mark.parametrize(
    "modes, device_mode, samples, asks",
    [
        ((ACCEPT_MODE_ARGMAX_IDS,), "decode_only", False, ACCEPT_MODE_ARGMAX_IDS),
        (
            (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_SAMPLED_IDS),
            "decode_only",
            True,
            ACCEPT_MODE_SAMPLED_IDS,
        ),
        (
            (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_SAMPLED_IDS),
            "all",
            True,
            ACCEPT_MODE_SAMPLED_IDS,
        ),
        # No device sampling: a verify would carry no sampling parameters.
        (
            (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_SAMPLED_IDS),
            None,
            False,
            ACCEPT_MODE_ARGMAX_IDS,
        ),
        ((ACCEPT_MODE_SAMPLED_IDS,), None, False, ACCEPT_MODE_SAMPLED_IDS),
    ],
)
def test_the_runner_latches_sampled_ids_from_the_plan(
    monkeypatch, modes, device_mode, samples, asks
):
    monkeypatch.setattr(
        model_runner_module.TTPlatform,
        "sample_on_device_mode",
        device_mode,
        raising=False,
    )
    vllm_config = uniform._runner_config()
    store_tt_spec_plan(
        vllm_config,
        uniform._plan(accept_modes=modes, verify_requires_speculable_rows=True),
    )
    runner = TTModelRunner(
        vllm_config,
        mesh_device=None,
        trace_mode="decode_only",
        enable_model_warmup=False,
        num_devices=1,
    )
    assert runner._spec_verify_samples is samples
    assert runner._spec_accept_mode == asks


# endregion The latch

# region The publish gate


def _publish(latched: bool, **batch_kwargs) -> dict[str, list[int]]:
    runner = object.__new__(TTModelRunner)
    runner.input_batch = uniform._FakeBatch(**batch_kwargs)
    runner._spec_verify_requires_speculable_rows = True
    runner._spec_verify_samples = latched
    runner._proposed_draft_token_ids = {}
    TTModelRunner._publish_draft(runner, "req", [1])
    return runner._proposed_draft_token_ids


def test_a_sampled_request_is_speculable_once_latched():
    assert _publish(True, random=["req"]) == {"req": [1]}
    assert _publish(False, random=["req"]) == {}


def test_a_sampled_peer_no_longer_holds_the_batch_back_once_latched():
    assert _publish(True, random=["peer"]) == {"req": [1]}
    assert _publish(False, random=["peer"]) == {}


@pytest.mark.parametrize("penalty", ["presence", "frequency", "repetition"])
def test_a_penalty_still_holds_back_when_latched(penalty):
    """The device never sees penalized logits (supports_device_penalties)."""
    assert _publish(True, **{penalty: ["peer"]}) == {}
    assert _publish(True, **{penalty: ["req"]}) == {}


@pytest.mark.parametrize("case", sorted(uniform.HOST_FILTERED))
def test_a_host_filter_still_holds_back_when_latched(case):
    runner = object.__new__(TTModelRunner)
    runner.input_batch = uniform._FakeBatch()
    runner.requests = {
        "req": SimpleNamespace(sampling_params=SamplingParams(temperature=1.0)),
        "peer": SimpleNamespace(sampling_params=uniform.HOST_FILTERED[case]),
    }
    runner.input_batch.req_id_to_index = {"req": 0, "peer": 1}
    runner._spec_verify_requires_speculable_rows = True
    runner._spec_verify_samples = True
    runner._proposed_draft_token_ids = {}
    TTModelRunner._publish_draft(runner, "req", [1])
    assert runner._proposed_draft_token_ids == {}


# endregion The publish gate

# region The acceptance math (injected logits)

V = 24


def _min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """vLLM's min-p on a probability row (keep p >= min_p * max p), renormalised."""
    keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
    out = torch.where(keep, probs, torch.zeros_like(probs))
    return out / out.sum(dim=-1, keepdim=True)


def _target(temperature, top_k, top_p, min_p, logits):
    """The law the operator asked for at each column: [2, V] float64."""
    p = _constrained_probs(
        logits.unsqueeze(0),
        torch.tensor([temperature]),
        None if top_k is None else torch.tensor([top_k]),
        None if top_p is None else torch.tensor([top_p]),
    )[0].double()
    return p if min_p is None else _min_p(p, min_p)


def _exact_law(p0: torch.Tensor, p1: torch.Tensor, d: int) -> dict[tuple, float]:
    """Law of the committed prefix of a K = 1 row with draft d."""
    law = {(d, int(y)): float(p0[d] * p1[y]) for y in range(V) if p1[y] > 0}
    law.update({(int(x),): float(p0[x]) for x in range(V) if x != d and p0[x] > 0})
    return law


def _chi2_tv(counts: dict[tuple, int], law: dict[tuple, float], n: int):
    unknown = set(counts) - set(law)
    assert not unknown, f"committed prefixes outside the target's support: {unknown}"
    keys = list(law)
    obs = np.array([counts.get(k, 0) for k in keys], dtype=np.float64)
    exp = np.array([law[k] for k in keys]) * n
    big = exp >= 5
    chi = float((((obs[big] - exp[big]) ** 2) / exp[big]).sum())
    rest_o, rest_e = obs[~big].sum(), exp[~big].sum()
    dof = int(big.sum())
    if rest_e > 0:
        chi += float((rest_o - rest_e) ** 2 / rest_e)
    else:
        dof -= 1
    tv = 0.5 * float(np.abs(obs / n - exp / n).sum())
    rng = np.random.default_rng(7)
    probs = np.array([law[k] for k in keys])
    null = [
        0.5 * np.abs(rng.multinomial(n, probs / probs.sum()) / n - probs).sum()
        for _ in range(200)
    ]
    return chi, max(dof, 1), tv, float(np.quantile(null, 0.999))


CASES = {
    "T 0.6": (0.6, None, None, None),
    "T 1.0": (1.0, None, None, None),
    "T 1.0 top-p 0.95": (1.0, None, 0.95, None),
    "T 0.8 top-k 5 top-p 0.8": (0.8, 5, 0.8, None),
    "T 1.0 top-k 8": (1.0, 8, None, None),
    "T 1.0 min-p 0.1 (host transform)": (1.0, None, None, 0.1),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("draft", ["argmax", "second", "tail"])
def test_the_id_walk_over_sampled_ids_is_lossless(case, draft):
    """``accept_greedy_drafts`` on ``[x ~ p0, y ~ p1(.|d)]`` commits the target law.

    ``x`` and ``y`` are drawn with independent noise, as the device does (two
    positions, two counter pairs). The committed prefix law is compared with
    the exact law of the target and with the plugin's reference rejection
    sampler fed the same logits and the point-mass draft. Chi-square and TV
    must lie inside multinomial noise, and the acceptance rate must be p0(d).
    """
    T, k, tp, mp = CASES[case]
    gen = torch.Generator().manual_seed(hash(case) % (2**31))
    base = torch.randn(V, generator=gen) * 2.0
    order = torch.argsort(base, descending=True)
    d = int({"argmax": order[0], "second": order[1], "tail": order[V - 1]}[draft])
    after_d = torch.randn(V, generator=gen) * 2.0
    logits = torch.stack([base, after_d])  # column 0 after the anchor, column 1 after d
    p = _target(T, k, tp, mp, logits)
    p0, p1 = p[0], p[1]
    law = _exact_law(p0, p1, d)
    n = 40_000

    draws = torch.Generator().manual_seed(1234)
    x = torch.multinomial(p0.float(), n, replacement=True, generator=draws)
    y = torch.multinomial(p1.float(), n, replacement=True, generator=draws)
    sampled = torch.stack([x, y], dim=1).to(torch.int32)
    drafts = torch.full((n, 1), d, dtype=torch.int32)
    committed, counts = accept_greedy_drafts(
        sampled, drafts, torch.ones(n, dtype=torch.int32)
    )
    spec: dict[tuple, int] = {}
    for row, c in zip(committed.tolist(), counts.tolist()):
        key = tuple(row[:c])
        spec[key] = spec.get(key, 0) + 1
    chi, dof, tv, tv_q = _chi2_tv(spec, law, n)
    z = (chi - dof) / math.sqrt(2 * dof)
    assert z < 5.0, f"{case} / {draft}: chi2 {chi:.1f} on {dof} dof"
    assert tv <= tv_q, (
        f"{case} / {draft}: TV {tv:.4f} above the null's 99.9th percentile {tv_q:.4f}"
    )
    acc = float((counts == 2).double().mean())
    assert (
        abs(acc - float(p0[d]))
        < 5 * math.sqrt(max(p0[d] * (1 - p0[d]), 1e-12) / n) + 1e-9
    )

    if mp is None:  # the reference sampler knows temperature / top-k / top-p only
        ref_logits = logits.unsqueeze(0).expand(n, 2, V).contiguous()
        ref = accept_speculated_tokens(
            ref_logits,
            drafts,
            torch.ones(n, dtype=torch.int32),
            torch.full((n,), T),
            None if k is None else torch.full((n,), k, dtype=torch.int32),
            None if tp is None else torch.full((n,), tp),
            generators={},
        )
        refc: dict[tuple, int] = {}
        for row, c in zip(
            ref.committed_token_ids.tolist(), ref.accepted_counts.tolist()
        ):
            key = tuple(row[:c])
            refc[key] = refc.get(key, 0) + 1
        chi_r, dof_r, tv_r, _ = _chi2_tv(refc, law, n)
        assert (chi_r - dof_r) / math.sqrt(2 * dof_r) < 5.0
        assert tv_r <= tv_q


def test_a_greedy_row_walks_the_same_in_both_id_modes():
    """A greedy row's sampled id is its argmax, so the walk is unchanged."""
    ids = torch.tensor([[7, 9], [7, 3], [2, -1]], dtype=torch.int32)
    drafts = torch.tensor([[7], [8], [PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    nv = torch.tensor([1, 1, 0], dtype=torch.int32)
    committed, counts = accept_greedy_drafts(ids, drafts, nv)
    assert counts.tolist() == [2, 1, 1]
    assert committed.tolist() == [[7, 9], [7, -1], [2, -1]]


# endregion The acceptance math

# region End to end: seeded identity through the runner


def _uniform(seed: int, position: int) -> float:
    """A counter-keyed uniform: a pure function of (seed, position), like F4's."""
    h = (
        seed * 0x9E3779B97F4A7C15 + position * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB
    ) & (2**64 - 1)
    h ^= h >> 31
    h = (h * 0x94D049BB133111EB) & (2**64 - 1)
    h ^= h >> 29
    return ((h >> 11) & ((1 << 53) - 1)) * 2.0**-53


class CounterSampledTarget(uniform.ScoredTarget):
    """ScoredTarget that samples on "device": every draw is the inverse CDF of
    the row's constrained distribution at a uniform keyed by (seed, position),
    position being the INPUT position (the sampled token is the next one), and
    a verify in ``sampled_ids`` mode draws each column the same way."""

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_SAMPLED_IDS),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=True,
            verify_requires_speculable_rows=True,
        )

    def __init__(self):
        super().__init__()
        self.accepted = 0
        self.verified = 0

    def _draw(self, token: int, position: int, temperature, top_k, top_p, seed) -> int:
        logits = self.logits(torch.tensor(token), torch.tensor(position))
        if float(temperature) < 1e-5:
            return int(logits.argmax())
        p = _constrained_probs(
            logits.view(1, 1, -1),
            torch.tensor([float(temperature)]),
            torch.tensor([int(top_k)]) if top_k not in (None, 0, -1) else None,
            torch.tensor([float(top_p)]) if top_p is not None else None,
        )[0, 0].double()
        c = torch.cumsum(p, 0)
        u = _uniform(int(seed), int(position))
        return int(
            min(
                int(
                    torch.searchsorted(
                        c, torch.tensor(u * float(c[-1]), dtype=c.dtype), right=True
                    )
                ),
                self.vocab_size - 1,
            )
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        sampling_params=None,
        **kwargs,
    ):
        del kwargs
        assert sampling_params is not None, (
            "a device-sampling launch sends sampling_params on every decode step"
        )
        sp = sampling_params
        rows = tokens.shape[0]

        def params(i):
            return sp.temperature[i], sp.top_k[i], sp.top_p[i], sp.seed[i]

        if spec_mode is None:
            self.plain_calls += 1
            out = [
                self._draw(int(tokens[i, 0]), int(start_pos[i]), *params(i))
                if int(start_pos[i]) >= 0
                else 0
                for i in range(rows)
            ]
            return torch.tensor(out, dtype=torch.int32).view(rows, 1)
        assert spec_mode == ACCEPT_MODE_SAMPLED_IDS, spec_mode
        check_spec_side_tensors(
            num_valid_drafts, accepted_counts, rows, tokens.shape[1] - 1
        )
        self.verify_calls.append({"num_valid_drafts": num_valid_drafts.clone()})
        ids = torch.full(tokens.shape, PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for i in range(rows):
            if int(start_pos[i, 0]) < 0:
                ids[i, 0] = 0
                continue
            for j in range(1 + int(num_valid_drafts[i])):
                ids[i, j] = self._draw(
                    int(tokens[i, j]), int(start_pos[i, j]), *params(i)
                )
            if int(num_valid_drafts[i]):
                self.verified += 1
                self.accepted += int(ids[i, 0] == tokens[i, 1])
        return VerifyOutput(spec_mode=spec_mode, sampled_ids=ids)

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        out = super().propose_draft_tokens(
            num_drafts, committed_tokens, committed_positions, accepted_counts, hidden
        )
        return DraftOutput(draft_token_ids=out.draft_token_ids[:, :1].contiguous())


def _device_runner(model, *, speculating: bool):
    runner = uniform._runner(model, speculating=speculating)
    runner.check_perform_device_sampling = lambda **_: True
    if speculating:
        runner._num_speculative_tokens = 1
        runner._spec_verify_samples = True
        runner._spec_accept_mode = ACCEPT_MODE_SAMPLED_IDS
    return runner


def _device_step(runner, *req_ids, drafts=None):
    model_input = uniform._build(runner, *req_ids, drafts=drafts)
    assert model_input.perform_device_sampling
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(req_ids)],
        perform_device_sampling=True,
        is_decode=True,
        spec_hidden=submission.spec_hidden,
    )
    output = runner._finish_front_packed_sync(None, fwd=fwd)
    return dict(zip(output.req_ids, output.sampled_token_ids)), model_input.spec_mode


# At most MAX_NUM_REQS (3) requests share the batch.
SEEDED = {
    "T 0.7 seed 11": SamplingParams(temperature=0.7, seed=11),
    "T 1.0 top-p 0.95 seed 5": SamplingParams(temperature=1.0, top_p=0.95, seed=5),
    "T 0.8 top-k 20 top-p 0.8 seed 3": SamplingParams(
        temperature=0.8, top_k=20, top_p=0.8, seed=3
    ),
}


def _run(
    params_by_req: dict[str, SamplingParams], *, speculating: bool, tokens: int = 40
):
    model = CounterSampledTarget()
    runner = _device_runner(model, speculating=speculating)
    for i, (req_id, params) in enumerate(params_by_req.items()):
        uniform._add_request(runner, req_id, 1 + 50 * i, params)
    emitted = {r: [] for r in params_by_req}
    drafts: dict[str, list[int]] = {}
    modes = set()
    while min(len(v) for v in emitted.values()) < tokens:
        committed, mode = _device_step(runner, *params_by_req, drafts=drafts)
        modes.add(mode)
        for req_id, toks in committed.items():
            emitted[req_id].extend(toks)
        drafts = uniform._handoff(runner) if speculating else {}
    return {r: v[:tokens] for r, v in emitted.items()}, model, modes


def test_a_speculating_launch_commits_what_plain_sampling_commits():
    """Seeded identity: the same tokens, seed for seed, every request at once.

    The verify's column 0 is the draw plain decoding makes at the anchor
    position, the draft row's draw is the one plain decoding makes at the next
    position after an accepted draft, and a rejected draft's draw is discarded
    while its position's noise is used again by the next step's anchor. So with
    noise keyed by (seed, position) the two launches agree token for token.
    """
    spec, model, modes = _run(SEEDED, speculating=True)
    plain, _, _ = _run(SEEDED, speculating=False)
    assert ACCEPT_MODE_SAMPLED_IDS in modes, "nothing verified"
    assert model.verified > 0 and 0 < model.accepted < model.verified
    assert spec == plain


def test_a_greedy_request_beside_sampled_ones_is_unchanged():
    mixed = {
        "greedy": SamplingParams(temperature=0.0),
        **dict(list(SEEDED.items())[:2]),
    }
    spec, model, _ = _run(mixed, speculating=True)
    plain, _, _ = _run(mixed, speculating=False)
    assert model.verified > 0
    assert spec == plain


def test_an_argmax_launch_keeps_sampled_requests_off_verify_steps():
    """Without the latch (argmax_ids) a sampled request still holds speculation back."""
    model = CounterSampledTarget()
    runner = _device_runner(model, speculating=True)
    runner._spec_verify_samples = False
    runner._spec_accept_mode = ACCEPT_MODE_ARGMAX_IDS
    uniform._add_request(runner, "s", 1, SamplingParams(temperature=1.0, seed=1))
    drafts = {}
    for _ in range(6):
        _, mode = _device_step(runner, "s", drafts=drafts)
        assert mode is None
        drafts = uniform._handoff(runner)
        assert drafts == {}


# endregion End to end

# region Launches and steps without device sampling


def test_custom_logits_processors_never_reach_a_speculative_runner(monkeypatch):
    """``--logits-processors`` with speculation is refused at runner build.

    vLLM's ``build_logitsprocs`` raises for custom logits processors on a
    speculative launch, so the latch's own guard against them
    (``_spec_verify_samples`` requires none) is defence in depth, and the
    per-step hold-back below covers every other host-sampled step.
    """
    monkeypatch.setattr(
        model_runner_module.TTPlatform,
        "sample_on_device_mode",
        "decode_only",
        raising=False,
    )
    vllm_config = uniform._runner_config()
    vllm_config.model_config.logits_processors = ["my.module:Proc"]
    store_tt_spec_plan(
        vllm_config,
        uniform._plan(
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_SAMPLED_IDS),
            verify_requires_speculable_rows=True,
        ),
    )
    with pytest.raises(ValueError, match="speculative decoding"):
        TTModelRunner(
            vllm_config,
            mesh_device=None,
            trace_mode="decode_only",
            enable_model_warmup=False,
            num_devices=1,
        )


def test_a_host_sampled_step_with_a_sampled_request_is_held_back():
    """A latched launch never sends a sampled_ids verify without sampling params.

    The step is built while ``check_perform_device_sampling`` says host: with a
    sampled request in the batch it must decode ordinarily (no ``spec_mode``),
    even though drafts are scheduled; once the device samples again, it
    verifies.
    """
    model = CounterSampledTarget()
    runner = _device_runner(model, speculating=True)
    uniform._add_request(runner, "s", 1, SamplingParams(temperature=1.0, seed=1))
    token, position = uniform._tail(runner, "s")
    drafts = {"s": model.greedy(token, position, 1)}
    runner.check_perform_device_sampling = lambda **_: False
    assert model_runner_module._sampled_verify_needs_host_sampling(runner, [0]) is True
    model_input = uniform._build(runner, "s", drafts=drafts)
    assert not model_input.perform_device_sampling
    assert model_input.spec_mode is None


def test_a_host_sampled_step_of_greedy_requests_still_verifies():
    model = CounterSampledTarget()
    runner = _device_runner(model, speculating=True)
    uniform._add_request(runner, "g", 1, SamplingParams(temperature=0.0))
    runner.check_perform_device_sampling = lambda **_: False
    assert model_runner_module._sampled_verify_needs_host_sampling(runner, [0]) is False


def test_an_unlatched_launch_never_asks():
    runner = object.__new__(TTModelRunner)
    runner._spec_verify_samples = False
    runner.input_batch = uniform._FakeBatch(random=["s"])
    assert model_runner_module._sampled_verify_needs_host_sampling(runner, [0]) is False


# endregion Launches and steps without device sampling
