"""Host sampler for TT decode: vLLM's ``Sampler`` with a compact top-k / top-p path.

The device hands the host the full logits row (Qwen3.8-27B: 248320 wide). vLLM's
``TopKTopPSampler`` implements top-p with a full-vocabulary sort per row and draws
exponential noise over the whole row, which at the served shapes costs 14 ms (B=1)
to 41 ms (B=8) per decode step on this host -- as much as a third of the TPOT under
the model's default sampling (temperature 1.0, top_k 20, top_p 0.95). Greedy rows
cost 0.2 ms.

``TTTopKTopPSampler.forward`` computes the same distribution on the candidate set
only: when every row has 1 <= top_k <= ``TT_HOST_SAMPLER_KMAX`` (default 256) it
takes the ``top_k + 32`` largest logits per row (the margin keeps the tokens tied
with the k-th value, which the upstream path keeps too), applies the identical
top-k / top-p masking rule on that compact ascending-sorted set, and samples with
the same exponential-race estimator over the compact probabilities. The surviving
token set, their probabilities and therefore the sampling distribution are those of
the upstream path (the only differences are fp32 reduction order in the softmax /
cumsum and the noise stream: the per-request generator now draws ``top_k + 32``
values instead of a full row, so a *seeded* random request produces a different --
still deterministic -- stream than upstream did). Rows with more than
``top_k + 32`` tied candidates, top_k == vocab (no top-k), or a ``logprobs_mode``
that needs the processed full-vocab logits fall back to the upstream path.
Greedy rows never reach this code (``Sampler.sample`` argmaxes them first).

``TT_HOST_SAMPLER_FAST=0`` disables the compact path (plain vLLM sampler).
"""

from __future__ import annotations

import os

import torch
from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

_TIE_MARGIN = 32


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


class TTTopKTopPSampler(TopKTopPSampler):
    """``TopKTopPSampler`` with a compact candidate-set path for small top_k."""

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
        kmax: int | None = None,
    ) -> None:
        super().__init__(logprobs_mode, use_fp64_gumbel)
        if kmax is None:
            kmax = int(os.environ.get("TT_HOST_SAMPLER_KMAX", "256"))
        self.kmax = kmax
        self._upstream_forward = self.forward
        self.forward = self.forward_compact  # type: ignore[method-assign]
        self.compact_steps = 0
        self.fallback_steps = 0

    def forward_compact(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        vocab = logits.shape[-1]
        if (
            k is None
            or self.logprobs_mode in ("processed_logits", "processed_logprobs")
            or int(k.max()) > self.kmax
            or int(k.min()) < 1
            or vocab <= self.kmax + _TIE_MARGIN
        ):
            self.fallback_steps += 1
            return self._upstream_forward(logits, generators, k, p)
        k_long = k.to(torch.long)
        width = int(k_long.max()) + _TIE_MARGIN
        top_vals, top_idx = logits.topk(width, dim=-1)  # descending
        kth = top_vals.gather(1, (k_long - 1).unsqueeze(1))  # k-th largest per row
        keep = top_vals >= kth  # ties with the k-th value are kept, as upstream
        # More than `width` candidates tie at the boundary: the compact set would
        # drop some of them, so take the exact (full-sort) path for this step.
        if bool((keep.all(dim=1) & (top_vals[:, -1] >= kth[:, 0])).any()):
            self.fallback_steps += 1
            return self._upstream_forward(logits, generators, k, p)
        comp = top_vals.masked_fill(~keep, -float("inf"))
        if p is not None:
            # Same rule as apply_top_k_top_p_pytorch on the ascending-sorted row:
            # the masked tokens carry probability 0 and sit first, so the
            # cumulative sum over the survivors is unchanged.
            asc = comp.flip(1)
            probs_sort = asc.softmax(dim=-1, dtype=torch.float32)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False  # at least one
            asc = asc.masked_fill(top_p_mask, -float("inf"))
            comp = asc.flip(1)
        probs = comp.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(
            probs, dtype=torch.float64 if self.use_fp64_gumbel else torch.float32
        )
        if len(generators) != probs.shape[0]:
            q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
        pick = probs.div_(q.to(probs.dtype)).argmax(dim=-1, keepdim=True)
        self.compact_steps += 1
        return top_idx.gather(1, pick).squeeze(1), None


def make_host_sampler(
    logprobs_mode: LogprobsMode = "raw_logprobs", use_fp64_gumbel: bool = False
) -> Sampler:
    """Plugin host ``Sampler``: compact top-k/top-p unless TT_HOST_SAMPLER_FAST=0."""
    sampler = Sampler(logprobs_mode=logprobs_mode, use_fp64_gumbel=use_fp64_gumbel)
    if _env_flag("TT_HOST_SAMPLER_FAST", "1"):
        sampler.topk_topp_sampler = TTTopKTopPSampler(logprobs_mode, use_fp64_gumbel)
    return sampler
