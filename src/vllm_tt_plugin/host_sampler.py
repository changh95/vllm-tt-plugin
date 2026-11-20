# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Native CPU sampling with bounded, speculative noise for seeded requests."""

import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    empty_exponential_noise_like,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import Sampler

# Largest top-k served by the top-k-first path; a larger k (or none: vLLM passes
# the vocab size) takes vLLM's full-vocab sort.
_TOPK_FIRST_MAX_K = 1024
# Candidates taken beyond max(k): top-k keeps every logit tied with the k-th
# largest, and device logits are bf16, so ties at the boundary are common.
_TOPK_FIRST_TIE_MARGIN = 64


def topk_first_filter(logits, k, p):
    """The candidates apply_top_k_top_p would leave, without sorting the vocab.

    Returns (values, vocab indices), each [batch, max(k) + margin]: the top
    logits of every row in ascending order, with the ones apply_top_k_top_p
    masks set to -inf. The masks follow vLLM's: top-k keeps every logit >= the
    row's k-th largest (ties included), and top-p runs on the kept logits in
    ascending order over float32 softmax and cumsum (the removed ones
    contribute 0 to both). Where the top-p cut falls inside a run of equal
    logits, vLLM's unstable sort decides which of them go; here the lower vocab
    indices go first, as in a stable ascending sort, so the kept count and
    probabilities match and the choice is deterministic. Returns None when a
    row has more logits tied at its k-th value than the candidates hold, which
    only the full sort keeps.
    """
    k = k.to(torch.long)
    width = min(logits.shape[1], int(k.max()) + _TOPK_FIRST_TIE_MARGIN)
    values, indices = logits.topk(width, dim=-1)
    kth = values.gather(1, (k - 1).unsqueeze(1))
    if not torch.equal((logits >= kth).sum(dim=1), (values >= kth).sum(dim=1)):
        return None
    # ascending by value, ties by vocab index
    by_index = indices.argsort(dim=-1)
    values, indices = values.gather(1, by_index), indices.gather(1, by_index)
    values, order = values.sort(dim=-1, stable=True)
    indices = indices.gather(1, order)
    values = values.masked_fill(values < kth, -float("inf"))
    if p is not None:
        probs_sum = values.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        values = values.masked_fill(top_p_mask, -float("inf"))
    return values, indices


@dataclass
class _PrefetchedNoise:
    state: torch.Tensor
    vocab: int
    size_bytes: int
    future: Future


class TTTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.forward = self.forward_native
        self._sampling_pool = None
        self._noise_cache = OrderedDict()
        self._cached_bytes = 0
        self._max_prefetch_bytes = 64 * 1024 * 1024
        self._sampling_lock = Lock()

    def _prepare_noise(self, state, vocab):
        # Workers only advance private clones. Preparing or discarding noise
        # must not consume a live request's RNG, including after cancellation.
        clone = torch.Generator(device="cpu").set_state(state)
        dtype = torch.float64 if self.use_fp64_gumbel else torch.float32

        def generate():
            noise = torch.empty(vocab, dtype=dtype)
            noise.exponential_(generator=clone)
            return noise, clone.get_state()

        size = vocab * (8 if self.use_fp64_gumbel else 4)
        return _PrefetchedNoise(
            state, vocab, size, self._sampling_pool.submit(generate)
        )

    def forward_native(self, logits, generators, k, p):
        batch, vocab = logits.shape
        if not generators and self._topk_first_applies(logits, k):
            sampled = self._sample_topk_first(logits, k, p)
            if sampled is not None:
                return sampled, None
        if (
            logits.device.type != "cpu"
            or batch == 0
            or vocab < 4096
            or set(generators) != set(range(batch))
            or len({id(gen) for gen in generators.values()}) != batch
            or any(gen.device.type != "cpu" for gen in generators.values())
        ):
            return super().forward_native(logits, generators, k, p)

        # A request generator may move to a different batch row. Cache by the
        # generator itself, and verify its complete state before reusing noise.
        with self._sampling_lock:
            return self._sample_seeded(logits, generators, k, p)

    def _topk_first_applies(self, logits, k):
        # Seeded rows keep the full-vocab noise of their generators; the
        # processed logprob modes return full-vocab processed logits.
        batch, vocab = logits.shape
        return (
            k is not None
            and batch > 0
            and logits.device.type == "cpu"
            and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
            and int(k.min()) >= 1
            and int(k.max()) <= min(_TOPK_FIRST_MAX_K, vocab - 1)
        )

    def _sample_topk_first(self, logits, k, p):
        """Unseeded rows with top-k: the same distribution as vLLM's
        forward_native, with top-p and the exponential-noise draw over the
        max(k) candidates instead of the vocab. None: take the full path."""
        filtered = topk_first_filter(logits, k, p)
        if filtered is None:
            return None
        values, indices = filtered
        probs = values.softmax(dim=-1, dtype=torch.float32)
        q = empty_exponential_noise_like(probs, self.use_fp64_gumbel)
        q.exponential_()
        chosen = sample_with_exponential_noise(probs, q)
        return indices.gather(1, chosen.unsqueeze(1)).view(-1)

    def _sample_seeded(self, logits, generators, k, p):
        batch, vocab = logits.shape
        if self._sampling_pool is None:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            self._sampling_pool = ThreadPoolExecutor(
                max_workers=min(32, available), thread_name_prefix="tt-sampling"
            )

        prepared = []
        for row in range(batch):
            generator = generators[row]
            state = generator.get_state()
            entry = self._noise_cache.pop(generator, None)
            if entry is not None:
                self._cached_bytes -= entry.size_bytes
            if (
                entry is None
                or entry.vocab != vocab
                or not torch.equal(entry.state, state)
            ):
                if entry is not None:
                    entry.future.cancel()
                entry = self._prepare_noise(state, vocab)
            prepared.append(entry)

        # Resolve generation before submitting sampling tasks to the same pool;
        # a worker must never wait on work queued behind other waiting workers.
        noise_and_states = [entry.future.result() for entry in prepared]

        def sample_row(row):
            values = apply_top_k_top_p(
                logits[row : row + 1],
                k[row : row + 1] if k is not None else None,
                p[row : row + 1] if p is not None else None,
            )
            scores = None
            if self.logprobs_mode == "processed_logits":
                scores = values
            elif self.logprobs_mode == "processed_logprobs":
                scores = values.log_softmax(dim=-1, dtype=torch.float32)
            probs = values.softmax(dim=-1, dtype=torch.float32)
            noise, next_state = noise_and_states[row]
            generators[row].set_state(next_state)
            token = sample_with_exponential_noise(probs, noise.unsqueeze(0))
            return token, scores

        # Each Torch operation may also use intra-op threads. Group rows into
        # a few tasks rather than dispatching one competing Torch job per row.
        if batch == 1:
            results = [sample_row(0)]
        else:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            workers = max(1, min(4, available // torch.get_num_threads(), batch))
            chunk_size = (batch + workers - 1) // workers

            def sample_chunk(start):
                return [
                    sample_row(row)
                    for row in range(start, min(start + chunk_size, batch))
                ]

            results = [
                result
                for chunk in self._sampling_pool.map(
                    sample_chunk, range(0, batch, chunk_size)
                )
                for result in chunk
            ]

        # This work can overlap the next TT forward pass. The cache is bounded
        # by noise bytes, including pending futures; evicted clones are harmless.
        noise_bytes = vocab * (8 if self.use_fp64_gumbel else 4)
        prefetch_count = min(batch, self._max_prefetch_bytes // noise_bytes)
        for row in range(batch - prefetch_count, batch):
            generator = generators[row]
            entry = self._prepare_noise(noise_and_states[row][1], vocab)
            self._noise_cache[generator] = entry
            self._cached_bytes += entry.size_bytes
            while self._cached_bytes > self._max_prefetch_bytes:
                _, old = self._noise_cache.popitem(last=False)
                self._cached_bytes -= old.size_bytes
                old.future.cancel()

        tokens = torch.cat([result[0] for result in results])
        scores = (
            torch.cat([result[1] for result in results])
            if results[0][1] is not None
            else None
        )
        return tokens, scores


class TTHostSampler(Sampler):
    @staticmethod
    def apply_temperature(logits, temp, all_random):
        # CPU metadata needs no synchronization. Avoid touching the entire
        # vocabulary when division would leave every value unchanged.
        if (
            all_random
            and logits.device.type == "cpu"
            and temp.device.type == "cpu"
            and torch.all(temp == 1.0).item()
        ):
            return logits
        return Sampler.apply_temperature(logits, temp, all_random)

    @staticmethod
    def apply_penalties(logits, sampling_metadata, output_token_ids):
        # vLLM builds [rows, vocab] bin counts and masks for the prompt and the
        # output every step and scales every logit; on CPU, touch only the
        # tokens each penalized row has seen. Same operations in the same order
        # and dtypes as vLLM's apply_penalties, so the logits are bit-identical.
        if sampling_metadata.no_penalties or logits.device.type != "cpu":
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)
        vocab = logits.shape[1]
        presence = sampling_metadata.presence_penalties
        frequency = sampling_metadata.frequency_penalties
        repetition = sampling_metadata.repetition_penalties
        for row in range(logits.shape[0]):
            p, f, r = presence[row], frequency[row], repetition[row]
            if p == 0 and f == 0 and r == 1:
                continue
            # async scheduling leaves -1 placeholders that vLLM ignores
            out = torch.tensor(
                [t for t in output_token_ids[row] if 0 <= t < vocab], dtype=torch.long
            )
            if r != 1:
                prompt = sampling_metadata.prompt_token_ids[row]
                seen = torch.cat([prompt[prompt < vocab].to(torch.long), out]).unique()
                values = logits[row, seen]
                scaling = torch.where(values > 0, 1.0 / r, r)
                logits[row, seen] = (values * scaling).to(logits.dtype)
            if out.numel():
                ids, counts = out.unique(return_counts=True)
                logits[row, ids] -= f * counts
                logits[row, ids] -= p
        return logits


def create_host_sampler():
    sampler = TTHostSampler()
    sampler.topk_topp_sampler = TTTopKTopPSampler(
        sampler.logprobs_mode, sampler.use_fp64_gumbel
    )
    return sampler
