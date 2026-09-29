# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)


def _extract_tt_config(
    config: dict[str, Any], config_name: str
) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a JSON object")
    if "tt" not in config:
        return {}, False
    tt_config = config["tt"]
    if not isinstance(tt_config, dict):
        raise ValueError(f"{config_name}['tt'] must be a JSON object")
    return tt_config, True


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT config from vLLM's generic additional config namespace."""
    additional_config, _ = _extract_tt_config(
        getattr(vllm_config, "additional_config", {}) or {}, "additional_config"
    )
    return dict(additional_config)


# Internal key recording the resolved TT lane count. Stored at the top level of
# additional_config -- deliberately outside the user "tt" namespace -- so it
# never collides with user config and reads as platform-derived state rather
# than user input. Written by store_tt_lane_count, read by
# get_tt_data_parallel_size.
_RESOLVED_LANE_COUNT_KEY = "_tt_resolved_lane_count"
_OUTPUT_TOKENS_PER_STEP_KEY = "_tt_output_tokens_per_step"


def get_tt_data_parallel_size(vllm_config: "VllmConfig") -> int:
    """Effective TT lane count for batching, KV sizing, and merged execution.

    Standard multi-process DP runs one independent TT mesh per rank, so the TT
    model itself sees no internal DP and the effective TT lane count remains 1.
    With a single engine (``data_parallel_size == 1``) the value is the lane
    count resolved by the Galaxy DP-to-lanes conversion (see ``platform.py``)
    and recorded via ``store_tt_lane_count``; absent that, the count is 1.
    Not user-facing.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_RESOLVED_LANE_COUNT_KEY, 1))


def store_tt_lane_count(vllm_config: "VllmConfig", lanes: int) -> None:
    """Record the resolved in-process TT lane count on the config.

    Writes an internal, top-level key into ``additional_config`` (kept out of
    the user "tt" namespace) so ``get_tt_data_parallel_size`` observes it both
    here and in the worker subprocess -- ``additional_config`` is a declared
    config field, so it survives the copy/pickle to that process. Internal
    handoff from the Galaxy DP-to-lanes conversion; not user-facing.
    """
    if lanes < 1:
        raise ValueError(f"resolved TT lane count must be >= 1, got {lanes}")
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_RESOLVED_LANE_COUNT_KEY] = lanes


def get_tt_output_tokens_per_step(vllm_config: "VllmConfig") -> int:
    """Return the normalized model output width, defaulting to AR behavior.

    ``TTPlatform.check_and_update_config`` resolves the model capability once
    and stores it on the serializable vLLM config. Scheduler and worker
    construction therefore do not need to import model-loader code.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_OUTPUT_TOKENS_PER_STEP_KEY, 1))


def require_tt_output_tokens_per_step(vllm_config: "VllmConfig") -> int:
    """Return the resolved output width, failing if setup did not store it."""
    additional = getattr(vllm_config, "additional_config", None)
    if (
        not isinstance(additional, dict)
        or _OUTPUT_TOKENS_PER_STEP_KEY not in additional
    ):
        raise RuntimeError(
            "TT output_tokens_per_step was not initialized on VllmConfig"
        )
    return int(additional[_OUTPUT_TOKENS_PER_STEP_KEY])


def is_tt_block_output_model(vllm_config: "VllmConfig") -> bool:
    """Whether this config describes a model that commits multi-token blocks."""
    return get_tt_output_tokens_per_step(vllm_config) > 1


_ADAPTIVE_BLOCK_OUTPUT_KEY = "tt_adaptive_block_output"


def is_tt_adaptive_block_output_model(vllm_config: "VllmConfig") -> bool:
    """Whether the model commits a block ONLY when it decodes alone (batch==1).

    A plain block-output model owns a single request state and requires
    ``max_num_seqs 1`` / no data-parallelism. An ADAPTIVE block-output model
    emits its multi-token block only on steps that schedule exactly one decode
    request, and falls back to plain 1-token baseline decode whenever two or
    more requests batch together. That lifts the ``max_num_seqs 1`` and (later)
    data-parallel restrictions: at low concurrency each request gets the block
    speedup, at higher concurrency the server is a plain batched baseline
    (never worse). The scheduler reserves the K-token placeholder block only for
    a solo decode step (see TTScheduler), matching the model's batch gate.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional.get(_ADAPTIVE_BLOCK_OUTPUT_KEY, False))


def store_tt_adaptive_block_output(vllm_config: "VllmConfig", flag: bool) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_OUTPUT_KEY] = bool(flag)


_ADAPTIVE_BLOCK_BATCHED_KEY = "tt_adaptive_block_batched"


def is_tt_adaptive_block_batched(vllm_config: "VllmConfig") -> bool:
    """Whether an adaptive block model commits its block on BATCHED decode steps too.

    A plain adaptive block model blocks only when it decodes alone. A model that
    runs its speculative session for every decoding request at once (one
    multi-user verify per step) declares this flag: every decode step is a block
    step for all of its requests, each committing exactly
    ``output_tokens_per_step`` tokens (EOS-filled at a stop). Prefill steps still
    commit one host-sampled anchor per request. The scheduler reserves the block
    placeholders for every request of a decode step, so a step that mixes
    prefill and decode requests is rejected -- TTScheduler's default mode never
    builds one.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional.get(_ADAPTIVE_BLOCK_BATCHED_KEY, False))


def store_tt_adaptive_block_batched(vllm_config: "VllmConfig", flag: bool) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_BATCHED_KEY] = bool(flag)


_ADAPTIVE_BLOCK_RAGGED_KEY = "tt_adaptive_block_ragged"

# Row padding of a ragged block: every row of the rectangular
# ``[num_reqs, output_tokens_per_step]`` step output carries ``1..W`` real
# token ids followed by this value. Token ids are never negative.
TT_RAGGED_BLOCK_PAD_TOKEN_ID = -1


def is_tt_adaptive_block_ragged(vllm_config: "VllmConfig") -> bool:
    """Whether a batched adaptive block model commits RAGGED per-request widths.

    Under ``tt_adaptive_block_batched`` every request of a decode step commits
    exactly ``W = output_tokens_per_step`` tokens, so a slot that stops early
    is held until every slot has a full block. A model that declares this flag
    delivers each request's tokens as produced instead: a decode-only block
    step still returns one rectangular int32 ``[num_reqs, W]`` tensor, but row
    ``i`` holds ``1 <= n_i <= W`` real token ids followed by
    ``TT_RAGGED_BLOCK_PAD_TOKEN_ID`` padding. The runner counts ``n_i`` as the
    non-negative ids of the row, strips the padding and appends only those
    ``n_i`` tokens; the scheduler accepts ``1 <= n_i <= W`` per request but
    still consumes the whole ``W`` placeholder reservation, so the request's
    computed tokens advance by ``n_i`` while the per-step reservation (and the
    KV lookahead) stays ``W``. Prefill anchors stay width-1. Requires
    ``tt_adaptive_block_batched``.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional.get(_ADAPTIVE_BLOCK_RAGGED_KEY, False))


def store_tt_adaptive_block_ragged(vllm_config: "VllmConfig", flag: bool) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_RAGGED_KEY] = bool(flag)


_ADAPTIVE_BLOCK_MAX_PROMPT_KEY = "tt_adaptive_block_max_prompt_tokens"


def get_tt_adaptive_block_max_prompt_tokens(vllm_config: "VllmConfig") -> int:
    """Prompt-length frontier for the adaptive block path (0 = no limit).

    An adaptive block model whose fused capture cannot fit beyond some prompt
    length serves longer prompts as plain baseline (width-1 steps) for their
    whole lifetime. The scheduler must reserve width-1 for those requests even
    on solo decode steps, so the model declares the SAME frontier it gates on.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_ADAPTIVE_BLOCK_MAX_PROMPT_KEY, 0))


def store_tt_adaptive_block_max_prompt_tokens(
    vllm_config: "VllmConfig", limit: int
) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_MAX_PROMPT_KEY] = int(limit)


_BLOCK_KV_LOOKAHEAD_KEY = "tt_block_output_kv_lookahead_tokens"


def get_tt_block_output_kv_lookahead_tokens(vllm_config: "VllmConfig") -> int:
    """KV slots a block-output model may write past the step's scheduled token.

    Upstream allocates KV blocks for ``num_new_tokens`` (one token on a decode
    step) plus ``num_lookahead_tokens``. A block-output model writes the whole
    committed block -- and a speculative one also its rejected-draft tail --
    into the paged KV inside that single step, so those slots must already be
    allocated when the model runs, not one step later. The model declares how
    far past the scheduled token it writes; TTScheduler feeds it to
    allocate_slots as lookahead. 0 = upstream behaviour (block-only models
    whose canvas does not touch the paged KV).
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_BLOCK_KV_LOOKAHEAD_KEY, 0))


def store_tt_block_output_kv_lookahead_tokens(
    vllm_config: "VllmConfig", lookahead: int
) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_BLOCK_KV_LOOKAHEAD_KEY] = int(lookahead)


def store_tt_output_tokens_per_step(
    vllm_config: "VllmConfig", output_tokens_per_step: int
) -> None:
    """Store the validated per-request output width on the vLLM config."""
    if (
        isinstance(output_tokens_per_step, bool)
        or not isinstance(output_tokens_per_step, int)
        or output_tokens_per_step < 1
    ):
        raise ValueError(
            "resolved TT output_tokens_per_step must be an integer >= 1, got "
            f"{output_tokens_per_step!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_OUTPUT_TOKENS_PER_STEP_KEY] = output_tokens_per_step


def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Return the global TT batch capacity for model/KV sizing.

    Standard DP is per-rank and single-process lane mode already stores the
    global engine capacity in ``max_num_seqs`` after the Galaxy conversion, so
    the TT model should always size itself to the visible engine-local batch.
    """
    return int(vllm_config.scheduler_config.max_num_seqs)


def get_tt_per_lane_max_num_seqs(vllm_config: "VllmConfig") -> int:
    """Return the per-lane/per-rank scheduling and wire-format capacity.

    Outside lane mode the global ``max_num_seqs`` is already the per-rank
    capacity. In single-process lane mode it is the validated per-lane split
    (see ``validate_tt_lane_config``).
    """
    if not uses_tt_lane_coordinator(vllm_config):
        return int(vllm_config.scheduler_config.max_num_seqs)
    return validate_tt_lane_config(vllm_config)


def validate_tt_lane_config(vllm_config: "VllmConfig") -> int:
    """Validate single-process lane-mode batch sizing; return per-lane capacity.

    Lane mode partitions the global ``max_num_seqs`` evenly across the lanes
    (one in-process DP replica each), so the global value must be a positive
    multiple of the lane count; raises ``ValueError`` otherwise. Assumes lane
    mode is active (callers gate on ``uses_tt_lane_coordinator``).

    Exposed as a named helper so ``platform.check_and_update_config`` can run
    this check at config time -- calling it for its raising side effect so a
    misconfiguration fails fast with a clear message -- rather than calling the
    per-lane getter and discarding its result.
    """
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    lanes = get_tt_data_parallel_size(vllm_config)
    if max_num_seqs % lanes != 0:
        raise ValueError(
            "max_num_seqs must be divisible by the TT lane count in "
            f"single-process lane mode; got max_num_seqs={max_num_seqs}, "
            f"lanes={lanes}."
        )
    per_lane = max_num_seqs // lanes
    if per_lane < 1:
        raise ValueError(
            "max_num_seqs must provide at least one request per TT lane; got "
            f"max_num_seqs={max_num_seqs}, lanes={lanes}."
        )
    return per_lane


def uses_tt_lane_coordinator(vllm_config: "VllmConfig") -> bool:
    return (
        vllm_config.parallel_config.data_parallel_size == 1
        and get_tt_data_parallel_size(vllm_config) > 1
    )


# Resolved scheduler-driven chunked-prefill policy
# (see ``resolve_tt_prefill_chunk_policy``):
# ``[chunk_tokens, decode_steps_per_chunk]``, or absent when the policy is off.
_PREFILL_CHUNK_POLICY_KEY = "_tt_prefill_chunk_policy"
_DEFAULT_CHUNKED_PREFILL_DECODE_STEPS = 4


def get_tt_prefill_chunk_policy(vllm_config: "VllmConfig") -> tuple[int, int] | None:
    """Return ``(chunk_tokens, decode_steps_per_chunk)`` when the TT chunk policy is on.

    The policy is on only for a model that declares ``tt_prefill_chunk_tokens``
    (and ``supports_chunked_prefill``) while chunked prefill stays enabled; the
    scheduler then splits at most one long prompt at a time into aligned chunks
    and interleaves them with decode steps (``TTScheduler``), and the runner
    tells the model which prefill rows resume a chunk.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    policy = additional.get(_PREFILL_CHUNK_POLICY_KEY)
    if policy is None:
        return None
    return int(policy[0]), int(policy[1])


def store_tt_prefill_chunk_policy(
    vllm_config: "VllmConfig", chunk_tokens: int, decode_steps: int
) -> None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_PREFILL_CHUNK_POLICY_KEY] = [int(chunk_tokens), int(decode_steps)]


def clear_tt_prefill_chunk_policy(vllm_config: "VllmConfig") -> None:
    """Turn the TT chunk policy off, together with its block-output contract
    and extras (a flag that outlives the policy would relax the runner's
    width checks and keep sticky slots with nothing chunking)."""
    additional = getattr(vllm_config, "additional_config", None)
    if isinstance(additional, dict):
        additional.pop(_PREFILL_CHUNK_POLICY_KEY, None)
        additional.pop(_PREFILL_CHUNK_EXTRAS_KEY, None)


# Extras of the resolved TT chunk policy (``resolve_tt_prefill_chunk_policy``),
# absent when the policy is off; cleared with it. Keys:
# - ``block_output``: the model declared ``tt_block_output_chunked_prefill``
#   (a block-output model that resumes a split prompt; prefill anchors stay
#   width 1, intermediate rows emit nothing, continuations keep their state
#   slot).
# - ``rider_tokens``: prompt-token budget of the short prompts that may share a
#   chunk step while requests decode (None = no cap).
# - ``max_riders``: how many short prompts may share such a step (None = no
#   count cap). A rider's prefill cost is mostly fixed (the smallest masked
#   bucket), so the count bounds the step better than tokens for short riders.
# - ``cadence_after_final``: the decode cadence also separates a partial's
#   final chunk from the next long prompt's first chunk.
# - ``burst_longs``: with this many long prompts pending (0 = never), prefill
#   first like the default policy (no split, no cadence; with no partial in
#   flight every waiting prompt that fits the token budget is admitted whole).
# - ``min_tokens``: a prompt is long (chunked, one in flight) only with more
#   than max(chunk, min_tokens) tokens left; a shorter prompt above the chunk
#   ("medium") is prefilled whole, never riding a chunk step.
# - ``chunk_without_decoders``: a partial keeps advancing one chunk per step
#   when nothing decodes (instead of its whole remainder in one step), so a
#   request arriving meanwhile waits at most one chunk.
# - ``oversized_rider_step``: ``rider_tokens`` is a hard per-step budget; the
#   oldest short prompt above it gets a prefill step of its own (the partial
#   and every other prompt held), which counts toward the cadence like a chunk
#   step.
_PREFILL_CHUNK_EXTRAS_KEY = "_tt_prefill_chunk_extras"
_DEFAULT_BLOCK_OUTPUT_RIDER_TOKENS = 512
_DEFAULT_BLOCK_OUTPUT_MAX_RIDERS = 1
_DEFAULT_BLOCK_OUTPUT_BURST_LONGS = 2
_DEFAULT_BLOCK_OUTPUT_MIN_TOKENS = 8192


def get_tt_prefill_chunk_extras(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return the chunk policy's extras (defaults when the policy is off or
    was stored without them: no block-output contract, no rider cap, no
    post-final cadence, no burst fallback)."""
    extras = {
        "block_output": False,
        "rider_tokens": None,
        "max_riders": None,
        "cadence_after_final": False,
        "burst_longs": 0,
        "min_tokens": 0,
        "chunk_without_decoders": False,
        "oversized_rider_step": False,
    }
    if get_tt_prefill_chunk_policy(vllm_config) is None:
        return extras
    additional = getattr(vllm_config, "additional_config", None) or {}
    extras.update(additional.get(_PREFILL_CHUNK_EXTRAS_KEY) or {})
    return extras


def is_tt_block_output_chunked_prefill(vllm_config: "VllmConfig") -> bool:
    """Whether a block-output model resumes split prompts under the TT chunk
    policy (``tt_block_output_chunked_prefill``; False whenever the policy is
    off)."""
    return bool(get_tt_prefill_chunk_extras(vllm_config)["block_output"])


def resolve_tt_prefill_chunk_policy(
    vllm_config: "VllmConfig", model_chunk_tokens: int, block_output: bool = False
) -> tuple[int, int]:
    """Resolve and store the chunk policy of a model that declared its chunk unit.

    Operator knobs (``additional_config.tt``):

    - ``prefill_chunk_tokens``: the chunk size, a positive multiple of the
      model's ``tt_prefill_chunk_tokens`` (default: that unit). Chunk starts
      are multiples of it, which is what the model resumes from.
    - ``chunked_prefill_decode_steps``: decode-only steps after each chunk step
      while requests are decoding (default 4; 0 = no cadence).
    - ``chunked_prefill_rider_tokens``: prompt tokens of short prompts that may
      share a chunk step while requests decode (the oldest waiting short prompt
      always may, unless ``chunked_prefill_oversized_rider_step``); the rest
      wait for the next chunk step. Default: 512 for a block-output model
      (``block_output``), no cap otherwise.
    - ``chunked_prefill_max_riders``: how many short prompts may share such a
      chunk step (the oldest always may). Default: 1 for a block-output model,
      no cap otherwise.
    - ``chunked_prefill_cadence_after_final``: also hold the next long prompt's
      first chunk for the cadence after a partial's final chunk, so decoders
      never see two chunk steps back to back. Default: on for a block-output
      model, off otherwise.
    - ``chunked_prefill_burst_longs``: with at least this many long prompts
      pending (the partial included), prefill first as without the policy (the
      remainder in one step, no cadence; with no partial in flight, every
      waiting prompt that fits the token budget in one step): bursts keep the
      unchunked throughput, TTFT and TPOT at the cost of the decode stall.
      Default: 2 for a block-output model, 0 (never) otherwise.
    - ``chunked_prefill_min_tokens``: a prompt is chunked only when more than
      max(chunk, this) of its tokens remain; a shorter prompt is prefilled
      whole (with a partial in flight, in a prefill step of its own). Default:
      8192 for a block-output model, 0 (the chunk size) otherwise.
    - ``chunked_prefill_chunk_without_decoders``: while a partial is in flight
      and nothing decodes, keep advancing it one chunk per step (no cadence)
      instead of running its whole remainder in one step, so a request that
      arrives meanwhile waits for one chunk, not the remainder. Default: on for
      a block-output model, off otherwise.
    - ``chunked_prefill_oversized_rider_step``: make ``rider_tokens`` a hard
      budget: the oldest waiting short prompt above it does not ride a chunk
      step but gets a prefill step of its own (the partial and every other
      prompt held), counted toward the cadence like a chunk step; the next
      gate-open step advances the long prompt. Default: on for a block-output
      model, off otherwise.

    Rewrites the scheduler config so the base scheduler's token budget never
    splits a prompt (only ``long_prefill_token_threshold`` does, at the chunk
    size): at most one long prompt plus ``max_num_seqs - 1`` prompts of at most
    one chunk share a step, so the budget is raised to at least
    ``max_model_len + max_num_seqs * chunk_tokens``.
    """
    unit = int(model_chunk_tokens)
    if unit <= 0:
        raise ValueError(f"tt_prefill_chunk_tokens must be positive, got {unit}")
    tt_config = get_tt_config(vllm_config)
    chunk = tt_config.get("prefill_chunk_tokens", unit)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
        raise ValueError(
            "additional_config.tt.prefill_chunk_tokens must be a positive "
            f"integer, got {chunk!r}"
        )
    if chunk % unit != 0:
        raise ValueError(
            f"additional_config.tt.prefill_chunk_tokens={chunk} must be a multiple "
            f"of the model's tt_prefill_chunk_tokens={unit}"
        )
    steps = tt_config.get(
        "chunked_prefill_decode_steps", _DEFAULT_CHUNKED_PREFILL_DECODE_STEPS
    )
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError(
            "additional_config.tt.chunked_prefill_decode_steps must be an integer "
            f">= 0, got {steps!r}"
        )
    block_size = getattr(getattr(vllm_config, "cache_config", None), "block_size", None)
    if block_size and chunk % int(block_size) != 0:
        raise ValueError(
            f"prefill chunk {chunk} must be a multiple of the KV block size "
            f"{block_size}"
        )
    scheduler_config = vllm_config.scheduler_config
    max_model_len = vllm_config.model_config.max_model_len
    budget = max_model_len + int(scheduler_config.max_num_seqs) * chunk
    if scheduler_config.max_num_batched_tokens < budget:
        logger.warning(
            "TT chunked prefill: raising max_num_batched_tokens %d -> %d "
            "(max_model_len + max_num_seqs * chunk) so only the chunk threshold "
            "splits a prompt.",
            scheduler_config.max_num_batched_tokens,
            budget,
        )
        scheduler_config.max_num_batched_tokens = budget
    rider_tokens = tt_config.get(
        "chunked_prefill_rider_tokens",
        _DEFAULT_BLOCK_OUTPUT_RIDER_TOKENS if block_output else None,
    )
    if rider_tokens is not None and (
        isinstance(rider_tokens, bool)
        or not isinstance(rider_tokens, int)
        or rider_tokens < 0
    ):
        raise ValueError(
            "additional_config.tt.chunked_prefill_rider_tokens must be an integer "
            f">= 0 or null, got {rider_tokens!r}"
        )
    max_riders = tt_config.get(
        "chunked_prefill_max_riders",
        _DEFAULT_BLOCK_OUTPUT_MAX_RIDERS if block_output else None,
    )
    if max_riders is not None and (
        isinstance(max_riders, bool)
        or not isinstance(max_riders, int)
        or max_riders < 1
    ):
        raise ValueError(
            "additional_config.tt.chunked_prefill_max_riders must be an integer "
            f">= 1 or null, got {max_riders!r}"
        )
    cadence_after_final = tt_config.get(
        "chunked_prefill_cadence_after_final", bool(block_output)
    )
    if not isinstance(cadence_after_final, bool):
        raise ValueError(
            "additional_config.tt.chunked_prefill_cadence_after_final must be a "
            f"boolean, got {cadence_after_final!r}"
        )
    burst_longs = tt_config.get(
        "chunked_prefill_burst_longs",
        _DEFAULT_BLOCK_OUTPUT_BURST_LONGS if block_output else 0,
    )
    if (
        isinstance(burst_longs, bool)
        or not isinstance(burst_longs, int)
        or burst_longs < 0
        or burst_longs == 1
    ):
        raise ValueError(
            "additional_config.tt.chunked_prefill_burst_longs must be 0 (off) or an "
            f"integer >= 2, got {burst_longs!r}"
        )
    min_tokens = tt_config.get(
        "chunked_prefill_min_tokens",
        _DEFAULT_BLOCK_OUTPUT_MIN_TOKENS if block_output else 0,
    )
    if (
        isinstance(min_tokens, bool)
        or not isinstance(min_tokens, int)
        or min_tokens < 0
    ):
        raise ValueError(
            "additional_config.tt.chunked_prefill_min_tokens must be an integer "
            f">= 0, got {min_tokens!r}"
        )
    flags = {}
    for key in ("chunk_without_decoders", "oversized_rider_step"):
        value = tt_config.get(f"chunked_prefill_{key}", bool(block_output))
        if not isinstance(value, bool):
            raise ValueError(
                f"additional_config.tt.chunked_prefill_{key} must be a boolean, "
                f"got {value!r}"
            )
        flags[key] = value
    scheduler_config.long_prefill_token_threshold = chunk
    store_tt_prefill_chunk_policy(vllm_config, chunk, steps)
    additional = vllm_config.additional_config
    additional[_PREFILL_CHUNK_EXTRAS_KEY] = {
        "block_output": bool(block_output),
        "rider_tokens": rider_tokens,
        "max_riders": max_riders,
        "cadence_after_final": cadence_after_final,
        "burst_longs": burst_longs,
        "min_tokens": min_tokens,
        **flags,
    }
    return chunk, steps
