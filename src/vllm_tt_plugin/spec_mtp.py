# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding with a model-owned drafter (Qwen3.6 MTP head): policy
and plumbing on the plugin side.

The TT model runs the whole draft -> verify -> commit loop itself (tt-metal
``qwen36/tt/spec_decoder.py``); vLLM only does the bookkeeping: the scheduler
schedules ``1 + K_s`` tokens per request (its last token plus the ``K_s`` draft
tokens the runner proposed after the previous step), allocates their KV
positions, receives ``1..1+K_s`` committed tokens back (``num_computed_tokens``
and the async placeholders are adjusted by the rejections in
``Scheduler.update_from_output``) and takes the next drafts through
``take_draft_token_ids``. This module holds what both the scheduler and the
runner need without importing tt-metal:

* ``TTSpecStepInput``: the per-step payload the runner attaches to
  ``TTModelInput.spec`` (request per row, drafts per row, greedy eligibility,
  the scheduler's flush request).
* the admission-hold sidecars: the runner publishes ``HoldInfo`` on the
  ``ModelRunnerOutput`` after every decode step, the scheduler answers with a
  flush request on the ``SchedulerOutput`` of a decode-only step it inserts
  before an admission that would break the model's pending lazy prefix (see
  docs/SPECULATIVE.md).
* ``request_forces_plain``: which sampling parameters keep a request off the
  greedy verify path (the whole step then runs as plain decode).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

_TT_SPEC_HOLD_ATTR = "_tt_spec_hold"
_TT_SPEC_FLUSH_ATTR = "_tt_spec_flush"


@dataclass(frozen=True)
class TTSpecStepInput:
    """Speculative-decoding inputs of one decode step (``TTModelInput.spec``)."""

    # Request id per padded decode row (``None`` = pad row).
    row_req_ids: list[str | None]
    # Scheduled draft tokens per padded row (``None`` for pad rows; a ``-1``
    # entry ends the usable prefix -- an unfilled scheduler placeholder).
    drafts: list[list[int] | None]
    # Every live request may take the greedy verify path (device greedy
    # sampling, no host-only features); otherwise the step is plain decode.
    eligible: bool
    # The scheduler inserted this decode-only step to commit the model's
    # pending accepted prefixes before an admission: zero drafts for everyone.
    flush: bool = False


@dataclass(frozen=True)
class HoldInfo:
    """Runner -> scheduler: whether the next admission must wait for a flush.

    ``pending_any``: some live request has an accepted-but-lazily-committed
    prefix (a plain step or a smaller verify grid cannot commit it).
    ``slots_before_crossing``: how many admissions the current verify band
    absorbs without breaking that prefix (free rows below the band's width);
    ``None`` = any number (the pending rows survive every reachable band).
    """

    pending_any: bool = False
    slots_before_crossing: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pending_any": bool(self.pending_any),
            "slots_before_crossing": self.slots_before_crossing,
        }

    @classmethod
    def from_any(cls, info: Any) -> HoldInfo:
        if info is None:
            return cls()
        if isinstance(info, HoldInfo):
            return info
        if isinstance(info, dict):
            return cls(
                pending_any=bool(info.get("pending_any", False)),
                slots_before_crossing=info.get("slots_before_crossing"),
            )
        return cls(
            pending_any=bool(getattr(info, "pending_any", False)),
            slots_before_crossing=getattr(info, "slots_before_crossing", None),
        )


def set_tt_spec_hold(output: ModelRunnerOutput, info: Any) -> None:
    """Attach the runner's hold info to a model runner output (plain dict: the
    output crosses the executor boundary by pickling in multiprocess mode)."""
    setattr(output, _TT_SPEC_HOLD_ATTR, HoldInfo.from_any(info).as_dict())


def get_tt_spec_hold(output: ModelRunnerOutput) -> HoldInfo | None:
    info = getattr(output, _TT_SPEC_HOLD_ATTR, None)
    return None if info is None else HoldInfo.from_any(info)


def set_tt_spec_flush(scheduler_output: SchedulerOutput, flush: bool = True) -> None:
    setattr(scheduler_output, _TT_SPEC_FLUSH_ATTR, bool(flush))


def get_tt_spec_flush(scheduler_output: SchedulerOutput) -> bool:
    return bool(getattr(scheduler_output, _TT_SPEC_FLUSH_ATTR, False))


def request_forces_plain(
    sampling_params: SamplingParams | None, use_structured_output: bool = False
) -> bool:
    """True when a request cannot take the greedy verify path: anything but
    temperature-0 device sampling without host-only features. Mirrors the
    runner's eligibility check (``spec_eligible``) at the request level, so
    the scheduler can predict that an admission will force plain decode."""
    if use_structured_output:
        return True
    sp = sampling_params
    if sp is None:
        return True
    if getattr(sp, "structured_outputs", None) is not None:
        return True
    if float(getattr(sp, "temperature", 0.0) or 0.0) != 0.0:
        return True
    if getattr(sp, "logprobs", None) is not None:
        return True
    if getattr(sp, "prompt_logprobs", None) is not None:
        return True
    if float(getattr(sp, "presence_penalty", 0.0) or 0.0) != 0.0:
        return True
    if float(getattr(sp, "frequency_penalty", 0.0) or 0.0) != 0.0:
        return True
    if float(getattr(sp, "repetition_penalty", 1.0) or 1.0) != 1.0:
        return True
    if float(getattr(sp, "min_p", 0.0) or 0.0) != 0.0:
        return True
    if getattr(sp, "logit_bias", None):
        return True
    if getattr(sp, "allowed_token_ids", None):
        return True
    if getattr(sp, "bad_words", None):
        return True
    return int(getattr(sp, "min_tokens", 0) or 0) > 0


def hold_needed(hold: HoldInfo | None, n_ready: int, any_plain_forcing: bool) -> bool:
    """The scheduler's decision: hold ``n_ready`` admissions for one flush step?"""
    if hold is None or not hold.pending_any or n_ready <= 0:
        return False
    if any_plain_forcing:
        return True
    return (
        hold.slots_before_crossing is not None and n_ready > hold.slots_before_crossing
    )


def drafts_for_rows(
    scheduled_spec_decode_tokens: dict[str, list[int]] | None,
    row_req_ids: Sequence[str | None],
    pad_to: int,
) -> list[list[int] | None]:
    """Per padded row: the scheduled draft tokens (``[]`` when none; ``None`` for
    a pad row), in the order the runner built the rows."""
    sched = scheduled_spec_decode_tokens or {}
    out: list[list[int] | None] = []
    for req_id in row_req_ids:
        if req_id is None:
            out.append(None)
        else:
            out.append([int(t) for t in sched.get(req_id, ())])
    out.extend([None] * max(0, pad_to - len(out)))
    return out


def is_spec_step_result(obj: Any) -> bool:
    """Duck-typed check for the model's speculative step result (tt-metal
    ``SpecStepResult``: ``committed`` / ``next_drafts`` per grid row) so the
    plugin never imports tt-metal."""
    return (
        hasattr(obj, "committed") and hasattr(obj, "next_drafts") and hasattr(obj, "w")
    )
