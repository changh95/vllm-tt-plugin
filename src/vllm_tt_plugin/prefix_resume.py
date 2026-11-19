# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefix-cache hits for models that keep prefill state outside vLLM's KV cache.

A model whose prefill carries per-request state between chunks that vLLM does
not hold (a sliding-window layer's history tail, for example) cannot take an
arbitrary prefix-cache hit: at the hit position it would have the cached KV but
not that state, and recomputing the state from a bounded window is not exact.
Such a model declares::

    model_capabilities["prefix_resume"] = {
        "granularity": G,  # resume points lie on this token grid
        "min_suffix": T,  # at least T prompt tokens are always recomputed
        "tail_slots": K,  # the model holds K saved states ("tails")
        "prompt_only": True,  # cache only prompt tokens' blocks
        "req_ids": True,  # pass each row's request id to the model
    }

The scheduler then owns a directory of the model's K saved tails and is the
only party that decides what the model stores and restores:

- **Store.** When a request's prefill chunk ends at ``e`` on the G grid
  (``G <= e <= num_prompt_tokens``) and every chunk of that request so far
  started on the grid (a *canonical* run: what a cold prefill with the full
  budget does), the step carries an order "save this row's tail in slot k".
  At most two entries per request (its two latest); otherwise least recently
  used, never a slot another order of the same step names. If the directory
  already holds the same point (block hash and ``e``), that entry is marked
  used instead: canonical runs leave identical tails there.
- **Cap.** A new or resumed request's prefix-cache hit ``h`` becomes the
  largest directory entry ``e <= min(h, num_tokens - T, num_prompt_tokens)``
  whose block hash at ``e`` matches the request's own (the hash chain implies
  an identical token prefix and cache salt), re-looked-up through the KV cache
  coordinator so every group's blocks stay consistent; with no such entry the
  hit is 0 and the request prefills cold.
- **Restore.** A request resuming at ``e`` carries an order "restore slot k at
  e" in the step that schedules its first chunk.

Orders ride the ``SchedulerOutput`` (a TT field) to the model runner, which
hands them to ``prefill_forward`` per row (``tail_orders``) next to the rows'
request ids (``req_ids``). Steps execute in the order they are scheduled, so a
store is on the device before any later restore of it; within one step the
model runs a storing row before a row that restores the same slot (pinning
keeps a slot from being both restored and re-stored in one step).

With ``prompt_only`` the KV cache coordinator caches only prompt tokens'
blocks: blocks written by decode programs are never a hit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.config import get_tt_data_parallel_size, store_tt_prefix_resume
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_tt_logger(__name__)

PREFIX_RESUME_CAPABILITY = "prefix_resume"
# The SchedulerOutput field (scheduler.TT_SCHEDULER_OUTPUT_FIELDS) carrying a
# step's orders, {req_id: {"restore": (slot, end, generation), "store": ...}}.
TAIL_ORDERS_ATTR = "_tt_prefix_tail_orders"
_KEYS = ("granularity", "min_suffix", "tail_slots", "prompt_only", "req_ids")


@dataclass(frozen=True)
class PrefixResumeConfig:
    granularity: int
    min_suffix: int
    tail_slots: int
    prompt_only: bool
    req_ids: bool


def resolve_prefix_resume(
    declared: Any, *, block_size: int, model_name: str
) -> PrefixResumeConfig:
    """Validate a model's ``prefix_resume`` capability."""
    if not isinstance(declared, dict) or set(declared) != set(_KEYS):
        raise ValueError(
            f"Model {model_name} declares model_capabilities['prefix_resume']="
            f"{declared!r}; it must be a dict with exactly the keys {_KEYS}"
        )
    ints = {k: declared[k] for k in ("granularity", "min_suffix", "tail_slots")}
    for key, value in ints.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Model {model_name}: prefix_resume[{key!r}] must be an integer "
                f">= 0, got {value!r}"
            )
    for key in ("prompt_only", "req_ids"):
        if not isinstance(declared[key], bool):
            raise ValueError(
                f"Model {model_name}: prefix_resume[{key!r}] must be a bool"
            )
    granularity = ints["granularity"]
    if granularity <= 0 or granularity % block_size:
        raise ValueError(
            f"Model {model_name}: prefix_resume['granularity']={granularity} "
            f"must be a positive multiple of the {block_size}-token block"
        )
    return PrefixResumeConfig(**declared)


def resolve_and_store_prefix_resume(
    vllm_config: VllmConfig,
    model_capabilities: dict | None,
    model_class: type,
    supports_prefix_caching: bool,
) -> None:
    """Validate and store ``model_capabilities["prefix_resume"]`` (the platform's
    check_and_update_config calls this).

    Absent: nothing changes. Present: the model must also declare
    supports_prefix_caching, the engine must run one TT lane (lanes would each
    own a tail directory over the one model's slots), and the resolution is
    stored only while prefix caching is on (with it off there are no hits to
    cap and nothing to store).
    """
    declared = (model_capabilities or {}).get(PREFIX_RESUME_CAPABILITY)
    if declared is None:
        store_tt_prefix_resume(vllm_config, None)
        return
    model_name = f"{model_class.__module__}.{model_class.__name__}"
    if not supports_prefix_caching:
        raise ValueError(
            f"Model {model_name} declares model_capabilities['prefix_resume'] "
            "without model_capabilities['supports_prefix_caching']=True"
        )
    lanes = get_tt_data_parallel_size(vllm_config)
    if lanes != 1:
        raise ValueError(
            f"Model {model_name} declares model_capabilities['prefix_resume'], "
            f"which supports one TT lane per engine, but this engine runs {lanes}"
        )
    resolved = resolve_prefix_resume(
        declared,
        block_size=vllm_config.cache_config.block_size,
        model_name=model_name,
    )
    enabled = vllm_config.cache_config.enable_prefix_caching
    store_tt_prefix_resume(vllm_config, asdict(resolved) if enabled else None)
    if enabled:
        logger.info("Prefix resume: %s", resolved)


@dataclass
class TailEntry:
    slot: int
    end: int
    block_hash: Any
    owner: str
    generation: int
    last_use: int


@dataclass
class TailDirectory:
    """The scheduler's record of what each of the model's tail slots holds."""

    slots: int
    entries: dict[int, TailEntry] = field(default_factory=dict)
    clock: int = 0
    generation: int = 0

    def find(self, block_hashes, limit: int, granularity: int, block_size: int):
        """Entries on the request's hash chain at or below ``limit``, largest first."""
        by_point = {(e.block_hash, e.end): e for e in self.entries.values()}
        out = []
        end = limit // granularity * granularity
        while end >= granularity:
            index = end // block_size - 1
            if index < len(block_hashes):
                entry = by_point.get((block_hashes[index], end))
                if entry is not None:
                    out.append(entry)
            end -= granularity
        return out

    def lookup(self, block_hash, end: int) -> TailEntry | None:
        """The entry saved at exactly this hash-chain point, if any."""
        for entry in self.entries.values():
            if entry.end == end and entry.block_hash == block_hash:
                return entry
        return None

    def touch(self, slot: int) -> None:
        self.clock += 1
        self.entries[slot].last_use = self.clock

    def choose_slot(self, owner: str, pinned: set[int]) -> int | None:
        own = sorted(
            (e for e in self.entries.values() if e.owner == owner),
            key=lambda e: e.end,
        )
        if len(own) >= 2:
            victim = next((e.slot for e in own if e.slot not in pinned), None)
            return victim
        free = [
            s for s in range(self.slots) if s not in self.entries and s not in pinned
        ]
        if free:
            return free[0]
        lru = sorted(
            (e for e in self.entries.values() if e.slot not in pinned),
            key=lambda e: e.last_use,
        )
        return lru[0].slot if lru else None

    def store(self, slot: int, end: int, block_hash, owner: str) -> TailEntry:
        self.generation += 1
        self.clock += 1
        entry = TailEntry(slot, end, block_hash, owner, self.generation, self.clock)
        self.entries[slot] = entry
        return entry


class PrefixResume:
    """Scheduler-side owner of the tail directory (see the module docstring)."""

    def __init__(self, config: PrefixResumeConfig, kv_cache_manager: KVCacheManager):
        self.config = config
        self.kv = kv_cache_manager
        self.block_size = self._hash_block_size(kv_cache_manager)
        if config.granularity % self.block_size:
            raise ValueError(
                f"prefix_resume granularity {config.granularity} is not a multiple "
                f"of the {self.block_size}-token prefix-cache hash block"
            )
        self.directory = TailDirectory(config.tail_slots)
        self.canonical: dict[str, bool] = {}
        self.pending_restores: dict[str, tuple[int, int, int]] = {}
        self.counters = {
            "capped": 0,
            "cold": 0,
            "restores": 0,
            "stores": 0,
            "dedups": 0,
            "relookups": 0,
        }
        if config.prompt_only and kv_cache_manager.enable_caching:
            self._install_prompt_only(kv_cache_manager)

    @staticmethod
    def _hash_block_size(kv: KVCacheManager) -> int:
        """The token size of one entry of ``Request.block_hashes``."""
        return int(kv.coordinator.block_pool.hash_block_size)

    @staticmethod
    def _install_prompt_only(kv: KVCacheManager) -> None:
        coordinator = kv.coordinator
        cache_blocks = coordinator.cache_blocks

        def cache_prompt_blocks(request, num_computed_tokens: int) -> None:
            return cache_blocks(
                request, min(num_computed_tokens, request.num_prompt_tokens)
            )

        coordinator.cache_blocks = cache_prompt_blocks

    # ------------------------------------------------------------------ cap

    def cap_hit(self, request: Request, hit):
        """Cap vLLM's local hit ``(blocks, h, boundary, diverged)`` to a saved tail."""
        blocks, h, boundary, diverged = hit
        if h <= 0:
            return hit
        limit = min(
            h, request.num_tokens - self.config.min_suffix, request.num_prompt_tokens
        )
        for entry in self.directory.find(
            request.block_hashes, limit, self.config.granularity, self.block_size
        ):
            if entry.end == h:
                capped = (blocks, h, boundary, diverged)
            else:
                computed, num, uncached = self.kv.coordinator.find_longest_cache_hit(
                    request.block_hashes, entry.end
                )
                self.counters["relookups"] += 1
                if num != entry.end:
                    continue
                capped = (
                    self.kv.create_kv_cache_blocks(computed),
                    num,
                    num + uncached if uncached else 0,
                    False,
                )
            self.pending_restores[request.request_id] = (
                entry.slot,
                entry.end,
                entry.generation,
            )
            self.counters["capped"] += int(entry.end < h)
            return capped
        self.pending_restores.pop(request.request_id, None)
        self.counters["cold"] += 1
        return self.kv.empty_kv_cache_blocks, 0, 0, False

    # ------------------------------------------------------------------ orders

    def finalize(
        self, scheduler_output: SchedulerOutput, requests: dict, is_decode: bool
    ):
        """The step's store and restore orders, ``{req_id: {"restore": (slot, end, gen),
        "store": (slot, end, gen)}}``; updates the directory."""
        orders: dict[str, dict[str, tuple[int, int, int]]] = {}
        scheduled = scheduler_output.num_scheduled_tokens
        first = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        first |= set(
            getattr(scheduler_output.scheduled_cached_reqs, "resumed_req_ids", ()) or ()
        )
        pinned: set[int] = set()
        for req_id, (slot, end, generation) in self.pending_restores.items():
            entry = self.directory.entries.get(slot)
            if req_id not in scheduled or req_id not in first:
                continue
            if entry is None or entry.generation != generation:
                raise RuntimeError(
                    f"Tail slot {slot} changed between the hit lookup of {req_id} "
                    "and its scheduling"
                )
            orders[req_id] = {"restore": (slot, end, generation)}
            pinned.add(slot)
            self.directory.touch(slot)
            self.counters["restores"] += 1
        self.pending_restores.clear()
        if is_decode:
            return orders
        g = self.config.granularity
        for req_id, n in scheduled.items():
            request = requests.get(req_id)
            if request is None:
                continue
            end = request.num_computed_tokens
            start = end - n
            fresh = req_id in first
            canonical = start % g == 0 and (fresh or self.canonical.get(req_id, False))
            self.canonical[req_id] = canonical
            if not canonical or end % g or end < g or end > request.num_prompt_tokens:
                continue
            block_hash = request.block_hashes[end // self.block_size - 1]
            same = self.directory.lookup(block_hash, end)
            if same is not None:
                # Canonical-only stores make tails at one hash-chain point
                # identical, so a saved one is refreshed, not written again.
                self.directory.touch(same.slot)
                self.counters["dedups"] += 1
                continue
            slot = self.directory.choose_slot(req_id, pinned)
            if slot is None:
                continue
            entry = self.directory.store(slot, end, block_hash, req_id)
            orders.setdefault(req_id, {})["store"] = (slot, end, entry.generation)
            pinned.add(slot)
            self.counters["stores"] += 1
        if len(self.canonical) > 4 * max(1, len(requests)):
            self.canonical = {k: v for k, v in self.canonical.items() if k in requests}
        return orders


def get_tail_orders(scheduler_output: SchedulerOutput) -> dict:
    return dict(getattr(scheduler_output, TAIL_ORDERS_ATTR, {}))


def tail_orders_for_rows(orders: dict | None, row_req_ids: list[str] | None):
    """Per-row ``tail_orders`` for ``prefill_forward``: ``None`` or a dict per row."""
    if not orders:
        return None
    if row_req_ids is None:
        raise RuntimeError(f"Tail orders {orders} for a step without request ids")
    rows = [orders.get(req_id) for req_id in row_req_ids]
    missing = set(orders) - set(row_req_ids)
    if missing:
        raise RuntimeError(
            f"Tail orders for requests {sorted(missing)} outside the step's rows"
        )
    return rows


def prefix_resume_kwargs(config: dict | None, model_input, *, prefill: bool) -> dict:
    """The model call's prefix-resume kwargs: ``req_ids`` (if the model asked for
    them) and, on a prefill, ``tail_orders`` (if any row carries one)."""
    orders = getattr(model_input, "prefix_tail_orders", None) if prefill else None
    if config is None:
        if orders is not None:
            raise RuntimeError(
                "Tail orders reached a model that declares no prefix_resume"
            )
        return {}
    kwargs = {}
    if config["req_ids"]:
        if model_input.row_req_ids is None:
            raise RuntimeError("A prefix-resume model needs the rows' request ids")
        kwargs["req_ids"] = list(model_input.row_req_ids)
    if orders is not None:
        kwargs["tail_orders"] = list(orders)
    return kwargs
