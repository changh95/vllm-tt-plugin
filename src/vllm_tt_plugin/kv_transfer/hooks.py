# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Model hook protocol ``TTKVTransferable`` (PHASE2_DESIGN.md 5.1) + the default
paged-KV-only mixin for attention-only TT models.

The worker (``worker.py``) calls exactly these names. The Qwen implementation
lives in the model tree (``M/kv_transfer.py:Qwen36KVTransfer``); this module
only defines the contract, a pure-Python manifest builder both implementations
can share, and ``DefaultKVTransferable``.

No ``ttnn`` at module level: ``DefaultKVTransferable`` imports it lazily.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from vllm.utils.math_utils import cdiv

from vllm_tt_plugin.kv_transfer.metadata import LAYOUT_VERSION, Manifest, PartSpec

# Bytes of one tile (32 x 32 elements) per ttnn dtype name.
_TILE_NBYTES = {"bfloat8_b": 1088, "bfloat4_b": 576, "bfloat16": 2048, "float32": 4096}
_ELEM_NBYTES = {"bfloat16": 2, "float32": 4, "int32": 4, "uint32": 4}


def tile_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Bytes of a TILE-layout tensor of ``shape`` (last two dims tile-aligned)."""
    numel = 1
    for d in shape:
        numel *= int(d)
    if dtype not in _TILE_NBYTES:
        raise ValueError(f"unknown tile dtype {dtype!r}")
    if numel % 1024 != 0:
        raise ValueError(f"{shape} is not tile-aligned")
    return (numel // 1024) * _TILE_NBYTES[dtype]


def row_major_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    numel = 1
    for d in shape:
        numel *= int(d)
    return numel * _ELEM_NBYTES[dtype]


def build_manifest(
    num_tokens: int,
    block_ids: list[int],
    *,
    block_size: int = 64,
    chunk_tokens: int = 2048,
    num_attn_layers: int = 16,
    num_gdn_layers: int = 48,
    kv_heads: int = 4,
    head_dim: int = 256,
    kv_dtype: str = "bfloat8_b",
    rec_shape: tuple[int, int, int] = (48, 128, 128),
    rec_dtype: str = "float32",
    taps_shape: tuple[int, int] = (4, 10240),
    taps_dtype: str = "bfloat16",
    model_sig: str = "",
    prompt_hash: str = "",
) -> Manifest:
    """The ONE on-the-wire layout (4.1): block-major K/V chunks of
    ``chunk_tokens // block_size`` blocks, one fp32 rec row and one host tap
    row-set per GDN layer. ``chunk_offsets[c] = c * chunk_nbytes``.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    nblk = cdiv(num_tokens, block_size)
    if len(block_ids) < nblk:
        raise ValueError(
            f"{len(block_ids)} blocks for {num_tokens} tokens (need {nblk})"
        )
    bpc = max(1, chunk_tokens // block_size)
    nchunks = cdiv(nblk, bpc)
    kv_shape = (bpc, kv_heads, block_size, head_dim)
    kv_chunk_nbytes = tile_nbytes(kv_shape, kv_dtype)
    parts: list[PartSpec] = []
    for i in range(num_attn_layers):
        for tag in ("k", "v"):
            parts.append(
                PartSpec(
                    name=f"kv.L{i}.{tag}",
                    kind="kv_blocks",
                    shape=kv_shape,
                    dtype=kv_dtype,
                    layout="TILE",
                    nchunks=nchunks,
                    chunk_nbytes=kv_chunk_nbytes,
                    chunk_offsets=[c * kv_chunk_nbytes for c in range(nchunks)],
                    nbytes=nchunks * kv_chunk_nbytes,
                )
            )
    rec_full = (1, *rec_shape)
    rec_nbytes = tile_nbytes(rec_full, rec_dtype)
    taps_nbytes = row_major_nbytes(taps_shape, taps_dtype)
    for j in range(num_gdn_layers):
        parts.append(
            PartSpec(
                name=f"gdn.L{j}.rec",
                kind="gdn_rec",
                shape=rec_full,
                dtype=rec_dtype,
                layout="TILE",
                nchunks=1,
                chunk_nbytes=rec_nbytes,
                chunk_offsets=[0],
                nbytes=rec_nbytes,
            )
        )
        parts.append(
            PartSpec(
                name=f"gdn.L{j}.taps",
                kind="gdn_taps",
                shape=taps_shape,
                dtype=taps_dtype,
                layout="ROW_MAJOR",
                nchunks=1,
                chunk_nbytes=taps_nbytes,
                chunk_offsets=[0],
                nbytes=taps_nbytes,
            )
        )
    return Manifest(
        layout_version=LAYOUT_VERSION,
        model_sig=model_sig,
        num_tokens=num_tokens,
        nblk=nblk,
        block_size=block_size,
        chunk_tokens=bpc * block_size,
        kv_dtype=kv_dtype,
        rec_dtype=rec_dtype,
        parts=parts,
        prompt_hash=prompt_hash,
    )


@runtime_checkable
class TTKVTransferable(Protocol):
    """Implemented by the vLLM adapter class (``Qwen36ForCausalLM`` delegates to
    ``M/kv_transfer.py:Qwen36KVTransfer``). Every device call is eager, cq 0,
    engine thread, in place (I5)."""

    kv_transfer_hybrid_state: bool

    def describe_request_state(
        self, num_tokens: int, block_ids: list[int]
    ) -> Manifest: ...

    def warmup_kv_transfer(
        self, *, role: str, mode: str, chunk_tokens: int, slots: range
    ) -> None: ...

    def export_request_state(
        self, block_ids: list[int], num_tokens: int, slot: int, sinks: Mapping[str, Any]
    ) -> None: ...

    def import_kv_blocks(
        self,
        sources: Mapping[str, Any],
        block_ids: list[int],
        num_tokens: int,
        *,
        chunk_range: slice | None = None,
    ) -> int:
        """K/V chunks only (request-private; step-END hook). Returns chunks done."""
        ...

    def validate_gdn_parts(self, sources: Mapping[str, Any]) -> None:
        """Host-only checks; raises -> ``fail(job)`` BEFORE ``finished_recving``."""
        ...

    def install_gdn_state(self, sources: Mapping[str, Any], slot: int) -> None:
        """rec + taps into ``slot``; JOIN-step step-BEGIN hook only (I11)."""
        ...

    def import_request_state(
        self,
        sources: Mapping[str, Any],
        block_ids: list[int],
        num_tokens: int,
        slot: int,
        *,
        chunk_range: slice | None = None,
    ) -> None:
        """= import_kv_blocks (all chunks) + validate_gdn_parts + install_gdn_state;
        tests and TT_PD_VERIFY_IMPORT only."""
        ...


HOOK_METHODS = (
    "describe_request_state",
    "warmup_kv_transfer",
    "export_request_state",
    "import_kv_blocks",
    "validate_gdn_parts",
    "install_gdn_state",
)


@runtime_checkable
class TTKVMirrorSeams(Protocol):
    """OPTIONAL producer seams (p1d1_opt lane B; the Qwen hook implements them, the
    worker probes them with ``getattr`` and falls back to the step-end gather):

    * ``begin_export(block_ids, num_tokens, sinks) -> bool`` at step BEGIN, before the
      request's prefill: the model mirrors every prefill chunk's K/V into ``sinks``
      as it goes (True = mirror active, False = ``export_request_state`` gathers all).
    * ``end_export(sinks=None)`` closes that window (``sinks``: only if it is the open
      one; a stale close must not drop a newer window).
    * ``set_kv_transfer_pump(fn, wants=None)``: ``fn()`` (the transport pump) runs on
      the engine thread at every non-final prefill chunk boundary; ``wants()`` is a
      host-only check whether it could send anything now (gates the per-chunk device
      sync).  See ``M/kv_transfer.py`` for the model side."""

    def begin_export(
        self, block_ids: list[int], num_tokens: int, sinks: Mapping[str, Any]
    ) -> bool: ...

    def end_export(self, sinks: Mapping[str, Any] | None = None) -> None: ...

    def set_kv_transfer_pump(self, fn: Any, wants: Any = None) -> None: ...


MIRROR_METHODS = ("begin_export", "end_export", "set_kv_transfer_pump")


def implements_kv_transfer(model: Any) -> bool:
    return all(callable(getattr(model, m, None)) for m in HOOK_METHODS)


def implements_kv_mirror(model: Any) -> bool:
    """Every optional mirror seam present (informational; the worker probes each)."""
    return all(callable(getattr(model, m, None)) for m in MIRROR_METHODS)


class DefaultKVTransferable:
    """K/V parts only over ``runner.kv_caches`` (list of ``[k, v]`` per layer,
    ``P/model_runner.py:506``) for attention-only TT models. ``slot`` is
    ignored; the GDN hooks are no-ops. Follows 5.3/5.4 (dumpfile chunk build =
    unit-block ``slice`` x bpc + one concat; import = staging + ``paged_fill_cache``).
    Untested on device in Phase 2a (the Qwen hook is the target).
    """

    kv_transfer_hybrid_state = False

    def __init__(
        self,
        kv_caches: list,
        *,
        mesh_device: Any = None,
        block_size: int = 64,
        chunk_tokens: int = 2048,
        pad_block: int | None = None,
        model_sig: str = "",
    ):
        self.kv_caches = list(kv_caches)
        self.mesh_device = mesh_device
        self.block_size = block_size
        self.chunk_tokens = chunk_tokens
        self._pad_block = pad_block
        self.model_sig = model_sig
        self._kv_staging: list[Any] = []

    # -- geometry ------------------------------------------------------------
    def _geometry(self) -> tuple[int, int, int, str]:
        k0 = self.kv_caches[0][0]
        shape = tuple(
            int(d) for d in k0.shape
        )  # [num_blocks(+1), heads, block_size, head_dim]
        dtype = str(k0.dtype).split(".")[-1].lower()
        dtype = {
            "bfloat8_b": "bfloat8_b",
            "bfloat16": "bfloat16",
            "float32": "float32",
        }.get(dtype, dtype)
        return shape[0], shape[1], shape[3], dtype

    def describe_request_state(self, num_tokens: int, block_ids: list[int]) -> Manifest:
        _, heads, head_dim, dtype = self._geometry()
        return build_manifest(
            num_tokens,
            block_ids,
            block_size=self.block_size,
            chunk_tokens=self.chunk_tokens,
            num_attn_layers=len(self.kv_caches),
            num_gdn_layers=0,
            kv_heads=heads,
            head_dim=head_dim,
            kv_dtype=dtype,
            model_sig=self.model_sig,
        )

    def warmup_kv_transfer(
        self, *, role: str, mode: str, chunk_tokens: int, slots: range
    ) -> None:
        import ttnn

        if not self._kv_staging and self.mesh_device is not None and "consumer" in role:
            nb, heads, head_dim, _ = self._geometry()
            k0 = self.kv_caches[0][0]
            bpc = chunk_tokens // self.block_size
            for _ in range(2):
                self._kv_staging.append(
                    ttnn.zeros(
                        (1, heads, bpc * self.block_size, head_dim),
                        dtype=k0.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                    )
                )
        if self.mesh_device is not None:
            ttnn.synchronize_device(self.mesh_device)

    def _pad(self) -> int:
        if self._pad_block is not None:
            return self._pad_block
        return self._geometry()[0] - 1

    def export_request_state(self, block_ids, num_tokens, slot, sinks) -> None:
        import ttnn

        nblk = cdiv(num_tokens, self.block_size)
        ids = list(block_ids[:nblk])
        bpc = self.chunk_tokens // self.block_size
        _, heads, head_dim, dtype = self._geometry()
        blk_nbytes = tile_nbytes((1, heads, self.block_size, head_dim), dtype)
        tmp: list[Any] = []
        for li, (k, v) in enumerate(self.kv_caches):
            for tag, t in (("k", k), ("v", v)):
                sink = sinks[f"kv.L{li}.{tag}"]
                for c in range(cdiv(nblk, bpc)):
                    chunk_ids = ids[bpc * c : bpc * c + bpc]
                    if getattr(sink, "supports_regions", False):
                        for k0, b0, n in _contiguous_runs(chunk_ids):
                            sink.write_region_from_device(
                                t,
                                b0 * blk_nbytes,
                                n * blk_nbytes,
                                chunk=c,
                                dst_offset_bytes=k0 * blk_nbytes,
                                blocking=False,
                            )
                    else:
                        rows = chunk_ids + [self._pad()] * (bpc - len(chunk_ids))
                        parts = [
                            ttnn.slice(
                                t,
                                (b, 0, 0, 0),
                                (b + 1, heads, self.block_size, head_dim),
                            )
                            for b in rows
                        ]
                        blk = ttnn.concat(parts, dim=0)
                        sink.write_from_device(blk, chunk=c, blocking=False)
                        tmp += parts
                        tmp.append(blk)
        if self.mesh_device is not None:
            ttnn.synchronize_device(self.mesh_device)
        for t in tmp:
            ttnn.deallocate(t)

    def import_kv_blocks(
        self, sources, block_ids, num_tokens, *, chunk_range=None
    ) -> int:
        import torch
        import ttnn

        nblk = cdiv(num_tokens, self.block_size)
        bpc = self.chunk_tokens // self.block_size
        nchunks = cdiv(nblk, bpc)
        _, heads, head_dim, _ = self._geometry()
        dev = self.mesh_device
        done = 0
        chunks = range(nchunks) if chunk_range is None else range(nchunks)[chunk_range]
        for c in chunks:
            ids = list(block_ids[bpc * c : bpc * c + bpc])
            pt_rows = ids + [self._pad()] * (bpc - len(ids))
            pt = ttnn.from_torch(
                torch.tensor([pt_rows], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
            )
            for li, (k, v) in enumerate(self.kv_caches):
                for tag, t in (("k", k), ("v", v)):
                    src = sources[f"kv.L{li}.{tag}"].chunk(c)
                    st = self._kv_staging[c % 2]
                    if getattr(src, "is_head_major", False):
                        src.read_into_device(st)
                    else:
                        blk = src.read_device(dev)
                        hm = ttnn.reshape(
                            ttnn.permute(blk, (1, 0, 2, 3)),
                            (1, heads, bpc * self.block_size, head_dim),
                        )
                        ttnn.copy(hm, st)
                        ttnn.deallocate(hm)
                        ttnn.deallocate(blk)
                    ttnn.experimental.paged_fill_cache(t, st, pt, batch_idx=0)
            ttnn.deallocate(pt)
            done += 1
        return done

    def validate_gdn_parts(self, sources) -> None:
        return None

    def install_gdn_state(self, sources, slot: int) -> None:
        return None

    def import_request_state(
        self, sources, block_ids, num_tokens, slot, *, chunk_range=None
    ):
        self.import_kv_blocks(sources, block_ids, num_tokens, chunk_range=chunk_range)
        self.validate_gdn_parts(sources)
        bpc = self.chunk_tokens // self.block_size
        nchunks = cdiv(cdiv(num_tokens, self.block_size), bpc)
        if (
            chunk_range is None
            or chunk_range.stop is None
            or chunk_range.stop >= nchunks
        ):
            self.install_gdn_state(sources, slot)


def _contiguous_runs(block_ids: list[int]) -> list[tuple[int, int, int]]:
    """``(index in chunk, first block, run length)`` for each run of consecutive ids."""
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(block_ids):
        j = i
        while j + 1 < len(block_ids) and block_ids[j + 1] == block_ids[j] + 1:
            j += 1
        runs.append((i, block_ids[i], j - i + 1))
        i = j + 1
    return runs
