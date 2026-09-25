# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for the P/D payload wire format (``tt_mooncake_connector``).

``pack_payload`` lays one request's KV pairs and its GDN snapshot into one uint8
buffer that Mooncake ships as-is; ``unpack_payload`` hands the decoder views into
that buffer. The GDN snapshot is the device-major pair the model's
``_snapshot_gdn_scratch_host`` produces, ``rec [n_dev, L, Nv, Dk, Dv]`` and
``taps [n_dev, L, K, C]``, carried as the two entries ``gdn.rec`` / ``gdn.taps``.
No device execution: the connector's packing is plain torch.
"""

import pytest
import torch

from vllm_tt_plugin.kv_connector.tt_mooncake_connector import (
    pack_payload,
    payload_nbytes,
    unpack_payload,
)

N_DEV, L, K = 2, 3, 4
NV, DK, DV, C = 2, 8, 8, 16
N_ATTN, N_BLOCKS, BLOCK, HEADS, HEAD_DIM = 2, 3, 4, 2, 8


def _kv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(N_BLOCKS, BLOCK, HEADS, HEAD_DIM, generator=g).to(
                torch.bfloat16
            ),
            torch.randn(N_BLOCKS, BLOCK, HEADS, HEAD_DIM, generator=g).to(
                torch.bfloat16
            ),
        )
        for _ in range(N_ATTN)
    ]


def _snapshot(seed=0):
    g = torch.Generator().manual_seed(seed)
    rec = torch.randn(N_DEV, L, NV, DK, DV, generator=g, dtype=torch.float32)
    taps = torch.randn(N_DEV, L, K, C, generator=g).to(torch.bfloat16)
    return rec, taps


def _inside(view: torch.Tensor, buf: torch.Tensor) -> bool:
    lo = buf.data_ptr()
    hi = lo + buf.numel() * buf.element_size()
    return (
        lo
        <= view.data_ptr()
        < view.data_ptr() + view.numel() * view.element_size()
        <= hi
    )


def test_round_trip_returns_views_into_the_buffer():
    kv, (rec, taps) = _kv(), _snapshot()
    num_tokens = N_BLOCKS * BLOCK - 1

    buf, header = pack_payload(kv, rec, taps, num_tokens, N_BLOCKS)

    assert header["nbytes"] == payload_nbytes(kv, rec, taps)
    assert header["num_tokens"] == num_tokens
    assert header["n_blocks"] == N_BLOCKS
    assert header["n_attn_layers"] == N_ATTN
    assert header["n_gdn_layers"] == L
    assert header["n_conv"] == K
    assert buf.dtype == torch.uint8 and buf.numel() >= header["nbytes"]
    names = [e["name"] for e in header["tensors"]]
    assert names[-2:] == ["gdn.rec", "gdn.taps"]

    kv2, rec2, taps2 = unpack_payload(buf, header)

    assert len(kv2) == N_ATTN
    for (k, v), (k2, v2) in zip(kv, kv2):
        assert torch.equal(k, k2) and torch.equal(v, v2)
        assert k2.dtype == torch.bfloat16 and _inside(k2, buf) and _inside(v2, buf)
    assert rec2.shape == rec.shape and rec2.dtype == torch.float32
    assert taps2.shape == taps.shape and taps2.dtype == torch.bfloat16
    assert torch.equal(rec, rec2) and torch.equal(taps, taps2)
    # views, not copies: the decoder uploads straight out of the receive buffer
    assert _inside(rec2, buf) and _inside(taps2, buf)


def test_pack_into_pooled_buffer_writes_in_place():
    kv, (rec, taps) = _kv(1), _snapshot(1)
    nbytes = payload_nbytes(kv, rec, taps)
    pooled = torch.zeros(nbytes + 4096, dtype=torch.uint8)

    buf, header = pack_payload(kv, rec, taps, 5, N_BLOCKS, out=pooled)

    assert buf is pooled and header["nbytes"] == nbytes
    _, rec2, taps2 = unpack_payload(pooled, header)
    assert torch.equal(rec, rec2) and torch.equal(taps, taps2)
    # the tail past the payload is untouched
    assert int(pooled[nbytes:].abs().sum()) == 0


def test_pack_into_too_small_pooled_buffer_raises():
    kv, (rec, taps) = _kv(), _snapshot()
    nbytes = payload_nbytes(kv, rec, taps)
    with pytest.raises(ValueError, match="pooled buffer"):
        pack_payload(
            kv, rec, taps, 5, N_BLOCKS, out=torch.empty(nbytes - 1, dtype=torch.uint8)
        )


def test_per_layer_list_snapshot_is_rejected():
    kv, (rec, taps) = _kv(), _snapshot()
    rec_list = [rec[:, li] for li in range(L)]
    taps_list = [[taps[:, li, m].unsqueeze(1) for m in range(K)] for li in range(L)]
    with pytest.raises(TypeError, match="tensor pair"):
        pack_payload(kv, rec_list, taps_list, 5, N_BLOCKS)
    with pytest.raises(TypeError, match="tensor pair"):
        payload_nbytes(kv, rec_list, taps_list)


def test_wrong_rank_snapshot_is_rejected():
    kv, (rec, taps) = _kv(), _snapshot()
    with pytest.raises(ValueError, match=r"expected rec \[n_dev, L, Nv, Dk, Dv\]"):
        pack_payload(kv, rec[0], taps, 5, N_BLOCKS)
    with pytest.raises(ValueError, match=r"taps \[n_dev, L, K, C\]"):
        pack_payload(kv, rec, taps.reshape(N_DEV, L * K, C), 5, N_BLOCKS)


def test_rec_taps_device_layer_mismatch_is_rejected():
    kv, (rec, taps) = _kv(), _snapshot()
    with pytest.raises(ValueError, match=r"disagree on \[n_dev, L\]"):
        pack_payload(kv, rec, taps[:, : L - 1], 5, N_BLOCKS)
    with pytest.raises(ValueError, match=r"disagree on \[n_dev, L\]"):
        payload_nbytes(kv, rec[:1], taps)


def test_legacy_per_layer_payload_is_rejected_on_unpack():
    """A producer still shipping ``gdn.<li>.rec`` / ``gdn.<li>.conv.<m>`` entries
    must fail loudly on a device-major consumer instead of decoding garbage."""
    kv, (rec, taps) = _kv(), _snapshot()
    buf, header = pack_payload(kv, rec, taps, 5, N_BLOCKS)
    legacy = dict(header)
    legacy["tensors"] = [
        e for e in header["tensors"] if not e["name"].startswith("gdn.")
    ] + [
        dict(e, name="gdn.0.rec" if e["name"] == "gdn.rec" else "gdn.0.conv.0")
        for e in header["tensors"]
        if e["name"].startswith("gdn.")
    ]
    with pytest.raises(ValueError, match="gdn.0.rec"):
        unpack_payload(buf, legacy)


# ---- version 2: the MTP drafter's KV layer(s) + hidden row ---------------------------


def test_v2_mtp_layer_and_hidden_row_round_trip():
    """The last ``n_mtp_layers`` pairs travel as ``mtp.kv.<j>`` (outside
    ``n_attn_layers``),
    the hidden row as ``mtp.hidden``; ``unpack_payload`` appends the MTP pairs after the
    main ones and ``unpack_mtp_hidden`` hands back the row (a view)."""
    from vllm_tt_plugin.kv_connector.tt_mooncake_connector import (
        payload_mtp_layers,
        unpack_mtp_hidden,
    )

    kv, (rec, taps) = _kv(), _snapshot()
    g = torch.Generator().manual_seed(3)
    mtp_pair = (
        torch.randn(N_BLOCKS, BLOCK, HEADS, HEAD_DIM, generator=g).to(torch.bfloat16),
        torch.randn(N_BLOCKS, BLOCK, HEADS, HEAD_DIM, generator=g).to(torch.bfloat16),
    )
    hidden = torch.randn(40, generator=g).to(torch.bfloat16)
    kv17 = kv + [mtp_pair]

    buf, header = pack_payload(
        kv17, rec, taps, 7, N_BLOCKS, mtp_hidden=hidden, n_mtp_layers=1
    )

    assert header["version"] == 2
    assert (
        header["n_attn_layers"] == N_ATTN
    )  # main layers only: a v1 consumer still decodes
    assert header["mtp"] == {"n_layers": 1, "hidden": True}
    assert header["nbytes"] == payload_nbytes(kv17, rec, taps, hidden)
    names = [e["name"] for e in header["tensors"]]
    assert names == [f"kv.{li}.{x}" for li in range(N_ATTN) for x in "kv"] + [
        "mtp.kv.0.k",
        "mtp.kv.0.v",
        "gdn.rec",
        "gdn.taps",
        "mtp.hidden",
    ]
    assert payload_mtp_layers(header) == 1

    kv2, rec2, taps2 = unpack_payload(buf, header)
    assert len(kv2) == N_ATTN + 1
    for (k, v), (k2, v2) in zip(kv17, kv2):
        assert torch.equal(k, k2) and torch.equal(v, v2)
    assert torch.equal(rec, rec2) and torch.equal(taps, taps2)
    h2 = unpack_mtp_hidden(buf, header)
    assert torch.equal(h2, hidden) and h2.dtype == torch.bfloat16 and _inside(h2, buf)


def test_v2_payload_without_mtp_has_no_mtp_field_and_reads_like_v1():
    from vllm_tt_plugin.kv_connector.tt_mooncake_connector import (
        payload_mtp_layers,
        unpack_mtp_hidden,
    )

    kv, (rec, taps) = _kv(), _snapshot()
    buf, header = pack_payload(kv, rec, taps, 7, N_BLOCKS)
    assert "mtp" not in header and header["n_attn_layers"] == N_ATTN
    assert payload_mtp_layers(header) == 0 and unpack_mtp_hidden(buf, header) is None
    kv2, _, _ = unpack_payload(buf, header)
    assert len(kv2) == N_ATTN


def test_v1_consumer_view_of_a_v2_payload_sees_the_main_layers_only():
    """A consumer that predates version 2 reads ``kv.<li>`` for li < n_attn_layers and
    the GDN pair, ignoring unknown tensor names: reconstruct exactly that from a v2
    header."""
    kv, (rec, taps) = _kv(), _snapshot()
    hidden = torch.zeros(8, dtype=torch.bfloat16)
    buf, header = pack_payload(
        kv + [kv[0]], rec, taps, 7, N_BLOCKS, mtp_hidden=hidden, n_mtp_layers=1
    )
    by_name = {
        e["name"]: buf[e["offset"] : e["offset"] + e["nbytes"]]
        .view(getattr(torch, e["dtype"]))
        .view(*e["shape"])
        for e in header["tensors"]
    }
    old_kv = [
        (by_name[f"kv.{li}.k"], by_name[f"kv.{li}.v"])
        for li in range(header["n_attn_layers"])
    ]
    assert len(old_kv) == N_ATTN
    for (k, v), (k2, v2) in zip(kv, old_kv):
        assert torch.equal(k, k2) and torch.equal(v, v2)
    assert torch.equal(by_name["gdn.rec"], rec) and torch.equal(
        by_name["gdn.taps"], taps
    )


def test_v2_rejects_bad_mtp_arguments():
    kv, (rec, taps) = _kv(), _snapshot()
    with pytest.raises(ValueError, match="n_mtp_layers"):
        pack_payload(kv, rec, taps, 7, N_BLOCKS, n_mtp_layers=N_ATTN + 1)
    with pytest.raises(ValueError, match=r"\[dim\] row"):
        pack_payload(kv, rec, taps, 7, N_BLOCKS, mtp_hidden=torch.zeros(2, 3))
