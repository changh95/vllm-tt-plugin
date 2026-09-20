# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Unit tests for the P/D disaggregation proxy (PHASE2_DESIGN.md 2.4, step 8).

Two ``httpx.MockTransport`` backends stand in for the prefill node (P) and the
decode node (D); the FastAPI app is driven through ``httpx.ASGITransport``.
No device, no vLLM import.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from vllm_tt_plugin.kv_transfer import pd_proxy
from vllm_tt_plugin.kv_transfer.pd_proxy import (
    D_RETRY_SUFFIX,
    MIN_REMOTE_TOKENS_FLOOR,
    P_LEG_SUFFIX,
    PREFILL_KV_TRANSFER_PARAMS,
    PDProxy,
    PDProxyConfig,
    build_prefill_body,
    classify_sse_event,
    create_app,
)

P_URL = "http://prefill.test"
D_URL = "http://decode.test"

P_PARAMS = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_engine_id": "p0",
    "remote_block_ids": [3, 4, 5],
    "remote_num_tokens": 191,
    "remote_prompt_hash": "ab" * 16,
    "remote_transport": {
        "kind": "shm",
        "mode": "dumpfile",
        "root": "/dev/shm/tt_pd/p0",
    },
    "xfer_id": "x" * 32,
    "remote_request_id": "chatcmpl-rid-p",
    "lease_expiry": 1.0e9,
}

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 20
)  # ~900 chars -> ~225 tokens


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Mock backends
# --------------------------------------------------------------------------- #


class Backend:
    """Records every request and answers from a programmable handler."""

    def __init__(self, name: str):
        self.name = name
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.handler: (
            Callable[[httpx.Request, int], Awaitable[httpx.Response]] | None
        ) = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else {}
        self.bodies.append(body)
        if self.handler is None:
            raise AssertionError(f"{self.name}: unexpected request {request.url}")
        return await self.handler(request, len(self.requests) - 1)

    def client(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self), base_url=base_url)

    @property
    def header_ids(self) -> list[str]:
        return [r.headers.get("X-Request-Id") for r in self.requests]


def p_ok(params=P_PARAMS, delay: float = 0.0):
    async def handler(request: httpx.Request, i: int) -> httpx.Response:
        if delay:
            await asyncio.sleep(delay)
        payload = {
            "id": "chatcmpl-rid-p",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 191,
                "completion_tokens": 1,
                "total_tokens": 192,
            },
        }
        if params is not None:
            payload["kv_transfer_params"] = dict(params)
        return httpx.Response(200, json=payload)

    return handler


def d_json(text: str = "hello", finish_reason: str = "stop", status: int = 200):
    async def handler(request: httpx.Request, i: int) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "id": "chatcmpl-rid",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 192,
                    "completion_tokens": 3,
                    "total_tokens": 195,
                },
            },
        )

    return handler


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def role_chunk() -> bytes:
    return sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
    )


def content_chunk(text: str) -> bytes:
    return sse(
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    )


def finish_chunk(reason: str = "stop") -> bytes:
    return sse({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


def error_event() -> bytes:
    # What vLLM 0.26 emits for GenerationError (kv_load_failure_policy=fail).
    return sse(
        {
            "error": {
                "message": "Internal server error",
                "type": "InternalServerError",
                "code": 500,
            }
        }
    )


DONE = b"data: [DONE]\n\n"


class _Stream(httpx.AsyncByteStream):
    def __init__(self, parts, gaps: float | dict[int, float] = 0.0):
        self._parts = parts
        self._gaps = gaps

    async def __aiter__(self):
        for i, part in enumerate(self._parts):
            gap = self._gaps.get(i, 0.0) if isinstance(self._gaps, dict) else self._gaps
            if gap:
                await asyncio.sleep(gap)
            yield part

    async def aclose(self):
        return None


def d_stream(parts, gaps=0.0, status: int = 200):
    async def handler(request: httpx.Request, i: int) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "text/event-stream"},
            stream=_Stream(list(parts), gaps),
        )

    return handler


def d_sequence(*handlers):
    async def handler(request: httpx.Request, i: int) -> httpx.Response:
        return await handlers[min(i, len(handlers) - 1)](request, i)

    return handler


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class Harness:
    def __init__(self, **config_kw):
        cfg = dict(
            prefill_url=P_URL,
            decode_url=D_URL,
            min_remote_tokens=2,
            sse_keepalive=0.0,
            retry_buffer_timeout=2.0,
        )
        cfg.update(config_kw)
        self.config = PDProxyConfig(**cfg)
        self.p = Backend("P")
        self.d = Backend("D")
        self.proxy = PDProxy(
            self.config, client_p=self.p.client(P_URL), client_d=self.d.client(D_URL)
        )
        self.app = create_app(self.config, proxy=self.proxy)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://proxy.test"
        )

    async def post(
        self, api: str, body: dict, rid: str | None = "rid1"
    ) -> httpx.Response:
        headers = {"X-Request-Id": rid} if rid else {}
        return await self.client.post(f"/v1{api}", json=body, headers=headers)

    async def aclose(self):
        await self.client.aclose()
        await self.proxy.client_p.aclose()
        await self.proxy.client_d.aclose()

    @property
    def rid(self) -> str:
        """The backend-facing id the proxy MINTED for the (single) request the
        backends saw: never the client's ``X-Request-Id`` (``rid1``)."""
        ids = self.p.header_ids + self.d.header_ids
        assert ids, "no backend request yet"
        base = ids[0]
        for suffix in (P_LEG_SUFFIX, D_RETRY_SUFFIX):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        assert len(base) == 32 and base != "rid1", base
        return base


def chat_body(**kw) -> dict:
    body = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": LONG_TEXT},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    body.update(kw)
    return body


def completion_body(**kw) -> dict:
    body = {"model": "Qwen/Qwen3.8-27B", "prompt": LONG_TEXT, "max_tokens": 64}
    body.update(kw)
    return body


def sse_events(raw: bytes) -> list[bytes]:
    return [e + b"\n\n" for e in raw.split(b"\n\n") if e]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_min_remote_tokens_hard_floor():
    assert MIN_REMOTE_TOKENS_FLOOR == 2
    assert PDProxyConfig(min_remote_tokens=0).min_remote_tokens == 2
    assert PDProxyConfig(min_remote_tokens=1).min_remote_tokens == 2
    assert PDProxyConfig(min_remote_tokens=2).min_remote_tokens == 2
    assert PDProxyConfig(min_remote_tokens=512).min_remote_tokens == 512


def test_cli_args_map_to_config():
    args = pd_proxy.parse_args(
        [
            "--port",
            "8000",
            "--prefill-url",
            "http://127.0.0.1:8100/",
            "--decode-url",
            "http://127.0.0.1:8200",
            "--min-remote-tokens",
            "1",
            "--p-fallback",
            "local",
            "--no-d-retry",
        ]
    )
    cfg = pd_proxy.config_from_args(args)
    assert cfg.prefill_url == "http://127.0.0.1:8100"
    assert cfg.decode_url == "http://127.0.0.1:8200"
    assert cfg.min_remote_tokens == 2
    assert cfg.p_fallback == "local"
    assert cfg.d_retry is False
    assert cfg.p_timeout is None and cfg.d_timeout is None


def test_default_clients_have_unbounded_read_timeout():
    proxy = PDProxy(PDProxyConfig())
    try:
        assert proxy.client_p.timeout.read is None
        assert proxy.client_d.timeout.read is None
        assert proxy.client_p.timeout.connect == 10.0
    finally:
        run(proxy.aclose())


# --------------------------------------------------------------------------- #
# P-leg body shape (pure)
# --------------------------------------------------------------------------- #


def test_build_prefill_body_shape():
    body = chat_body(
        stream=True,
        stream_options={"include_usage": True},
        max_completion_tokens=77,
        min_tokens=5,
        logprobs=True,
        top_logprobs=3,
        response_format={"type": "json_object"},
        tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        seed=1,
        stop=["\n"],
    )
    body_p, restore = build_prefill_body(body)
    assert body_p["stream"] is False
    assert body_p["max_tokens"] == 1
    assert body_p["max_completion_tokens"] == 1
    for dropped in (
        "stream_options",
        "logprobs",
        "top_logprobs",
        "response_format",
        "tool_choice",
        "min_tokens",
    ):
        assert dropped not in body_p
    assert restore == {"min_tokens": 5}
    assert body_p["kv_transfer_params"] == PREFILL_KV_TRANSFER_PARAMS
    assert body_p["kv_transfer_params"]["do_remote_decode"] is True
    assert body_p["kv_transfer_params"]["do_remote_prefill"] is False
    # Kept: tools and every message (rendered into the prompt), sampling knobs.
    assert body_p["tools"] == body["tools"]
    assert body_p["messages"] == body["messages"]
    assert body_p["seed"] == 1 and body_p["stop"] == ["\n"]
    # The original is untouched.
    assert (
        body["stream"] is True and body["max_tokens"] == 64 and body["min_tokens"] == 5
    )


# --------------------------------------------------------------------------- #
# End-to-end (two legs) non-streaming
# --------------------------------------------------------------------------- #


def test_chat_nonstream_two_legs_ids_params_and_passthrough():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json("hello world")

    async def go():
        body = chat_body(min_tokens=3, stream_options={"include_usage": True})
        resp = await h.post("/chat/completions", body)
        return resp

    try:
        resp = run(go())
        assert resp.status_code == 200
        assert resp.headers["x-request-id"] == "rid1"
        assert resp.json()["choices"][0]["message"]["content"] == "hello world"
        # P leg
        assert len(h.p.requests) == 1
        assert h.p.requests[0].url.path == "/v1/chat/completions"
        assert h.p.header_ids == [h.rid + P_LEG_SUFFIX]
        bp = h.p.bodies[0]
        assert bp["stream"] is False and bp["max_tokens"] == 1
        assert "min_tokens" not in bp and "stream_options" not in bp
        assert bp["kv_transfer_params"] == PREFILL_KV_TRANSFER_PARAMS
        # D leg: original body + restored min_tokens + P's params, bare id.
        assert len(h.d.requests) == 1
        assert h.d.header_ids == [h.rid]
        bd = h.d.bodies[0]
        assert bd["kv_transfer_params"] == P_PARAMS
        assert bd["min_tokens"] == 3
        assert bd["max_tokens"] == 64
        assert bd["stream_options"] == {"include_usage": True}
        assert bd["messages"] == chat_body()["messages"]
    finally:
        run(h.aclose())


def test_completions_api_path_and_token_id_prompt():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()
    try:
        resp = run(h.post("/completions", completion_body(prompt=list(range(300)))))
        assert resp.status_code == 200
        assert h.p.requests[0].url.path == "/v1/completions"
        assert h.d.requests[0].url.path == "/v1/completions"
        assert h.d.bodies[0]["kv_transfer_params"] == P_PARAMS
    finally:
        run(h.aclose())


def test_generated_request_id_when_header_absent():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()
    try:
        resp = run(h.post("/chat/completions", chat_body(), rid=None))
        rid = resp.headers["x-request-id"]
        assert rid and len(rid) == 32
        assert h.p.header_ids == [rid + P_LEG_SUFFIX]
        assert h.d.header_ids == [rid]
    finally:
        run(h.aclose())


def test_client_request_id_is_echoed_never_forwarded_and_reuse_is_safe():
    """Audit minor: a client-chosen ``X-Request-Id`` used to be the engine id
    of all three backend legs, so a reused id (concurrently, or inside P's
    finished_sending window) hit ``Scheduler.add_request``'s duplicate-id
    assert on a backend. The proxy now mints the backend id and only echoes
    the client's."""
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()

    async def go():
        return await asyncio.gather(
            h.post("/chat/completions", chat_body(), rid="same"),
            h.post("/chat/completions", chat_body(), rid="same"),
        )

    try:
        r1, r2 = run(go())
        assert r1.status_code == r2.status_code == 200
        assert r1.headers["x-request-id"] == r2.headers["x-request-id"] == "same"
        d_ids = h.d.header_ids
        assert len(d_ids) == 2 and len(set(d_ids)) == 2, "distinct per request"
        assert all(len(i) == 32 and i != "same" for i in d_ids)
        assert sorted(h.p.header_ids) == sorted(i + P_LEG_SUFFIX for i in d_ids)
        assert all("same" not in i for i in d_ids + h.p.header_ids)
    finally:
        run(h.aclose())


def test_error_payloads_echo_the_client_request_id():
    h = Harness()
    h.p.handler = p_ok()

    async def boom(request, i):
        raise httpx.ConnectError("down", request=request)

    h.d.handler = boom
    try:
        resp = run(h.post("/chat/completions", chat_body(), rid="mine"))
        assert resp.status_code == 502
        assert resp.headers["x-request-id"] == "mine"
        assert resp.json()["request_id"] == "mine"
        assert h.d.header_ids == [h.rid] and h.rid != "mine"
    finally:
        run(h.aclose())


def test_client_supplied_kv_transfer_params_are_dropped():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()
    try:
        body = chat_body(
            kv_transfer_params={"do_remote_prefill": True, "xfer_id": "evil"}
        )
        resp = run(h.post("/chat/completions", body))
        assert resp.status_code == 200
        assert h.p.bodies[0]["kv_transfer_params"] == PREFILL_KV_TRANSFER_PARAMS
        assert h.d.bodies[0]["kv_transfer_params"] == P_PARAMS
    finally:
        run(h.aclose())


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_short_prompt_goes_straight_to_decode():
    h = Harness(min_remote_tokens=512)
    h.d.handler = d_json()
    try:
        body = chat_body(messages=[{"role": "user", "content": "hi"}], min_tokens=2)
        resp = run(h.post("/chat/completions", body))
        assert resp.status_code == 200
        assert h.p.requests == []
        assert h.d.header_ids == [h.rid]
        assert "kv_transfer_params" not in h.d.bodies[0]
        assert h.d.bodies[0] == body
    finally:
        run(h.aclose())


def test_long_prompt_at_threshold_goes_remote():
    # 900-char prompt / 4 chars per token = ~225 (+ 8 template) -> remote at 200.
    h = Harness(min_remote_tokens=200)
    h.p.handler = p_ok()
    h.d.handler = d_json()
    try:
        assert h.proxy.token_counter.estimate("/chat/completions", chat_body()) >= 200
        run(h.post("/chat/completions", chat_body()))
        assert len(h.p.requests) == 1 and len(h.d.requests) == 1
    finally:
        run(h.aclose())


def test_short_streaming_prompt_bypasses_prefill_without_keepalives():
    h = Harness(min_remote_tokens=512, sse_keepalive=0.01)
    h.d.handler = d_stream([role_chunk(), content_chunk("a"), finish_chunk(), DONE])
    try:
        body = chat_body(messages=[{"role": "user", "content": "hi"}], stream=True)
        resp = run(h.post("/chat/completions", body))
        assert resp.status_code == 200
        assert h.p.requests == []
        assert b"keep-alive" not in resp.content
        assert sse_events(resp.content) == [
            role_chunk(),
            content_chunk("a"),
            finish_chunk(),
            DONE,
        ]
    finally:
        run(h.aclose())


# --------------------------------------------------------------------------- #
# Streaming pass-through and keep-alives
# --------------------------------------------------------------------------- #


def test_streaming_passthrough_with_keepalives_during_prefill_leg():
    h = Harness(sse_keepalive=0.01)
    h.p.handler = p_ok(delay=0.08)
    parts = [
        role_chunk(),
        content_chunk("Hel"),
        content_chunk("lo"),
        finish_chunk(),
        DONE,
    ]
    h.d.handler = d_stream(parts)
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True, min_tokens=1)))
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-request-id"] == "rid1"
        raw = resp.content
        # Keep-alive comments precede the first data event.
        first_data = raw.index(b"data:")
        assert raw[:first_data].count(b": keep-alive\n\n") >= 2
        assert b"keep-alive" not in raw[first_data:]
        assert raw[first_data:] == b"".join(parts)
        # D got the original streaming body + params; P got stream=False.
        assert h.d.bodies[0]["stream"] is True
        assert h.d.bodies[0]["kv_transfer_params"] == P_PARAMS
        assert h.d.bodies[0]["min_tokens"] == 1
        assert h.p.bodies[0]["stream"] is False
        assert h.p.header_ids == [h.rid + "-p"] and h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_streaming_keepalive_disabled():
    h = Harness(sse_keepalive=0.0)
    h.p.handler = p_ok(delay=0.03)
    parts = [role_chunk(), content_chunk("x"), DONE]
    h.d.handler = d_stream(parts)
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == b"".join(parts)
    finally:
        run(h.aclose())


def test_streaming_chunks_split_mid_event_are_passed_verbatim():
    h = Harness()
    h.p.handler = p_ok()
    whole = role_chunk() + content_chunk("abc") + finish_chunk() + DONE
    # Split at arbitrary byte offsets, including inside a JSON payload.
    parts = [whole[:7], whole[7:40], whole[40:41], whole[41:]]
    h.d.handler = d_stream(parts)
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == whole
    finally:
        run(h.aclose())


# --------------------------------------------------------------------------- #
# Missing params / P failures
# --------------------------------------------------------------------------- #


def test_prefill_without_params_falls_back_to_local_decode():
    h = Harness()
    h.p.handler = p_ok(params=None)
    h.d.handler = d_json()
    try:
        resp = run(h.post("/chat/completions", chat_body(min_tokens=2)))
        assert resp.status_code == 200
        assert len(h.p.requests) == 1
        assert h.d.header_ids == [h.rid]
        assert "kv_transfer_params" not in h.d.bodies[0]
        assert h.d.bodies[0]["min_tokens"] == 2
    finally:
        run(h.aclose())


def test_prefill_http_error_gives_502_and_no_decode_leg():
    h = Harness()

    async def p_400(request, i):
        return httpx.Response(400, json={"error": {"message": "prompt too long"}})

    h.p.handler = p_400
    try:
        resp = run(h.post("/chat/completions", chat_body()))
        assert resp.status_code == 502
        assert resp.headers["x-request-id"] == "rid1"
        assert "prompt too long" in resp.json()["error"]["message"]
        assert h.d.requests == []
    finally:
        run(h.aclose())


def test_prefill_transport_error_gives_502():
    h = Harness()

    async def p_down(request, i):
        raise httpx.ConnectError("refused", request=request)

    h.p.handler = p_down
    try:
        resp = run(h.post("/chat/completions", chat_body()))
        assert resp.status_code == 502
        assert h.d.requests == []
    finally:
        run(h.aclose())


def test_prefill_failure_with_local_fallback_decodes_without_params():
    h = Harness(p_fallback="local")

    async def p_500(request, i):
        return httpx.Response(500, json={"error": {"message": "boom"}})

    h.p.handler = p_500
    h.d.handler = d_json()
    try:
        resp = run(h.post("/chat/completions", chat_body(min_tokens=2)))
        assert resp.status_code == 200
        assert h.d.header_ids == [h.rid]
        assert "kv_transfer_params" not in h.d.bodies[0]
        assert h.d.bodies[0]["min_tokens"] == 2
    finally:
        run(h.aclose())


def test_prefill_failure_streaming_emits_sse_error_event():
    h = Harness()

    async def p_500(request, i):
        return httpx.Response(500, text="boom")

    h.p.handler = p_500
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.status_code == 200  # SSE already started
        events = sse_events(resp.content)
        assert len(events) == 2 and events[1] == DONE
        err = json.loads(events[0][len(b"data: ") :])
        assert err["error"]["code"] == 502
        assert h.d.requests == []
    finally:
        run(h.aclose())


# --------------------------------------------------------------------------- #
# Rejection retry (D leg)
# --------------------------------------------------------------------------- #


def test_decode_500_retries_once_without_params_nonstream():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_sequence(d_json(status=500), d_json("second"))
    try:
        resp = run(h.post("/chat/completions", chat_body(min_tokens=2)))
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "second"
        assert h.p.header_ids == [h.rid + "-p"]
        assert h.d.header_ids == [h.rid, h.rid + D_RETRY_SUFFIX]
        assert h.d.bodies[0]["kv_transfer_params"] == P_PARAMS
        assert "kv_transfer_params" not in h.d.bodies[1]
        assert h.d.bodies[1]["min_tokens"] == 2
    finally:
        run(h.aclose())


def test_decode_finish_reason_error_retries_nonstream():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_sequence(d_json("", finish_reason="error"), d_json("second"))
    try:
        resp = run(h.post("/chat/completions", chat_body()))
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "second"
        assert h.d.header_ids == [h.rid, h.rid + "-r1"]
    finally:
        run(h.aclose())


def test_decode_retry_failure_is_passed_through_not_retried_again():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json(status=500)
    try:
        resp = run(h.post("/chat/completions", chat_body()))
        assert resp.status_code == 500
        assert h.d.header_ids == [h.rid, h.rid + "-r1"]
    finally:
        run(h.aclose())


def test_decode_4xx_is_passed_through_without_retry():
    h = Harness()
    h.p.handler = p_ok()

    async def d_400(request, i):
        return httpx.Response(400, json={"error": {"message": "bad", "code": 400}})

    h.d.handler = d_400
    try:
        resp = run(h.post("/chat/completions", chat_body()))
        assert resp.status_code == 400
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_no_retry_when_no_params_were_sent():
    h = Harness(min_remote_tokens=512)
    h.d.handler = d_json(status=500)
    try:
        body = chat_body(messages=[{"role": "user", "content": "hi"}])
        resp = run(h.post("/chat/completions", body))
        assert resp.status_code == 500
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_retry_disabled_by_config():
    h = Harness(d_retry=False)
    h.p.handler = p_ok()
    h.d.handler = d_json(status=500)
    try:
        resp = run(h.post("/chat/completions", chat_body()))
        assert resp.status_code == 500
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_streaming_error_event_before_content_triggers_retry():
    h = Harness()
    h.p.handler = p_ok()
    good = [role_chunk(), content_chunk("ok"), finish_chunk(), DONE]
    h.d.handler = d_sequence(
        d_stream([role_chunk(), error_event(), DONE]), d_stream(good)
    )
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True, min_tokens=2)))
        assert resp.status_code == 200
        # Nothing from the failed first stream reaches the client.
        assert resp.content == b"".join(good)
        assert h.d.header_ids == [h.rid, h.rid + "-r1"]
        assert h.d.bodies[0]["kv_transfer_params"] == P_PARAMS
        assert "kv_transfer_params" not in h.d.bodies[1]
        assert h.d.bodies[1]["stream"] is True and h.d.bodies[1]["min_tokens"] == 2
    finally:
        run(h.aclose())


def test_streaming_finish_reason_error_chunk_triggers_retry():
    h = Harness()
    h.p.handler = p_ok()
    good = [role_chunk(), content_chunk("ok"), DONE]
    h.d.handler = d_sequence(
        d_stream([role_chunk(), finish_chunk("error"), DONE]), d_stream(good)
    )
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == b"".join(good)
        assert h.d.header_ids == [h.rid, h.rid + "-r1"]
    finally:
        run(h.aclose())


def test_streaming_500_status_triggers_retry():
    h = Harness()
    h.p.handler = p_ok()
    good = [role_chunk(), content_chunk("ok"), DONE]
    h.d.handler = d_sequence(d_json(status=500), d_stream(good))
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == b"".join(good)
        assert h.d.header_ids == [h.rid, h.rid + "-r1"]
    finally:
        run(h.aclose())


def test_streaming_no_retry_after_first_content_chunk():
    h = Harness()
    h.p.handler = p_ok()
    parts = [role_chunk(), content_chunk("partial"), error_event(), DONE]
    h.d.handler = d_stream(parts)
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == b"".join(parts)
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_streaming_retry_buffer_timeout_flushes_and_passes_through():
    h = Harness(retry_buffer_timeout=0.05)
    h.p.handler = p_ok()
    # Role chunk, then a long gap before the first content: the proxy must
    # flush after the timeout and never retry, even if an error follows.
    parts = [role_chunk(), content_chunk("late"), error_event(), DONE]
    h.d.handler = d_stream(parts, gaps={1: 0.2})
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == b"".join(parts)
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_streaming_retry_stream_error_is_passed_through_not_retried_again():
    h = Harness()
    h.p.handler = p_ok()
    bad = [role_chunk(), error_event(), DONE]
    h.d.handler = d_stream(bad)
    try:
        resp = run(h.post("/chat/completions", chat_body(stream=True)))
        assert resp.content == b"".join(bad)
        assert h.d.header_ids == [h.rid, h.rid + "-r1"]
    finally:
        run(h.aclose())


def test_completions_stream_text_counts_as_content():
    h = Harness()
    h.p.handler = p_ok()
    parts = [
        sse({"choices": [{"index": 0, "text": "Hi", "finish_reason": None}]}),
        error_event(),
        DONE,
    ]
    h.d.handler = d_stream(parts)
    try:
        resp = run(h.post("/completions", completion_body(stream=True)))
        assert resp.content == b"".join(parts)
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_classify_sse_event():
    assert classify_sse_event(role_chunk()) == "other"
    assert classify_sse_event(content_chunk("x")) == "content"
    assert classify_sse_event(finish_chunk("stop")) == "content"
    assert classify_sse_event(finish_chunk("error")) == "error"
    assert classify_sse_event(error_event()) == "error"
    assert classify_sse_event(DONE) == "done"
    assert classify_sse_event(b": keep-alive\n\n") == "other"
    assert classify_sse_event(b"data: not json\n\n") == "other"
    tool = sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "1"}]}}]})
    assert classify_sse_event(tool) == "content"
    reasoning = sse({"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]})
    assert classify_sse_event(reasoning) == "content"


# --------------------------------------------------------------------------- #
# 400 rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "api,body,needle",
    [
        (
            "/completions",
            completion_body(prompt=["a" * 900, "b" * 900]),
            "list of prompts",
        ),
        ("/completions", completion_body(prompt=[["a", "b"]]), "list of prompts"),
        ("/chat/completions", chat_body(n=2), "n > 1"),
        ("/chat/completions", chat_body(best_of=2), "best_of"),
        ("/completions", completion_body(best_of=3), "best_of"),
        ("/completions", completion_body(prompt_logprobs=1), "prompt_logprobs"),
        ("/chat/completions", chat_body(prompt_logprobs=0), "prompt_logprobs"),
        ("/completions", completion_body(echo=True, logprobs=1), "echo"),
        ("/chat/completions", chat_body(max_tokens=0), "max_tokens=0"),
        (
            "/chat/completions",
            chat_body(max_completion_tokens=0),
            "max_completion_tokens=0",
        ),
        (
            "/chat/completions",
            chat_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": LONG_TEXT},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA"},
                            },
                        ],
                    }
                ]
            ),
            "multimodal",
        ),
    ],
)
def test_rejections_400_reach_no_backend(api, body, needle):
    h = Harness()
    try:
        resp = run(h.post(api, body))
        assert resp.status_code == 400
        assert needle in resp.json()["error"]["message"]
        assert resp.headers["x-request-id"] == "rid1"
        assert h.p.requests == [] and h.d.requests == []
    finally:
        run(h.aclose())


def test_allowed_variants_are_not_rejected():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()
    try:
        # n=1, best_of=1, echo without logprobs, text-part content, token-id prompt.
        resp = run(h.post("/chat/completions", chat_body(n=1, best_of=1)))
        assert resp.status_code == 200
        resp = run(h.post("/completions", completion_body(echo=True)))
        assert resp.status_code == 200
        body = chat_body(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": LONG_TEXT}]}
            ]
        )
        resp = run(h.post("/chat/completions", body))
        assert resp.status_code == 200
    finally:
        run(h.aclose())


def test_invalid_json_body_is_400():
    h = Harness()
    try:
        resp = run(
            h.client.post(
                "/v1/chat/completions",
                content=b"{not json",
                headers={"content-type": "application/json", "X-Request-Id": "rid1"},
            )
        )
        assert resp.status_code == 400
        assert h.p.requests == [] and h.d.requests == []
    finally:
        run(h.aclose())


# --------------------------------------------------------------------------- #
# Client disconnect
# --------------------------------------------------------------------------- #


def test_nonstream_client_disconnect_after_prefill_skips_decode_leg():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()

    async def disconnected() -> bool:
        return True

    try:
        resp = run(
            h.proxy.handle(
                "/chat/completions",
                chat_body(),
                {"X-Request-Id": "rid1"},
                is_disconnected=disconnected,
            )
        )
        assert resp.status_code == pd_proxy.CLIENT_CLOSED_REQUEST
        assert h.p.header_ids == [h.rid + "-p"]
        assert h.d.requests == []
    finally:
        run(h.aclose())


def test_nonstream_connected_client_proceeds_to_decode_leg():
    h = Harness()
    h.p.handler = p_ok()
    h.d.handler = d_json()

    async def connected() -> bool:
        return False

    try:
        resp = run(
            h.proxy.handle(
                "/chat/completions",
                chat_body(),
                {"X-Request-Id": "rid1"},
                is_disconnected=connected,
            )
        )
        assert resp.status_code == 200
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


def test_stream_client_disconnect_during_prefill_cancels_prefill_and_skips_decode():
    """Starlette cancels the response generator when the client goes away;
    the proxy must cancel the in-flight P request and never open the D leg."""
    h = Harness(sse_keepalive=0.01)
    h.d.handler = d_json()
    state = {"p_cancelled": False}

    async def slow_p(request, i):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["p_cancelled"] = True
            raise
        return httpx.Response(200, json={"kv_transfer_params": P_PARAMS, "choices": []})

    h.p.handler = slow_p

    async def go():
        resp = await h.proxy.handle(
            "/chat/completions", chat_body(stream=True), {"X-Request-Id": "rid1"}
        )
        received: list[bytes] = []

        async def consume():
            async for chunk in resp.body_iterator:
                received.append(chunk)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # P leg in flight, keep-alives flowing
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)
        return received

    try:
        received = run(go())
        assert received and all(c == b": keep-alive\n\n" for c in received)
        assert state["p_cancelled"] is True
        assert h.p.header_ids == [h.rid + "-p"]
        assert h.d.requests == []
    finally:
        run(h.aclose())


def test_stream_client_disconnect_during_decode_closes_upstream():
    h = Harness()
    h.p.handler = p_ok()
    closed = {"value": False}

    class _Tracking(_Stream):
        async def aclose(self):
            closed["value"] = True

    async def d_slow(request, i):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_Tracking(
                [role_chunk(), content_chunk("a"), content_chunk("b"), DONE],
                gaps={2: 10.0},
            ),
        )

    h.d.handler = d_slow

    async def go():
        resp = await h.proxy.handle(
            "/chat/completions", chat_body(stream=True), {"X-Request-Id": "rid1"}
        )
        received: list[bytes] = []

        async def consume():
            async for chunk in resp.body_iterator:
                received.append(chunk)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.02)  # detached upstream teardown
        return received

    try:
        received = run(go())
        assert b"".join(received) == role_chunk() + content_chunk("a")
        assert closed["value"] is True
        assert h.d.header_ids == [h.rid]
    finally:
        run(h.aclose())


# --------------------------------------------------------------------------- #
# Passthrough routes
# --------------------------------------------------------------------------- #


def test_models_and_health_routes():
    h = Harness()

    async def p_misc(request, i):
        assert request.url.path == "/health"
        return httpx.Response(200)

    async def d_misc(request, i):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "Qwen/Qwen3.8-27B"}]}
            )
        assert request.url.path == "/health"
        return httpx.Response(200)

    h.p.handler = p_misc
    h.d.handler = d_misc
    try:
        resp = run(h.client.get("/v1/models"))
        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == "Qwen/Qwen3.8-27B"
        resp = run(h.client.get("/health"))
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
    finally:
        run(h.aclose())


def test_health_reports_503_when_a_node_is_down():
    h = Harness()

    async def p_ok_health(request, i):
        return httpx.Response(200)

    async def d_down(request, i):
        raise httpx.ConnectError("refused", request=request)

    h.p.handler = p_ok_health
    h.d.handler = d_down
    try:
        resp = run(h.client.get("/health"))
        assert resp.status_code == 503
        assert resp.json()["decode"]["status"] == "error"
        assert resp.json()["prefill"]["status"] == "ok"
    finally:
        run(h.aclose())
