# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""OpenAI-compatible prefill/decode disaggregation proxy (PHASE2_DESIGN.md 2.4).

One client request becomes up to three backend requests, each with its own
engine id derived from the ``X-Request-Id`` header the backend receives. The
proxy MINTS that id (``uuid4().hex``) for every client request: a client's own
``X-Request-Id`` is only echoed back in the proxy's response header, error
payloads and logs, never handed to P or D (a reused client id would hit
``Scheduler.add_request``'s duplicate-id assert on a backend -- engine crash).

* ``{rid}-p``  the prefill leg on P (``max_tokens=1``, ``stream=false``,
  ``kv_transfer_params`` with ``do_remote_decode``); P's one token is
  discarded and its response's ``kv_transfer_params`` are taken;
* ``{rid}``    the decode leg on D (the original body + P's params);
* ``{rid}-r1`` the one retry on D without params after D rejected the load
  (HTTP >= 500, a choice with ``finish_reason == "error"``, or an in-stream
  ``{"error": ...}`` event before the first content chunk).

Prompts whose length estimate is below ``--min-remote-tokens`` (hard minimum
2) go straight to D, which prefills locally.

The module is stdlib + httpx + fastapi/starlette only: it runs as its own
process (``python -m vllm_tt_plugin.kv_transfer.pd_proxy``) and never imports
vLLM or ttnn.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("vllm_tt_plugin.kv_transfer.pd_proxy")

# The T == 1 prompt would export a segment D immediately demotes (2.4 step 1).
MIN_REMOTE_TOKENS_FLOOR = 2

P_LEG_SUFFIX = "-p"
D_RETRY_SUFFIX = "-r1"

# 2.2 step 1: the producer-side request shape.
PREFILL_KV_TRANSFER_PARAMS: dict[str, Any] = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}

# 2.4 step 2: dropped on the P leg (never restored). Structured-output controls
# would compile a grammar and wait in WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR on
# P for a token that is discarded; logprobs of the discarded token are noise.
PREFILL_DROPPED_FIELDS: tuple[str, ...] = (
    "stream_options",
    "logprobs",
    "top_logprobs",
    "response_format",
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_grammar",
    "guided_choice",
    "tool_choice",
)
# Popped for the P leg and restored for the D leg.
PREFILL_RESTORED_FIELDS: tuple[str, ...] = ("min_tokens", "min_completion_tokens")

SSE_KEEPALIVE_COMMENT = b": keep-alive\n\n"
SSE_DONE = b"data: [DONE]\n\n"

# Status used when the client went away before the D leg (nginx convention).
CLIENT_CLOSED_REQUEST = 499

_API_PATHS = ("/chat/completions", "/completions")

_FORWARDED_REQUEST_HEADERS = ("authorization", "accept", "user-agent")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PDProxyConfig:
    prefill_url: str = "http://127.0.0.1:8100"
    decode_url: str = "http://127.0.0.1:8200"
    min_remote_tokens: int = 512
    # SSE comment keep-alives emitted to a streaming client while the P leg
    # runs; 0 disables.
    sse_keepalive: float = 5.0
    # Streaming D leg: how long after D's first byte the proxy keeps buffering
    # (waiting for the first content chunk) so an error event can still
    # trigger the retry.
    retry_buffer_timeout: float = 2.0
    # "error": P failure -> 502; "local": P failure -> D without params.
    p_fallback: str = "error"
    # httpx read/write/pool timeouts per leg; None = unbounded (2.4 keeps the
    # toy proxy's explicit timeout=None: the P leg is tens of seconds at 32k).
    p_timeout: float | None = None
    d_timeout: float | None = None
    connect_timeout: float | None = 10.0
    # Prompt-length heuristic when no tokenizer is configured.
    chars_per_token: int = 4
    tokenizer: str | None = None
    # One retry to D without params after a D-side load rejection.
    d_retry: bool = True
    log_timing: bool = False

    def __post_init__(self) -> None:
        if self.min_remote_tokens < MIN_REMOTE_TOKENS_FLOOR:
            logger.warning(
                "min_remote_tokens=%d is below the hard minimum %d; clamping",
                self.min_remote_tokens,
                MIN_REMOTE_TOKENS_FLOOR,
            )
            self.min_remote_tokens = MIN_REMOTE_TOKENS_FLOOR
        if self.p_fallback not in ("error", "local"):
            raise ValueError(
                f"p_fallback must be 'error' or 'local', got {self.p_fallback!r}"
            )
        if self.chars_per_token < 1:
            raise ValueError(
                f"chars_per_token must be >= 1, got {self.chars_per_token}"
            )
        if self.retry_buffer_timeout < 0:
            raise ValueError("retry_buffer_timeout must be >= 0")
        if self.sse_keepalive < 0:
            raise ValueError("sse_keepalive must be >= 0")
        self.prefill_url = self.prefill_url.rstrip("/")
        self.decode_url = self.decode_url.rstrip("/")


# --------------------------------------------------------------------------- #
# Request validation, prompt-length estimate, body shaping (pure functions)
# --------------------------------------------------------------------------- #


class ProxyRequestError(Exception):
    """A client-side rejection (400 unless stated otherwise)."""

    def __init__(
        self, message: str, status_code: int = 400, err_type: str = "BadRequestError"
    ):
        super().__init__(message)
        self.status_code = status_code
        self.err_type = err_type


def _iter_message_text(messages: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(messages, list):
        return texts
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
                elif isinstance(part, str):
                    texts.append(part)
        for key in ("reasoning_content", "reasoning"):
            extra = msg.get(key)
            if isinstance(extra, str):
                texts.append(extra)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            texts.append(json.dumps(tool_calls))
    return texts


def _has_multimodal_parts(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type", "text") != "text":
                return True
    return False


def validate_request(api: str, body: Mapping[str, Any]) -> None:
    """2.4 step 1 rejections. Raises ProxyRequestError (400)."""
    if not isinstance(body, Mapping):
        raise ProxyRequestError("request body must be a JSON object")
    n = body.get("n")
    if n is not None and n != 1:
        raise ProxyRequestError(
            "n > 1 is not supported behind the P/D proxy", err_type="BadRequestError"
        )
    best_of = body.get("best_of")
    if best_of is not None and best_of != 1:
        raise ProxyRequestError("best_of is not supported behind the P/D proxy")
    if body.get("prompt_logprobs") is not None:
        raise ProxyRequestError(
            "prompt_logprobs is not supported under remote prefill "
            "(prompt logprobs for the prefilled tokens do not exist on the decode node)"
        )
    for key in ("max_tokens", "max_completion_tokens"):
        if body.get(key) == 0:
            raise ProxyRequestError(f"{key}=0 is not supported behind the P/D proxy")
    if api == "/completions":
        prompt = body.get("prompt")
        if isinstance(prompt, list) and not all(isinstance(t, int) for t in prompt):
            raise ProxyRequestError(
                "a list of prompts is not supported behind the P/D proxy "
                "(send one request per prompt)"
            )
        if body.get("echo") and body.get("logprobs") is not None:
            raise ProxyRequestError(
                "echo=true together with logprobs is not supported under remote prefill"
            )
    elif api == "/chat/completions":
        if _has_multimodal_parts(body.get("messages")):
            raise ProxyRequestError(
                "multimodal content parts are not supported behind the P/D proxy"
            )


class TokenCounter:
    """Prompt-length estimate for routing (a wrong estimate is harmless, 2.4)."""

    def __init__(self, chars_per_token: int = 4, tokenizer_path: str | None = None):
        self._chars_per_token = max(1, chars_per_token)
        self._tokenizer = None
        if tokenizer_path:
            # Optional and imported lazily: the proxy must not depend on
            # transformers being importable for the heuristic path.
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text) // self._chars_per_token)

    def estimate(self, api: str, body: Mapping[str, Any]) -> int:
        if api == "/completions":
            prompt = body.get("prompt")
            if isinstance(prompt, list):
                return len(prompt)  # token ids
            if isinstance(prompt, str):
                return self.count_text(prompt)
            return 0
        messages = body.get("messages")
        total = sum(self.count_text(t) for t in _iter_message_text(messages))
        # Chat-template overhead per message (role markers etc).
        if isinstance(messages, list):
            total += 4 * len(messages)
        tools = body.get("tools")
        if tools:
            total += self.count_text(json.dumps(tools))
        return total


def build_prefill_body(
    body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """2.4 step 2: the P-leg body and the fields to restore on the D leg."""
    body_p = copy.deepcopy(dict(body))
    body_p["stream"] = False
    body_p["max_tokens"] = 1
    if "max_completion_tokens" in body_p:
        body_p["max_completion_tokens"] = 1
    for key in PREFILL_DROPPED_FIELDS:
        body_p.pop(key, None)
    restore: dict[str, Any] = {}
    for key in PREFILL_RESTORED_FIELDS:
        if key in body_p:
            restore[key] = body_p.pop(key)
    body_p["kv_transfer_params"] = dict(PREFILL_KV_TRANSFER_PARAMS)
    return body_p, restore


def build_decode_body(
    body: Mapping[str, Any],
    params: Mapping[str, Any] | None,
    restore: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """2.4 step 3: the D-leg body (original request + P's params)."""
    body_d = copy.deepcopy(dict(body))
    if restore:
        body_d.update(restore)
    body_d.pop("kv_transfer_params", None)
    if params:
        body_d["kv_transfer_params"] = dict(params)
    return body_d


# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #


def _choice_commits(choice: Mapping[str, Any]) -> bool:
    """True when the choice carries content (or a non-error finish): after it
    the proxy can no longer retry, so buffering stops."""
    if choice.get("text"):
        return True
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        for key in (
            "content",
            "reasoning_content",
            "reasoning",
            "tool_calls",
            "function_call",
        ):
            if delta.get(key):
                return True
    finish = choice.get("finish_reason")
    return finish is not None and finish != "error"


def classify_sse_event(event: bytes) -> str:
    """One complete SSE event -> 'error' | 'content' | 'done' | 'other'."""
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return "done"
        try:
            obj = json.loads(payload)
        except ValueError:
            return "other"
        if not isinstance(obj, Mapping):
            return "other"
        if "error" in obj:
            return "error"
        choices = obj.get("choices") or []
        if any(
            isinstance(c, Mapping) and c.get("finish_reason") == "error"
            for c in choices
        ):
            return "error"
        if any(isinstance(c, Mapping) and _choice_commits(c) for c in choices):
            return "content"
    return "other"


def _json_choices_errored(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if "error" in payload:
        return True
    choices = payload.get("choices") or []
    return any(
        isinstance(c, Mapping) and c.get("finish_reason") == "error" for c in choices
    )


def _error_payload(
    message: str, status: int, err_type: str, rid: str
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "code": status,
            "param": None,
        },
        "request_id": rid,
    }


def _sse_error_event(message: str, status: int, err_type: str, rid: str) -> bytes:
    return (
        b"data: "
        + json.dumps(_error_payload(message, status, err_type, rid)).encode()
        + b"\n\n"
    )


# --------------------------------------------------------------------------- #
# Proxy
# --------------------------------------------------------------------------- #


@dataclass
class _RequestContext:
    rid: str
    api: str
    body: dict[str, Any]
    headers: dict[str, str]
    stream: bool
    # The client's own X-Request-Id, if it sent one: echoed, never forwarded.
    client_rid: str | None = None
    t_start: float = field(default_factory=time.monotonic)
    estimate: int = 0
    route: str = "remote"
    t_p_done: float | None = None
    t_first_content: float | None = None
    restore: dict[str, Any] = field(default_factory=dict)

    @property
    def echo_rid(self) -> str:
        """The id the CLIENT sees (its own X-Request-Id when it sent one)."""
        return self.client_rid or self.rid

    def log(self, level: int, event: str, **kv: Any) -> None:
        if self.client_rid is not None:
            kv = {"client_rid": self.client_rid, **kv}
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.log(level, "rid=%s api=%s event=%s %s", self.rid, self.api, event, extra)


def _httpx_timeout(read: float | None, connect: float | None) -> httpx.Timeout:
    return httpx.Timeout(timeout=read, connect=connect)


class PDProxy:
    """Request pipeline; the FastAPI app (create_app) is a thin wrapper.

    ``client_p`` / ``client_d`` are injectable so tests can use
    ``httpx.MockTransport``; when absent they are built from the config with
    ``timeout=None``-style unbounded read timeouts (2.4).
    """

    def __init__(
        self,
        config: PDProxyConfig,
        client_p: httpx.AsyncClient | None = None,
        client_d: httpx.AsyncClient | None = None,
        token_counter: TokenCounter | None = None,
    ):
        self.config = config
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        self._owns_p = client_p is None
        self._owns_d = client_d is None
        self.client_p = client_p or httpx.AsyncClient(
            base_url=config.prefill_url,
            timeout=_httpx_timeout(config.p_timeout, config.connect_timeout),
            limits=limits,
        )
        self.client_d = client_d or httpx.AsyncClient(
            base_url=config.decode_url,
            timeout=_httpx_timeout(config.d_timeout, config.connect_timeout),
            limits=limits,
        )
        self.token_counter = token_counter or TokenCounter(
            config.chars_per_token, config.tokenizer
        )

    async def aclose(self) -> None:
        if self._owns_p:
            await self.client_p.aclose()
        if self._owns_d:
            await self.client_d.aclose()

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    @staticmethod
    def client_request_id(headers: Mapping[str, str]) -> str | None:
        """The client's own ``X-Request-Id`` (echo only), or None."""
        for key, value in headers.items():
            if key.lower() == "x-request-id" and value:
                return value
        return None

    @staticmethod
    def mint_request_id() -> str:
        """The backend-facing id: fresh per client request, never client-chosen."""
        return uuid.uuid4().hex

    @classmethod
    def request_id(cls, headers: Mapping[str, str]) -> str:
        """The id to ECHO to the client: its own header when present, else a
        fresh one. Not what the backends receive (see ``mint_request_id``)."""
        return cls.client_request_id(headers) or cls.mint_request_id()

    async def handle(
        self,
        api: str,
        body: Any,
        headers: Mapping[str, str],
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> Response:
        """Serve one /v1{api} request. ``is_disconnected`` is polled (non-stream
        path only) after the P leg; the streaming path relies on the server
        cancelling the response generator on disconnect."""
        if api not in _API_PATHS:
            raise ValueError(f"unsupported api {api!r}")
        client_rid = self.client_request_id(headers)
        rid = self.mint_request_id()  # backend legs: {rid}-p, {rid}, {rid}-r1
        echo = client_rid or rid
        if not isinstance(body, dict):
            return self._error_response(echo, "request body must be a JSON object", 400)
        if "kv_transfer_params" in body:
            # Clients must not steer D at a foreign segment (R16); D validates
            # anyway, but drop it here so the P leg sees a clean body.
            logger.warning("rid=%s event=client_kv_transfer_params_dropped", rid)
            body = {k: v for k, v in body.items() if k != "kv_transfer_params"}
        ctx = _RequestContext(
            rid=rid,
            api=api,
            body=body,
            headers=self._forward_headers(headers),
            stream=bool(body.get("stream", False)),
            client_rid=client_rid,
        )
        try:
            validate_request(api, body)
        except ProxyRequestError as e:
            ctx.log(logging.INFO, "rejected", status=e.status_code, reason=str(e))
            return self._error_response(echo, str(e), e.status_code, e.err_type)

        ctx.estimate = self.token_counter.estimate(api, body)
        remote = ctx.estimate >= self.config.min_remote_tokens
        ctx.route = "remote" if remote else "local"
        ctx.log(
            logging.INFO,
            "accepted",
            stream=ctx.stream,
            estimate=ctx.estimate,
            route=ctx.route,
        )

        if ctx.stream:
            return StreamingResponse(
                self._stream_pipeline(ctx, do_prefill=remote),
                media_type="text/event-stream",
                headers={
                    "X-Request-Id": echo,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await self._json_pipeline(
            ctx, do_prefill=remote, is_disconnected=is_disconnected
        )

    # ------------------------------------------------------------------ #
    # P leg
    # ------------------------------------------------------------------ #

    async def _prefill_leg(self, ctx: _RequestContext) -> dict[str, Any] | None:
        """Returns P's kv_transfer_params (None when P returned none).
        Raises ProxyRequestError(502) on P failure."""
        body_p, ctx.restore = build_prefill_body(ctx.body)
        headers = dict(ctx.headers)
        headers["X-Request-Id"] = ctx.rid + P_LEG_SUFFIX
        t0 = time.monotonic()
        try:
            resp = await self.client_p.post(
                f"/v1{ctx.api}", json=body_p, headers=headers
            )
        except httpx.HTTPError as e:
            ctx.log(
                logging.ERROR,
                "prefill_transport_error",
                error=f"{type(e).__name__}:{e}",
            )
            raise ProxyRequestError(
                f"prefill node unreachable: {type(e).__name__}", 502, "BadGateway"
            ) from e
        try:
            if resp.status_code >= 400:
                snippet = resp.text[:300].replace("\n", " ")
                ctx.log(
                    logging.ERROR,
                    "prefill_http_error",
                    status=resp.status_code,
                    body=snippet,
                )
                raise ProxyRequestError(
                    f"prefill node returned HTTP {resp.status_code}: {snippet}",
                    502,
                    "BadGateway",
                )
            try:
                payload = resp.json()
            except ValueError as e:
                ctx.log(logging.ERROR, "prefill_bad_json")
                raise ProxyRequestError(
                    "prefill node returned a non-JSON body", 502, "BadGateway"
                ) from e
        finally:
            await resp.aclose()
        ctx.t_p_done = time.monotonic()
        params = (
            payload.get("kv_transfer_params") if isinstance(payload, Mapping) else None
        )
        finish = None
        if isinstance(payload, Mapping):
            choices = payload.get("choices") or []
            if choices and isinstance(choices[0], Mapping):
                finish = choices[0].get("finish_reason")
        if not params:
            ctx.log(
                logging.WARNING,
                "prefill_no_params",
                msg="prefill node returned no kv_transfer_params; "
                "decode node will prefill locally",
            )
            params = None
        ctx.log(
            logging.INFO,
            "prefill_done",
            p_ms=round((ctx.t_p_done - t0) * 1000, 1),
            finish_reason=finish,
            remote_num_tokens=(params or {}).get("remote_num_tokens"),
            xfer_id=(params or {}).get("xfer_id"),
        )
        return dict(params) if params else None

    async def _run_prefill(
        self, ctx: _RequestContext
    ) -> tuple[dict[str, Any] | None, str | None]:
        """(params, error_message). With p_fallback=local a P failure yields
        (None, None) and the D leg prefills locally."""
        try:
            return await self._prefill_leg(ctx), None
        except ProxyRequestError as e:
            if self.config.p_fallback == "local":
                ctx.log(logging.WARNING, "prefill_fallback_local", reason=str(e))
                ctx.restore = {}
                return None, None
            return None, str(e)

    # ------------------------------------------------------------------ #
    # Non-streaming pipeline
    # ------------------------------------------------------------------ #

    async def _json_pipeline(
        self,
        ctx: _RequestContext,
        do_prefill: bool,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> Response:
        params: dict[str, Any] | None = None
        if do_prefill:
            params, err = await self._run_prefill(ctx)
            if err is not None:
                return self._error_response(ctx.echo_rid, err, 502, "BadGateway")
            if is_disconnected is not None and await is_disconnected():
                # 2.4 step 3: no D leg; P's segment leaks until the lease.
                ctx.log(
                    logging.WARNING,
                    "client_disconnected",
                    stage="before_decode_leg",
                    xfer_id=(params or {}).get("xfer_id"),
                )
                return Response(status_code=CLIENT_CLOSED_REQUEST)

        body_d = build_decode_body(ctx.body, params, ctx.restore)
        resp = await self._decode_post(ctx, body_d, ctx.rid)
        if isinstance(resp, Response):
            return resp
        if params and self.config.d_retry and self._json_response_rejected(resp):
            ctx.log(
                logging.WARNING,
                "decode_rejected_retry",
                status=resp.status_code,
                retry_id=ctx.rid + D_RETRY_SUFFIX,
            )
            body_r = build_decode_body(ctx.body, None, ctx.restore)
            resp = await self._decode_post(ctx, body_r, ctx.rid + D_RETRY_SUFFIX)
            if isinstance(resp, Response):
                return resp
        self._log_timing(ctx, status=resp.status_code)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers={"X-Request-Id": ctx.echo_rid},
        )

    async def _decode_post(
        self, ctx: _RequestContext, body_d: dict[str, Any], header_id: str
    ) -> httpx.Response | Response:
        headers = dict(ctx.headers)
        headers["X-Request-Id"] = header_id
        try:
            resp = await self.client_d.post(
                f"/v1{ctx.api}", json=body_d, headers=headers
            )
        except httpx.TimeoutException as e:
            ctx.log(logging.ERROR, "decode_timeout", error=f"{type(e).__name__}")
            return self._error_response(
                ctx.echo_rid, "decode node timed out", 504, "GatewayTimeout"
            )
        except httpx.HTTPError as e:
            ctx.log(
                logging.ERROR, "decode_transport_error", error=f"{type(e).__name__}:{e}"
            )
            return self._error_response(
                ctx.echo_rid,
                f"decode node unreachable: {type(e).__name__}",
                502,
                "BadGateway",
            )
        return resp

    @staticmethod
    def _json_response_rejected(resp: httpx.Response) -> bool:
        if resp.status_code >= 500:
            return True
        if resp.status_code >= 400:
            return False
        try:
            return _json_choices_errored(resp.json())
        except ValueError:
            return False

    # ------------------------------------------------------------------ #
    # Streaming pipeline
    # ------------------------------------------------------------------ #

    async def _stream_pipeline(
        self, ctx: _RequestContext, do_prefill: bool
    ) -> AsyncIterator[bytes]:
        params: dict[str, Any] | None = None
        stage = "prefill" if do_prefill else "decode_open"
        try:
            if do_prefill:
                p_task = asyncio.ensure_future(self._run_prefill(ctx))
                try:
                    keepalive = self.config.sse_keepalive
                    while True:
                        if keepalive <= 0:
                            await asyncio.wait({p_task})
                            break
                        done, _ = await asyncio.wait({p_task}, timeout=keepalive)
                        if done:
                            break
                        yield SSE_KEEPALIVE_COMMENT
                except BaseException:
                    # Client disconnect (the server cancelled this generator).
                    # Cancel the in-flight POST so P sees a disconnect and
                    # aborts the prefill unless it already exported. Do NOT
                    # await the task here: the server's cancel scope keeps
                    # re-cancelling this task, and an `await p_task` would
                    # forward every re-cancel into p_task, interrupting
                    # httpcore's shielded connection close.
                    _cancel_detached(p_task)
                    raise
                params, err = p_task.result()
                if err is not None:
                    yield _sse_error_event(err, 502, "BadGateway", ctx.echo_rid)
                    yield SSE_DONE
                    return

            stage = "decode_open"
            body_d = build_decode_body(ctx.body, params, ctx.restore)
            allow_retry = bool(params) and self.config.d_retry
            async for chunk in self._decode_stream(
                ctx, body_d, ctx.rid, allow_retry=allow_retry
            ):
                stage = "decode_stream"
                yield chunk
        except asyncio.CancelledError:
            ctx.log(logging.WARNING, "client_disconnected", stage=stage)
            raise
        finally:
            if ctx.t_first_content is not None or stage == "decode_stream":
                self._log_timing(ctx, status=200)

    async def _decode_stream(
        self,
        ctx: _RequestContext,
        body_d: dict[str, Any],
        header_id: str,
        allow_retry: bool,
    ) -> AsyncIterator[bytes]:
        headers = dict(ctx.headers)
        headers["X-Request-Id"] = header_id
        req = self.client_d.build_request(
            "POST", f"/v1{ctx.api}", json=body_d, headers=headers
        )
        try:
            resp = await self.client_d.send(req, stream=True)
        except httpx.TimeoutException:
            ctx.log(logging.ERROR, "decode_timeout")
            yield _sse_error_event(
                "decode node timed out", 504, "GatewayTimeout", ctx.echo_rid
            )
            yield SSE_DONE
            return
        except httpx.HTTPError as e:
            ctx.log(
                logging.ERROR, "decode_transport_error", error=f"{type(e).__name__}:{e}"
            )
            yield _sse_error_event(
                f"decode node unreachable: {type(e).__name__}",
                502,
                "BadGateway",
                ctx.echo_rid,
            )
            yield SSE_DONE
            return

        if resp.status_code >= 400:
            body = await resp.aread()
            await resp.aclose()
            if resp.status_code >= 500 and allow_retry:
                ctx.log(
                    logging.WARNING,
                    "decode_rejected_retry",
                    status=resp.status_code,
                    retry_id=ctx.rid + D_RETRY_SUFFIX,
                )
                async for chunk in self._retry_stream(ctx):
                    yield chunk
                return
            ctx.log(logging.ERROR, "decode_http_error", status=resp.status_code)
            snippet = body[:300].decode("utf-8", "replace").replace("\n", " ")
            yield _sse_error_event(
                f"decode node returned HTTP {resp.status_code}: {snippet}",
                resp.status_code,
                "BadGateway" if resp.status_code >= 500 else "BadRequestError",
                ctx.echo_rid,
            )
            yield SSE_DONE
            return

        # Reader task -> queue, so the buffering phase can time out without
        # cancelling an httpx read mid-chunk.
        queue: asyncio.Queue[bytes | None | BaseException] = asyncio.Queue()

        async def _reader() -> None:
            try:
                async for chunk in resp.aiter_bytes():
                    await queue.put(chunk)
                await queue.put(None)
            except BaseException as e:  # propagate to the consumer
                await queue.put(e)
                if isinstance(e, asyncio.CancelledError):
                    raise

        reader = asyncio.ensure_future(_reader())
        try:
            buffered = bytearray()
            pending = bytearray()  # partial SSE event
            decision = "passthrough" if not allow_retry else None
            deadline: float | None = None
            while decision is None:
                timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    ctx.log(
                        logging.INFO, "retry_buffer_timeout", buffered=len(buffered)
                    )
                    decision = "passthrough"
                    break
                if item is None:
                    decision = "passthrough"  # stream ended before any content
                    break
                if isinstance(item, BaseException):
                    raise item
                if deadline is None:
                    deadline = time.monotonic() + self.config.retry_buffer_timeout
                buffered += item
                pending += item
                while True:
                    idx = pending.find(b"\n\n")
                    if idx < 0:
                        break
                    event = bytes(pending[: idx + 2])
                    del pending[: idx + 2]
                    kind = classify_sse_event(event)
                    if kind == "error":
                        decision = "retry"
                        break
                    if kind in ("content", "done"):
                        decision = "passthrough"
                        break

            if decision == "retry":
                await _close_upstream(reader, resp)
                ctx.log(
                    logging.WARNING,
                    "decode_rejected_retry",
                    status=resp.status_code,
                    retry_id=ctx.rid + D_RETRY_SUFFIX,
                    buffered=len(buffered),
                )
                async for chunk in self._retry_stream(ctx):
                    yield chunk
                return

            if buffered:
                self._note_first_content(ctx)
                yield bytes(buffered)
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                self._note_first_content(ctx)
                yield item
        finally:
            if reader.done():
                await resp.aclose()
            else:
                # Leaving early (client disconnect): tear the upstream down
                # in a detached task, since this task's awaits are being
                # re-cancelled (see _stream_pipeline).
                _detach(_close_upstream(reader, resp))

    async def _retry_stream(self, ctx: _RequestContext) -> AsyncIterator[bytes]:
        body_r = build_decode_body(ctx.body, None, ctx.restore)
        async for chunk in self._decode_stream(
            ctx, body_r, ctx.rid + D_RETRY_SUFFIX, allow_retry=False
        ):
            yield chunk

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def _note_first_content(self, ctx: _RequestContext) -> None:
        if ctx.t_first_content is None:
            ctx.t_first_content = time.monotonic()

    def _log_timing(self, ctx: _RequestContext, status: int) -> None:
        if not self.config.log_timing:
            return
        now = time.monotonic()
        p_ms = (
            None
            if ctx.t_p_done is None
            else round((ctx.t_p_done - ctx.t_start) * 1000, 1)
        )
        ttfb_ms = (
            None
            if ctx.t_first_content is None
            else round((ctx.t_first_content - ctx.t_start) * 1000, 1)
        )
        ctx.log(
            logging.INFO,
            "timing",
            route=ctx.route,
            estimate=ctx.estimate,
            stream=ctx.stream,
            status=status,
            p_ms=p_ms,
            ttfb_ms=ttfb_ms,
            total_ms=round((now - ctx.t_start) * 1000, 1),
        )

    @staticmethod
    def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in _FORWARDED_REQUEST_HEADERS:
                out[key] = value
        return out

    @staticmethod
    def _error_response(
        rid: str, message: str, status: int, err_type: str = "BadRequestError"
    ) -> JSONResponse:
        return JSONResponse(
            _error_payload(message, status, err_type, rid),
            status_code=status,
            headers={"X-Request-Id": rid},
        )

    # ------------------------------------------------------------------ #
    # Passthrough routes
    # ------------------------------------------------------------------ #

    async def models(self) -> Response:
        try:
            resp = await self.client_d.get("/v1/models")
        except httpx.HTTPError as e:
            return self._error_response(
                "-", f"decode node unreachable: {type(e).__name__}", 502, "BadGateway"
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def health(self) -> Response:
        async def probe(client: httpx.AsyncClient) -> dict[str, Any]:
            try:
                r = await client.get("/health", timeout=httpx.Timeout(5.0))
                return {
                    "status": "ok" if r.status_code == 200 else "error",
                    "code": r.status_code,
                }
            except httpx.HTTPError as e:
                return {"status": "error", "error": type(e).__name__}

        p, d = await asyncio.gather(probe(self.client_p), probe(self.client_d))
        ok = p["status"] == "ok" and d["status"] == "ok"
        return JSONResponse(
            {"status": "ok" if ok else "error", "prefill": p, "decode": d},
            status_code=200 if ok else 503,
        )


def _consume_task_result(task: asyncio.Future) -> None:
    """Done-callback for detached tasks: retrieve the outcome so asyncio does
    not log 'exception was never retrieved'; failures were logged upstream."""
    if not task.cancelled():
        task.exception()


def _detach(coro: Awaitable[Any]) -> asyncio.Future:
    task = asyncio.ensure_future(coro)
    task.add_done_callback(_consume_task_result)
    return task


def _cancel_detached(task: asyncio.Future) -> None:
    if not task.done():
        task.cancel()
    task.add_done_callback(_consume_task_result)


async def _close_upstream(reader: asyncio.Future, resp: httpx.Response) -> None:
    """Stop the reader task and close the upstream response (connection)."""
    if not reader.done():
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader
    await resp.aclose()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #


def create_app(
    config: PDProxyConfig,
    proxy: PDProxy | None = None,
    *,
    startup: Callable[[], Awaitable[None]] | None = None,
    shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """The FastAPI app.  ``startup`` runs inside the lifespan BEFORE the app starts
    serving (uvicorn prints "Application startup complete." only after it returns;
    an exception fails the startup), ``shutdown`` runs after the proxy closed --
    the container supervisor brings the P/D pair up and down through these."""
    owns_proxy = proxy is None
    proxy = proxy or PDProxy(config)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):  # pragma: no cover - process teardown
        if startup is not None:
            await startup()
        try:
            yield
        finally:
            if owns_proxy:
                await proxy.aclose()
            if shutdown is not None:
                await shutdown()

    app = FastAPI(title="TT P/D disaggregation proxy", lifespan=_lifespan)
    app.state.proxy = proxy
    app.state.config = config

    async def _serve(api: str, request: Request) -> Response:
        rid = PDProxy.request_id(request.headers)
        try:
            body = await request.json()
        except Exception:
            return PDProxy._error_response(rid, "request body is not valid JSON", 400)
        return await proxy.handle(api, body, request.headers, request.is_disconnected)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _serve("/chat/completions", request)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _serve("/completions", request)

    @app.get("/v1/models")
    async def models() -> Response:
        return await proxy.models()

    @app.get("/health")
    async def health() -> Response:
        return await proxy.health()

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pd_proxy",
        description="OpenAI-compatible prefill/decode disaggregation proxy "
        "for the TT KV connector",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200")
    parser.add_argument(
        "--min-remote-tokens",
        type=int,
        default=512,
        help="prompts estimated shorter than this go straight to the decode node "
        f"(hard minimum {MIN_REMOTE_TOKENS_FLOOR})",
    )
    parser.add_argument(
        "--sse-keepalive",
        type=float,
        default=5.0,
        help="seconds between SSE keep-alive comments during the P leg; 0 disables",
    )
    parser.add_argument(
        "--retry-buffer-timeout",
        type=float,
        default=2.0,
        help="seconds after D's first byte to keep buffering for a retryable "
        "error event",
    )
    parser.add_argument(
        "--p-fallback",
        choices=("error", "local"),
        default="error",
        help="on prefill-node failure: return 502 (error) or decode locally (local)",
    )
    parser.add_argument(
        "--p-timeout",
        type=float,
        default=None,
        help="read timeout for the prefill leg in seconds (default: none)",
    )
    parser.add_argument(
        "--d-timeout",
        type=float,
        default=None,
        help="read timeout for the decode leg in seconds (default: none)",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--chars-per-token",
        type=int,
        default=4,
        help="prompt-length heuristic when --tokenizer is not given",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="optional HF tokenizer path for the prompt-length estimate",
    )
    parser.add_argument(
        "--no-d-retry",
        action="store_true",
        help="disable the one retry to D without params after a load rejection",
    )
    parser.add_argument("--log-timing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PDProxyConfig:
    return PDProxyConfig(
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
        min_remote_tokens=args.min_remote_tokens,
        sse_keepalive=args.sse_keepalive,
        retry_buffer_timeout=args.retry_buffer_timeout,
        p_fallback=args.p_fallback,
        p_timeout=args.p_timeout,
        d_timeout=args.d_timeout,
        connect_timeout=args.connect_timeout,
        chars_per_token=args.chars_per_token,
        tokenizer=args.tokenizer,
        d_retry=not args.no_d_retry,
        log_timing=args.log_timing,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = config_from_args(args)
    logger.info(
        "pd_proxy starting port=%d prefill=%s decode=%s min_remote_tokens=%d "
        "p_fallback=%s",
        args.port,
        config.prefill_url,
        config.decode_url,
        config.min_remote_tokens,
        config.p_fallback,
    )
    import uvicorn

    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
