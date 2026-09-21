# Prefill/decode disaggregation with `TTMooncakeConnector`

`vllm_tt_plugin.kv_connector.tt_mooncake_connector.TTMooncakeConnector` is a vLLM
`KVConnectorBase_V1` for TT models: one vLLM instance prefills (`kv_role=kv_producer`),
another decodes (`kv_role=kv_consumer`), and a request's state moves between them through
[Mooncake](https://github.com/kvcache-ai/Mooncake)'s transfer engine, or, when both run
on one host, through a shared-memory mapping of the producer's staging buffer. A proxy in
front sends each request to the producer (`max_tokens=1`,
`kv_transfer_params.do_remote_decode`) and to the consumer (`do_remote_prefill` with the
producer's side channel and a `transfer_id`); it may do so serially, with the
`kv_transfer_params` the producer returned (the round trip of vLLM's own Mooncake/NIXL
connectors and its `toy_proxy_server`), or concurrently under a `transfer_id` it chose
itself (see *Proxy fan-out*).

## What moves

TT models keep their KV cache in ttnn tensors on a mesh the model owns (vLLM sees TP=1 and one
`FullAttentionSpec` group), and hybrid models keep per-request state that is not KV at all. The
connector therefore does not touch vLLM's `kv_caches`; it asks the model:

- `export_kv_blocks(block_ids)` / `import_kv_blocks(kv, block_ids)` for the request's paged KV
  blocks of every attention layer, and
- a per-slot snapshot of any recurrent state (Qwen3.x Gated-DeltaNet: the fp32 recurrent state
  and the conv taps of 48 layers, parked by the model under `pd_gdn_capture[slot]` during the
  prefill and written into the decode slot by `import_gdn_slot` on the consumer).

The producer prefills `N-1` prompt tokens (mamba/GDN rule: the consumer recomputes the last
token from the transferred state, so its first step is a decode step, never a prefill), stages
KV + state into one contiguous host buffer registered with Mooncake, and serves a ZMQ side channel
(`GET`/`DONE`/`CANCEL`; a ROUTER socket, so a `GET` for a transfer that is not staged yet is
parked, up to 4 s, and answered the moment the producer stages it while other clients are served).
The consumer writes the KV into its paged cache (`paged_fill_cache`) and parks the state until the
request is given a decode slot. How the bytes get there depends on where the two run:

- **Same host (default, `QWEN36_PD_SHM=1`).** The producer's staging pool is made of
  `/dev/shm/qwen36-pd-<pid>-<id>` files mapped `MAP_SHARED` (the Mooncake registration is the
  same pointer). The `GET` reply carries `{"host": <hostname:boot_id>, "shm": {"name", "offset",
  "nbytes"}}`; a consumer whose host identity matches maps the segment read-only (one mapping per
  pooled buffer, cached), hands the runner views into it, and sends `DONE` only when the runner has
  written the decode slot (the release callback the connector parks with the snapshot). No copy is
  made: the same bytes are uploaded from the mapping, so results are bit-identical to the pull.
  The producer's staging lifetime grows by the consumer's import latency (~50 ms). A producer that
  dies without `shutdown` leaves its files behind; the bundle proxy (`pd_app.py`) removes
  `qwen36-pd-*` files of dead pids at start, and `unlink_stale_shm_segments()` does the same.
- **Otherwise** (`QWEN36_PD_SHM=0` on either side, a different host, or a segment that does not
  open) the consumer pulls the buffer with `transfer_sync_read` into a registered receive buffer
  (TCP on one host, RDMA across hosts) and sends `DONE` as soon as the copy is complete.

Every `[pd] pulled ...` log line says which path ran (`via shm (... map X ms ...)` vs
`pull X ms = Y GB/s`), and both sides log the same payload digest.

Reference implementation of the model side: tt-metal
`models/demos/blackhole/qwen36/tt/pd_transfer.py`.

## Configuration

Producer (prefill instance):

```json
{"kv_connector": "TTMooncakeConnector",
 "kv_connector_module_path": "vllm_tt_plugin.kv_connector.tt_mooncake_connector",
 "kv_role": "kv_producer",
 "kv_connector_extra_config": {"side_channel_host": "127.0.0.1", "side_channel_port": 18100,
                               "mooncake_protocol": "tcp"}}
```

Consumer (decode instance): the same with `"kv_role": "kv_consumer"` and no side-channel fields.
Pass either as `--kv-transfer-config`. `side_channel_host`/`side_channel_port` are what the
producer advertises to consumers in `kv_transfer_params` (`remote_host`/`remote_port`), so on two
hosts set the producer's reachable address and `"mooncake_protocol": "rdma"`.

Both instances run the same model with the same `--block-size`. The producer needs only enough
`--max-num-seqs` for prefills in flight (8 is plenty); the consumer sizes the decode batch.

Environment knobs (all optional): `QWEN36_PD_SHM=0` disables the same-host shared-memory
hand-off (Mooncake pull instead); `QWEN36_PD_EXPORT_WARMUP=0` skips the producer's export
warm-up at start; `QWEN36_PD_GDN_IMPORT=trace|fillcache|host` selects the consumer's state import
path (trace = per-slot captured traces, pre-captured at start); `QWEN36_PD_VERIFY=1` re-reads
every imported slot and compares it with the host snapshot (debug); `QWEN36_PD_ALLOW_PTRACE=1`
lets `py-spy` attach to the detached engine core.

## Proxy fan-out

The serial round trip puts the consumer's admission (tokenize, schedule, allocate, start the
pull thread) after the producer's whole prefill plus two HTTP hops. A proxy can instead post to
both at once:

1. Mint a `transfer_id` (any unique string). vLLM's `InputProcessor` rewrites the engine request
   id as `<X-Request-Id>-<8 random hex>`, differently on every instance, which is why the
   producer's `request.request_id` cannot serve as the key.
2. POST to the producer with `kv_transfer_params = {"do_remote_decode": true,
   "do_remote_prefill": false, "transfer_id": <id>}` (`max_tokens=1`, non-streaming). The
   producer files the staging under that id and echoes it in the returned
   `kv_transfer_params.transfer_id` (without one it uses its engine request id, as before).
3. At the same time POST the client's request to the consumer with `kv_transfer_params =
   {"do_remote_prefill": true, "do_remote_decode": false, "remote_host": <producer side channel
   host>, "remote_port": <its port>, "transfer_id": <id>}`. `num_tokens` may be omitted: the
   consumer derives it from its own tokenization (prompt length minus the recomputed last token)
   and the worker checks it against the payload header; on a mismatch it logs an error, drops
   the payload and lets the runner prefill locally. Its `GET` parks on the side channel until the
   producer stages.
4. Await the producer. On a non-200 (or missing `kv_transfer_params`), cancel/close the consumer
   request: the consumer aborts it (its connector stops the in-flight pull, reports it so the
   scheduler frees the blocks it held back, and sends `CANCEL` if the pull never started) and
   returns the producer's error to the client. If the producer did not echo the `transfer_id`
   (an older connector), abandon the consumer request the same way and run the serial round trip
   with the producer's parameters.
5. Stream the consumer's response.

Both bundled proxies implement this: `models/demos/blackhole/qwen36/server/pd_app.py`
(`QWEN36_PD_PROXY_FANOUT=0` for the serial path) and `scripts/pd_proxy.py` (`--side-channel-host/
--side-channel-port`, `--no-fanout`) in the tt-metal/qwen36 workspace.

## Scheduling notes

- The TT scheduler alternates prefill-only and decode-only steps. A request whose prefill happened
  remotely is admitted into the base scheduler's mixed step (running decodes plus the import) rather
  than through the prefill-only path, which would advance the running requests' recurrent slots as
  padding rows.
- Connector metadata is re-shipped every `build_connector_meta` until the worker acknowledges the
  transfer, because the TT scheduler may discard a zero-token prefill-only output; the worker
  de-duplicates. A zero-token step still runs so finished transfers are reported.
- Block-table rows are zero-padded beyond the request's blocks so a masked-bucket prefill never
  writes padding K/V into another request's blocks.

## Requirements

`mooncake-transfer-engine` (PyPI). Its wheel links `libcuda.so.1`, `libcudart.so.12`,
`libcurl.so.4` and rdma-core; on a host without an NVIDIA driver provide `libcudart` from
`nvidia-cuda-runtime-cu12` and a `libcuda.so.1` stub on `LD_LIBRARY_PATH` (the Qwen3.8-27B
P150x8 bundle ships both as `tt-mooncake-sysdeps`).
