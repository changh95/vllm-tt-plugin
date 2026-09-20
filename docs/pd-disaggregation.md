# Prefill/decode disaggregation with `TTMooncakeConnector`

`vllm_tt_plugin.kv_connector.tt_mooncake_connector.TTMooncakeConnector` is a vLLM
`KVConnectorBase_V1` for TT models: one vLLM instance prefills (`kv_role=kv_producer`),
another decodes (`kv_role=kv_consumer`), and a request's state moves between them through
[Mooncake](https://github.com/kvcache-ai/Mooncake)'s transfer engine. A proxy in front
sends each request to the producer first (`max_tokens=1`, `kv_transfer_params.do_remote_decode`)
and then to the consumer with the `kv_transfer_params` the producer returned, the same
round trip as vLLM's own Mooncake/NIXL connectors and its `toy_proxy_server`.

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
(`GET`/`DONE`/`CANCEL`). The consumer pulls the buffer with `transfer_sync_read` (TCP on one host,
RDMA across hosts), writes the KV into its paged cache (`paged_fill_cache`) and parks the state
until the request is given a decode slot.

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

Environment knobs (all optional): `QWEN36_PD_EXPORT_WARMUP=0` skips the producer's export
warm-up at start; `QWEN36_PD_GDN_IMPORT=trace|fillcache|host` selects the consumer's state import
path (trace = per-slot captured traces, pre-captured at start); `QWEN36_PD_VERIFY=1` re-reads
every imported slot and compares it with the host snapshot (debug); `QWEN36_PD_ALLOW_PTRACE=1`
lets `py-spy` attach to the detached engine core.

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
