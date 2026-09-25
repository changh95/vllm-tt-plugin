# Speculative decoding with a model-owned drafter (Qwen3.6 MTP)

This note describes how the TT plugin runs vLLM speculative decoding for a model
that owns its whole draft -> verify -> commit loop: today Qwen3.6-27B (`Qwen3.8-27B`
checkpoint) with its native MTP head as the drafter, on the decode engine of the
4+4 prefill/decode stack (tt-metal `models/demos/blackhole/qwen36`). The plugin
keeps vLLM's bookkeeping honest; it never runs a drafter or a rejection sampler
itself.

Switches (both must be set; either alone leaves the served path byte-identical
to plain decoding):

- `QWEN36_SPEC_MTP=1` in the environment of both engines (the prefill engine
  runs the MTP prefill and ships the head's KV + the request's hidden row; the
  decode engine allocates the head and imports them).
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --no-async-scheduling` on the decode engine (`scripts/serve_pd.sh` adds it
  when `QWEN36_SPEC_MTP=1`, `QWEN36_SPEC_K` overrides the draft count).

The platform accepts a `speculative_config` only for a model class declaring
`model_capabilities["supports_speculative_mtp"]`, only with `method: mtp`, and
only with synchronous scheduling (the drafts of a step are produced by that
step's verify). The model reads `tt_speculative_k` (set by the runner from the
config) at warm-up.

## Contract with vLLM

vLLM 0.13's V1 scheduler already carries everything a model-owned drafter needs:

| vLLM side | Plugin / model side |
|---|---|
| `Request.spec_token_ids` set by `Scheduler.update_draft_token_ids` after every step (`EngineCore.post_step` -> `Worker.take_draft_token_ids`) | The runner returns `DraftTokenIds` for every request the step emitted tokens for: the model's drafts, or `[]` (plain step / no drafter state). |
| `schedule()` gives a decode request `1 + K_s` tokens (its last token + `K_s` drafts), allocates their KV positions, advances `num_computed_tokens` by `1 + K_s`, adds `1 + K_s` output placeholders (`AsyncScheduler`, used in sync mode too) | The verify writes KV at positions `P_s .. P_s + K_s`; rejected positions are overwritten by the next step. |
| `update_from_output` receives `1 .. 1 + K_s` tokens per request; `num_rejected = K_s - (len - 1)` is subtracted from `num_computed_tokens` and from the placeholders; every token goes through the stop checks | The runner emits `[d_1 .. d_a, bonus]` for `a` accepted drafts (`a = 0`: the plain next token). A step never emits more than `1 + K_s` tokens for a request (the placeholder counter asserts it). |
| `SpecDecodingStats` (acceptance per step, logged by `SpecDecodingLogging` with the engine stats) | Computed by vLLM from the output lengths; no extra work. |

The model runner therefore only (1) builds `TTModelInput.spec` on a decode step
(`spec_mtp.TTSpecStepInput`: request per padded row, scheduled drafts per row,
greedy eligibility, the scheduler's flush request), (2) recognises a model
`SpecStepResult` in the synchronous decode path and applies its variable-length
token lists (`_finish_spec_step`), (3) stores the next drafts for
`take_draft_token_ids`, and (4) publishes the hold sidecar described below.

Eligibility: a step takes the verify path only when every live request is
greedy on device (temperature 0, no penalties, no logprobs, no structured
output, no host-only sampling features) -- the verify's per-row argmax is then
the plain decode's token, so committed streams are bitwise the plain ones. One
request outside that set makes the whole step a plain decode (its drafts are
rejected). `spec_mtp.request_forces_plain` is the request-level predicate the
scheduler uses to predict such a step.

## The model's state machine (tt-metal `qwen36/tt/spec_serving.py`)

The verify grid is the decode slot grid: grid user `s` is decode slot `s` (the
GDN kernel commits grid user `s` into state slot `s`), so a `(w, T)` plan covers
slots `0 .. w-1`, with padding users for empty slots (position -1: KV update and
SDPA skipped; the attention / GDN outputs of their rows are zeroed by a `where`
mask so a stale slot can never leak NaN into the body's row-mixing 0/1 matmuls).

Ladder by `w_grid = highest live slot + 1` (rows per user `T = k + 1`,
`R = w*T <= 32` so every plan runs the decode step's own ops):

| `w_grid` | bucket `w` | `T` |
|---|---|---|
| 1 / 2 / 3-4 / 5-8 | 1 / 2 / 4 / 8 | 4 (k = 3) |
| 9-10 | 10 | 3 |
| 11-16 | 16 | 2 |
| 17-32 | plain decode (`QWEN36_SPEC_ALLOW_FRACTURED=1` adds `(32, 2)`: R = 64, the fractured reduce-scatter numerics that flip greedy near-ties) | -- |

`QWEN36_SPEC_LADDER="1:4,2:4,4:4,8:4,10:3,16:2"` overrides. `T` is clamped to
`num_speculative_tokens + 1`.

Lazy GDN prefix: the multi-token GDN kernel commits the previous step's accepted
rows `1..a_s` from the plan's `qkv_prev` buffers (row layout `s*T + j`). On a plan
change the pending rows are carried into the new plan's buffers with an exact
0/1 row-permutation matmul (`verify_grid.migration_matrix`), which is possible
iff every pending `a_s <= T_new - 1`: always true for bucket changes inside a
band and for band-down changes (fewer users, larger `T`). A band-UP change (an
admission above the band's width, smaller `T`) may not fit, and a plain step
never does. Such transitions need a **flush** first: a verify step at the current
plan with zero drafts (every scheduled draft rejected, one token per user,
pending rows committed).

## The admission-hold protocol

A flush cannot include a user the current plan does not cover, and every
scheduled request must emit `1..1+K_s` tokens, so the flush has to run BEFORE
the admission. The runner publishes `HoldInfo` on every decode `ModelRunnerOutput`
(`spec_mtp.set_tt_spec_hold`):

- `pending_any`: some live request has a lazily committed prefix;
- `slots_before_crossing`: free rows below the current band's width (how many
  admissions stay inside the band); `None` when the pending rows survive every
  reachable band.

`TTScheduler.update_from_output` stores it; before admitting waiting work while
running decodes exist, `_spec_hold_step` schedules one decode-only step flagged
`flush` (`spec_mtp.set_tt_spec_flush`) iff `pending_any` and (a ready request
forces plain decode or `n_ready > slots_before_crossing`). The model runs the
flush, the next output reports `pending_any = False`, the admission proceeds.
A crossing the scheduler did not hold raises `SpecProtocolError` in the model
(a bug signal; never silent state corruption). Cost: one extra decode step
(~one token per user) per band-up crossing or per plain-forcing admission.

## Per-step device flow (tt-metal `qwen36/tt/spec_decoder.py`)

1. `SpecServingState.begin_step`: sync slot owners, pick the plan (or plain),
   check / plan the `qkv_prev` migration, truncate each request's drafts to `k`
   (a `-1` placeholder ends the usable prefix), find freshly admitted users
   whose hidden row arrived (`MTPHead.set_hidden_in`, from the P/D payload's
   `mtp.hidden` or the local prefill hook).
2. Catch-up draft step for those users: one MTP step `(t'_s, h_{P_s - 1})` at
   `P_s - 1` fills the head's KV hole the prefill left; its draft is unused (the
   scheduler gave a new request no draft slots).
3. Verify (traced per plan): rows `[t'_s, d_1 .. d_k]` at `P_s + j`; commit
   `argmax[0..a_s]`; the accepted rows' hidden states are selected into the
   drafter (exact 0/1 matmul) and `k` chained draft steps (traced per width)
   produce the next step's drafts.
4. `SpecStepResult` (committed tokens and next drafts per grid row, `HoldInfo`)
   goes back through `decode_forward`; the runner packs it.

Every program runs compiled before any trace is captured: the decoder is built
and compiled in the first decode warm-up phase (`enable_trace=False`) and its
traces are captured in the second, before the decode-bucket captures
(`tests/VERIFY_W32_AUDIT.md` rule).

## Evidence

- CPU: `tt-metal qwen36/tests/test_spec_serving_cpu.py` (ladder, migration
  matrix, hold protocol over a simulated engine with joins / leaves / flushes /
  plain-forced steps against a slot oracle; 25 tests) and
  `vllm-tt-plugin/tests/test_spec_mtp.py` (policy, sidecars, a real
  `TTScheduler` with a `mtp` speculative config: drafts scheduled, rejections
  accounted, the hold step inserted; the runner's variable-length output).
- Device (half B, TP=4): `qwen36/tests/test_spec_serving_scratch.py` -- a
  46-step scenario ramping 3 -> 9 -> 17 users and draining, with a
  non-greedy user forcing plain steps: bucket changes, a held band-up crossing
  (flush), migrations, padding users, catch-up steps and plain decode above 16
  users -- every committed stream bitwise equal to the plain traced decode
  (`logs/spec_serving2.log`).
- Served: see the table in the delivery report (`scripts/serve_pd.sh start`
  with `QWEN36_SPEC_MTP=1`).

## Not done yet

- Only greedy requests speculate; sampled requests force the whole step to plain
  decode (device rejection sampling would lift this).
- Above 16 concurrent users the ladder falls back to plain decode (R > 32 needs
  the fractured verify path whose numerics flip greedy near-ties; opt in with
  `QWEN36_SPEC_ALLOW_FRACTURED=1`).
- Logprobs on the speculative path (the verify reads back only the per-row
  argmax and max).
- A DFlash-style parallel drafter would slot in behind the same `draft` call.
