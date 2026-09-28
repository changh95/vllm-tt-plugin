# Speculative decoding with a model-owned drafter (Qwen3.6 MTP head / DFlash2)

This note describes how the TT plugin runs vLLM speculative decoding for a model
that owns its whole draft -> verify -> commit loop: today Qwen3.6-27B (`Qwen3.8-27B`
checkpoint) on the decode engine of the 4+4 prefill/decode stack (tt-metal
`models/demos/blackhole/qwen36`), with one of two drafters: the checkpoint's
native MTP head, or the DFlash2 block-diffusion drafter
(`z-lab/Qwen3.8-27B-DFlash2`). The plugin keeps vLLM's bookkeeping honest; it
never runs a drafter or a rejection sampler itself.

Switches (the first two must be set; either alone leaves the served path
byte-identical to plain decoding):

- `QWEN36_SPEC_MTP=1` in the environment of both engines (the master switch:
  the prefill engine computes the drafter's per-request state and ships it, the
  decode engine builds the drafter and imports it).
- `--speculative-config '{"method":"mtp","num_speculative_tokens":K}'
  --no-async-scheduling` on the decode engine (`scripts/serve_pd.sh` adds it
  when `QWEN36_SPEC_MTP=1`; `QWEN36_SPEC_K` overrides `K`, default 3 for the
  MTP head and 7 for DFlash2). `K` is the ladder's MAXIMUM draft count: the
  model proposes `<= K` drafts per user per step (the band's `k`), vLLM
  schedules exactly what was proposed. `method` stays `mtp` for both drafters
  (vLLM only does the token bookkeeping; the drafter is the model's choice).
- `QWEN36_SPEC_DRAFTER=mtp|dflash2|hybrid` (default `mtp`) in the environment of
  both engines selects the drafter (tt-metal `tt/aux_hidden.py::spec_drafter`);
  `hybrid` keeps both resident on the decode engine and picks one per step by the
  batch width (the bundle's setting; see "The hybrid policy" below).

The platform accepts a `speculative_config` only for a model class declaring
`model_capabilities["supports_speculative_mtp"]`, only with `method: mtp`, and
only with synchronous scheduling (the drafts of a step are produced by that
step's verify). The model reads `tt_speculative_k` (set by the runner from the
config) at warm-up.

## The two drafters

| | MTP head (`mtp`) | DFlash2 (`dflash2`) |
|---|---|---|
| draft | `k` chained steps of the checkpoint's one-layer MTP head (traced per width), 2.4-3.4 ms each | ONE traced 8-row block step at the bucket width (1 anchor + 7 mask rows per user, non-causal block attention over the drafter's context K/V, target LM head, codebook selector walk on host): 7 drafts per user at any `k`, 8.2 ms (w=1) .. 16 ms (w=16) |
| per-request state | the head's KV (the payload's 17th attention layer, `mtp.kv.0`) + the request's last post-norm hidden row (`mtp.hidden`) | 5 layers of CONTEXT K/V = projections of the target's aux hidden rows (layers 5/19/33/47/61) at every committed position; the prompt's rows come from the prefill engine as the payload's KV group `dflash2` (`docs/pd-disaggregation.md`), every later row from the verify step itself |
| verify plan | `VerifyStep(keep_hidden=True)`: the accepted row's hidden state is selected into the head (exact 0/1 matmul) | `VerifyStep(keep_aux_hidden=True)`: the grid rows' aux hiddens (`plan.out_aux`) feed a traced `commit(plan, positions)` per plan that writes the drafter's K/V at `P_s + j` (rejected rows are overwritten by the next commit / block before they are read) |
| admission | catch-up draft step at `P_s - 1` from the imported hidden row (fills the head's KV hole) | none: the connector imports the prompt's context blocks before admission (`pd_transfer.import_kv_groups`), the runner reports it (`spec_note_admission` -> `SpecDecoder.note_context`), the first verify step at `P_s = N` (row 0 = the prefill engine's first token) commits position `N`, the first block draft reads a gap-free context. A request whose payload had no group decodes with no drafts (its row is padding in the draft step). |
| DRAM per chip (D) | one layer + 1/16 of the KV pool | 540 MiB weights + 2720 B/token (2.7 GiB at the 1,052,672-token pool) |
| ladder (default) | `1:4,2:4,4:4,8:4,10:3,16:2`, R <= 32 (bitwise) | `1:8,2:8,4:8,8:8,16:2`: T = 8 (k = 7) up to 8 users -- buckets 1/2/4 are R <= 32 (bitwise), bucket 8 is R = 64 (fractured path, near-tie-bounded); T = 2 (k = 1, R = 32, bitwise) for 9..16; plain above |

tt-metal `tt/spec_decoder.py` puts both behind one adapter interface
(`_MtpDrafter` / `_DFlash2Drafter`); the step loop, the ladder machinery, the
flush / migration protocol and the plugin side are shared. With
`QWEN36_SPEC_DRAFTER` unset (or `mtp`) every op of the MTP loop runs in the
2026-09-25 order; with `QWEN36_SPEC_MTP` unset nothing speculative is built.

### The bitwise / near-tie bound: `QWEN36_SPEC_ALLOW_FRACTURED`

A verify grid of R <= 32 rows runs the decode step's own fused-all-reduce ops,
so its argmax rows ARE the plain decode's: committed streams are bitwise the
plain greedy ones. R > 32 runs the fractured reduce-scatter path whose
different summation order flips greedy near-ties (plain-logit gaps of 0.125 /
0.25 = one bf16 ulp at |logit| 16..32; tt-metal `tests/DFLASH2_RESULTS.md`,
`tests/MTP_SPEC_RESULTS.md`): a rare token differs, the stream stays a valid
greedy continuation of a near-tied distribution, and the divergence is
draft-independent (the same flips at the same prompt positions for every
drafter). The knob:

| `QWEN36_SPEC_ALLOW_FRACTURED` | MTP ladder | DFlash2 ladder |
|---|---|---|
| unset (default) | R <= 32: `1:4,2:4,4:4,8:4,10:3,16:2` | R <= 64: `1:8,2:8,4:8,8:8,16:2` -- the (8,T=8) plan is the fractured one (E2E: near-ties only, x2.13 tokens/s at 8 users) |
| `0` | same as unset | R <= 32 only: `1:8,2:8,4:8,8:4,16:2` ((8,T=4): x1.86, bitwise) |
| `1` | + `32:2` (R = 64) | + `32:2` (R = 64): 17..32 users speculate at k = 1 |

`QWEN36_SPEC_LADDER="w:T,..."` overrides either default; its plans must respect
the knob's row bound (`1:8,2:8,4:8,8:8,16:4` selects the R = 64 k = 3 band for
9..16 users instead of the bitwise T = 2 one).

## The hybrid policy (`QWEN36_SPEC_DRAFTER=hybrid`)

The A/B below (2026-09-28) has the block drafter winning only at small batches
and the MTP head above: on GSM8K text DFlash2's bitwise ladder does 9.4 / 10.3 ms
TPOT at 1 / 4 users against MTP's 12.1 / 13.0, but 15.3 vs 14.4 at 8 and 31 vs 23
at 16 (its 16 ms block step buys one draft in the T = 2 band), and the R <= 64
ladder that would rescue the 8-user band is not self-consistent. The hybrid keeps
BOTH drafters resident on D and lets the ladder choose per plan by width
(tt-metal `tt/spec_serving.py` `Ladder.drafter_for`):

| `w_grid` | plan | drafter |
|---|---|---|
| 1 / 2 / 3-4 | (1,8) / (2,8) / (4,8) -- T = 8, k = 7, R <= 32 | DFlash2 |
| 5-8 / 9-10 / 11-16 | (8,4) / (10,3) / (16,2) -- the mtp ladder's bands | MTP head |
| 17-32 | plain decode (`QWEN36_SPEC_ALLOW_FRACTURED=1` adds the MTP (32,2) tail) | -- |

Default ladder `1:8,2:8,4:8,8:4,10:3,16:2`, every plan R <= 32 (bitwise by
construction; `QWEN36_SPEC_LADDER` overrides keep the width rule: buckets up to 4
draft with DFlash2, wider ones with MTP). `QWEN36_SPEC_K` defaults to 7 (the
DFlash2 band's maximum; the MTP bands clamp their own T). A drafter change is a
plan change and goes through the protocol below unchanged: the 4 -> 5 crossing
(T 8 -> 4) is a band-up change held for a flush whenever a prefix is pending,
5 -> 4 migrates the pending MTP-band rows (<= 3 <= 7) into the (4,8) plan, 16 ->
17 flushes into plain decode as before. Nothing is re-prefilled at a switch
because both drafters' per-request state is kept current on EVERY spec step
(tt-metal `tt/spec_decoder.py` `_HybridDrafter`):

- P computes and ships BOTH states: the MTP prefill hook (head KV `mtp.kv.0`
  over `0..N-2` + `mtp.hidden`) and the DFlash2 aux hook (KV group `dflash2`)
  both fire per prefilled segment; the version-3 payload carries both fields
  and D imports both (`import_mtp_hidden` + the group import, as in the
  single-drafter modes).
- Verify plans keep both the post-norm rows (`out_hidden`) and the aux rows
  (`out_aux`). After every verify: the DFlash2 `commit(plan, P)` writes the
  drafter's context K/V for the grid rows (also when MTP drafts); the MTP head's
  K/V for the committed rows is written by `MTPHead.keep_current(plan, argmax)`
  -- one traced program per plan that runs the head's LAYER (no lm_head) over the
  grid rows with `fc(norm(embed(x_{P_s+j+1})), norm(h_{P_s+j}))` (the verify's
  argmax and post-norm row of row (s, j)) and writes K/V at `P_s + j` through the
  plan's positions / page table -- on every DFlash2-drafted step and on the first
  MTP-drafted step after one (or after a plain step); inside the MTP band the
  chain writes the head's K/V exactly as in mtp mode. The MTP hidden select and
  chain run only when MTP drafts; the DFlash2 block step only when it drafts.
- The MTP catch-up step of a freshly admitted user (its imported hidden row ->
  the head's K/V at `N-1`) runs whichever drafter is active; the DFlash2 context
  report (`spec_note_admission`) is taken as in dflash2 mode; a request without
  DFlash2 context proposes nothing while DFlash2 drafts and drafts normally under
  MTP.
- Cost (served, D log `[hybrid: ...]` lines): DFlash2-drafted steps pay the
  keep-current, 0.7 ms (one decoder layer over <= 32 rows) next to the 0.85 ms
  commit; MTP-drafted steps pay the DFlash2 commit, 1.1 ms, next to the 0.3-0.4 ms
  select. DRAM on D after warm-up: 5698 MiB free per chip (both drafters + the
  full 1,052,672-token DFlash2 context pool: mtp alone leaves 9078, dflash2 alone
  6497; the pool was not shrunk). P side: both hooks add ~11 ms to a short
  prefill (PREFILL_TIMING total 118.6 ms vs 107.2 mtp / 112.5 dflash2 at <= 256
  tokens; 4k prompts 743 vs 582 / 742 ms) and ~1-4 MiB to the payload (164.3 MiB
  per 179-token request vs 160.5 / 163.5; 4k prompts 500 vs 420 / 484 MiB; the
  plain v10 payload is ~158 MiB at that length), export 8.5 ms vs 8.1 / 8.3.
- With `mtp` / `dflash2` selected, none of this runs: those loops are op for op
  the 2026-09-28 ones.

Exactness: CPU tests of the hybrid ladder and its 4 <-> 5 / 16 <-> 17 switches
(`test_spec_serving_cpu.py`: flush on the way up, migration on the way down,
streams greedy); device scenario (half B, `test_spec_serving_scratch.py
SPEC_DRAFTER=hybrid`, 3 -> 5 -> 9 -> 17 users and back with a no-context user
and a plain-forcing user): 50 steps (8 DFlash2-drafted, 9 MTP-drafted, 30
plain), 3 drafter switches ((4,8) -> flush -> (8,4); (10,3) -> flush -> plain;
(16,2) -> (2,8) by migration), every committed stream bitwise the plain traced
decode, the unheld crossing raises `SpecProtocolError`
(tt-metal `logs/spec_serving_hybrid1.log`). Served gate (all 8 chips): the table
under "Served gate: hybrid vs the single drafters" below.

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

Ladder by `w_grid = highest live slot + 1` (rows per user `T = k + 1`; the
smallest bucket covering the highest live slot):

| `w_grid` | bucket `w` | MTP `T` | DFlash2 `T` (default) |
|---|---|---|---|
| 1 / 2 / 3-4 | 1 / 2 / 4 | 4 (k = 3) | 8 (k = 7), R <= 32 bitwise |
| 5-8 | 8 | 4 (k = 3) | 8 (k = 7), R = 64 near-tie-bounded |
| 9-10 | 10 (MTP) | 3 | 2 (bucket 16) |
| 11-16 | 16 | 2 | 2 |
| 17-32 | plain decode (`QWEN36_SPEC_ALLOW_FRACTURED=1` adds `(32, 2)`) | -- | -- |

`QWEN36_SPEC_LADDER` overrides (see the knob table above). `T` is clamped to
`num_speculative_tokens + 1`, so a smaller `QWEN36_SPEC_K` shortens every band
(DFlash2 with `K=3`: the first 3 tokens of the block's selected path, T = 4
plans). Flush / migration semantics do not depend on the drafter: a `T` change
needs the pending rows to fit (`a_s <= T_new - 1`) or a flush first, exactly as
below; with the T = 8 band a pending prefix can be 7 rows, so every admission
that leaves the band (the 9th user) is held for one flush step.

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
   whose drafter state arrived (MTP: `MTPHead.set_hidden_in` from the P/D
   payload's `mtp.hidden` or the local prefill hook; DFlash2: the runner's
   `spec_note_admission` from the payload's `kv_groups["dflash2"]` metadata).
2. MTP only: catch-up draft step for those users: one MTP step
   `(t'_s, h_{P_s - 1})` at `P_s - 1` fills the head's KV hole the prefill
   left; its draft is unused (the scheduler gave a new request no draft slots).
3. Verify (traced per plan): rows `[t'_s, d_1 .. d_k]` at `P_s + j`; commit
   `argmax[0..a_s]`. MTP: the accepted rows' hidden states are selected into
   the drafter (exact 0/1 matmul) and `k` chained draft steps (traced per
   width) produce the next step's drafts. DFlash2: `drafter.commit(plan, P)`
   (traced per plan) projects the grid rows' aux hiddens into the drafter's
   context K/V at `P_s + j`, then one traced block step at the bucket width
   (anchor = the last committed token at `P_s + a_s`, block rows at
   `P_s + a_s + 1 ..`) yields 7 drafts per user; the first `k` are proposed.
   Users without context (no group in their payload) are padding rows of the
   draft step and propose nothing.
4. `SpecStepResult` (committed tokens and next drafts per grid row, `HoldInfo`)
   goes back through `decode_forward`; the runner packs it.

Warm-up order on D (`qwen36_vllm.py`): the drafter is built right after the
KV caches (`allocate_kv_cache`: MTP head, or `DFlash2Drafter` = weights +
5 x [k, v] paged context caches mirroring the main cache's blocks, registered
as the payload KV group `dflash2`); decode warm-up phase 1 builds the ladder's
verify plans and draft buffers, compiles every program (for DFlash2 also the
per-plan commit and the KV-group import programs of every block bucket,
`pd_transfer.kv_group_import_warmup`), phase 2 captures the draft-step traces,
then each verify trace followed by its commit trace (which reads the verify
output buffer's fixed address), all before the decode-bucket captures.

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
- Served (2026-09-25, P/D 4+4 stack, `scripts/serve_pd.sh` defaults: decode
  bucketing off, both runs at the same AICLK; `scripts/spec_served_suite.sh`):

  | 128/128, concurrency | plain TPOT ms / tok/s | spec TPOT ms / tok/s | speed-up |
  |---|---|---|---|
  | 1 | 35.5 / 27.3 | 13.3 / 67.4 | 2.5x |
  | 4 | 36.0 / 103 | 15.0 / 226 | 2.2x |
  | 8 | 36.6 / 195 | 16.5 / 394 | 2.0x |
  | 16 | 38.1 / 353 | 24.6 / 516 | 1.5x (k = 1 band) |
  | 32 | 40.1 / 617 | 40.3 / 623 | plain above the ladder |

  GSM8K real text: 13.2 / 15.5 / 23.9 ms TPOT at 1 / 8 / 16 vs 35.4 / 36.6 /
  38.1 plain; vLLM acceptance length 3.5 (k = 3) on GSM8K, 2.9-3.1 on random
  prompts; lm-eval GSM8K 0.82 both ways with all 200 generations identical
  token for token; 5900 speculative steps, 14 held admissions (flushes), 26
  migrations, no protocol error. TTFT unchanged (P side).

## Served A/B: DFlash2 vs MTP (2026-09-28, 4+4 P/D stack, all 8 chips)

`scripts/spec_ab_gate.sh` / `scripts/spec_gate_variant.sh` (AICLK under firmware
control, decode bucketing on, D pool 1,052,672 tokens; `QWEN36_SPEC_K` 7 / 3;
`vllm bench serve`, mean TPOT / mean TTFT / aggregate output tok/s, streaming,
temperature 0). Three configurations: `dflash2` with the default R <= 64 ladder,
`dflash2` with `QWEN36_SPEC_ALLOW_FRACTURED=0` (bitwise ladder, the bundle's),
`mtp` (default ladder).

| 128/128 random, users | dflash2 R<=64 TPOT ms / tok/s | dflash2 bitwise TPOT / tok/s | mtp TPOT / tok/s |
|---|---|---|---|
| 1 | 11.7 / 75 | 11.7 / 75 | 12.1 / 74 |
| 4 | 14.8 / 231 | 14.9 / 231 | 13.8 / 245 |
| 8 | 21.1 / 313 | 17.9 / 370 | 15.4 / 429 |
| 16 | 32.4 / 430 | 32.7 / 431 | 23.1 / 574 |
| 32 (plain) | 39.7 / 701 | 39.5 / 699 | 38.0 / 723 |

| GSM8K text (OSL 128), users | dflash2 R<=64 TPOT ms / tok/s | dflash2 bitwise TPOT / tok/s | mtp TPOT / tok/s |
|---|---|---|---|
| 1 | 9.4 / 91 | 9.4 / 93 | 12.1 / 74 |
| 4 | 10.4 / 313 | 10.3 / 316 | 13.0 / 265 |
| 8 | 14.1 / 472 | 15.3 / 445 | 14.4 / 471 |
| 16 | 31.7 / 446 | 31.1 / 442 | 22.9 / 599 |
| 32 (plain) | 39.0 / 672 | 39.3 / 694 | 37.6 / 694 |

| | dflash2 R<=64 | dflash2 bitwise | mtp |
|---|---|---|---|
| TTFT 128 / 1k / 4k (latency_probe, ms) | 180 / 310 / 870-1080 | 168-188 / 310-368 / 870-1110 | 160 / 268 / 690-940 |
| vLLM acceptance length (`SpecDecoding metrics`), lm-eval GSM8K phase (8 users) | 5.9-6.3 (k = 7 at 8 users; per-position 0.93, 0.87, 0.82, 0.76, 0.71, 0.65, 0.58) | 3.5-3.6 at 8 users (k = 3 band; 0.94 / 0.87 / 0.81); 3.3-4.2 at 1-4 users on the OSL-128 GSM8K bench prompts (k = 7; positions 0.83-0.93, 0.56-0.79, 0.36-0.60, 0.25-0.47, 0.14-0.27, 0.09-0.13, 0.04-0.06) | 3.4-3.6 (k = 3; 0.95 / 0.86 / 0.76) |
| GSM8K lm-eval 200, flexible-extract | 0.835 +- 0.026 | 0.82 +- 0.027 | 0.82 +- 0.027 |
| det_probe (3 x 3 prompts) | deterministic | deterministic | deterministic |
| self-consistency, conc 1..32 x 64 tokens | **10 / 63 streams differ** (conc 8: 1/8, 16: 5/16, 32: 4/32; near-tie flips of the (8,T=8) plan) | **ALL MATCH** (63/63) | ALL MATCH |
| spec / plain steps, flushes, migrations, protocol errors | 4750 / 2427, 9, 41, 0 | 5850 / 2430, 22, 37, 0 | 6600 / 2410, 17, 46, 0 |

Reading: on real text the block drafter accepts ~6 of 7 drafts per step and wins
at 1-4 users (GSM8K TPOT 9.4 vs 12.1 ms at one user, 10.3 vs 13.0 at four:
1.3x fewer ms per token, 1.2-1.25x the tokens/s). At 5-8 users the bitwise
ladder's (8,T=4) plan (37 + 1 + 12 ms verify + commit + draft, 3 drafts of the
7) is a wash against MTP's (8,T=4) (37 + 0.4 + 8.3): 15.3 vs 14.4 ms on GSM8K,
17.9 vs 15.4 on random prompts (low acceptance either way); the R = 64
(8,T=8) plan (52 + 1.2 + 12.4 ms) ties MTP on GSM8K text and loses on random
prompts. The T=2 band (9..16 users) pays the whole block step (16 ms) for one
draft and is slower than MTP's k=1 band (31-33 vs 23 ms TPOT); above 16 both
decode plainly (~39 ms). TTFT is P-side and unchanged within noise (the DFlash2
payload carries 20 KB/token more). The R = 64 plan is not self-consistent under
a changing batch (the fractured path's near-tie flips reach ~1 stream in 6
within 64 tokens; `tests/DFLASH2_RESULTS.md` measured them at gap <= 0.25), so
the bundle ships `QWEN36_SPEC_ALLOW_FRACTURED=0`: bitwise everywhere, ALL MATCH.
Where DFlash2 should go next: a per-user block step (the 8 x w rows are
computed for every user; a 16-row path would halve the 16-26 ms at w >= 16),
and the 5-8 user band could return to (8,T=8) once the fractured verify path is
made bitwise (a fused all-reduce for R = 64).

## Served gate: hybrid vs the single drafters (2026-09-28, 4+4 P/D stack, all 8 chips)

`scripts/spec_gate_variant.sh spec_hybrid hybrid` (the A/B's settings: AICLK under
firmware control, decode bucketing on, D pool 1,052,672 tokens, `QWEN36_SPEC_K=7`),
`logs/served_spec_hybrid/summary.txt`; the `mtp` / `dflash2 bitwise` columns are
the A/B runs above.

| 128/128 random, users | hybrid TPOT ms / tok/s | dflash2 bitwise | mtp |
|---|---|---|---|
| 1 | 11.9 / 74 | 11.7 / 75 | 12.1 / 74 |
| 4 | 15.2 / 226 | 14.9 / 231 | 13.8 / 245 |
| 8 | 16.2 / 405 | 17.9 / 370 | 15.4 / 429 |
| 16 | 24.6 / 552 | 32.7 / 431 | 23.1 / 574 |
| 32 (plain) | 39.6 / 684 | 39.5 / 699 | 38.0 / 723 |

| GSM8K text (OSL 128), users | hybrid TPOT ms / tok/s | dflash2 bitwise | mtp |
|---|---|---|---|
| 1 | 9.5 / 90 | 9.4 / 93 | 12.1 / 74 |
| 4 | 10.5 / 307 | 10.3 / 316 | 13.0 / 265 |
| 8 | 15.1 / 442 | 15.3 / 445 | 14.4 / 471 |
| 16 | 24.0 / 573 | 31.1 / 442 | 22.9 / 599 |
| 32 (plain) | 38.9 / 690 | 39.3 / 694 | 37.6 / 694 |

| | hybrid | dflash2 bitwise | mtp |
|---|---|---|---|
| TTFT 128 / 1k / 4k (latency_probe, ms) | 176-184 / 336-397 / 893-1161 | 168-188 / 310-368 / 870-1110 | 160 / 268 / 690-940 |
| vLLM acceptance length | 5.15 at 1-4 users on the GSM8K bench (DFlash2, k = 7; per position 0.88, 0.77, 0.68, 0.58, 0.48, 0.42, 0.35); 3.52-3.54 in the lm-eval phase (8 users, MTP k = 3; 0.94, 0.85, 0.74) | 3.3-4.2 (k = 7, 1-4 users); 3.5-3.6 (8 users, k = 3) | 3.4-3.6 (k = 3) |
| GSM8K lm-eval 200, flexible-extract | 0.82 +- 0.027, all 200 generations identical to the mtp and dflash2 runs | 0.82 +- 0.027 | 0.82 +- 0.027 |
| det_probe (3 x 3 prompts) | deterministic | deterministic | deterministic |
| self-consistency, conc 1..32 x 64 tokens | **ALL MATCH** (63/63) | ALL MATCH | ALL MATCH |
| spec / plain steps, flushes, migrations, drafter switches, protocol errors | 6100 (1503 DFlash2 + 4597 MTP) / 2361, 29, 31, 19, 0 | 5850 / 2430, 22, 37, -, 0 | 6600 / 2410, 17, 46, -, 0 |
| per-step device time (D log) | (1,8) DFlash2: verify 34.2 + commit 0.85 + keep 0.71 + draft 7.6 ms; (8,4) MTP: 33.5 + commit 1.1 + select 0.4 + 7.9; (16,2) MTP: 33.9 + 1.7 + 3.1 | (8,4): 33.5 + 1.2 + 12.4 | (8,4): 35.7 + 0.3 + 8.0 |
| DRAM after warm-up, D / P (MiB free per chip) | 5698 / 13071 | 6497 / 13558 | 9078 / 15051 |

Reading: the hybrid takes each band's better drafter within ~0.3-0.8 ms TPOT of
that drafter's own run (the DFlash2 band pays the 0.7 ms keep-current, the MTP
bands the 1.1 ms DFlash2 commit, each spread over 2-4.5 tokens per step): GSM8K
9.5 / 10.5 / 15.1 / 24.0 ms at 1 / 4 / 8 / 16 users against the previous best of
9.4 / 10.3 / 14.4 / 22.9 and against 12.1 / 13.0 / 15.3 / 31.1 for the loser at
each width. Exactness held everywhere (ALL MATCH, deterministic, the GSM8K
generations token-identical to both single-drafter runs) across 19 served
drafter switches. TTFT is ~10-20 % higher than mtp's on 1k-4k prompts because P
runs both hooks (the DFlash2 projection dominates; the MTP prefill adds ~5 ms).

## Long-context TPOT: the DFlash2 context window (2026-09-28)

The first hybrid sweep (`logs/points_pd_v11_merged` in the experiments tree) had
long-prompt TPOT about 2x the MTP sweep's at every listed concurrency (16k/128:
14-15 -> 27-35 ms; 32k: 15 -> 34-40; 64k: 13 -> 55-59; 128k: 18 -> 70) with
128/128 unchanged, and D's acceptance length at 1.04-1.25 late in the sweep. An
A/B of four stacks on the same tree (tt-metal `tests/DFLASH2_RESULTS.md`
section 8: hybrid; mtp; hybrid resident but never drafting; no speculation)
separated the two candidate causes:

- The resident drafters cost the decode nothing: plain steps with both drafters
  built, P shipping both states and D importing them run at 24.2 / 24.8 / 25.2 ms
  TPOT for one user at 128 / 16k / 32k context, the no-speculation floor's
  24.2 / 24.7 / 25.3 (decode bucketing on). `keep_current` / the DFlash2 commit
  run on speculative steps only.
- The mtp drafter on the same tree is at the previous level (13.6 / 15.0 ms at
  16k / 32k, acceptance 3.05-3.37 tok/step on real text at every context length).
- The regression is the DFlash2 band: the device drafter attended the WHOLE
  context while the checkpoint is trained with a 2048-token sliding window
  (config.json `sliding_window: 2048`, `layer_types: sliding_attention`), so its
  drafts degraded with the prompt length (real text: 4.7 tok/step at 1k-2k, 4.3
  at 4k, 2.4 at 8k, 2.0 at 16k, 1.75 at 32k; random prompts 1.0-1.2 at 64k-128k)
  while its block SDPA grew (8 -> 15 ms at 128k). The "32-user" long-prompt
  points of the sweep never had more than 1-2 live users on D (P prefills a 16k
  prompt every 1.5-3 s, a 128-token decode lasts 2-5 s), so they ran in the
  DFlash2 band, not the plain one; the sweep's true 32-live-user long-context
  points (8192/1024, 10000/1024) never regressed.

Fix (tt-metal `tt/dflash2_head.py`): the block step's paged SDPA decode gets
`sliding_window_size=2048` (the kernel attends keys `[cur_pos + 1 - 2048,
cur_pos]` per block row; the 8 rows share `cur_pos = P + 7`, so a row sees at
most 7 fewer of the oldest keys than the reference's per-row window and nothing
the reference does not). `QWEN36_DFLASH2_DEVICE_WINDOW` (default = the
checkpoint's 2048; `0` = the whole context, for A/B only) is read on D. Device
evidence: the drafter's SDPA configuration matches a windowed torch reference
(PCC 0.9997) and differs from the whole-context result past 2048 tokens
(`tests/test_dflash2_window_op.py`); the device draft logits on a 3000-token
GSM8K prompt match the fp32 host reference, which applies the window (PCC mean
0.9991, argmax agreement 0.92, selected-path agreement 0.84;
`tests/test_dflash2_spec_scratch.py DF_PCC_LEN=3000 DF_BPU=64`). Served
re-measure: section 8.3 of the tt-metal note (the table there is the one to
quote).

## Not done yet

- Only greedy requests speculate; sampled requests force the whole step to plain
  decode (device rejection sampling would lift this).
- Above 16 concurrent users the ladder falls back to plain decode (R > 32 needs
  the fractured verify path whose numerics flip greedy near-ties; opt in with
  `QWEN36_SPEC_ALLOW_FRACTURED=1`).
- Logprobs on the speculative path (the verify reads back only the per-row
  argmax and max).
- The prefill engine ships the DFlash2 context K/V of EVERY prompt position by
  default (`QWEN36_DFLASH2_CONTEXT_WINDOW=0`) although the device drafter reads
  only the last 2048 (since 2026-09-28, see "Long-context TPOT" below); `2048`
  on both engines ships the window's tail only (an opt-in until it has its own
  served gate).
- The prefill engine builds the whole `DFlash2Drafter` (its context caches are
  unused there: ~1.4 GB/chip at the 525k-token prefill pool); a projector-only
  construction would save them.
