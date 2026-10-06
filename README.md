# Rust-MLX inference

Native Rust text inference for Qwen4Exp, backed by mlx-rs 0.32.0 / MLX 0.32.2. Python produces independent research oracles only; the executables do not start Python. Mixed affine 4/5/6/8-bit weights, hyper-connections, GatedDeltaNet, QSA, mmap n-gram PLE and native MTP are supported on the requested checkpoint. Requires Apple Silicon, Xcode's Metal toolchain and CMake.

```sh
PATH="$PWD/.venv/bin:$PATH" cargo build --release
MODEL=/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp
target/release/mtp-infer --model "$MODEL" --chat --prompt "Write a Fibonacci function in Rust." --max-tokens 256 --runs 1 --stream
target/release/server --model "$MODEL"
```

The server listens on `127.0.0.1:8080`. It implements `/health`, `/v1/models`, `/v1/completions` and `/v1/chat/completions`, including SSE streaming. One owning MLX thread consumes a bounded FIFO; requests get independent caches. The API supports greedy decoding (`temperature=0`, `top_p=1`, `n=1`) and text system/user/assistant messages. Unsupported sampling, tools, vision and unknown fields are rejected. The checkpoint's own chat template is rendered in Rust; `enable_thinking` and `reasoning_effort` (`low`, `medium`, `xhigh`) are honored. Reasoning appears as the checkpoint's text, including thinking tags; no separate reasoning-content parser yet. `mtp:false` selects the reference generator. The default context limit includes prompt and completion (4096); queue capacity is 8. This local development API has no authentication; keep the default loopback binding.

The model-scoped exact-prompt LRU defaults to 4 entries / 8192 total prompt tokens. Every continuation clones independent cache handles. Requests can bypass it with `prefix_cache:false`; startup `--prefix-cache-entries 0` disables it. Responses report `rust_mlx.prefix_cache_hit` and cached prompt token usage. A cache hit avoids prompt preparation; MTP still primes its own private cache. This is separate from raw decode speed.

Optional continuous batching supports plain decoding with up to eight active requests:

```sh
target/release/server --model "$MODEL" --no-mtp --batch-size 8
```

The scheduler groups rows at equal cache offsets, supports different prompt lengths and pauses backpressured clients independently. Prompt preparation is serialized. MTP requests require the default batch1 server; MTP batching is not implemented. Per-request decode wall time includes scheduling and other requests' prefills. Offline aggregate batch throughput excludes prefills and is reported separately from HTTP latency and individual conversation speed.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explique les emprunts Rust."}],"max_tokens":256,"temperature":0,"stream":true}'
```

`infer` is the plain reference CLI; `mtp-infer` uses exact greedy acceptance and cache rollback. Both accept `--prompt-ids`, `--prefill-chunk`, `--ignore-eos`, `--warmup-tokens`, `--runs`, `--output`, `--chat`, `--no-thinking` and `--reasoning-effort`. Every measured run starts with fresh caches. `mtp-infer --draft-depth 3` is best on the initial raw prompt; `--sweep-depth` measures depths1–7. Use non-streaming JSON for timing. MTP rate is generated tokens after the first divided by decode wall time, including priming, draft, verification and synchronization. The older `infer` baseline counts one timed unconsumed final forward; `workload-bench` uses the same timing definition for both modes.

On M5 Max 40 GPU cores / 128 GB, raw prompt10/output256, the latest validated session measures **45.28 tok/s plain and68.43 with exact MTP depth3** (the previous stable session gave46.51/70.56) in four alternating pairs, all256 IDs identical. Eight plain batch rows measure **105.27 tok/s aggregate**, about13.16 per conversation, in four complete256-token runs. Earlier batch sessions were around99 aggregate; absolute rates vary. Four chat workloads show paired HC gains9.20/9.54/9.87/8.90%, with identical256-token trajectories. Their latest full-head rates vary about51–64tok/s. The independent mlx-vlm adapted plain baseline is about41tok/s, with a slower35tok/s fourth sample. **100 tok/s in one conversation has not been reached.** Previously untouched expert/PLE pages and system storage indexing affect measurements. Full samples, timing definitions, quantization, defaults, rejected experiments and comparison limits are in the [performance report](docs/performance.md) and [research log](docs/research-log.md).

Pinned MLX0.32.2 needs an allocation guard for native depthwise convolutions: the logical GDN/PLE weights stay unchanged, with a padded backing buffer preventing tail reads outside the last group. Both the native failure and the guarded full-model Metal validation are recorded in `results/conv-allocation-validation.json`.

Correctness: target prefill logits and 16 greedy tokens match independent official mlx-vlm 0.7.6 on MLX 0.32.2 bit for bit. MTP head tensors, verification logits/hidden/states and rollback prefixes are exact in the recorded cases. Draft depths1–7 produce the same 256 tokens as the reference. The multi-prompt suite also matches. The Float32 research fixture currently differs2.682209e-7 from its saved prefill oracle on the unchanged native path; its native/optimized fallback transitions remain exact, and no tolerance was widened. The requested BF16 model uses error0. These checks do not establish correctness for every context; 2 096-token prefill logits and 8 decode tokens also match the native oracle exactly across the sparse-QSA threshold. Equal-offset batches2/4/8 preserve logits, hidden states and every cache exactly across 16 steps on the target and small fixture. The combined HC/QMV batch also preserves complete cache parity after a 2 096-token prefill and across eight decode steps. Full256-token batch reports compare every row with independent singleton trajectories.

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
target/release/parity --model tests/fixtures/dense --oracle tests/fixtures/dense/oracle.json
target/release/hybrid-parity --model tests/fixtures/hybrid --oracle tests/fixtures/hybrid/oracle.json
target/release/hybrid-parity --model "$MODEL" --oracle results/target-oracle-mlx32.2.json --max-error 0
target/release/workload-bench --model "$MODEL" --prompts tests/fixtures/workload-prompts.json --chat --ignore-eos --output results/workloads.json
```

Shared-weight verifier QMV, compatible batch-row QMV and restricted combined HC down/inject are enabled by default. Each retains shape/dtype/quantization guards and native fallback. Blocks of5–8 positions use independent chunks of at most4 to avoid a reproduced long-block Metal validation failure; every row keeps its original reduction tree. `RUST_MLX_VERIFY_QMV=0`, `RUST_MLX_BATCH_QMV=0` and `RUST_MLX_HC_PROJECTION=0` independently restore native dispatch. `workload-bench --ab-qmv/--ab-hc`, `mtp-infer --ab-kernel qmv|hc` and `batch-bench --ab-qmv/--ab-hc` run alternating comparisons with complete warmups.

Experimental controls stay off: `RUST_MLX_ROWS_GEMV`, `RUST_MLX_QMV_STREAM_X`, `RUST_MLX_BATCH_PLE`, `RUST_MLX_ASYNC_LAYERS`, `RUST_MLX_COMPILE_HYPER`, `RUST_MLX_COMPILE_GDN`, `RUST_MLX_PACKED_GDN`, `RUST_MLX_FUSED_MOE`. Component improvements do not qualify a global default. Pure BF16 GDN compilation passes exact target logits and complete caches but improves raw MTP only1.72%; `mtp-infer --ab-kernel gdn` reproduces its alternating comparison. Float32 input retains native dispatch because compilation changes last bits. Draft shortlist settings `--draft-vocab-limit 32768 --draft-vocab-refresh-rounds 8` are optional in `mtp-infer`: they preserve exact target output through full-vocabulary verification, but alter draft acceptance and give workload-dependent gains (raw4.20%; French1.69%). Full-head drafting remains default. `RUST_MLX_GDN_OPS` retains a portable ops recurrence with a different floating-point reduction order. Custom Metal ports are credited in NOTICE/third-party; weights are neither modified nor redistributed.

`scripts/check.sh` checks formatting, strict workspace/all-target/all-feature Clippy and release tests. `scripts/validate-metal.sh` enables process-local GPU shader validation and abort-on-fault; its timings are excluded from speed claims. Diagnostic component capture in results/head-component.json covers the actual vocabulary head only, not a full-model GPU profile. The Python comparison scripts require their separate research environments; production executables do not.

Further solo-decode research is tracked in [round two](docs/optimization-round2.md). The first two prototypes preserve exact outputs but stay disabled: `mtp-infer --ab-kernel gpu-draft` gives0.93% median paired gain, and `--ab-kernel sorted-moe` gives-1.13%, each in four alternating256-token pairs. `verify-profile` records synchronized T4 phase diagnostics and actual expert overlap; these are not throughput percentages. Current source passes25 release tests,24 portable instrumented tests and expanded actual-checkpoint checks. All five directions are evaluated: combined adaptive depth/vocabulary gains2.96% raw (four chats16.62/1.00/0.55/5.17%), greedy-only head0.69%, and block KV remains near parity through4117-token contexts. None passes the global5% gate; defaults retain native cache/full draft vocabulary/fixed depth3. Optional CLI controls are `--ab-kernel adaptive-depth|adaptive-vocab|adaptive|greedy-head|kv-blocks`, with corresponding workload flags. QMV recalculates offsets from original buffers after preserved mixed-session T4 shader failures;64-bit addressing is default,32-bit control stays off. Current integration, serving and expanded actual-model evidence is in `results/round2-final-validation.json`.
