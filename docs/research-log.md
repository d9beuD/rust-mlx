# Research log

## 2026-10-06 — Initial environment and contract

Empty repository, Apple M5 Max (40 GPU cores), 128 GB unified memory, macOS 27, Xcode 27, Rust 1.98.1. Requested checkpoint: Qwen3.8-Flash-Next-oQ4e-mtp, qwen4_exp, 48 hybrid layers, 512 experts with 10 selected per token, hyper-connections, PLE n-gram table, QSA and native MTP. Mixed affine 4/5/6/8-bit weights, 106 GB reported checkpoint. Download in progress.

Decision: inspect current upstream and validate a dense model first as required; implement the hybrid target next. Python is an oracle/research dependency only. Never report 100 tokens/s without a correct, complete timed generation. The goal is a measurable usable engine; the speed target is an aspiration, not permission to change numerical semantics.

## 2026-10-06 — Dense and hybrid baseline correctness

Dense fixture: deterministic two-layer Qwen3, float32, native mlx-lm 0.32.0 / MLX 0.32.3 oracle. Rust MLX 0.32.2: exact prefill logits (max abs error 0), 32/32 greedy tokens identical. Command: `cargo run --release --bin parity -- --model tests/fixtures/dense --oracle tests/fixtures/dense/oracle.json`.

Hybrid fixture: official mlx-vlm 0.7.6 Qwen4Exp, four layers, quantized affine int4 group32 projections, eight experts/top2, PLE and sparse QSA (budget16) crossed during decode, GDN float32 state. Max prefill logit error 2.9802322e-7, mean 7.9868187e-8, 24/24 greedy tokens identical. Native Rust path currently uses MLX operations for the recurrence and convolution. Command: `cargo run --release --bin hybrid-parity -- --model tests/fixtures/hybrid --oracle tests/fixtures/hybrid/oracle.json --dump results/hybrid-logits.json`.

The hardware did not initially have the separate Xcode Metal compiler asset; installed using `xcodebuild -downloadComponent MetalToolchain`, then mlx-sys built successfully. CMake is pinned in the research .venv. This is a build dependency; the inference executable does not start Python.

Target weights complete (all 21 shards present; each safetensors data range bounds checked by the mmap loader). N-gram lookups use Rust CPU hash + read-only maps and only copy required rows into MLX, avoiding full table GPU allocation. The independent target oracle is running separately, using official mlx-vlm and the same read-only checkpoint rows.

## 2026-10-06 — Target first parity rejection

Official mlx-vlm on requested checkpoint (all weights retained, mmap PLE): generated coherent text. Raw 16-step run includes 1.323 s initial decode shape compilation, so its aggregate 9.13 tok/s is not a warm speed claim. The next 15 steps averaged 28.62 ms (~34.94 tok/s); this is a preliminary short observation, not the required repeated benchmark.

Rust's unfused BF16 graph was rejected: max logit error 1.3025811, mean 0.1967785, cosine 0.993669, same first argmax but changed rankings. Do not widen the tolerance. Hypothesis: Python's compiled SiLU/SwiGLU and decay boundaries avoid intermediate BF16 rounding that the naïve Rust graph introduces. Match upstream compile boundaries and remeasure. The mmap table has an additional weight_scale tensor (1.0 in this checkpoint); honor it rather than ignoring an unrecognized parameter.

## 2026-10-06 — Exact requested-checkpoint reference established

Parity passes on the full checkpoint: prefill logits max abs error **0**, 16/16 greedy tokens identical to official mlx-vlm 0.7.6 / MLX **0.32.2**, matching Rust's MLX. Dense remains exact (32 tokens); hybrid fixture now exact (24 tokens including sparse-QSA transition). Every isolated real-weight layer stage matches bit for bit. The diagnostic's original training-mode GDN was corrected to eval mode; the production oracle was already in eval mode.

Corrections: weak-scalar dtype semantics; compiled activation, norm and hyper-pointwise graph boundaries including reshape placement; decode-equivalent narrow gate projections; BF16 grouped norms and residual association; native GDN SIMD reduction; explicit stored float32 inverse RoPE frequencies; fixed-width indexed QSA prefill (even short prefixes); float32 depthwise convolution reduction during BF16 decode; shape-dependent compilation where reshapes inspect dimensions. Keep ops GDN and per-row PLE fallback. The Metal ports establish oracle semantics rather than claiming new kernel inventions.

Commands and raw evidence: results/target-parity-complete.log (ignored transient log), results/target-oracle-mlx32.2.json; scripts/target_trace.py and src/bin/trace.rs produce diagnostic snapshots (large transient safetensors). tests/native_kernels.rs checks real BF16 dimensions, nonzero recurrent state, causality/sentinel/tail QSA against independent vectors. Full format/strict-Clippy/release tests passed before the subsequent profiling/experimental additions; rerun for the final commit.

First repeated correct Rust baseline, batch1/prompt10/output256/greedy/MTP off/fresh KV: 33.66, 37.45, 37.15 tok/s. Warmup32. Raw report results/target-baseline-256.json. First recorded run still loads previously untouched expert/PLE pages; call this out, do not hide it. Profiling with synchronization after each category changes scheduling and is diagnostic only (results/target-profile.json).

Experimental batched PLE: lookup microbench 236→158 us for a decode row set, 10.12→0.67 ms for 128-token rows. Exact outputs and target trajectory. End-to-end alternating A/B on prompt10 shows steady decode ~37.4 tok/s for both; the apparent aggregate gain comes from a first baseline run with cold pages. **Do not promote it for decode.** Keep experimental while measuring longer prefill. Next candidate overlaps CPU graph construction and GPU evaluation by async-evaluating layer outputs, as the upstream batch-invariant runtime does.

Async layer submission passes exact parity but does not improve this run: alternating 4+4 samples average 29.166 (baseline) vs 29.273 tok/s (candidate), with all 256-token outputs identical. The run is slower and noisier than earlier reports. A read-only system snapshot shows Spotlight/storage analysis using three CPU cores heavily, no reported thermal warning, and no active competing oMLX inference. Do not stop those user/system processes; repeat under quieter conditions and record the observed workload. **Do not promote async layer submission based on these data.**

## 2026-10-06 — Exact MTP and rejected short-block candidates

Native MTP head prefill and two steps match independent mlx-vlm tensors bit for bit. Teacher-forced verification lengths 2–8 match sequential logits, hidden states and caches exactly; rollback every retained prefix 0–4 preserves the next logits, GDN states/convolution, full-attention KV and PLE convolution exactly. MTP depth 1–7 preserves all 256 baseline greedy tokens. Reports: mtp-oracle.json, verify-parity.json, rollback-parity.json, target-mtp-depth-sweep-256.json.

Depth3 is best on this raw prompt: three sweep samples 55.33, 54.94, 54.91 tok/s; earlier warm samples 56.79, 56.75. Other depths 1/2/4/5/6/7 average 45.25/52.83/51.61/50.28/47.57/45.63. Includes draft, verifier, priming and synchronization; fresh caches. First cold untouched expert pages remain visible. This is a workload-specific improvement, not a general speed claim or 100 tok/s.

Whole-hyperconnection compilation changes no output but averages 37.406 vs 37.584 tok/s (~0.48%), below promotion threshold. Keep experimental. Packed GDN is a credited Apple mlx-lm MIT port with the same explicit reduction tree; portable tests at T1/4/17/129 and extreme gates, full prefill, verifier and rollback pass exact comparisons. Its T1 microbench is slightly slower (160.57→163.17 us); investigate longer blocks before any promotion. Packed reports are separate from baseline reports.

Next: a single MLX-owned worker with bounded request queue and cancellation-aware UTF-8 streaming; native checkpoint chat-template rendering and explicit greedy API limits. Then longer-context and end-to-end qualification. These are reversible local runtime additions, retaining the measured reference graph.

## 2026-10-06 — Serving and row batching plan

Checkpoint chat template matches Jinja2 exactly on 12 cases covering thinking on/off, reasoning low/medium/xhigh, system messages and multi-turn reasoning history. Local OpenAI-style completions/chat, SSE, one MLX owner, bounded FIFO, strict greedy limits and UTF-8 decode are implemented. Server smoke saved in results/server-smoke.json; the initial disconnect test hit max_context, so that assertion must be rerun with a valid request before treating cancellation as verified.

Implement equal-offset decode batches first: concatenate independent cache rows, hash PLE histories independently, use decode-equivalent projection rows, then split updated states. Empty/mismatched caches and out-of-vocabulary tokens must be rejected. Multi-token verifier mode remains separate from batching so batch decode does not allocate rollback histories. Validate B2/B4/B8 against independent B1 trajectories and complete caches before using the path in serving. Variable-length scheduling and MTP batching remain separate milestones.

Packed GDN short-block A/B: 54.906 baseline vs 54.941 tok/s candidate, four paired cycles and full 256-token warmups, exact output. Final pair slowed together to ~50; no decode promotion. Fused MoE gate/up A/B: 54.087 baseline vs 52.025 candidate (~3.8% regression); no promotion. Component benchmark explains this: T1 small improvement, T4/T8 slower. Keep both switches for research; avoid blanket activation.

Chat workload suite (4 prompts, 256 fixed output tokens, thinking on, fresh caches) passes all baseline/MTP trajectory comparisons. Warm stable MTP rates vary ~40–52 tok/s versus ~37 plain. First full-output runs still touch new expert pages after warmup64 and are slower; next suite uses warmup256. Ignore-EOS is explicit in these fixed-length experiments. A 35s xctrace launch produced zero target GPU events and no completed inference (target killed at recording limit). This is a failed profiling attempt, not usable hotspot evidence; raw trace is transient.

Initial F32 fixture batching was rejected at step0: ~2.4e-7 logit and ~6e-7 cache drift. The quantized gather projection takes a different F32 accumulation path from singleton qmm. Keep F32 projections tokenwise; qualify BF16 batching separately on the actual model. Do not widen the parity gate to make this pass.

Long-context parity: prompt2 096/chunk128 crosses QSA's sparse threshold. Full last-token prefill logits max/mean error0 and 8/8 greedy tokens match official mlx-vlm / MLX0.32.2 exactly. Native oracle prefill7.71s includes cold shapes/pages. Evidence target-long-oracle.json and target-long-parity.json.

After preserving F32 projection geometry, equal-offset batches2/4/8 are bit-identical to independent B1: all logits, wide hidden, GDN states/convolution, full KV/raw indexer keys/summaries, PLE state and CPU history; 16 autoregressive steps per batch, target and four-layer fixture. This is qualification/diagnostic timing, not an HTTP throughput claim. Evidence batch-parity-target.json and batch-parity-fixture.json. The serving scheduler remains batch1 pending end-to-end queue tests and performance measurements.

Server disconnect was rerun against a valid 4 000-token request: generation stopped with "client disconnected" and the next fresh request returned the correct first baseline token. Smoke report now reflects an actual cancellation. Full quality script passes format, strict Clippy all targets/features and release tests (including native kernels/chat). This validated milestone is committed before the next projection-kernel experiment.
