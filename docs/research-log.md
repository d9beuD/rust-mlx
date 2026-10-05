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
