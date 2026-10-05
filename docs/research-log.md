# Research log

## 2026-10-06 — Initial environment and contract

Empty repository, Apple M5 Max (40 GPU cores), 128 GB unified memory, macOS 27, Xcode 27, Rust 1.98.1. Requested checkpoint: Qwen3.8-Flash-Next-oQ4e-mtp, qwen4_exp, 48 hybrid layers, 512 experts with 10 selected per token, hyper-connections, PLE n-gram table, QSA and native MTP. Mixed affine 4/5/6/8-bit weights, 106 GB reported checkpoint. Download in progress.

Decision: inspect current upstream and validate a dense model first as required; implement the hybrid target next. Python is an oracle/research dependency only. Never report 100 tokens/s without a correct, complete timed generation. The goal is a measurable usable engine; the speed target is an aspiration, not permission to change numerical semantics.

## 2026-10-06 — Dense and hybrid baseline correctness

Dense fixture: deterministic two-layer Qwen3, float32, native mlx-lm 0.32.0 / MLX 0.32.3 oracle. Rust MLX 0.32.2: exact prefill logits (max abs error 0), 32/32 greedy tokens identical. Command: `cargo run --release --bin parity -- --model tests/fixtures/dense --oracle tests/fixtures/dense/oracle.json`.

Hybrid fixture: official mlx-vlm 0.7.6 Qwen4Exp, four layers, quantized affine int4 group32 projections, eight experts/top2, PLE and sparse QSA (budget16) crossed during decode, GDN float32 state. Max prefill logit error 2.9802322e-7, mean 7.9868187e-8, 24/24 greedy tokens identical. Native Rust path currently uses MLX operations for the recurrence and convolution. Command: `cargo run --release --bin hybrid-parity -- --model tests/fixtures/hybrid --oracle tests/fixtures/hybrid/oracle.json --dump results/hybrid-logits.json`.

The hardware did not initially have the separate Xcode Metal compiler asset; installed using `xcodebuild -downloadComponent MetalToolchain`, then mlx-sys built successfully. CMake is pinned in the research .venv. This is a build dependency; the inference executable does not start Python.

Target weights complete (all 21 shards present; each safetensors data range bounds checked by the mmap loader). N-gram lookups use Rust CPU hash + read-only maps and only copy required rows into MLX, avoiding full table GPU allocation. The independent target oracle is running separately, using official mlx-vlm and the same read-only checkpoint rows.
