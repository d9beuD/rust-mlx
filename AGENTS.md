# Rust-MLX

Read docs/architecture.md, docs/research-log.md and GATES.md before changing inference. This is one Cargo package, using mlx-rs 0.32.0 and mlx-sys 0.6.0 (MLX 0.32.2). Safe Rust owns graphs and state; unsafe is confined to the MLX C boundary and every unsafe block must explain ownership/lifetime and bounds with SAFETY. Never fork MLX or build a C++ bridge without a measured missing capability.

Build with Xcode + Metal toolchain and CMake. In this workspace CMake is provided by .venv/bin; use `PATH="$PWD/.venv/bin:$PATH" cargo build --release`. Python is a research oracle only.

Required checks: cargo fmt --check; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo test --workspace --release. Run scripts/check.sh with CMake on PATH. Test every new custom kernel against native MLX, including the actual bf16 model shapes, then microbenchmark and end-to-end A/B. Experimental paths never become default on microbench evidence alone. Keep a native MLX reference fallback.

Record successes and failures in docs/research-log.md with exact commands and commit. Raw reports belong in results, including model, mixed quantization, hardware, macOS, MLX, prompt/output IDs, lengths, MTP, sampler, cache, repetitions and variance. Warm-up then alternate baseline/candidate. Below 2% end-to-end can be noise; aim for >=5%. Never compare cached-prefix performance as raw model speed.

Requested model is 106 GB qwen4_exp with 48 hybrid layers, zero-centered grouped RMS norms, 512 experts, 10 active, n-gram PLE sharded table, QSA and MTP. The checkpoint already uses sanitized language_model.model.*, mtp.* and vision_tower.* keys. Each module may have its own bits/group size. Vision is outside the first text inference milestone. CPU mmap PLE rows should avoid materializing the entire n-gram table on GPU. Exact distribution and cache commit/rollback take priority over performance.

Do not stop or modify unrelated user processes. Existing oMLX on port 8001 requires authentication; avoid reading secrets. Record competing workloads when measuring.
