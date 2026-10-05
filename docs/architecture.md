# Architecture

One Cargo package plus a library and CLI binaries, deliberately avoiding a dozen empty crates. Safe Rust owns loading, model graphs, cache, generation and measurement. mlx-rs 0.32.0 / mlx-sys 0.6.0 own tensor computation through MLX 0.32.2. Custom Metal uses MLX-C's existing `mlx_fast_metal_kernel` API, with scoped RAII ownership. Python lives only in scripts for independent research oracles and comparison.

Dense Qwen3 is the first correctness gate. Target Qwen4Exp is text-only: zero-centered grouped RMS norms and hyper-connections, MoE with shared expert, sigmoid gated GDN, QSA sparse indexer, PLE hashed n-gram lookup and dilated convolution, and one-layer MTP. The requested checkpoint is already sanitized to `language_model.model.*`, `mtp.*`, and `vision_tower.*`; honor per-module mixed affine quantization rather than assuming uniform int4.

No optimization becomes the default without numerical and end-to-end evidence. Reference dispatch remains accessible. Benchmark outputs include model/config, prompt token IDs, output token IDs, lengths, sampler, MTP, cache, repetitions and environment. Incomplete and noisy results remain labeled.
