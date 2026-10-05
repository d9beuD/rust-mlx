# Architecture

One Cargo package plus a library and CLI binaries, deliberately avoiding a dozen empty crates. Safe Rust owns loading, model graphs, cache, generation and measurement. mlx-rs 0.32.0 / mlx-sys 0.6.0 own tensor computation through MLX 0.32.2. Custom Metal uses MLX-C's existing `mlx_fast_metal_kernel` API, with scoped RAII ownership. Python lives only in scripts for independent research oracles and comparison.

Dense Qwen3 is the first correctness gate. Target Qwen4Exp is text-only: zero-centered grouped RMS norms and hyper-connections, MoE with shared expert, sigmoid gated GDN, QSA sparse indexer, PLE hashed n-gram lookup and dilated convolution, and one-layer MTP. The requested checkpoint is already sanitized to `language_model.model.*`, `mtp.*`, and `vision_tower.*`; honor per-module mixed affine quantization rather than assuming uniform int4.

No optimization becomes the default without numerical and end-to-end evidence. Reference dispatch remains accessible. Benchmark outputs include model/config, prompt token IDs, output token IDs, lengths, sampler, MTP, cache, repetitions and environment. Incomplete and noisy results remain labeled.

MTP keeps its private QSA cache and hidden state independent of the target. Verification uses decode-equivalent projection reductions and recurrent state histories; committing a prefix selects stored states, trims KV/QSA summaries and convolution windows, and restores CPU n-gram history. Greedy target verification preserves the exact output trajectory. The plain generator is independently callable with the same rate convention.

The local HTTP server constructs the model inside one OS worker thread. Tokio handles sockets/SSE and passes owned text requests through a bounded channel; tensors and thread-local compiled kernels never cross that boundary. Dropped response channels cancel generation at token emission. Chat formatting runs the checkpoint Jinja template through MiniJinja, with independent Jinja2 regression cases.

Experimental equal-offset batching merges independent cache rows, builds one decode graph with row-equivalent projections and independent PLE hashing, evaluates before commit, and splits state back to each request. Float32 projection rows remain individual qmm to preserve their accumulation order. The server currently uses FIFO batch1, pending measured batching qualification.
