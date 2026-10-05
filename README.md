# Rust-MLX inference

Native Rust text inference for the requested Qwen4Exp checkpoint, backed by mlx-rs 0.32.0 / MLX 0.32.2. Python is used only to produce independent research oracles. The runtime honors mixed affine quantization, hyper-connections, GatedDeltaNet, QSA, mmap n-gram PLE and growing caches. Requires Apple Silicon, Xcode's Metal toolchain and CMake.

```sh
PATH="$PWD/.venv/bin:$PATH" cargo build --release --bin infer
target/release/infer --model /Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp --prompt "Write a Fibonacci function in Rust." --max-tokens 256 --runs 1 --stream
```

`infer` accepts `--prompt-ids`, `--prefill-chunk`, `--ignore-eos`, `--warmup-tokens`, `--runs` and `--output` for reproducible JSON. It currently generates greedily, batch size one, MTP off. Every run gets fresh KV state; warmup heats weights/kernels, without prefix reuse. Decode rate counts timed forward steps; the first generated token comes from prefill and the final timed step computes one unconsumed distribution. Streaming adds terminal/tokenizer overhead; use non-streaming reports for comparisons.

Correctness: full requested checkpoint prefill logits are bit-identical to official mlx-vlm 0.7.6 on MLX 0.32.2, with 16/16 greedy tokens identical. Dense Qwen3 and a small hybrid fixture also match exactly (32/24 tokens). Native-kernel regression tests include the real BF16 dimensions and nonzero recurrent state. These checks do not establish correctness for every prompt/context.

Initial correct baseline: 33.66–37.45 tok/s on M5 Max 40 GPU cores, 128 GB, prompt10/output256, MTP off. **100 tok/s has not been reached.** See [research log](docs/research-log.md) and raw [results](results) for variance, rejected candidates and conditions. Work on MTP, serving and further optimization is ongoing.

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
target/release/parity --model tests/fixtures/dense --oracle tests/fixtures/dense/oracle.json
target/release/hybrid-parity --model tests/fixtures/hybrid --oracle tests/fixtures/hybrid/oracle.json
target/release/hybrid-parity --model /Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp --oracle results/target-oracle-mlx32.2.json
```

Experimental switches: `RUST_MLX_BATCH_PLE=1` batches identical quantized PLE row formats; `RUST_MLX_ASYNC_LAYERS=1` submits intermediate layer graphs asynchronously; `RUST_MLX_GDN_OPS=1` uses the portable MLX-ops recurrence with a different floating-point reduction. Neither performance candidate is promoted on the current decode evidence. `infer --ab-ple` or `--ab-async` alternates candidate/reference with separate warmups in one process. These flags are research controls, not guaranteed speedups. Model weights are never modified or redistributed; licenses and adaptations are in NOTICE/third-party.
