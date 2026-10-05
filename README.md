# Rust-MLX inference

Native Rust text inference for Qwen4Exp, backed by mlx-rs 0.32.0 / MLX 0.32.2. Python produces independent research oracles only; the executables do not start Python. Mixed affine 4/5/6/8-bit weights, hyper-connections, GatedDeltaNet, QSA, mmap n-gram PLE and native MTP are supported on the requested checkpoint. Requires Apple Silicon, Xcode's Metal toolchain and CMake.

```sh
PATH="$PWD/.venv/bin:$PATH" cargo build --release
MODEL=/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp
target/release/mtp-infer --model "$MODEL" --chat --prompt "Write a Fibonacci function in Rust." --max-tokens 256 --runs 1 --stream
target/release/server --model "$MODEL"
```

The server listens on `127.0.0.1:8080`. It implements `/health`, `/v1/models`, `/v1/completions` and `/v1/chat/completions`, including SSE streaming. One owning MLX thread consumes a bounded FIFO; requests get independent caches. The API supports greedy decoding (`temperature=0`, `top_p=1`, `n=1`) and text system/user/assistant messages. Unsupported sampling, tools, vision and unknown fields are rejected. The checkpoint's own chat template is rendered in Rust; `enable_thinking` and `reasoning_effort` (`low`, `medium`, `xhigh`) are honored. Reasoning appears as the checkpoint's text, including thinking tags; no separate reasoning-content parser yet. `mtp:false` selects the reference generator. The default context limit includes prompt and completion (4096); queue capacity is 8. This local development API has no authentication; keep the default loopback binding.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explique les emprunts Rust."}],"max_tokens":256,"temperature":0,"stream":true}'
```

`infer` is the plain reference CLI; `mtp-infer` uses exact greedy acceptance and cache rollback. Both accept `--prompt-ids`, `--prefill-chunk`, `--ignore-eos`, `--warmup-tokens`, `--runs`, `--output`, `--chat`, `--no-thinking` and `--reasoning-effort`. Every measured run starts with fresh caches. `mtp-infer --draft-depth 3` is best on the initial raw prompt; `--sweep-depth` measures depths1–7. Use non-streaming JSON for timing. MTP rate is generated tokens after the first divided by decode wall time, including priming, draft, verification and synchronization. The older `infer` baseline counts one timed unconsumed final forward; `workload-bench` uses the same timing definition for both modes.

On M5 Max 40 GPU cores / 128 GB, raw prompt10/output256, the correct reference runs at about 37 tok/s and MTP depth3 at about 55–57 tok/s. Chat tasks in the current suite vary around 40–52 tok/s with MTP. **100 tok/s has not been reached.** First full-length runs can be slower because previously untouched expert/PLE pages were cold. These are workload-specific measurements, with raw tokens, environment, quantization and variance saved in [results](results). See the [research log](docs/research-log.md) for rejected optimizations and limitations.

Correctness: target prefill logits and 16 greedy tokens match independent official mlx-vlm 0.7.6 on MLX 0.32.2 bit for bit. MTP head tensors, verification logits/hidden/states and rollback prefixes are exact in the recorded cases. Draft depths1–7 produce the same 256 tokens as the reference. The multi-prompt suite also matches. These checks do not establish correctness for every context; 2 096-token prefill logits and 8 decode tokens also match the native oracle exactly across the sparse-QSA threshold. Equal-offset batches2/4/8 preserve logits, hidden states and every cache exactly across 16 steps on the target and small fixture. Batching remains an experimental library interface, not the server scheduler.

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
target/release/parity --model tests/fixtures/dense --oracle tests/fixtures/dense/oracle.json
target/release/hybrid-parity --model tests/fixtures/hybrid --oracle tests/fixtures/hybrid/oracle.json
target/release/hybrid-parity --model "$MODEL" --oracle results/target-oracle-mlx32.2.json --max-error 0
target/release/workload-bench --model "$MODEL" --prompts tests/fixtures/workload-prompts.json --chat --ignore-eos --output results/workloads.json
```

Research controls: `RUST_MLX_BATCH_PLE`, `RUST_MLX_ASYNC_LAYERS`, `RUST_MLX_COMPILE_HYPER`, `RUST_MLX_PACKED_GDN`, `RUST_MLX_FUSED_MOE` activate experimental paths. None is promoted globally on current decode evidence. `RUST_MLX_GDN_OPS` retains a portable ops recurrence, with different floating-point reduction order. `infer --ab-ple/--ab-async/--ab-hyper/--ab-packed` and `mtp-infer --ab-kernel packed|moe` alternate candidate/reference in one process. Custom Metal ports are credited in NOTICE/third-party; weights are neither modified nor redistributed.
