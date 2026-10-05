# Gates: Rust MLX inference engine

OWNS: src/**, tests/**, kernels/**, scripts/**, docs/**, results/**, Cargo.toml, Cargo.lock, AGENTS.md, README.md

Scope: native Rust inference, verified against an independent MLX oracle, measured optimizations on the requested checkpoint.

- [x] G1: pinned dependencies and studied upstream sources documented
  EVIDENCE: docs/prior-art.md, docs/upstream-lock.json, Cargo.lock
- [x] G2: Rust formatting, strict Clippy and release tests pass
  CHECK: scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: results/quality-current.log; format, strict workspace/all-target/all-feature Clippy, release tests all pass
- [x] G3: numerical kernel tests pass against native MLX
  EVIDENCE: tests/native_kernels.rs; real BF16 GDN output/state and QSA prefill match the independent oracle exactly
- [x] G4: dense model autoregressive trajectory matches MLX oracle
  EVIDENCE: dense parity: max error 0, 32/32 tokens; hybrid fixture max error 0, 24/24 tokens
- [x] G5: requested Qwen4Exp checkpoint generates correctly with its quantization and cache
  EVIDENCE: results/target-oracle-mlx32.2.json; target parity max error 0, 16/16 tokens
- [x] G6: raw repeated baseline and candidate performance reports saved with environment metadata
  EVIDENCE: results/target-baseline-256.json, target-ab-ple-256.json, target-ab-async-256.json, microbench.json; fresh KV, warmup, alternating candidates
- [ ] G7: optimized path improves a measured workload and preserves a reference fallback
  EVIDENCE: pending
- [ ] G8: MTP accept/reject and cache rollback are verified
  EVIDENCE: pending
- [ ] G9: usable CLI, streaming server, benchmarks and known limitations documented
  EVIDENCE: pending
- [ ] G10: 100 tokens per second aspiration evaluated honestly on the requested model
  EVIDENCE: pending
