# Gates: Rust MLX inference engine

OWNS: src/**, tests/**, kernels/**, scripts/**, docs/**, results/**, Cargo.toml, Cargo.lock, AGENTS.md, README.md

Scope: native Rust inference, verified against an independent MLX oracle, measured optimizations on the requested checkpoint.

- [ ] G1: pinned dependencies and studied upstream sources documented
  EVIDENCE: pending
- [ ] G2: Rust formatting, strict Clippy and release tests pass
  CHECK: scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: pending
- [ ] G3: numerical kernel tests pass against native MLX
  EVIDENCE: pending
- [ ] G4: dense model autoregressive trajectory matches MLX oracle
  EVIDENCE: pending
- [ ] G5: requested Qwen4Exp checkpoint generates correctly with its quantization and cache
  EVIDENCE: pending
- [ ] G6: raw repeated baseline and candidate performance reports saved with environment metadata
  EVIDENCE: pending
- [ ] G7: optimized path improves a measured workload and preserves a reference fallback
  EVIDENCE: pending
- [ ] G8: MTP accept/reject and cache rollback are verified
  EVIDENCE: pending
- [ ] G9: usable CLI, streaming server, benchmarks and known limitations documented
  EVIDENCE: pending
- [ ] G10: 100 tokens per second aspiration evaluated honestly on the requested model
  EVIDENCE: pending
