# Gates: Rust MLX inference engine

OWNS: src/**, tests/**, kernels/**, scripts/**, docs/**, results/**, Cargo.toml, Cargo.lock, AGENTS.md, README.md

Scope: native Rust inference, verified against an independent MLX oracle, measured optimizations on the requested checkpoint.

- [x] G1: pinned dependencies and studied upstream sources documented
  EVIDENCE: docs/prior-art.md, docs/upstream-lock.json, Cargo.lock
- [x] G2: Rust formatting, strict Clippy and release tests pass
  CHECK: scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: results/quality-current.json; format, strict workspace/all-target/all-feature Clippy,19 release tests pass; code SHA verifies no subsequent core change; stable32e74c1 evidence archived separately
- [x] G3: numerical kernel tests pass against native MLX
  EVIDENCE: tests/native_kernels.rs; real BF16 GDN output/state and QSA prefill match the independent oracle exactly
- [x] G4: dense model autoregressive trajectory matches MLX oracle
  EVIDENCE: dense parity max error0,32/32 tokens; hybrid fixture24/24 tokens, current native F32 prefill differs2.682209e-7 from saved oracle (already present before compiled experiment); direct native/optimized transitions and caches error0; no tolerance widened
- [x] G5: requested Qwen4Exp checkpoint generates correctly with its quantization and cache
  EVIDENCE: results/target-oracle-mlx32.2.json; target parity max error 0, 16/16 tokens
- [x] G6: raw repeated baseline and candidate performance reports saved with environment metadata
  EVIDENCE: results/target-baseline-256.json, target-ab-ple-256.json, target-ab-async-256.json, microbench.json; fresh KV, warmup, alternating candidates
- [x] G7: optimized path improves a measured workload and preserves a reference fallback
  EVIDENCE: target-mtp-depth3-256.json, target-mtp-depth-sweep-256.json; exact native Metal reference with MTP improves the raw 256-token workload, plain reference remains available
- [x] G8: MTP accept/reject and cache rollback are verified
  EVIDENCE: verify-parity.json, rollback-parity.json, target-workloads-chat-256.json; exact logits/cache commit across accepted lengths0–4 and depth1–7 trajectories
- [x] G9: usable CLI, streaming server, benchmarks and known limitations documented
  EVIDENCE: README.md, docs/performance.md, results/server-qualification.json; default binaries, FIFO/batch/prefix/SSE/Unicode/real cancellation; reproducible raw reports and native fallbacks
- [x] G10: 100 tokens per second aspiration evaluated honestly on the requested model
  EVIDENCE: final-raw-bounded-plain-mtp-256.json median46.51 plain/70.56 MTP; final-batch-ab-qmv-bounded-256.json QMV-on median105.27 aggregate B8,13.16 per conversation;100 single-conversation not reached
- [x] G11: exact prefix reuse preserves repeated plain/MTP continuations and bounded eviction
  EVIDENCE: results/prefix-parity-target.json, results/prefix-parity-long.json; complete prompt keys, model-scoped LRU, disabled/oversize/clear/access order checked
- [x] G12: optional continuous plain batching passes real concurrent HTTP trajectories and cancellation
  EVIDENCE: results/server-batch-smoke.json; three8-client mixed-length/SSE series; cancellation during three surviving clients, all128-token outputs exact; Unicode EOF exact in server-utf8-batch8.json; default MTP remains FIFO
- [x] G13: available competitor comparisons use matching trajectories and label incompatibilities
  EVIDENCE: native-target-baseline-256.json all256 IDs exact; mtplx-adapted-baseline/norm-f32.json and omlx-adapted-default/baseline.json reject logit drift before timing; adapter/stock serving limits explicitly labeled
- [x] G14: instrumented Metal validation passes outside performance measurements
  CHECK: scripts/validate-metal.sh
  EXPECT: METAL_VALIDATION_PASSED
  EVIDENCE: results/metal-validation.json,18 passing instrumented kernel tests and actual validation marker; round2-initial-validation.json adds actual private chains and full-model expert verifier/rollback; stable32e74c1 evidence archived separately; component capture is results/head-component.json and ignored GPU trace

- [x] G15: compiled GDN attention experiment evaluated without loosening exactness
  EVIDENCE: compiled-gdn-target-direct.json, compiled-gdn-portable.json, compiled-gdn-statistics.json; BF16 exact logits/complete caches, F32 rejection retained with native fallback; four full256-token pairs gain1.72%, below5%, experiment off by default

- [x] G16: native grouped convolution backing allocation passes full-model Metal validation
  EVIDENCE: conv-allocation-validation.json; rejected upstream F32/BF16 traces retained, guarded actual oracle prefill+16 tokens error0 under abort-on-fault; final-raw-bounded-plain-mtp-256.json repeats default throughput

- [x] G17: large-output long-block QMV passes actual-model shader validation
  EVIDENCE: metal-verify-bounded.json, metal-rollback-bounded.json, metal-batch-bounded.json; instrumented actual-model verifier2–8/rollback0–4 and batches2/4/8 ×16 exact; long context2096 ×8 steps exact in batch-parity-bounded-long.json; final-batch-ab-qmv-bounded-256.json refreshes full throughput

- [x] G18: current code and history are published in a public repository on the user's GitHub account
  CHECK: .venv/bin/python scripts/verify_round2.py publication
  EXPECT: ROUND2_PUBLICATION_VERIFIED
  EVIDENCE: https://github.com/d9beuD/rust-mlx; initial main32e74c1, gh repository/public and remote commit verification
- [x] G19: four-position verification attribution and expert-weight reuse are experimentally evaluated
  CHECK: .venv/bin/python scripts/verify_round2.py experts
  EXPECT: ROUND2_EXPERTS_EXPLORATION_VERIFIED
  EVIDENCE: verify-profile-round2.json native logits exact across12 actual MTP rounds; expert selection duplication33.55% is theoretical, not bandwidth; sorted-moe-components.json BF16/native exact, uncontended100-pair microbench; target verifier2–8 twice and rollback0–4 shader validation exact; sorted-moe-raw-ab-256.json four alternating full256-token pairs, complete draft/target/acceptance parity; sorted-moe-statistics.json rejects default promotion
- [x] G20: GPU-resident MTP drafting is evaluated against scalar CPU drafting
  CHECK: .venv/bin/python scripts/verify_round2.py gpu
  EXPECT: ROUND2_GPU_EXPLORATION_VERIFIED
  EVIDENCE: gpu-draft-target-metal.json64 tokens/draft IDs/acceptance exact with abort-on-fault; gpu-draft-private-metal.log depths1–7 logits/hidden/private cache exact; gpu-draft-raw-ab-256.json four alternating full256-token pairs; gpu-draft-statistics.json median paired gain0.9286%, below5%, default remains scalar
- [ ] G21: cost-aware depth and vocabulary policies are experimentally evaluated
  EVIDENCE: pending; exact target output, recorded chosen depths/acceptance, raw and four chat workloads with complete costs
- [ ] G22: greedy output-head specialization is experimentally evaluated
  EVIDENCE: pending; actual BF16 head shapes, exact tie/rounding/IDs, microbenchmark and full decoding A/B
- [ ] G23: block-growing KV caches are experimentally evaluated in long-context solo decoding
  EVIDENCE: pending; exact complete-cache rollback, sparse threshold, multiple context lengths, full decode wall A/B
- [ ] G24: final research outcomes, production defaults and public code pass refreshed quality gates
  EVIDENCE: pending; scripts/check.sh, instrumented actual checkpoint, appropriate HTTP smoke tests, exact repeated timings, research log and published final commit
