# Gates: Rust MLX inference engine

OWNS: src/**, tests/**, kernels/**, scripts/**, docs/**, results/**, Cargo.toml, Cargo.lock, AGENTS.md, README.md

Scope: native Rust inference, verified against an independent MLX oracle, measured optimizations on the requested checkpoint.

- [x] G1: pinned dependencies and studied upstream sources documented
  EVIDENCE: docs/prior-art.md, docs/upstream-lock.json, Cargo.lock
- [x] G2: Rust formatting, strict Clippy and release tests pass
  CHECK: scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: results/quality-current.json; format, strict workspace/all-target/all-feature Clippy,29 release tests pass; core source SHA4b866a77; preceding25-test qualification archived in quality-current-before-matrix.json
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
  EVIDENCE: round2-final-default-raw-256.json latest45.28 plain/68.43 MTP; previous stable final-raw-bounded-plain-mtp-256.json46.51/70.56; final-batch-ab-qmv-bounded-256.json QMV-on median105.27 aggregate B8,13.16 per conversation;100 single-conversation not reached
- [x] G11: exact prefix reuse preserves repeated plain/MTP continuations and bounded eviction
  EVIDENCE: results/prefix-parity-target.json, results/prefix-parity-long.json; complete prompt keys, model-scoped LRU, disabled/oversize/clear/access order checked
- [x] G12: optional continuous plain batching passes real concurrent HTTP trajectories and cancellation
  EVIDENCE: results/server-batch-smoke.json; three8-client mixed-length/SSE series; cancellation during three surviving clients, all128-token outputs exact; Unicode EOF exact in server-utf8-batch8.json; default MTP remains FIFO
- [x] G13: available competitor comparisons use matching trajectories and label incompatibilities
  EVIDENCE: native-target-baseline-256.json all256 IDs exact; mtplx-adapted-baseline/norm-f32.json and omlx-adapted-default/baseline.json reject logit drift before timing; adapter/stock serving limits explicitly labeled
- [x] G14: instrumented Metal validation passes outside performance measurements
  CHECK: scripts/validate-metal.sh
  EXPECT: METAL_VALIDATION_PASSED
  EVIDENCE: results/metal-validation.json,28 passing portable instrumented tests including the expected cooperative numerical counterexample/native fallback; matrix-study-validation.json adds actual mixed-weight shapes, short verifier/rollback and HTTP checks; preceding24-test qualification is archived in metal-validation-before-matrix.json. A passing negative control does not qualify the rejected cooperative kernel.

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
- [x] G21: cost-aware depth and vocabulary policies are experimentally evaluated
  CHECK: .venv/bin/python scripts/verify_round2.py adaptive
  EXPECT: ROUND2_ADAPTIVE_EXPLORATION_VERIFIED
  EVIDENCE: adaptive-{depth,vocab}/adaptive remat raw reports and four-chat suites: four256-token alternating pairs per prompt, exact IDs, complete cost; combined raw2.957050%, variable chat gains, default off; real depth/full-head fallback transitions instrumented in adaptive-qmv-remat-metal.json; negative-control pair oracle passes
- [x] G22: greedy output-head specialization is experimentally evaluated
  CHECK: .venv/bin/python scripts/verify_round2.py head
  EXPECT: ROUND2_HEAD_EXPLORATION_VERIFIED
  EVIDENCE: greedy-head-final-actual-metal.log actual248320x2560 BF16 headT1–4 block/ID comparisons and reduction Inf/NaN/zero ties;100 component pairs; four256-token full A/B pairs, exact proposal/target IDs, paired0.693185%, default off; negative-control pair oracle passes
- [x] G23: block-growing KV caches are experimentally evaluated in long-context solo decoding
  CHECK: .venv/bin/python scripts/verify_round2.py kv
  EXPECT: ROUND2_KV_EXPLORATION_VERIFIED
  EVIDENCE: actual context2096 native/block prefill/logits/hidden/all caches,16 transitions,T2–8/every rollback prefix, private/prefix/batch instrumentation exact; boundary components100 pairs; complete four-pair raw contexts10/2107/4117 gains0.197794/-0.792734/0.216423%, native concatenation remains default; pair oracle negative control passes
- [x] G24: final research outcomes, production defaults and public code pass refreshed quality gates
  CHECK: .venv/bin/python scripts/verify_round2.py final
  EXPECT: ROUND2_FINAL_INTEGRATION_PUBLICATION_VERIFIED
  EVIDENCE: round2-final-validation.json source483b3cd,25 release/24 portable instrumented tests, expanded actual head/private/sparse/rollback/batch checks, HTTP/SSE/prefix/cancellation/Unicode; latest raw45.28/68.43; implementation2b67a7a public main verified; publication oracle rejects local unpublished changes and validates qualified source, report hashes, defaults and remote HEAD

G18–G24 describe the completed historical round-two scope (source483b3cd). Its final oracle is tied to that snapshot and is not a current-source qualification. Current-source evidence follows in G25–G28.

## M5 matrix-unit optimization (completed rejection study)

- [x] G25: implement and execute Metal matrix-unit kernels on real verifier geometries, with native numerical reference and safe fallback
  EVIDENCE: six variants,207 final component configurations using actual BF16/mixed affine weights and synthetic activation stages; matrix-study-summary.json records every numerical failure and paired gain. Geometry/hardware guards and unchanged native/default fallback are tested.
- [x] G26: implement and execute the macOS27 register/cooperative-tensor alternative and compare it to staged Metal tensor operations
  EVIDENCE: matrix-final-{components,hybrid,affine,packed}-metal.json; all variants compile and execute,207 instrumented geometries. Register multiply-accumulate counterexample remains unqualified; selected affine falls back under validation. Paired timing is separate and no prototype passes the exactness/performance promotion gates.
- [x] G27: integrate diagnostic dispatch into actual solo MTP, then qualify or reject it before promotion
  CHECK: .venv/bin/python scripts/analyze_matrix_study.py
  EXPECT: MATRIX_STUDY_EVIDENCE_VERIFIED
  EVIDENCE: Packed engagesT2–4 and preserves short logits/hidden/full caches plus rollback0–4, but the attempted warmed four-pair256-token A/B stops at first candidate mismatch103 (2830→1048). matrix-packed-rejected-256.json preserves complete IDs; no completed candidate four-pair/chat gain is claimed. Strict rejection and its negative control pass. Native four256-token control runs remain exact, median68.715tok/s, prototypes off. Rejection ends candidate throughput qualification; no new default.
- [x] G28: refresh quality, actual-model shader validation and documentation for the resulting code
  CHECK: PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: matrix-study-validation.json, quality-current.json, metal-validation.json and server-qualification.json; core4b866a77,29 release/28 portable instrumented tests, actual short verifier/rollback, FIFO/batch8 HTTP/SSE/prefix/cancellation and64 Unicode caps per mode. docs/optimization-study-m5-matrix.md gives commands, input provenance and rejected outcomes; performance-summary.json is unchanged.
