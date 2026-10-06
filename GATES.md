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

## Three decode directions (base0de9521)

- [x] G29: draft-only quantization and frequency-ranked vocabularies are implemented and measured through full greedy MTP
  EVIDENCE: twenty complete decode3-final-draft-* reports; ten affineQ6/Q4/full/fixed32K/64K/96K combinations, raw and four-chat256-token cohorts with four alternating fully warmed pairs per prompt. Independent multilingual ranking and upstream code artifact preserve canonical target IDs; acceptance/proposals and init costs recorded. FullQ4 raw+3.520%, code64/Q4+4.281% with French-8.515%; no global promotion. Instrumented actual draft-head checks and native mapping/chunk-boundary tests pass. scripts/analyze_decode3.py recomputes samples and rejects an intentionally corrupted token oracle. Final current-source G33 qualification is separate.
- [x] G30: a new exact verifier data-layout or producer/consumer fusion is implemented and evaluated
  EVIDENCE: native concatenated expert gate/up allocation, actual BF16 layer0 component oracle and evaluated backing ownership. Actual verifier2–8 twice and rollback0–4 plus continuation compare logits/hidden/full caches at0 under Metal abort-on-fault. Four separate original/packed process pairs in decode3-config-layout-raw-256.json give paired median+0.113881%; all target/proposal IDs/acceptance/depths unchanged. Component+1.8–4.8% does not qualify a default; native fallback retained. Final current-source G33 qualification is separate.
- [x] G31: a decode-oriented resident quantization variant is calibrated, persisted separately and evaluated
  EVIDENCE: calibrated Q4/Q5/Q6 overlays, 245 single-projection downstream sensitivity cases and mixed32 overlay (213 changed projections) complete; original checkpoint untouched and corpus disjoint. Four held-out studies cover28 documents/6239 predictions/seven64-token continuations. Q4/Q5/Q6/mixed32 perplexity changes+4.691/+0.203/-0.126/+3.987%; limited-corpus changes do not prove broad quality equivalence. Every variant has an independent native oracle, actual verifier/rollback shader checks and four raw process pairs. Mixed32 raw+3.036661%, Q4+0.462372%, Q5+0.393865%, Q6-7.548494%; no default promotion. Weight/code metadata, sensitivity, calibration/protection/atomicity tests, memory and cold complete process costs are retained. Double quantization remains explicit.
- [x] G32: integrate qualified outcomes of all three directions and compare complete solo configurations
  EVIDENCE: decode3-final-summary.json recomputes20 draft and9 configuration cohorts, four alternating warmed256-token pairs per prompt, fresh caches, separate process/layout controls and native oracles. Exact combo-full raw+3.800676% (70.064/72.707tok/s); chats+2.515662/+1.622170/+0.568452/+6.640881%. Approximate combo-mixed32 raw+1.992719%, chats+20.253057/+9.877384/+1.555607/+8.931998%, but held-out perplexity+3.986705%/all seven continuations differ. No new global default qualifies; cold complete process wall, model-init head/layout costs and peak memory are recorded, model-load-only wall explicitly not separated. No100tok/s solo claim. Original weights, default dispatch and performance-summary.json remain unchanged.
- [x] G33: current-source quality, GPU/HTTP safety, evidence and publication are verified
  CHECK: PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: source86cf8780 passes34 release/33 portable instrumented tests, strict format/all-target/all-feature Clippy,14 actual model verifier/rollback checks and two actual private draft-head shader checks. Default FIFO/plain batch8 HTTP/cancellation/Unicode tests pass; only task-owned servers stop. scripts/analyze_decode3.py --final recomputes all required scopes and rejects corrupt identity/mislabel controls. Implementation e76a705b964a24918579c6a29f8fdcc498d580b2 and evidence26c8201b1e550b8414ab7dbf5764afe26c4cdd9f are published by git push origin main (exit0); gh api independently returns26c8201 as public main. Prior source quality reports remain archived, raw reports include all samples/commands/binary/source identities. User .agents remains untracked/unpublished; checkpoints/private corpus/logits/captures are unchanged/unpublished. Documentation-only closure does not change the qualified core.
# Solo decode follow-up: MTPLX kernels and remaining directions (base676498c)

- [x] G34: attribute actual greedy MTP cycle costs and CPU/GPU scheduling on fresh-cache solo requests
  EVIDENCE: followup-attribution-summary.json binds fresh-cache four-run256-token baseline (median70.283tok/s), natural own-process CPU sampling and successful4s Metal trace (24,033 active compute intervals), plus native-exact synchronized short12-round/long2107-token4-round diagnostics. First trace/save timeout and export failure retained. Shader labels/DRAM counters unavailable; inclusive CPU samples and GPU gaps never treated as removable wall time. Instrumented rates excluded; subsequent core changes require separate current qualification.
- [x] G35: evaluate targeted MTPLX-style MoE/routing fusions with native mixed-BF16 reference
  EVIDENCE: pinned MTPLX9882703 routing/down ports with Apache notices; routing240 actual-bank cases and down15 selected-bank geometries match native BF16. Actual full verifier/rollback shader checks and four alternating256-token pairs preserve original IDs; routing+0.501%, down+0.433%, below promotion. docs/decode-followup-study.md and source-bound receipts distinguish synthetic components from canonical full trajectories; native fallback/defaults retained.
- [x] G36: evaluate PLE preparation overlap and QSA setup reuse without changing model/cache semantics
  EVIDENCE: followup-runtime-summary.json; early immutable PLE preparation, bounded RoPE position sharing and per-Kernel complete-key MLX-C config reuse pass8 actual verifier/rollback shader checks,37 release and36 portable instrumented tests. Four alternating warmed256-token raw pairs per policy and representative combined chats preserve canonical IDs; raw gains−0.113/+0.078/−0.133/+0.029%, below promotion. Default policies remain off; per-request history/rollback preserved.
- [x] G37: evaluate a distinct improved draft predictor and the feasibility of training/distillation
  EVIDENCE: fixed200-step independent112-fit/448-heldout BF16 draft-only bias pilot; heldout agreement47.5446→47.9911% but regularized objective worsens and raw acceptance stays177/234 over78 rounds. Four raw/four-chat256-token pairs remain target-exact; raw−0.600%, no promotion. Native quant-input gradients succeed and custom-kernel VJP probe fails; full MTP needs derivatives/native training graph. Corpus/prefix disjointness and artifact bounds/hash/target-unchanged tests pass; this is not full-transformer training or broad quality evidence.
- [x] G38: evaluate a weight-format/kernel pair beyond previously tested resident requantization
  EVIDENCE: lossless four-output Q4/g64 word layout on selected0/23/47 banks, original native codes/scales/biases retained.1.258GB extra and preparation/peak recorded; scalar addressing raw−9.308%. Hoisted uint4 variant passes15 actual geometries/full verifier/rollback/Metal plus four raw/four-chat256-token pairs; raw+0.432%, below gate. Initial third-mode/two-slot Rust panic preserved and corrected retry passes. No double quantization, changed target, all48-bank claim or default promotion.
- [x] G39: integrate all qualified results, preserve native/default fallbacks, and refresh current-source quality
  CHECK: PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: followup-current-qualification.json at final source33b62bf06487: strict format/Clippy,38 release/37 portable instrumented tests, actual native/routing/preparation/down/vector verifier/rollback plus private64-token biased drafting. Actual FIFO/batch8 HTTP cancellation/SSE/prefix/64 Unicode caps per mode pass; only owned servers stopped. Native/default fallbacks stay unchanged; implementationa165645 and evidence59d2ba3 committed, failures retained/replayable.
- [x] G40: audit and report every follow-up direction against current authoritative evidence
  EVIDENCE: docs/decode-followup-study.md and followup-final-summary.json; independent audit recomputes25 complete four-pair256-token cohorts, validates original IDs/oracle and source/binary/log binding, disjoint training/heldout contexts and current quality. Corrupted-token negative control rejects. All five research directions evaluated; no global5% promotion/100 solo claim, no summed components or DRAM ceiling inference. Exact commands, implementationa165645 and evidence59d2ba3 are in docs/research-log.md.

## Generated-trajectory MTP and remaining decode axes (completed)

- [x] G41: disjoint generated-trajectory collection and activation-dependent private draft training evaluated
  CHECK: .venv/bin/python scripts/analyze_mtp_next.py
  EXPECT: MTP_NEXT_EVIDENCE_INDEPENDENTLY_VERIFIED
  EVIDENCE:192 disjoint documents,128 fit/32 validation/32 blind; generated target-greedy trajectories trimmed atEOS,29439/7241/7229 valid positions. Rank32 residual uses frozen fullQ8 head gradients. Four epochs/460 updates stop after three validation regressions, epoch0 retained; agreement83.455→85.444% validation and84.078→85.945% blind. Full checkpoint/config/tokenizer identity, original target/head/wide-state immutability and finite/shape corruption tests pass;48h remains a ceiling.
- [x] G42: private adapter Rust/oracle parity, exact target IDs and depth1–7 cost evaluated
  CHECK: .venv/bin/python scripts/analyze_mtp_next.py
  EXPECT: MTP_NEXT_EVIDENCE_INDEPENDENTLY_VERIFIED
  EVIDENCE: current-source nonzero Python/Rust BF16 parityT1/2/3/4/8 plus actualQ8 input gradients under Metal validation. Natural32-prompt validation evaluates all seven depths and fixes depth2/rank32 before blind/external.32 blind prompts×four alternating pairs preserve IDs, weighted60.369→63.980tok/s (+5.981%); median prompt+5.665%, variable−3.239..17.151%. Predeclared first-blind forced1024-token confirmation preserves allIDs across four pairs, paired+16.425% (32.708→38.012tok/s). External raw−0.714%, chat+8.887/+6.689/−0.795/−3.362%; optional scoped gain, no global default/100 solo claim. Rank64 condition not met.
- [x] G43: MoE/shared/HC epilogue exactness, Metal validation and complete decode A/B evaluated
  CHECK: .venv/bin/python scripts/analyze_mtp_next.py
  EXPECT: MTP_NEXT_EVIDENCE_INDEPENDENTLY_VERIFIED
  EVIDENCE: originalBF16 boundaries retained; actual verifier2–8 twice, every rollback prefix0–8 and continuation exact under Metal validation.15 selected-bank actual-activation components exact. Four alternating256-token pairs preserve target/private proposal IDs/acceptance: raw−0.942%, four chats−1.581/−0.637/−0.953/−1.028%. No5% end-to-end benefit; conditional producer/down fusion not advanced and native fallback remains default.
- [x] G44: expert Q3/g64 and MXFP4/g32 actual-shape quality/cost screening completed
  CHECK: .venv/bin/python scripts/analyze_mtp_next.py
  EXPECT: MTP_NEXT_EVIDENCE_INDEPENDENTLY_VERIFIED
  EVIDENCE:30 nativeMLX selected-bank0/23/47 down cases,T1/2/3/4/8, captured verifier activations,10 warmups/100 alternating pairs. Q3/g64 median component+0.946%, maximum3.723%; MXFP4/g32 median−5.672%, all cases slower. Output-relativeL2 medians.179678/.100719 are component proxies, not full-model quality. No qualifying format advantage; conditional Rust port/approximate overlay not advanced, original/protected weights unchanged.
- [x] G45: final source passes required checks and every axis has reproducible evidence and explicit promotion decision
  CHECK: PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
  EXPECT: QUALITY_CHECKS_PASSED
  EVIDENCE: core91e3047d7915 passes strict format/Clippy,41 release/40 portable Metal tests, actual native/routing/runtime/down/verifier/rollback checks and private adapter checks. Default FIFO/plain batch8 HTTP/SSE/prefix/cancellation with3 survivors and64 Unicode caps each pass; task-owned servers stopped. Independent source/log/hash/trajectory audit and corrupted-token negative control pass. docs/mtp-next-study.md and mtp-next-summary.json give exact commands/cohorts/provenance/conditional rejection decisions; performance-summary/defaults unchanged. Implementation052a743; publication recorded separately in research-log.md.
