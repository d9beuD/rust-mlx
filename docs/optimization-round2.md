# Solo decode research, round two

The authorized scope is publication of the existing code on the user's public GitHub account, followed by experimental exploration of all five proposed directions. A rejected experiment counts as explored only with evidence of its actual numerical or performance result; documenting a hypothesis alone does not.

Initial public snapshot: https://github.com/d9beuD/rust-mlx at32e74c1. Preserve the original target checkpoint, mixed quantization, native fallbacks, exact greedy outputs and cache rollback. The untracked user-provided .agents directory is outside this publication.

The initial raw four-pair measurement gives70.5621 solo MTP tok/s. Verification accounts for84.74–84.90% of decode wall, draft generation9.99–10.14%, and draft synchronization4.58–4.80%. These phase timings include graph execution barriers, not isolated GPU execution. Instrumented component times must stay outside throughput statistics.

| Gate | Experiment | Acceptance evidence |
|---|---|---|
| G19 | Verifier T4 attribution, expert overlap and shared-weight reuse | Diagnostic component attribution and actual expert overlap; exact BF16/native outputs and histories; micro and repeated end-to-end comparison |
| G20 | GPU-resident autoregressive draft IDs | Exact individual draft IDs, acceptance and final trajectory; actual model Metal validation; four alternating256-token pairs |
| G21 | Adaptive depth and vocabulary | Deterministic policy with full target verification; all costs counted; raw and four chat workloads, chosen-depth/acceptance records |
| G22 | Greedy head projection/reduction | Preserve BF16 projection rounding and lowest-ID ties; actual248320x2560 head; native oracle, component timing and end-to-end |
| G23 | Block-growing KV | Immutable snapshot semantics and exact prefix rollback; short and sparse long contexts; multiple-context solo decode measurements |
| G24 | Integration and publication | Required Rust checks, actual-model shader validation, serving checks appropriate to defaults, final evidence and public commit |

Only a repeated end-to-end gain of at least5% with exactness qualifies a production default. Smaller gains and failures remain documented experiments. No performance target is a substitute for these checks.

## Results so far

GPU draft chains preserve every intermediate private MTP tensor/cache at depths1–7 under actual-checkpoint Metal validation. Four alternating256-token pairs preserve all proposed IDs,177/234 acceptance and target output. Native median70.2937, GPU-chain71.0190tok/s; median paired gain0.9286%. The experiment stays off by default. Raw samples are in results/gpu-draft-raw-ab-256.json.

T4 verifier diagnostics use actual proposals from that run and preserve native logits in12 rounds. Across48 layers,23040 selected expert slots correspond to15309 distinct experts per layer/round, a33.55% duplication opportunity. That number is a theoretical reduction in expert occurrences, not measured memory traffic or speed: caches and native kernels may already exploit reuse. GDN/MoE each account for roughly one quarter of the synchronized component sum, and the two HC phases together another quarter; these diagnostic shares change scheduling and are not natural-runtime percentages.

The first expert-major prototype sorts selected rows, calls native gather_qmm with sorted indices, then restores canonical order before weighted reduction. Fixture F32/BF16 and actual BF16T2/4/8 outputs are exact. Actual full-model verifier lengths2–8 twice and rollback prefixes0–4 also pass exact logits/hidden/cache checks with Metal abort-on-fault. A calibration microbenchmark overlapped a compiler job and remains separately labeled; the uncontended component rerun gives roughly597/626us atT2,487/494us atT4,675/638us atT8. Four complete alternating256-token pairs give native70.2366/sorted69.3416tok/s, median paired gain-1.1324%. Every proposal/acceptance/target token is exact and candidate engagement is3744 calls per run. Reject this prototype as a default.
