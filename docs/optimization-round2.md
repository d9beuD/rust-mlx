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

## Results

GPU draft chains preserve every intermediate private MTP tensor/cache at depths1–7 under actual-checkpoint Metal validation. Four alternating256-token pairs preserve all proposed IDs,177/234 acceptance and target output. Native median70.2937, GPU-chain71.0190tok/s; median paired gain0.9286%. The experiment stays off by default. Raw samples are in results/gpu-draft-raw-ab-256.json.

T4 verifier diagnostics use actual proposals from that run and preserve native logits in12 rounds. Across48 layers,23040 selected expert slots correspond to15309 distinct experts per layer/round, a33.55% duplication opportunity. That number is a theoretical reduction in expert occurrences, not measured memory traffic or speed: caches and native kernels may already exploit reuse. GDN/MoE each account for roughly one quarter of the synchronized component sum, and the two HC phases together another quarter; these diagnostic shares change scheduling and are not natural-runtime percentages.

The first expert-major prototype sorts selected rows, calls native gather_qmm with sorted indices, then restores canonical order before weighted reduction. Fixture F32/BF16 and actual BF16T2/4/8 outputs are exact. Actual full-model verifier lengths2–8 twice and rollback prefixes0–4 also pass exact logits/hidden/cache checks with Metal abort-on-fault. A calibration microbenchmark overlapped a compiler job and remains separately labeled; the uncontended component rerun gives roughly597/626us atT2,487/494us atT4,675/638us atT8. Four complete alternating256-token pairs give native70.2366/sorted69.3416tok/s, median paired gain-1.1324%. Every proposal/acceptance/target token is exact and candidate engagement is3744 calls per run. Reject this prototype as a default.


The other three directions are complete. Deterministic calibrated depth and rejection-driven vocabulary/full-head fallback preserve all target IDs with complete costs included. Depth/vocabulary/combined raw paired gains are0.24/1.52/2.96%; combined four-chat gains16.62/1.00/0.55/5.17%, so no global promotion. Independent policies regress on SQL; workload-dependent improvements remain optional. Full reports and chosen-depth/vocabulary histories are in results/adaptive*-ab-256.json and matching statistics.

Exact greedy8-bit head projection reduces BF16-rounded eight-row maxima/lowest IDs and then the global token. Actual248320x2560 BF16T1–4 native block maxima/IDs, zero ties and extreme reduction cases pass shader validation. The full four-pair raw gain0.69% does not qualify; component projection+greedy reduction is slightly slower than the selected full-logit path. Keep off.

Block-growing KV preserves immutable snapshots and every accepted-prefix continuation, including actual sparse context2096 and private MTP/prompt reuse/batch rows. Complete raw contexts10/2107/4117 give paired gains0.20/-0.79/0.22%, with identical proposals/acceptance/output. Growth-boundary components are slower; conditional MLX buffer donation cannot be assumed while retaining snapshots. Native concatenation stays default.

Mixed-session GPU validation exposed an existing actual8-bit T4 QMV invalid-address failure, including a fixed-depth/full-head reference after an adaptive candidate. Pointer rematerialization is retained as a safety change, with bounded4-position blocks and explicit size guards. Expanded actual-model validation passes recalculated64/32-bit offsets. The32-bit control gains-0.21% raw, so64-bit stays default. Root cause of the original compiler/instrumentation/register interaction remains unproven, and rejected logs remain in results/adaptive-qmv-metal-rejections.json.

Final source483b3cd passes25 release tests, strict format/Clippy,24 portable instrumented tests, current actual-model state/shader tests, and refreshed default/batch HTTP prefix/stream/cancellation/Unicode checks. Latest raw default session is45.28 plain/68.43 MTP; the earlier70.56 is preserved as its historical cohort, not a paired comparison. Complete reports are in results/round2-final-validation.json. All five experiments retain fallbacks and remain disabled.100tok/s in one conversation remains unmet.

Implementation and reports are published in commit2b67a7ac37a44aa863f5c16e5fe550cf084f0087. G1–G24 are met with no abandoned gates; final publication verification also rejects any unpublished local changes. The local user `.agents/` directory is preserved outside the repository.
