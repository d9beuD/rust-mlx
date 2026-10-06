# Measured inference performance

Current production selection: exact greedy MTP depth3, selected shared-weight QMV, restricted combined HC down/inject; compatible batch QMV on. Full-vocabulary drafting, shared GEMV off, streamed QMV off. Native depthwise weights retain guarded allocations and long QMV uses at-most4-position chunks; full-model shader validation passes for the recorded cases. QMV recomputes addresses from original buffers; the64-bit offset path is default. M5 Max40 GPU cores /128GB, macOS27.0.1, MLX0.32.2, mlx-rs0.32.0. Model: Qwen3.8-Flash-Next-oQ4e-mtp (Qwen4Exp architecture, mixed affine4/5/6/8-bit).

These are medians of all saved timed repetitions. Round-two and previous stable raw plain/MTP each use four alternating pairs; the latest matrix-study MTP-only control uses four complete runs; batch figures use all four QMV-on samples from the final guarded/bounded batch A/B. Every generated ID matches the original independent singleton trajectory. Fresh target/private caches and full256-token warmups; no prefix-cache hit or instrumented trace in these offline rates. `(generated-1)/decode wall` includes graph construction, argmax/evaluation and all MTP prime/draft/verify/sync work. Prefill is separate; no unused final forward. Batch uses `B*(generated-1)/decode`, not single-user speed.

| Configuration | Median tok/s | Scope | Raw report |
|---|---:|---|---|
| Rust latest MTP control | 68.72 | single conversation, raw prompt10/output256, four exact runs; no new kernel | [matrix-final-default-mtp-256.json](../results/matrix-final-default-mtp-256.json) |
| Rust round-two plain | 45.28 | single conversation, raw prompt10/output256, round-two session | [round2-final-default-raw-256.json](../results/round2-final-default-raw-256.json) |
| Rust round-two MTP depth3 | 68.43 | same raw workload, four complete pairs | [round2-final-default-raw-256.json](../results/round2-final-default-raw-256.json) |
| Rust previous stable plain | 46.51 | historical session, same raw prompt10/output256 | [final-raw-bounded-plain-mtp-256.json](../results/final-raw-bounded-plain-mtp-256.json) |
| Rust previous stable MTP depth3 | 70.56 | historical session, same raw workload | [final-raw-bounded-plain-mtp-256.json](../results/final-raw-bounded-plain-mtp-256.json) |
| Rust default plain batch2 | 66.28 | aggregate; 33.14 per conversation, output256/row | [final-batch-ab-qmv-bounded-256.json](../results/final-batch-ab-qmv-bounded-256.json) |
| Rust default plain batch4 | 93.30 | aggregate; 23.32 per conversation, output256/row | [final-batch-ab-qmv-bounded-256.json](../results/final-batch-ab-qmv-bounded-256.json) |
| Rust default plain batch8 | 105.27 | aggregate; 13.16 per conversation, output256/row | [final-batch-ab-qmv-bounded-256.json](../results/final-batch-ab-qmv-bounded-256.json) |
| Official mlx-vlm adapted plain | 41.18 | different session; mmap PLE row adapter, output256 | [native-target-baseline-256.json](../results/native-target-baseline-256.json) |

**100 tok/s in one conversation was not reached.** Final guarded/bounded default batch8 exceeds100 aggregate in four complete runs, about13.2 per conversation. Earlier HC batch session was about99 aggregate; retain that variance. Chat prompt structure changes outputs and acceptance: the full-head HC/QMV baseline in the latest pre-fix four-chat suite has medians58.68/51.32/51.96/64.31tok/s. Those are different workloads/sessions from the historical raw70.56 figure. Round-two raw45.28/68.43 comes from the fully validated recalculated-pointer source483b3cd, with competing macOS StorageManagement processes recorded. Cross-session changes are not paired kernel gains.

Promotions use paired end-to-end gains, not cross-session rate comparisons: verifier QMV raw10.41%; restricted HC raw12.83% and four chats9.20/9.54/9.87/8.90%; batch QMV8.72/16.96/9.10%; HC with batch QMV14.48/7.43/5.57%. After the allocation/long-block fixes, the fresh batch-QMV paired gains remain9.16/14.69/8.93%. Independent native fallbacks remain available with RUST_MLX_VERIFY_QMV=0, RUST_MLX_BATCH_QMV=0, RUST_MLX_HC_PROJECTION=0.

Rejected defaults: pure BF16 GDN attention compilation1.72% raw (exact target logits/states, F32 uses native fallback); shared GEMV0.95% raw and0.85/0.27/1.04% batch; static draft shortlist4.78% raw with a French regression; refreshed shortlist4.20% raw and1.69% French; packed GDN0.06%; whole-HC compilation0.48%; async layers0.37%; fused MoE about3.8% slower; long-prefill batched PLE about3.1%; streamed QMV regresses the large head. Optional refreshed shortlist raw median73.61tok/s preserves target IDs but changes draft acceptance. Keep it experimental. [All paired descriptive statistics](../results/performance-summary.json) preserve every sample and outlier.


Round-two research is complete and public at [d9beuD/rust-mlx](https://github.com/d9beuD/rust-mlx). The five directions retain exact target tokens and their native fallback. None qualifies as a new global default:

| Direction | Median paired raw gain | Decision |
|---|---:|---|
| Expert-major native MoE gather | -1.13% | Off; sorting does not itself share expert weight loads |
| GPU-resident draft IDs | +0.93% | Off; exact private chains1–7, little complete-round gain |
| Cost-aware depth plus vocabulary fallback | +2.96% | Off; four chats+16.62/+1.00/+0.55/+5.17%, workload dependent |
| Exact greedy head | +0.69% | Off; actual BF16 block maxima/IDs exact, component slightly slower |
| Block-growing KV | +0.20/-0.79/+0.22% at contexts10/2107/4117 | Off; exact snapshots/rollback/sparse state, no long-context gain |

Each row names its own measured cohort; they are not rates from one combined run. The policy/head cohorts precede the final address-mode diagnostic; their reports retain exact measured binary hashes. The KV pair uses the same optional address32 mode in both arms, whose independent raw gain versus address64 is-0.21%; this does not promote either experiment. Historical round-two source/Metal/HTTP evidence is in [round2-final-validation.json](../results/round2-final-validation.json), per-cohort descriptive statistics in `results/*-statistics.json`. The earlier QMV8 T4 mixed-session invalid-address logs are preserved in [adaptive-qmv-metal-rejections.json](../results/adaptive-qmv-metal-rejections.json). Recomputed-pointer modes pass expanded actual-model instrumentation; the underlying compiler/register/instrumentation cause is unproven.

Official mlx-vlm adapted plain matches all256 IDs; its unoptimized Python mmap PLE adapter is explicitly part of that measurement. Powered by MTPLX by Youssof Altoukhi ([upstream](https://github.com/youssofal/MTPLX)): the two adapted MTPLX norm-storage attempts fail strict prefill parity (max1.47472/1.78125). Pinned oMLX adapted default/HC-fused-off attempts also fail (max1.83594/1.92188). No throughput was collected after these gates failed. These are research adapter incompatibilities, not claims about their stock HTTP/MTP performance. See [MTPLX BF16 report](../results/mtplx-adapted-baseline.json), [MTPLX F32 report](../results/mtplx-adapted-norm-f32.json), [oMLX default](../results/omlx-adapted-default.json), [oMLX fused-off](../results/omlx-adapted-baseline.json).

Current validation: [29 release tests/strict Clippy/format](../results/quality-current.json); [28 portable instrumented Metal tests](../results/metal-validation.json), including an expected numerical counterexample and native fallback, not qualification of that kernel; [HTTP FIFO/batch/prefix/cancellation/Unicode qualification](../results/server-qualification.json). Full target oracle logits and cache comparisons use error0, including verifier lengths2–8, rollback0–4 and sparse context2096. Checks cover recorded cases, not every possible context. [Full-model hot-attach GPU timeline](../results/full-hot-metal-profile.json) captures111849 attributed active compute intervals; shader function names are unavailable, so per-kernel attribution remains incomplete. [Component capture](../results/head-component.json) covers only the vocabulary head.

Reproduce final raw measurement with:

```sh
target/release/workload-bench --model /Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp --prompts results/raw-prompt.json --output results/repeated-final-raw.json --max-tokens 256 --warmup-tokens 256 --runs 4 --depth 3 --ignore-eos
```

For batch replay use `batch-bench --ab-qmv --expected results/batch-bench-target-256.json --max-tokens 256 --runs 4`; its QMV-on samples represent defaults. Do not change batching, timing, prompt/output lengths or sampler when comparing rates. Exact commands, failed hypotheses and environment observations remain in the [research log](research-log.md).

M5 matrix-unit study: [six Rust/Metal variants](optimization-study-m5-matrix.md),207 final component configurations with actual mixed checkpoint weights and explicitly synthetic activation stages. Direct packed Q4 preserves the short verifier and rollback but diverges at token103 of the256-token trajectory; the rejected long replay is never a speed report. Cooperative-input multiply-accumulate has a separate instrumented numerical counterexample of unproven cause. All prototypes remain off. The latest four-run default MTP control is68.715tok/s, all canonical256 target IDs exact; this new cohort is not a paired kernel gain. Current source/Metal/HTTP provenance is in [matrix-study-validation.json](../results/matrix-study-validation.json); previous source483b3cd quality/server reports are archived as `*-before-matrix.json`. The quantization/assembly inventory and draft-head-only probes are in [the separate study](optimization-study-quantization-metal.md). No component, instrumented or failed-trajectory rate enters performance-summary.json.
