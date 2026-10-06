# Measured inference performance

Final production selection: exact greedy MTP depth3, selected shared-weight QMV, restricted combined HC down/inject; compatible batch QMV on. Full-vocabulary drafting, shared GEMV off, streamed QMV off. Native depthwise weights retain guarded allocations and long QMV uses at-most4-position chunks; full-model shader validation passes. M5 Max40 GPU cores /128GB, macOS27.0.1, MLX0.32.2, mlx-rs0.32.0. Model: Qwen3.8-Flash-Next-oQ4e-mtp (Qwen4Exp architecture, mixed affine4/5/6/8-bit).

These are medians of all saved timed repetitions. Raw final plain/MTP uses four alternating pairs; batch figures use all four QMV-on samples from the final guarded/bounded batch A/B. Every generated ID matches the original independent singleton trajectory. Fresh target/private caches and full256-token warmups; no prefix-cache hit or instrumented trace in these offline rates. `(generated-1)/decode wall` includes graph construction, argmax/evaluation and all MTP prime/draft/verify/sync work. Prefill is separate; no unused final forward. Batch uses `B*(generated-1)/decode`, not single-user speed.

| Configuration | Median tok/s | Scope | Raw report |
|---|---:|---|---|
| Rust default plain | 46.51 | single conversation, raw prompt10/output256 | [final-raw-bounded-plain-mtp-256.json](../results/final-raw-bounded-plain-mtp-256.json) |
| Rust default MTP depth3 | 70.56 | single conversation, same raw workload | [final-raw-bounded-plain-mtp-256.json](../results/final-raw-bounded-plain-mtp-256.json) |
| Rust default plain batch2 | 66.28 | aggregate; 33.14 per conversation, output256/row | [final-batch-ab-qmv-bounded-256.json](../results/final-batch-ab-qmv-bounded-256.json) |
| Rust default plain batch4 | 93.30 | aggregate; 23.32 per conversation, output256/row | [final-batch-ab-qmv-bounded-256.json](../results/final-batch-ab-qmv-bounded-256.json) |
| Rust default plain batch8 | 105.27 | aggregate; 13.16 per conversation, output256/row | [final-batch-ab-qmv-bounded-256.json](../results/final-batch-ab-qmv-bounded-256.json) |
| Official mlx-vlm adapted plain | 41.18 | different session; mmap PLE row adapter, output256 | [native-target-baseline-256.json](../results/native-target-baseline-256.json) |

**100 tok/s in one conversation was not reached.** Final guarded/bounded default batch8 exceeds100 aggregate in four complete runs, about13.2 per conversation. Earlier HC batch session was about99 aggregate; retain that variance. Chat prompt structure changes outputs and acceptance: the full-head HC/QMV baseline in the latest pre-fix four-chat suite has medians58.68/51.32/51.96/64.31tok/s. Those are different workloads/sessions from the raw70.56 figure.

Promotions use paired end-to-end gains, not cross-session rate comparisons: verifier QMV raw10.41%; restricted HC raw12.83% and four chats9.20/9.54/9.87/8.90%; batch QMV8.72/16.96/9.10%; HC with batch QMV14.48/7.43/5.57%. After the allocation/long-block fixes, the fresh batch-QMV paired gains remain9.16/14.69/8.93%. Independent native fallbacks remain available with RUST_MLX_VERIFY_QMV=0, RUST_MLX_BATCH_QMV=0, RUST_MLX_HC_PROJECTION=0.

Rejected defaults: pure BF16 GDN attention compilation1.72% raw (exact target logits/states, F32 uses native fallback); shared GEMV0.95% raw and0.85/0.27/1.04% batch; static draft shortlist4.78% raw with a French regression; refreshed shortlist4.20% raw and1.69% French; packed GDN0.06%; whole-HC compilation0.48%; async layers0.37%; fused MoE about3.8% slower; long-prefill batched PLE about3.1%; streamed QMV regresses the large head. Optional refreshed shortlist raw median73.61tok/s preserves target IDs but changes draft acceptance. Keep it experimental. [All paired descriptive statistics](../results/performance-summary.json) preserve every sample and outlier.

Official mlx-vlm adapted plain matches all256 IDs; its unoptimized Python mmap PLE adapter is explicitly part of that measurement. Powered by MTPLX by Youssof Altoukhi ([upstream](https://github.com/youssofal/MTPLX)): the two adapted MTPLX norm-storage attempts fail strict prefill parity (max1.47472/1.78125). Pinned oMLX adapted default/HC-fused-off attempts also fail (max1.83594/1.92188). No throughput was collected after these gates failed. These are research adapter incompatibilities, not claims about their stock HTTP/MTP performance. See [MTPLX BF16 report](../results/mtplx-adapted-baseline.json), [MTPLX F32 report](../results/mtplx-adapted-norm-f32.json), [oMLX default](../results/omlx-adapted-default.json), [oMLX fused-off](../results/omlx-adapted-baseline.json).

Validation: [18 release tests/strict Clippy/format](../results/quality-current.json); [17 instrumented Metal kernel tests](../results/metal-validation.json); [HTTP FIFO/batch/prefix/cancellation/Unicode qualification](../results/server-qualification.json). Full target oracle logits and cache comparisons use error0, including verifier lengths2–8, rollback0–4 and sparse context2096. Checks cover recorded cases, not every possible context. [Full-model hot-attach GPU timeline](../results/full-hot-metal-profile.json) captures111849 attributed active compute intervals; shader function names are unavailable, so per-kernel attribution remains incomplete. [Component capture](../results/head-component.json) covers only the vocabulary head.

Reproduce final raw measurement with:

```sh
target/release/workload-bench --model /Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp --prompts results/raw-prompt.json --output results/repeated-final-raw.json --max-tokens 256 --warmup-tokens 256 --runs 4 --depth 3 --ignore-eos
```

For batch replay use `batch-bench --ab-qmv --expected results/batch-bench-target-256.json --max-tokens 256 --runs 4`; its QMV-on samples represent defaults. Do not change batching, timing, prompt/output lengths or sampler when comparing rates. Exact commands, failed hypotheses and environment observations remain in the [research log](research-log.md).
