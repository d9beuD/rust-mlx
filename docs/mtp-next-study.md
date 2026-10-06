# Generated-trajectory draft correction and remaining decode axes

Study on the original Qwen3.8-Flash-Next-oQ4e-mtp checkpoint, MLX0.32.2,
M5Max40 GPU /128GB, macOS27.0.1. Production dispatch remains unchanged.
All throughput runs use one conversation, greedy sampling, fresh independent
private/target caches and no prefix cache. GPU instrumentation is separate.

## Private correction

A small activation-dependent residual predicts a correction to the frozen MTP
mixed output. Its input combines normalized mixed hidden, preceding wide hidden,
and embedding; rank32 has573440 F32 parameters. The correction is cast toBF16
before addition. The original target, full vocabulary head and wide private
state are unchanged. Verification preserves canonical greedy target IDs.
Artifacts bind config, tokenizer and every checkpoint shard by SHA256 and reject
invalid shapes, nonfinite parameters and changed checkpoints.

192 document-disjoint generated trajectories are fixed before training:128 fit,
32 validation,32 blind, excluding previous calibration/heldout documents. Eight
fixed task domains cover natural language, code and reasoning. Samples after
firstEOS are removed during training/metrics. There are29439/7241/7229 valid
prediction positions. Four epochs/460 updates stop after three validation
regressions; epoch0 is selected. The48h allowance is a ceiling. This is a small
residual pilot, not retraining the full transformer.

| One-step metric | Frozen native | Selected correction |
|---|---:|---:|
| Validation greedy agreement |83.4553%|85.4440%|
| Validation NLL |0.571219|0.509828|
| Blind greedy agreement |84.0780%|85.9455%|
| Blind NLL |0.542646|0.478033|

These teacher-state metrics do not measure speculative acceptance. All seven
fixed depths are subsequently evaluated on complete natural validation
trajectories. Depth2 is selected before blind/external tests. Rank64 training is
conditional on a5% depth3 validation speed benefit; its trigger is not met.

The32 blind prompts run four alternating pairs each, with EOS respected and
all target IDs exact. Weighted serial decode throughput improves60.3691 to
63.9799tok/s (+5.9813%). Median per-prompt paired gain is5.6649%;31 prompts have
nonzero post-first-token timing, with gains ranging−3.2393 to17.1509% and17
reaching5%. This establishes a corpus-specific benefit and does not establish
100tok/s or a universal default improvement.

## Exact MoE/shared/HC epilogue

A guarded Metal epilogue fuses shared weighting, routed/shared addition,
HC injection and residual addition. Every originalBF16 arithmetic boundary is
retained. Native projection, routing and expert reduction remain in place.
Unsupported geometry falls back natively; the experiment stays disabled by
default. Actual verifier lengths2–8 and every rollback prefix0–8 with
continuation match logits, hidden and all states under Metal validation.

Four warmed alternating256-token pairs yield raw−0.9425%; four chat cohorts
−1.581/−0.637/−0.953/−1.028%. Changes below2% may be noise. No cohort meets5%,
so the conditional producer/down fusion stage is not advanced.

## Quantization screening

Native MLX reconstructs existingQ4 selected expert-down banks0/23/47 and
requantizes them toQ3/g64 orMXFP4/g32. Actual captured verifier activations at
T1/2/3/4/8 run ten warmups and100 alternating pairs. All30 cases execute.

| Format | Median component gain | Range | Median output relativeL2 |
|---|---:|---:|---:|
|Q3/g64|+0.9455%|−0.4088..+3.7229%|0.179678|
|MXFP4/g32|−5.6721%|−10.0253..−1.5206%|0.100719|

No format reaches5% even locally. The conditional Rust port, approximate
checkpoint overlay and broad quality evaluation are therefore not advanced.
These errors concern selected-bank outputs, not whole-model accuracy or
perplexity. Requantization begins from the existing quantized checkpoint.

## Replay and provenance

Use the original checkpoint as `TARGET` and `.venv/bin` onPATH forCargo.
Private corpus/trajectory tensors, adapters, logs and captures remain local.
Public JSON receipts preserve IDs, commands, repetitions, model/hardware,
source/binary/log identities, mixed quantization, startup and cache settings.

```sh
PATH="$PWD/.venv/bin:$PATH" cargo build --release
.venv/bin/python scripts/prepare_mtp_trajectories.py --model "$TARGET"
target/release/mtp-collect --model "$TARGET" --corpus .unlazy/mtp-next/prompts.json --destination .unlazy/mtp-next/trajectories --output results/mtp-next-collection.json
.venv/bin/python scripts/run_mtp_next.py --model "$TARGET"
.venv/bin/python scripts/run_mtp_next_long.py --model "$TARGET"
.venv/bin/python scripts/run_mtp_next_qualification.py --model "$TARGET"
.venv/bin/python scripts/analyze_mtp_next.py
```

Runners preserve existing output names and fail instead of overwriting evidence;
archive the published research reports and use an empty output set to replay. The cached source corpus from the earlier decode study must be available before preparing trajectories. Collection has its own earlier binary
receipt. A first qualification attempt was rejected because the source changed
during its checks; its successful numerical checks are retained separately and
are not final qualification. Timing/final checks use frozen core
`91e3047d7915c20188acf2a42efe4aeb304c0c2838a3f9ec24719a367f21b7c3`.

External four-pair256-token cohorts preserve every original greedy ID:

| Request | Native mediantok/s | Corrected mediantok/s | Paired gain |
|---|---:|---:|---:|
|Raw reference|69.9727|69.3545|−0.7144%|
|Rust chat|68.1671|74.2071|+8.8870%|
|French chat|58.0655|61.7526|+6.6889%|
|SQL chat|65.9686|65.1824|−0.7951%|
|Fourth chat|69.0726|67.2522|−3.3619%|

The raw/global default gate is rejected. The optional mode may help these
natural workloads, but no automatic request classification is implemented.
The predeclared first blind document is also extended to1024 forced tokens,
including afterEOS. Four alternating pairs preserve all1024 IDs. Native median
32.7075→corrected38.0125tok/s; paired median+16.4248%, pair range
15.2530..16.6669%, native/candidate rate CV0.00578/0.00548. The continuation
has lower acceptance and absolute throughput than the short natural suite.
This confirms a localized long-generation benefit, not a100tok/s result.

Final frozen-source qualification passes strict formatting/Clippy,41 release
tests and40 portable instrumented Metal tests. Actual native, routing,
preparation and down/vector verifier/rollback checks pass. Default FIFO and
plain batch8 HTTP/SSE/prefix/cancellation tests pass, with three cancellation
survivors and64 Unicode caps per mode. Both modes include actual partial
codepoints; only owned servers are stopped. A refreshed current-source nonzero
adapter oracle and frozenQ8 head gradient check also pass Metal validation.

`scripts/analyze_mtp_next.py` independently recomputes paired/weighted rates,
checks corpus/file/source/log identities, all seven validation depths, canonical
external IDs, long/blind exactness and current qualification. An intentionally
corrupted candidate token is rejected. The authoritative summary is
`results/mtp-next-summary.json`; quality receipts are in
`results/mtp-next-current-qualification.json`.

Implementation commit:052a74343186fe0078ecbde2243d48a52c3abb4e.
Production defaults and `results/performance-summary.json` remain unchanged.

Evidence commit:5f2078e29303d1ee782ca7b255cd3a99988b19af, published on public `d9beuD/rust-mlx` main and independently verified through GitHub API. Model/adapter tensors and local logs remain unpublished.
