# Solo decode: MTPLX ports and five follow-up directions

The measured variants retain the original target checkpoint, greedy batch1 and fresh KV caches. No new variant passes the global5% end-to-end gate, and100tok/s solo remains unachieved. Successful ports demonstrate that the earlier incompatible research adapters do not prevent individual MTPLX kernels from being reproduced.

## Protocol and source snapshots

Base676498c; routing/preparation implementation7a28950 and evidence9b048ee. Each subsequent report has a source/binary/command receipt. Four alternating baseline/candidate pairs, each arm warmed256 tokens and generating256 canonical IDs; repeated short raw input and representative Rust/French/SQL/train chats. MTPdepth3; full target vocabulary; complete priming/draft/verification/commit/synchronization cost. Cached-prefix, profiler and HTTP rates are excluded. Hardware M5 Max40/128GB, macOS27.0.1, MLX0.32.2, mlx-rs0.32.0 and mlx-sys0.6.0. Mixed quantization remains per-module. Competing macOS storage/security workloads are recorded and untouched.

Raw median paired gains (percentage, not a sum of component effects):

| Experiment | Raw gain | Decision |
|---|---:|---|
| `route-tail` | +0.501% | Experimental; no default promotion |
| `ple-prepare` | -0.113% | Experimental; no default promotion |
| `rope-ids` | +0.078% | Experimental; no default promotion |
| `config-reuse` | -0.133% | Experimental; no default promotion |
| `runtime-prepare` | +0.029% | Experimental; no default promotion |
| `down-tail` | +0.433% | Experimental; no default promotion |
| `down-packed` | -9.308% | Experimental; no default promotion |
| `draft-bias` | -0.600% | Experimental; no default promotion |
| `down-packed-vector` | +0.432% | Experimental; no default promotion |

## 1. Attribution

`results/followup-attribution-summary.json` binds four raw runs (median70.283tok/s), natural own-process CPU samples and a successful short Metal timeline, plus synchronized phase diagnostics at short context and2107 tokens. Native timer scopes place verification near85% of decode wall, drafting10% and synchronization4.6%. Synchronized short/long phases rank GDN/MoE/HC work, but forced evaluation changes scheduling and cannot provide natural shader percentages.

The trace contains24,033 active compute intervals,3.516s busy over4.566s observed span. Function labels and DRAM counters are unavailable. Its GPU gaps include boundaries, graph preparation and waits; they are not an achievable23% speed gain. CPU eval samples include graph walking/submission and inclusive categories overlap. First trace/save timeout and failed partial export remain recorded; a shorter retry succeeds. Raw captures stay ignored.

## 2. MTPLX kernel reproduction

Ports adapt MTPLX revision9882703f3105363ddc37eca9f97aa09a1d387112, Apache-2.0, by Youssof Altoukhi. Notices are retained and programs credit MTPLX. Reviewed literals/f-strings are extracted without executing upstream Python. Routing preserves native router/shared projections, BF16 stable top10 and renormalization. Down/reduction adapts g32 to actual g64 and preserves native scalar partitions, SIMD sums, BF16 down/product casts and ten-expert reduction order. Unsupported shapes/dtypes retain native MLX.

Routing passes240 actual-bank components across48 banks and1/2/3/4/8 rows. Down and packing pass15 configurations on predetermined banks0/23/47, including8 rows, followed by actual full verifier2–8 twice and every rollback0–4 with continuation. Numerical component fixtures use synthetic sin(j*.013) BF16 layer0 stages, reused across actual banks; they are not natural activations from every layer. Full model checks and256-token trajectories separately use canonical token IDs. Routing raw+0.501% and down/reduction+0.433% do not justify defaults.

## 3. Preparation and scheduling

Early immutable PLE row preparation,8-entry RoPE integer-ID sharing keyed by batch/rows/offset/stride, and bounded per-Kernel MLX-C configuration reuse all preserve caches and exact outputs. Config keys include every output/template/grid/group setting; inputs and stream remain fresh. Pinned MLX-C copies settings by value, and RAII owns each config exactly once. Eight entries per Kernel bound residency. These are preparation/config experiments, not a reactivation of rejected batched PLE dequantization or async layers.

Four separate raw cohorts stay between−0.133% and+0.078%; combined raw+0.029%, chats−0.008/+0.397/−0.207/−0.610%. Full verifier/rollback and Metal checks pass. Leave every switch off. RoPE inverse frequencies were already cached; that existing feature is not presented as a new optimization.

## 4. Draft distillation pilot

A fixed pilot trains a full-vocabulary additive BF16 bias against native teacher distributions, without changing the target or MTP transformer. Fit: seven first-per-language calibration documents,16 fixed positions after64-token teacher-forced contexts (112 predictions). Independent heldout:28 documents/448 predictions; document hashes and64-token prefixes do not overlap. Timing prompts are outside fitting.200 fixed gradient-descent steps, lr0.05,0.1 squared-norm penalty and clamp±0.5; final step selected without heldout tuning. Teacher-forced private contexts are not natural autoregressive acceptance.

Bias max magnitude0.0498047, norm0.168103; additional BF16 values496,640 bytes. Fit teacher KL1.352091→1.350396, heldout1.636804→1.635750. Heldout argmax agreement47.5446→47.9911%, two additional matches. Regularized BF16 objective increases4.069118→4.070240: cast rounding and the penalty prevent a claim of monotonically improved training loss. Natural raw acceptance remains177/234 over78 rounds and paired speed−0.600%. Four chat paired gains are Rust/French/SQL/train +0.948%/+0.226%/-0.843%/-0.586%; every original target ID remains unchanged. Neither training nor timing establishes an improved general predictor. Stored pilot training_seconds includes fitting and metric evaluation, and artifact-load-only cost was not isolated.

`scripts/probe_training_boundary.py` executes small native MLX probes: quantized matmul input gradients work; a custom Metal kernel has no built-in VJP. Training the full private MTP therefore needs explicit derivatives or a native differentiable training graph, together with a larger curated teacher set. The leaf-bias pilot does not prove full training impossible or certify broad model quality. Rust handles all inference; Python is only the training/research oracle.

## 5. Format plus kernel

A lossless four-output word layout changes codes from `[512,2560,80]` to `[512,640,80,4]`, with scales/biases unchanged; no dequantization/requantization occurs. Preparation retains original evaluated contiguous banks as a real native control, and packs only predetermined layers0/23/47:1,258,291,200 additional code bytes. One raw preparation costs0.12355s; both measured arms retain that allocation, and peak MLX memory is70.663GiB in that cohort. No all48-bank speed or memory claim is made.

The first shader recomputes packed addresses inside each dot and regresses complete raw generation9.308%. A second version hoists address calculations and loads four outputs jointly. It restores native-level component costs, with some local short-block gains, yet raw end-to-end gain is0.432% and chats+0.135/−0.565/−1.020/−0.235%. Microbenchmark gains are not global gains. The first vector integration encounters a two-slot Rust table with a third mode and panics before kernel execution; corrected three-slot retry and all actual native/Metal/rollback checks pass. Both failure and retry receipts remain.

## Qualification and remaining research

Current-source qualification completes at SHA256 `33b62bf06487eb013cc5b05658afd8bd7326df0757034be31b1d808f7b9e51a9`:38 release tests,37 portable instrumented tests, strict formatting/Clippy, actual native/routing/preparation/down/vector verifier and rollback checks,64-token private biased drafting, and real default/batch8 HTTP cancellation/UTF-8 tests. Task-owned servers are stopped; unrelated processes remain untouched. `scripts/analyze_followup.py --final` independently audits all25 full-generation reports and rejects a corrupted token despite unchanged metadata. `results/followup-current-qualification.json` and `results/followup-final-summary.json` bind source/binaries/commands/logs; implementation/evidence commits are recorded in the research log.

Future directions with greater potential are training a materially better private MTP, co-designing that predictor with verification depth, and fusing expert output/shared branch/HC writes while preserving BF16 boundaries. A lower-bit expert or hardware-native format is a separate approximate target requiring quality evidence; these lossless experiments do not establish its benefit. More SIMD/address tuning can help a component but has not demonstrated the roughly30% wall-time reduction needed for100tok/s from the measured70.3 baseline. No bandwidth ceiling or universal impossibility is established.

MTPLX source context: [PR475](https://github.com/youssofal/MTPLX/pull/475), [g64 adaptations, PR488](https://github.com/youssofal/MTPLX/pull/488). The latter uses a different quantization recipe and distinguishes rounding-class kernels, acceptance law and relaxed acceptance modes. Its106.67tok/s figure cannot serve as our exact-greedy mixed-checkpoint comparator. Earlier MTPLX/oMLX research adapter failures are scoped to those adapters, not stock products.
