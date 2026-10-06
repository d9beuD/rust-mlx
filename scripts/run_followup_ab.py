#!/usr/bin/env python3
"""Source-bound four-pair short and representative-chat kernel experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from run_decode3_checks import core_digest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--bias-artifact")
    parser.add_argument("--suite", choices=["raw", "chat"], default="raw")
    args = parser.parse_args()
    assert not any(os.environ.get(k) for k in ["MTL_SHADER_VALIDATION", "MTL_DEBUG_LAYER"])
    prompts = json.loads((ROOT / f"results/{'raw-prompt' if args.suite == 'raw' else 'workload-prompts'}.json").read_text())
    snapshot = core_digest()
    records = []
    for i, prompt in enumerate(prompts):
        stem = f"followup-ab-{args.kernel}-{args.suite}-p{i}"
        output = ROOT / f"results/{stem}.json"
        log = ROOT / f"results/{stem}.log"
        assert not output.exists(), f"preserve{output}"
        command = ["target/release/mtp-infer", "--model", args.model, "--prompt", prompt,
                   "--max-tokens", "256", "--warmup-tokens", "256", "--runs", "4",
                   "--draft-depth", "3", "--ignore-eos", "--ab-kernel", args.kernel,
                   "--output", str(output)]
        if args.bias_artifact:
            command += ["--draft-logit-bias",args.bias_artifact]
        if args.suite == "chat":
            command += ["--chat", "--no-thinking"]
            oracle = ROOT / f"results/decode3-config-combo-full-chat-p{i}-c1-native.json"
        else:
            oracle = ROOT / "results/target-baseline-256.json"
        expected_report = json.loads(oracle.read_text())
        expected = expected_report["runs"][0].get("tokens") or expected_report["runs"][0]["generation"]["tokens"]
        # The CLI expects the plain-oracle schema, so provide an explicit owned
        # reference file while retaining the complete original report binding.
        native = ROOT / f".unlazy/{stem}-oracle.json"
        native.write_text(json.dumps({"runs": [{"tokens": expected}]}) + "\n")
        command += ["--expected", str(native)]
        start = time.time()
        with log.open("wb") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        receipt = {"command": command, "exit_code": process.returncode, "source_sha256": snapshot,
                   "binary_sha256": sha(ROOT / command[0]), "started_unix": start, "ended_unix": time.time(),
                   "log_sha256": sha(log), "report_sha256": sha(output) if output.exists() else None,
                   "oracle_report": str(oracle.relative_to(ROOT)), "oracle_sha256": sha(oracle)}
        (ROOT / f"results/{stem}-command.json").write_text(json.dumps(receipt, indent=2) + "\n")
        assert core_digest() == snapshot, "core changed during throughput"
        if process.returncode:
            raise SystemExit(process.returncode)
        report = json.loads(output.read_text())
        assert report["prompt_ids"] == expected_report["prompt_ids"]
        pairs = {}
        for row in report["runs"]:
            assert row["generation"]["tokens"] == expected and len(expected) == 256
            assert row["draft_depth"] == 3 and row["kernel_candidate"] == args.kernel
            pairs.setdefault(row["run"], {})[row["candidate_enabled"]] = row
        assert set(pairs) == {1,2,3,4} and all(set(p)=={False,True} for p in pairs.values())
        controls = [pairs[c][False]["decode_tokens_per_second"] for c in range(1,5)]
        candidates = [pairs[c][True]["decode_tokens_per_second"] for c in range(1,5)]
        gains = [(c/b - 1)*100 for b,c in zip(controls,candidates)]
        records.append({"prompt": i, "report": str(output.relative_to(ROOT)), "receipt": receipt,
                        "native_samples": controls, "candidate_samples": candidates,
                        "native_median": statistics.median(controls), "candidate_median": statistics.median(candidates),
                        "paired_gain_samples_percent": gains, "paired_gain_median_percent": statistics.median(gains),
                        "native_cv": statistics.stdev(controls)/statistics.mean(controls),
                        "candidate_cv": statistics.stdev(candidates)/statistics.mean(candidates),
                        "exact_original_256_ids": True})
        print(args.kernel, args.suite, i, records[-1]["paired_gain_median_percent"], flush=True)
    (ROOT / f"results/followup-ab-{args.kernel}-{args.suite}-summary.json").write_text(json.dumps({
        "model": args.model, "kernel": args.kernel, "suite": args.suite, "source_sha256": snapshot,
        "protocol": "four alternating baseline/candidate pairs in one model process; both warmed256; fresh caches; greedy depth3; complete priming/draft/verify/rollback/sync cost",
        "records": records, "complete": True, "promoted": False}, indent=2) + "\n")
    print("FOLLOWUP_AB_COHORT_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
