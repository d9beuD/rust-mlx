#!/usr/bin/env python3
"""Compare separately loaded complete configurations, never strided-view controls.

Native plain oracles are outside timing. Approximate variants use their own
oracle; matching those tokens is not original-checkpoint distribution parity.
"""
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
CASES = {
    "layout": ["--native-gate-up"],
    **{f"q{b}": ["--resident-overlay", f".unlazy/decode3-quant-resident-q{b}"] for b in (4, 5, 6)},
    "combo-exact": ["--native-gate-up", "--draft-head-bits", "4", "--draft-vocab-ids", "results/decode3-vocab-upstream64k.json"],
    "combo-full": ["--native-gate-up", "--draft-head-bits", "4"],
    "mixed32": ["--resident-overlay", ".unlazy/decode3-quant-mixed32"],
    "combo-mixed32": ["--resident-overlay", ".unlazy/decode3-quant-mixed32", "--native-gate-up", "--draft-head-bits", "4"],
    **{f"combo-q{b}": ["--resident-overlay", f".unlazy/decode3-quant-resident-q{b}", "--native-gate-up", "--draft-head-bits", "4", "--draft-vocab-ids", "results/decode3-vocab-upstream64k.json"] for b in (5, 6)},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, stem, env=None):
    log = ROOT / f"results/{stem}.log"
    started = time.time()
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {"command": command, "started_unix": started, "ended_unix": time.time(),
               "exit_code": result.returncode, "log_sha256": sha(log),
               "binary_sha256": sha(ROOT / command[0])}
    (ROOT / f"results/{stem}-command.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--suite", choices=["raw", "chat"], default="raw")
    parser.add_argument("--runs", type=int, default=4)
    a = parser.parse_args()
    assert a.runs >= 4
    snapshot = core_digest()
    runner_sha = sha(Path(__file__))
    assert not any(os.environ.get(k) for k in ("MTL_SHADER_VALIDATION", "MTL_DEBUG_LAYER")), "timing must be uninstrumented"
    prompts = json.loads((ROOT / f"results/{'raw-prompt' if a.suite == 'raw' else 'workload-prompts'}.json").read_text())
    approximate = "resident-overlay" in " ".join(CASES[a.case])
    overlay_flags = CASES[a.case][:2] if approximate else []
    oracle_dir = ROOT / ".unlazy/decode3-config-oracles"
    oracle_dir.mkdir(exist_ok=True)
    flags = ["--model", a.model, "--max-tokens", "256", "--warmup-tokens", "256", "--runs", "1", "--ignore-eos"]
    if a.suite == "chat":
        flags += ["--chat", "--no-thinking"]
    oracles = {}
    for profile, extra in [("original", [])] + ([(a.case, overlay_flags)] if approximate else []):
        identity = {"source_sha256": snapshot, "binary_sha256": sha(ROOT / "target/release/infer"),
                    "model_config_sha256": sha(Path(a.model) / "config.json"),
                    "tokenizer_sha256": sha(Path(a.model) / "tokenizer.json"),
                    "overlay_sha256": sha(ROOT / extra[1] / "overlay.json") if extra else None}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        for pi, prompt in enumerate(prompts):
            path = oracle_dir / f"{profile}-{key}-{a.suite}-{pi}.json"
            binding = path.with_suffix(".binding.json")
            if not path.exists():
                command = ["target/release/infer", *flags, "--prompt", prompt, *extra, "--output", str(path)]
                execute(command, f"decode3-oracle-{profile}-{key}-{a.suite}-{pi}",
                        {**os.environ, "RUST_MLX_VERIFY_QMV": "0", "RUST_MLX_HC_PROJECTION": "0", "RUST_MLX_BATCH_QMV": "0"})
                binding.write_text(json.dumps({"identity": identity, "prompt": prompt,
                                              "report_sha256": sha(path)}, indent=2) + "\n")
            proof = json.loads(binding.read_text())
            assert proof == {"identity": identity, "prompt": prompt, "report_sha256": sha(path)}, "stale native oracle"
            oracle = json.loads(path.read_text())
            assert oracle["model"] == a.model and len(oracle["runs"]) == 1 and len(oracle["runs"][0]["tokens"]) == 256
            assert oracle["runtime"]["ignore_eos"]
            assert not oracle["runtime"]["mtp"] and not oracle["runtime"]["prefix_cache"]
            assert bool(oracle["resident_variant"]) == bool(extra)
            if extra:
                overlay = json.loads((ROOT / extra[1] / "overlay.json").read_text())
                assert oracle["resident_variant"] == overlay
            oracles[profile, pi] = path
    final = ROOT / f"results/decode3-config-{a.case}-{a.suite}-256.json"
    assert not final.exists(), f"preserve existing complete cohort {final}"
    records = []
    for cycle in range(1, a.runs + 1):
        for pi, prompt in enumerate(prompts):
            for candidate in ([False, True] if cycle % 2 == 0 else [True, False]):
                stem = f"decode3-config-{a.case}-{a.suite}-p{pi}-c{cycle}-{'candidate' if candidate else 'native'}"
                output = ROOT / f"results/{stem}.json"
                assert not output.exists(), f"preserve process arm {output}"
                oracle_path = oracles[(a.case if approximate and candidate else "original"), pi]
                command = ["target/release/mtp-infer", *flags, "--prompt", prompt,
                           "--expected", str(oracle_path), "--output", str(output)]
                if candidate:
                    command += CASES[a.case]
                execute(command, stem)
                report = json.loads(output.read_text())
                assert len(report["runs"]) == 1 and report["runtime"]["mtp"]
                r = report["runs"][0]
                assert report["prompt_ids"] == json.loads(oracle_path.read_text())["prompt_ids"]
                assert report["runtime"]["warmup_tokens"] == 256 and not report["runtime"]["prefix_cache"]
                assert core_digest() == snapshot and sha(Path(__file__)) == runner_sha, "source changed during cohort"
                assert r["generation"]["tokens"] == json.loads(oracle_path.read_text())["runs"][0]["tokens"]
                assert len(r["generation"]["tokens"]) == 256
                records.append({"prompt": pi, "cycle": cycle, "candidate": candidate,
                                "rate": r["decode_tokens_per_second"], "report": str(output.relative_to(ROOT)),
                                "report_sha256": sha(output), "oracle_sha256": sha(oracle_path),
                                "oracle_report": str(oracle_path.relative_to(ROOT)), "prompt_ids": report["prompt_ids"],
                                "own_native_oracle_exact": True, "exact_original_distribution": not (candidate and approximate)})
                summary = {"case":a.case,"suite":a.suite,"candidate_flags":CASES[a.case],"model":a.model,
                           "source_sha256":snapshot,"runner_sha256":runner_sha,
                           "approximate_target":approximate,"max_tokens":256,"warmup_tokens":256,"runs":a.runs,
                           "batch_size":1,"prefix_cache":False,"mtp":True,"sampler":"greedy","records":records,
                           "protocol":"separate process/model/layout per arm; full256-token warmup then256 measured; alternate original/candidate arms; init outside decode, recorded separately",
                           "complete":False,"qualified":False}
                final.write_text(json.dumps(summary,indent=2)+"\n")
                print(a.case,a.suite,pi,cycle,candidate,r["decode_tokens_per_second"],flush=True)
    result = json.loads(final.read_text())
    stats = []
    for pi in range(len(prompts)):
        pairs = {(r["cycle"],r["candidate"]):r["rate"] for r in records if r["prompt"]==pi}
        gains = [(pairs[c,True]/pairs[c,False]-1)*100 for c in range(1,a.runs+1)]
        stats.append({"prompt":pi,"paired_gains_percent":gains,"paired_median_percent":statistics.median(gains)})
    result["statistics"]=stats
    result["complete"]=True
    final.write_text(json.dumps(result,indent=2)+"\n")
    print("DECODE3_CONFIG_COHORT_COMPLETED",flush=True)


if __name__ == "__main__":
    main()
