#!/usr/bin/env python3
"""Attach a diagnostic to owned, already warmed short-context generations.

Profiler timings are never throughput evidence. Repeated requests start fresh
target/draft caches; no long4096-token continuation substitutes for short decode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from run_decode3_checks import core_digest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=["cpu", "metal"], required=True)
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--label", default="short")
    args = parser.parse_args()
    assert 1 <= args.seconds <= 30
    assert args.label.replace("-", "").isalnum()
    stem = f"followup-profile-{args.kind}-{args.label}"
    report = ROOT / f"results/{stem}-generation.json"
    log = ROOT / f"results/{stem}-generation.log"
    assert not report.exists(), "preserve previous diagnostic"
    command = ["target/release/mtp-infer", "--model", args.model,
               "--max-tokens", "256", "--warmup-tokens", "256",
               "--runs", "16", "--draft-depth", "3", "--ignore-eos",
               "--expected", "results/target-baseline-256.json", "--output", str(report)]
    snapshot = core_digest()
    start = time.time()
    profile_error = None
    profile_exit = None
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            while "warmup 0:" not in log.read_text(errors="replace"):
                if process.poll() is not None:
                    raise RuntimeError(f"inference exited before warmup: {process.returncode}")
                if time.monotonic() > deadline:
                    raise RuntimeError("owned process did not warm up within240s")
                time.sleep(0.25)
            if args.kind == "cpu":
                artifact = ROOT / f"results/{stem}-sample.log"
                profile_command = ["sample", str(process.pid), str(args.seconds), "1",
                                   "-file", str(artifact)]
            else:
                artifact = ROOT / f"results/{stem}.trace"
                assert not artifact.exists(), "preserve previous trace"
                profile_command = ["xcrun", "xctrace", "record", "--template", "Metal System Trace",
                                   "--attach", str(process.pid), "--time-limit", f"{args.seconds}s",
                                   "--no-prompt", "--output", str(artifact)]
            with (ROOT / f"results/{stem}-profiler.log").open("wb") as output:
                try:
                    # Saving a Metal trace can greatly outlast its short capture
                    # window. The outer exec remains pollable during this wait.
                    profile = subprocess.run(profile_command, cwd=ROOT, stdout=output,
                                             stderr=subprocess.STDOUT, timeout=300)
                    profile_exit = profile.returncode
                except subprocess.TimeoutExpired as error:
                    profile_error = str(error)
            inference_exit = process.wait(timeout=240)
        finally:
            # Only the child created above belongs to this diagnostic. No broad
            # process-name termination; unrelated model servers stay untouched.
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    assert core_digest() == snapshot, "core changed during diagnostic"
    receipt = {"model": args.model, "kind": args.kind, "source_sha256": snapshot,
               "binary_sha256": sha(ROOT / command[0]), "command": command,
               "pid": process.pid, "profiler_command": profile_command,
               "profiler_exit": profile_exit, "profiler_error": profile_error, "inference_exit": inference_exit,
               "started_unix": start, "ended_unix": time.time(),
               "requested_profile_seconds": args.seconds, "artifact": str(artifact.relative_to(ROOT)),
               "generation_report": str(report.relative_to(ROOT)), "generation_log_sha256": sha(log),
               "instrumented": True, "throughput_qualified": False,
               "protocol": "attach after256-token warmup; sixteen fresh-cache256-token greedy MTP requests"}
    if report.exists():
        generation = json.loads(report.read_text())
        expected = json.loads((ROOT / "results/target-baseline-256.json").read_text())["runs"][0]["tokens"]
        assert len(generation["runs"]) == 16
        assert all(r["generation"]["tokens"] == expected for r in generation["runs"])
        receipt["all_sixteen_256_token_trajectories_exact"] = True
        receipt["generation_report_sha256"] = sha(report)
    (ROOT / f"results/{stem}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert inference_exit == 0 and profile_exit == 0, receipt
    print("FOLLOWUP_PROFILE_CAPTURED", flush=True)


if __name__ == "__main__":
    main()
