#!/usr/bin/env python3
"""Serial actual-model routing qualification and synchronized attribution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_decode3_checks import core_digest, VALIDATION

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix",default="followup-route")
    args = parser.parse_args()
    snapshot = core_digest()
    cases = [
        ("route-bench", "actual-components", [], "ACTUAL_ROUTE_TAIL_COMPONENTS_EXACT", True),
        ("verify-parity", "verify-metal", ["--route-tail"], "VERIFY_PARITY_PASSED", True),
        ("rollback-parity", "rollback-metal", ["--route-tail"], "ROLLBACK_PARITY_PASSED", True),
        ("verify-profile", "synchronized-phases", ["--proposals", "results/followup-baseline-raw-256.json", "--rounds", "12"], "VERIFY_PROFILE_PASSED", False),
    ]
    receipts = []
    for binary, name, extra, marker, validation in cases:
        stem = f"{args.prefix}-{name}"
        output = ROOT / f"results/{stem}.json"
        log = ROOT / f"results/{stem}.log"
        assert not output.exists(), f"preserve prior check{output}"
        command = [f"target/release/{binary}", "--model", args.model, "--output", str(output), *extra]
        start = time.time()
        with log.open("wb") as stream:
            process = subprocess.run(command, cwd=ROOT, env={**os.environ, **(VALIDATION if validation else {})},
                                     stdout=stream, stderr=subprocess.STDOUT)
        receipt = {"name": name, "command": command, "started_unix": start,
                   "ended_unix": time.time(), "source_sha256": snapshot,
                   "binary_sha256": sha(ROOT / command[0]), "exit_code": process.returncode,
                   "success_marker": marker, "passed": process.returncode == 0 and marker in log.read_text(errors="replace"),
                   "validation_environment": VALIDATION if validation else {},
                   "log_sha256": sha(log), "report_sha256": sha(output) if output.exists() else None,
                   "timing_qualified": False}
        (ROOT / f"results/{stem}-command.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipts.append(receipt)
        assert snapshot == core_digest(), "source changed during actual qualification"
        (ROOT / f"results/{args.prefix}-checks.json").write_text(json.dumps({
            "model": args.model, "source_sha256": snapshot, "checks": receipts,
            "complete": len(receipts) == len(cases), "passed": all(r["passed"] for r in receipts)}, indent=2) + "\n")
        print(name, receipt["passed"], flush=True)
        if not receipt["passed"]:
            raise SystemExit(process.returncode or 1)
    print("FOLLOWUP_ACTUAL_ROUTE_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
