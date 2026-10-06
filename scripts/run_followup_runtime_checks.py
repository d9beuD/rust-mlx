#!/usr/bin/env python3
"""Serial actual-model preparation qualification, without throughput claims."""
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
    args = parser.parse_args()
    snapshot = core_digest()
    cases = [(binary, f"{mode}-{label}", ["--runtime-prepare",mode],marker,True)
        for mode in ["ple-prepare","rope-ids","config-reuse","runtime-prepare"]
        for binary,label,marker in [("verify-parity","verify-metal","VERIFY_PARITY_PASSED"),
                                    ("rollback-parity","rollback-metal","ROLLBACK_PARITY_PASSED")]]
    receipts = []
    for binary, name, extra, marker, validation in cases:
        stem = f"followup-runtime-{name}"
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
        if receipt["passed"]:
            rows=json.loads(output.read_text())
            mode=extra[1]
            axes={"ple-prepare":[0],"rope-ids":[1],"config-reuse":[2],"runtime-prepare":[0,1,2]}[mode]
            assert all(row["runtime_prepare"]==mode for row in rows)
            assert all(rows[-1]["runtime_prepare_calls"][i]>0 for i in axes), "unengaged candidate"
            receipt["engagement_axes"]=axes
        receipts.append(receipt)
        assert snapshot == core_digest(), "source changed during actual qualification"
        (ROOT / "results/followup-runtime-checks.json").write_text(json.dumps({
            "model": args.model, "source_sha256": snapshot, "checks": receipts,
            "complete": len(receipts) == len(cases), "passed": all(r["passed"] for r in receipts)}, indent=2) + "\n")
        print(name, receipt["passed"], flush=True)
        if not receipt["passed"]:
            raise SystemExit(process.returncode or 1)
    print("FOLLOWUP_ACTUAL_RUNTIME_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
