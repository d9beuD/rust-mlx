#!/usr/bin/env python3
"""Actual-model native Rust checks, serialized; instrumented rates never qualify."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = {"MTL_SHADER_VALIDATION": "1", "MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING": "1",
              "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1", "MTL_SHADER_VALIDATION_ABORT_ON_FAULT": "1"}


def core_digest():
    paths = [p for name in ("src", "tests", "kernels") for p in (ROOT / name).rglob("*") if p.is_file()]
    paths += [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix()):
        h.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        h.update(path.read_bytes() + b"\0")
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--only", nargs="*", choices=["original", "layout", "q4", "q5", "q6", "mixed32", "combo-mixed32"])
    parser.add_argument("--prefix", default="decode3-final")
    args = parser.parse_args()
    modes = [("original", []), ("layout", ["--native-gate-up"])]
    modes += [(f"q{b}", ["--resident-overlay", f".unlazy/decode3-quant-resident-q{b}"]) for b in (4, 5, 6)]
    modes += [("mixed32", ["--resident-overlay", ".unlazy/decode3-quant-mixed32"]),
              ("combo-mixed32", ["--resident-overlay", ".unlazy/decode3-quant-mixed32", "--native-gate-up"])]
    snapshot = core_digest()
    for name, flags in modes:
        if args.only and name not in args.only:
            continue
        for binary, marker in [("verify-parity", "VERIFY_PARITY_PASSED"), ("rollback-parity", "ROLLBACK_PARITY_PASSED")]:
            stem = f"{args.prefix}-{name}-{binary}-metal"
            output = ROOT / f"results/{stem}.json"
            log = ROOT / f"results/{stem}.log"
            assert not output.exists(), f"preserve existing check {output}"
            command = [f"target/release/{binary}", "--model", args.model, "--output", str(output), *flags]
            start = time.time()
            with log.open("wb") as stream:
                result = subprocess.run(command, cwd=ROOT, env={**os.environ, **VALIDATION}, stdout=stream, stderr=subprocess.STDOUT)
            success = result.returncode == 0 and marker in log.read_text()
            record = {"command": command, "validation_environment": VALIDATION, "source_sha256": snapshot,
                      "started_unix": start, "ended_unix": time.time(), "exit_code": result.returncode,
                      "success_marker": marker, "passed": success, "timings_are_diagnostic": True,
                      "binary_sha256": hashlib.sha256((ROOT / command[0]).read_bytes()).hexdigest(),
                      "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
                      "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
            (ROOT / f"results/{stem}-command.json").write_text(json.dumps(record, indent=2) + "\n")
            print(name, binary, "passed", success, flush=True)
            assert core_digest() == snapshot, "source changed during qualification"
            if not success:
                raise SystemExit(result.returncode or 1)
    print("DECODE3_ACTUAL_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
