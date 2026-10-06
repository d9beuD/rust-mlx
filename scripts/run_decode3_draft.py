#!/usr/bin/env python3
"""Sequential full native Rust MTP cohorts. No MLX computation in Python."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--only", nargs="*", help="Run named cases; default all.")
    args = parser.parse_args()
    cases = [
        ("q4", ["--head-bits", "4"]),
        ("q6", ["--head-bits", "6"]),
        ("multi32", ["--vocab-ids", "results/decode3-vocab-multilingual-32768.json"]),
        ("multi64", ["--vocab-ids", "results/decode3-vocab-multilingual-65536.json"]),
        ("multi96", ["--vocab-ids", "results/decode3-vocab-multilingual-98304.json"]),
        ("q4-multi32", ["--head-bits", "4", "--vocab-ids", "results/decode3-vocab-multilingual-32768.json"]),
        ("q4-multi96", ["--head-bits", "4", "--vocab-ids", "results/decode3-vocab-multilingual-98304.json"]),
        ("q4-multi64", ["--head-bits", "4", "--vocab-ids", "results/decode3-vocab-multilingual-65536.json"]),
        ("q6-multi64", ["--head-bits", "6", "--vocab-ids", "results/decode3-vocab-multilingual-65536.json"]),
        ("q4-code64", ["--head-bits", "4", "--vocab-ids", "results/decode3-vocab-upstream64k.json"]),
    ]
    if args.only:
        assert set(args.only) <= {n for n, _ in cases}, "unknown candidate"
        cases = [(n, flags) for n, flags in cases if n in args.only]
    for name, flags in cases:
        for suite in ("raw", "chat"):
            prefix = f"decode3-final-draft-{name}-{suite}-256"
            output = ROOT / f"results/{prefix}.json"
            log = ROOT / f"results/{prefix}.log"
            assert not output.exists(), f"preserve existing cohort: {output}"
            command = ["target/release/draft-study", "--model", args.model, "--prompts",
                       f"results/{'raw-prompt' if suite == 'raw' else 'workload-prompts'}.json",
                       "--output", str(output), "--max-tokens", "256", "--warmup-tokens", "256",
                       "--runs", "4", "--depth", "3", "--ignore-eos", *flags]
            if suite == "chat":
                command.append("--chat")
            started = time.time()
            with log.open("wb") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record = {"command": command, "started_unix": started, "ended_unix": time.time(),
                      "exit_code": result.returncode,
                      "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                      "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None}
            (ROOT / f"results/{prefix}-command.json").write_text(json.dumps(record, indent=2) + "\n")
            print(name, suite, "exit", result.returncode, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
            report = json.loads(output.read_text())
            assert report["complete"] and all(r["exact_target_ids"] for r in report["records"])
    print("DECODE3_DRAFT_COHORTS_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
