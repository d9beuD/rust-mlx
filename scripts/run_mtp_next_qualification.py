#!/usr/bin/env python3
"""Serialize current-source checks and task-owned HTTP qualification.

Run only after throughput/other MLX jobs finish. No instrumented or HTTP timing
is included in raw throughput. Old qualification artifacts are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

import requests

from run_decode3_checks import core_digest, VALIDATION

ROOT = Path(__file__).resolve().parents[1]
METHOD = "SHA256 of sorted relative path + NUL + file bytes + NUL; all src/tests/kernels files, Cargo.toml, Cargo.lock"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    a = parser.parse_args()
    snapshot = core_digest()
    environment = {**os.environ, "PATH": str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"]}
    assert not any(environment.get(k) for k in VALIDATION), "start qualification without inherited GPU instrumentation"
    assert not any(k.startswith("RUST_MLX_") for k in environment), "qualify actual defaults without inherited overrides"
    runner_sha = sha(Path(__file__))
    receipts = []

    def check(command, name, marker=None, extra_env=None):
        log = ROOT / f"results/mtp-next-current-{name}.log"
        assert not log.exists(), f"preserve {log}"
        started = time.time()
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=ROOT, env={**environment, **(extra_env or {})},
                                    stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text()
        passed = result.returncode == 0 and (marker is None or marker in text)
        receipt = {"command": command, "log": str(log.relative_to(ROOT)), "sha256": sha(log),
                   "code_sha256": snapshot, "source_hash_method": METHOD, "runner_sha256": runner_sha,
                   "started_unix": started, "ended_unix": time.time(), "exit_code": result.returncode,
                   "passed": passed, "success_marker": marker,
                   "contains_metal_validation_enabled": bool(extra_env),
                   "validation_environment": extra_env or {}}
        (ROOT / f"results/mtp-next-current-{name}-command.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipts.append(receipt)
        assert core_digest() == snapshot and sha(Path(__file__)) == runner_sha, "source changed during qualification"
        if not passed:
            raise SystemExit(result.returncode or 1)
        print(name, "passed", flush=True)
        return receipt, text

    # Build actual executables; Cargo's test binary is not the deployed server.
    check(["cargo", "build", "--release"], "build")
    quality, text = check(["scripts/check.sh"], "quality", "QUALITY_CHECKS_PASSED")
    quality["release_tests_passed"] = sum(map(int, re.findall(r"test result: ok\. (\d+) passed", text)))
    quality["checks"] = ["cargo fmt --check", "cargo clippy --workspace --all-targets --all-features -- -D warnings",
                         "cargo test --workspace --release"]
    metal, text = check(["scripts/validate-metal.sh"], "portable-metal", "METAL_VALIDATION_PASSED", VALIDATION)
    metal["contains_metal_validation_enabled"] = True
    metal["instrumentation"] = VALIDATION
    metal["validation_tests_passed"] = sum(map(int, re.findall(r"test result: ok\. (\d+) passed", text)))
    metal["timing"] = "diagnostic only; excluded from throughput"
    metal["negative_control"] = "existing cooperative matrix discrepancy remains rejected; native fallback retained"
    for script, marker, prefix in [
        ("run_decode3_checks.py","DECODE3_ACTUAL_CHECKS_PASSED","mtp-next-current-native"),
        ("run_followup_route_checks.py","FOLLOWUP_ACTUAL_ROUTE_CHECKS_PASSED","mtp-next-current-route"),
        ("run_followup_runtime_checks.py","FOLLOWUP_ACTUAL_RUNTIME_CHECKS_PASSED","mtp-next-current-runtime"),
        ("run_followup_down_checks.py","FOLLOWUP_ACTUAL_DOWN_CHECKS_PASSED","mtp-next-current-down"),
    ]:
        extra=["--only","original"] if script=="run_decode3_checks.py" else []
        check([".venv/bin/python",f"scripts/{script}","--model",a.model,"--prefix",prefix,*extra],prefix,marker)
    bias_output="results/mtp-next-current-draft-bias-metal.json"
    check(["target/release/mtp-infer","--model",a.model,"--draft-logit-bias",".unlazy/followup-draft-bias",
           "--ab-kernel","draft-bias","--max-tokens","64","--warmup-tokens","64","--runs","1",
           "--draft-depth","3","--ignore-eos","--output",bias_output],"draft-bias-metal",None,VALIDATION)
    bias=json.loads((ROOT/bias_output).read_text())
    oracle=json.loads((ROOT/"results/target-baseline-256.json").read_text())["runs"][0]["tokens"][:64]
    assert len(bias["runs"])==2 and all(r["generation"]["tokens"]==oracle for r in bias["runs"])
    assert {r["draft_head_enabled"] for r in bias["runs"]}=={False,True}
    names = ["quality-current.json", "metal-validation.json", "server-qualification.json", "server-smoke.json",
             "server-batch-smoke.json", "server-utf8-batch1.json", "server-utf8-batch8.json"]
    for name in names:
        old = ROOT / "results" / name
        archived = ROOT / "results" / f"mtp-next-prior-{name}"
        assert not archived.exists(), f"preserve previous archive {archived}"
        if old.exists():
            shutil.copy2(old, archived)
    owned = []
    server_start = time.time()
    for mode, flags, scripts in [
        ("fifo", [], ["server_smoke.py", "server_utf8_smoke.py"]),
        ("batch8", ["--no-mtp", "--batch-size", "8"], ["server_batch_smoke.py", "server_utf8_smoke.py"]),
    ]:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8080))  # Never stop an existing process to claim this port.
        log = ROOT / f"results/mtp-next-current-server-{mode}.log"
        assert not log.exists()
        command = ["target/release/server", "--model", a.model, *flags]
        with log.open("wb") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    assert proc.poll() is None, f"task-owned server failed: {log}"
                    try:
                        response = requests.get("http://127.0.0.1:8080/health", timeout=1)
                        if response.status_code == 200:
                            assert response.json()["batch_size"] == (1 if mode == "fifo" else 8)
                            break
                    except requests.RequestException:
                        pass
                    assert time.monotonic() < deadline, "task-owned server did not become ready"
                    time.sleep(.25)
                for script in scripts:
                    check([".venv/bin/python", f"scripts/{script}"], f"{mode}-{script[:-3]}",
                          {"server_smoke.py": "SERVER_SMOKE_PASSED", "server_batch_smoke.py": "SERVER_BATCH_SMOKE_PASSED",
                           "server_utf8_smoke.py": "SERVER_UTF8_SMOKE_PASSED"}[script])
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
                owned.append({"pid": proc.pid, "command": command, "stopped": proc.poll() is not None,
                              "exit_code": proc.returncode, "log": str(log.relative_to(ROOT)), "log_sha256": sha(log)})
                (ROOT / "results/mtp-next-current-owned-servers.json").write_text(json.dumps(owned, indent=2) + "\n")
    reports = {}
    for name in names[3:]:
        path = ROOT / "results" / name
        assert path.stat().st_mtime >= server_start
        reports[path.stem] = {"sha256": sha(path), "mtime": path.stat().st_mtime}
    unicode = [json.loads((ROOT / f"results/server-utf8-batch{b}.json").read_text()) for b in (1, 8)]
    batch = json.loads((ROOT / "results/server-batch-smoke.json").read_text())
    assert all(len(u["records"]) == 64 and u["partial_codepoint_cases"] > 0 for u in unicode)
    assert len(batch["concurrent_cancellation_results"]) == 3
    server = {"passed": True, "code_sha256": snapshot, "source_hash_method": METHOD,
              "binary_sha256": sha(ROOT / "target/release/server"), "runtime": "actual defaults; greedy; no kernel overrides",
              "server_modes": ["default FIFO", "--no-mtp --batch-size8"], "reports": reports,
              "scripts": {n: sha(ROOT / "scripts" / f"{n}.py") for n in ("server_smoke", "server_batch_smoke", "server_utf8_smoke")},
              "owned_servers": owned, "owned_servers_stopped": all(p["stopped"] for p in owned),
              "timing": "HTTP/prefix/diagnostic timings excluded from raw inference speed",
              "partial_codepoint_cases": [u["partial_codepoint_cases"] for u in unicode],
              "concurrent_cancellation_survivors": 3,
              "success_logs": [r for r in receipts if "server_" in r["log"]], "captured_unix_seconds": time.time()}
    for name, data in [("quality-current.json", quality), ("metal-validation.json", metal), ("server-qualification.json", server)]:
        (ROOT / "results" / name).write_text(json.dumps(data, indent=2) + "\n")
    (ROOT/"results/mtp-next-current-qualification.json").write_text(json.dumps({"passed":True,"source_sha256":snapshot,"receipts":receipts,"quality":quality,"metal":metal,"server":server,"bias_shader_tokens_exact":True},indent=2)+"\n")
    print("MTP_NEXT_CURRENT_QUALIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
