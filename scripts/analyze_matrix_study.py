"""Recompute component statistics and verify the matrix-study rejection evidence."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest():
    files = sorted([p for folder in ("src", "tests", "kernels")
                    for p in (ROOT / folder).rglob("*") if p.is_file()]
                   + [ROOT / "Cargo.toml", ROOT / "Cargo.lock"])
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def recorded_sources_match(report):
    for line in report["source_file_sha256"].splitlines():
        value, name = line.split(None, 1)
        assert digest(ROOT / name.strip()) == value, f"changed component source: {name}"


def reject(p):
    assert p["complete"] is False and p["qualified"] is False
    assert p["candidate_enabled"] and p["matrix_calls"] > 0
    a, b = p["generation"]["tokens"], p["expected_tokens"]
    assert len(a) == len(b) == 256
    differences = [i for i, (a, b) in enumerate(zip(a, b)) if a != b]
    assert differences and differences[0] == p["first_mismatch"]
    return differences


def main():
    reports = ["matrix-final-staged-register-components.json",
               "matrix-final-hybrid-components.json", "matrix-final-affine-components.json",
               "matrix-final-packed-components.json"]
    rows, hashes = [], {}
    rng = random.Random(42)
    for name in reports:
        path = ROOT / "results" / name
        p = json.loads(path.read_text())
        assert p["complete"] and not p["instrumented"] and p["repetitions"] >= 30
        assert p["prompt_ids"] is None and p["synthetic_positions"] == 10
        assert p["environment"]["software"]["mlx"] == "0.32.2"
        recorded_sources_match(p)
        hashes[name] = digest(path)
        for r in p["benchmarks"]:
            a, b = r["original_seconds"], r["candidate_seconds"]
            assert len(a) == len(b) == p["repetitions"]
            assert all(math.isfinite(x) and x > 0 for x in a + b)
            assert r["exact"] == (r["different_elements"] == 0)
            gains = [100 * (a / b - 1) for a, b in zip(a, b)]
            boot = sorted(st.median(rng.choices(gains, k=len(gains))) for _ in range(2000))
            rows.append({k: r[k] for k in ("module", "rows", "mode", "splits", "bits", "group_size", "exact", "max_abs_error", "different_elements", "elements")} | {
                "report": name, "reference_median_us": st.median(a) * 1e6,
                "candidate_median_us": st.median(b) * 1e6,
                "reference_cv_percent": st.pstdev(a) / st.mean(a) * 100,
                "candidate_cv_percent": st.pstdev(b) / st.mean(b) * 100,
                "paired_median_gain_percent": st.median(gains),
                "paired_bootstrap_95_percent": [boot[49], boot[1949]]})
    failed = json.loads((ROOT / "results/matrix-packed-rejected-256.json").read_text())
    differences = reject(failed)
    # A report with no ID discrepancy must not qualify as rejection evidence.
    control = json.loads(json.dumps(failed))
    control["generation"]["tokens"] = control["expected_tokens"][:]
    try:
        reject(control)
    except AssertionError:
        pass
    else:
        raise AssertionError("rejection oracle accepted a false counterexample")
    native = json.loads((ROOT / "results/matrix-final-default-mtp-256.json").read_text())
    assert len(native["runs"]) == 4
    assert all(r["generation"]["tokens"] == failed["expected_tokens"] for r in native["runs"])
    assert all(not r["matrix_enabled"] and r["matrix_calls"] == 0 for r in native["runs"])
    quality = (ROOT / "results/matrix-quality-final.log").read_text()
    metal = (ROOT / "results/matrix-metal-final.log").read_text()
    assert "QUALITY_CHECKS_PASSED" in quality and "METAL_VALIDATION_PASSED" in metal
    checks = {}
    for name in ["matrix-final-default-verifier-metal.json", "matrix-final-packed-verifier-metal.json"]:
        p = json.loads((ROOT / "results" / name).read_text())
        assert [r["depth"] for r in p] == list(range(2, 9)) * 2
        assert all(r["logit_error"] == r["hidden_error"] == r["state_error"] == 0 for r in p)
        previous_calls = 0
        for r in p:
            if "packed" in name:
                assert r["matrix_packed"] and not r["matrix_affine"]
                assert (r["matrix_calls"] > previous_calls) == (r["depth"] <= 4)
            else:
                assert not r["matrix_packed"] and not r["matrix_affine"]
                assert r["matrix_calls"] == 0
            previous_calls = r["matrix_calls"]
        checks[name] = digest(ROOT / "results" / name)
    rollback = json.loads((ROOT / "results/matrix-final-packed-rollback-metal.json").read_text())
    assert [r["keep"] for r in rollback] == list(range(5))
    assert all(r["logit_error"] == r["state_error"] == 0 and r["matrix_calls"] > 0 for r in rollback)
    checks["matrix-final-packed-rollback-metal.json"] = digest(ROOT / "results/matrix-final-packed-rollback-metal.json")
    instrumented = {}
    for name, count in [("matrix-final-components-metal.json", 120),
                        ("matrix-final-hybrid-metal.json", 60),
                        ("matrix-final-affine-metal.json", 15),
                        ("matrix-final-packed-metal.json", 12)]:
        path = ROOT / "results" / name
        p = json.loads(path.read_text())
        assert p["complete"] and p["instrumented"] and p["repetitions"] == 2
        recorded_sources_match(p)
        assert len(p["benchmarks"]) == count
        assert all(r["exact"] == (r["different_elements"] == 0) for r in p["benchmarks"])
        instrumented[name] = {"sha256": digest(path), "configurations": count,
                              "numerically_exact": sum(r["exact"] for r in p["benchmarks"]),
                              "timing_qualified": False}
    manifest = json.loads((ROOT / "results/matrix-study-validation.json").read_text())
    assert manifest["code_sha256"] == source_digest()
    assert manifest["analysis_script_sha256"] == digest(Path(__file__))
    for name, value in manifest["reports_sha256"].items():
        assert digest(ROOT / name) == value, f"changed qualification report: {name}"
    for name in ["quality-current.json", "metal-validation.json", "server-qualification.json"]:
        p = json.loads((ROOT / "results" / name).read_text())
        assert p["passed"] and p["code_sha256"] == source_digest()
        logs = p.get("success_logs", [p])
        for item in logs:
            assert item["exit_code"] == 0 and item["passed"]
            path = ROOT / item["log"]
            assert digest(path) == item["sha256"] and item["success_marker"] in path.read_text()
    server = json.loads((ROOT / "results/server-qualification.json").read_text())
    for name, item in server["reports"].items():
        assert digest(ROOT / f"results/{name}.json") == item["sha256"]
    for name, value in server["scripts"].items():
        assert digest(ROOT / f"scripts/{name}.py") == value
    result = {"source_sha256": source_digest(), "analysis_script_sha256": digest(Path(__file__)),
              "component_reports_sha256": hashes, "statistics": rows,
              "source_hash_method": "sorted Path relative name + NUL + bytes + NUL; src/tests/kernels and Cargo.toml/Cargo.lock",
              "component_inputs": "synthetic sin(j*0.013) BF16 layer0 stages; actual checkpoint weights; no token prompt",
              "qualified_new_defaults": [], "candidate_trajectory_rejected": True,
              "rejected_first_mismatch": differences[0], "rejected_differing_tokens": len(differences),
              "rejection_replay_timing_qualified": False,
              "rejection_replay_competition": "portable Metal diagnostics overlapped; IDs only",
              "rejection_oracle_negative_control_passed": True,
              "default_current_mtp_median_tok_s": st.median(r["decode_tokens_per_second"] for r in native["runs"]),
              "default_all256_ids_exact": True, "full_model_instrumented_reports_sha256": checks,
              "instrumented_component_reports": instrumented,
              "release_tests_passed": sum(map(int,re.findall(r"test result: ok\. (\d+) passed",quality))),
              "portable_instrumented_tests_passed": sum(map(int,re.findall(r"test result: ok\. (\d+) passed",metal))),
              "cooperative_input_instrumentation": "known numerical counterexample reproduced and selected path falls back; not qualified exact or production-ready"}
    (ROOT / "results/matrix-study-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MATRIX_STUDY_EVIDENCE_VERIFIED")


if __name__ == "__main__":
    main()
