#!/usr/bin/env python3
"""Verify publication and measured exploration outcomes from authoritative artifacts."""
import argparse
import copy
import json
import statistics
import subprocess
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def validate_ab(data, kind):
    reference = load("results/target-baseline-256.json")["runs"][0]["tokens"]
    assert data["runtime"]["sampler"] == "greedy"
    assert data["runtime"]["batch_size"] == 1
    assert data["runtime"]["prefix_cache"] is False
    assert data["runtime"]["warmup_tokens"] == 256
    assert len(data["runs"]) == 8
    gains = []
    for cycle in range(1, 5):
        rows = [r for r in data["runs"] if r["run"] == cycle]
        assert len(rows) == 2
        native = next(r for r in rows if not r["candidate_enabled"])
        candidate = next(r for r in rows if r["candidate_enabled"])
        for key in ("tokens", "draft_tokens", "acceptance", "draft_lengths"):
            assert native["generation"][key] == candidate["generation"][key], key
        assert len(native["generation"]["tokens"]) == 256
        assert native["generation"]["tokens"] == reference
        if kind == "gpu":
            assert native["gpu_draft"] is False and candidate["gpu_draft"] is True
        else:
            assert native["sorted_moe_calls"] == 0
            assert candidate["sorted_moe_calls"] == 48 * len(candidate["generation"]["acceptance"])
        gains.append(100 * (candidate["decode_tokens_per_second"] / native["decode_tokens_per_second"] - 1))
    return statistics.median(gains)


def validate_experiment(kind):
    stem = "gpu-draft" if kind == "gpu" else "sorted-moe"
    data = load(f"results/{stem}-raw-ab-256.json")
    measured = validate_ab(data, kind)
    saved = load(f"results/{stem}-statistics.json")
    assert abs(measured - saved["median_paired_gain_percent"]) < 1e-8
    assert saved["default_promoted"] is False
    assert measured < 5
    # Positive control: the identical oracle must reject a corrupted output trajectory.
    bad = copy.deepcopy(data)
    bad["runs"][0]["generation"]["tokens"][0] ^= 1
    try:
        validate_ab(bad, kind)
    except AssertionError:
        pass
    else:
        raise AssertionError("trajectory oracle failed its negative control")
    if kind == "experts":
        profile = load("results/verify-profile-round2.json")
        assert profile["logits_exact"] is True and profile["rounds"] == 12
        assert len(profile["expert_overlap"]) == 48 * 12
        for r in load("results/sorted-moe-components.json")["benchmarks"]:
            assert r["exact"] is True and r["dtype"] == "BF16"
            assert len(r["native_seconds"]) == len(r["sorted_seconds"]) == 100
        verifier = load("results/sorted-moe-verify-metal.json")
        assert [r["depth"] for r in verifier] == list(range(2, 9)) * 2
        assert all(r["logit_error"] == r["hidden_error"] == r["state_error"] == 0 for r in verifier)
        rollback = load("results/sorted-moe-rollback-metal.json")
        assert [r["keep"] for r in rollback] == list(range(5))
        assert all(r["logit_error"] == r["state_error"] == 0 for r in rollback)
    print(f"ROUND2_{kind.upper()}_EXPLORATION_VERIFIED paired_gain={measured:.6f}%")


def measured_direction(kind):
    from analyze_round2 import analyze
    reference = load("results/target-baseline-256.json")["runs"][0]["tokens"]
    if kind == "adaptive":
        for stem in ("adaptive-depth", "adaptive-vocab", "adaptive"):
            for chat in (False, True):
                name = f"results/{stem}-{'chat-ab' if chat else 'remat-raw-ab'}-256.json"
                actual = analyze(name, chat=chat)
                saved = load(f"results/{stem}-{'chat' if chat else 'remat'}-statistics.json")
                assert actual == saved
                if not chat:
                    assert all(r["generation"]["tokens"] == reference for r in load(name)["runs"])
        combined = load("results/adaptive-qmv-remat-metal.json")
        candidates = [r["generation"] for r in combined["runs"] if r["candidate_enabled"]]
        assert len(candidates) == 2
        assert all(2 in g["draft_lengths"] and 3 in g["draft_lengths"] and 248320 in g["draft_vocab_sizes"] for g in candidates)
        assert all(r["generation"]["tokens"] == reference for r in combined["runs"])
    elif kind == "head":
        actual = analyze("results/greedy-head-raw-ab-256.json", same_drafts=True)
        assert actual == load("results/greedy-head-statistics.json")
        assert all(r["generation"]["tokens"] == reference for r in load("results/greedy-head-raw-ab-256.json")["runs"])
        for row in load("results/greedy-head-components.json")["benchmarks"]:
            assert row["exact"] and row["dtype"] == "BF16"
            assert len(row["native_seconds"]) == len(row["candidate_seconds"]) == 100
        for row in load("results/greedy-head-target-metal.json")["runs"]:
            assert row["generation"]["tokens"] == reference
            assert (row["greedy_head_calls"] > 0) == row["candidate_enabled"]
    elif kind == "kv":
        actual = analyze("results/kv-blocks-context-ab-256.json", chat=True, same_drafts=True)
        assert actual == load("results/kv-blocks-context-statistics.json")
        assert [r["prompt_length"] for r in actual["cohorts"]] == [10,2107,4117]
        for name in ("kv-blocks-target-metal", "kv-blocks-long-target-metal"):
            report = load(f"results/{name}.json")
            assert len(report["decode"]) == 16
            assert [r["depth"] for r in report["verifier"]] == list(range(2,9))
            assert all(r["logit_error"] == r["hidden_error"] == r["state_error"] == 0 for r in report["decode"]+report["verifier"])
            assert all(r["rollback_prefixes"] == r["depth"]+1 for r in report["verifier"])
        for row in load("results/kv-blocks-components.json")["benchmarks"]:
            assert row["exact"] and len(row["native_seconds"]) == len(row["candidate_seconds"]) == 100
    # The same pair oracle must reject a corrupted candidate, including chat cohorts.
    import tempfile
    report_path = {"adaptive": "results/adaptive-remat-raw-ab-256.json",
                   "head": "results/greedy-head-raw-ab-256.json",
                   "kv": "results/kv-blocks-context-ab-256.json"}[kind]
    bad = load(report_path)
    rows = bad["records"] if kind == "kv" else bad["runs"]
    key = "candidate" if kind == "kv" else "candidate_enabled"
    next(r for r in rows if r[key])["generation"]["tokens"][0] ^= 1
    with tempfile.TemporaryDirectory(prefix="rust-mlx-pair-oracle-") as directory:
        path = Path(directory)/"corrupt.json"
        path.write_text(json.dumps(bad))
        try:
            analyze(path, chat=kind == "kv")
        except AssertionError:
            pass
        else:
            raise AssertionError("pair oracle failed its negative control")
    print(f"ROUND2_{kind.upper()}_EXPLORATION_VERIFIED")


def final_integration():
    import hashlib
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    paths = [p for root in ("src", "tests", "kernels") for p in Path(root).rglob("*") if p.is_file()]
    paths += [Path("Cargo.toml"), Path("Cargo.lock")]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode()+b"\0"+path.read_bytes()+b"\0")
    report = load("results/round2-final-validation.json")
    assert report["code_sha256"] == digest.hexdigest()
    assert report["remaining_experiments"] == [] and report["default_promotions"] == []
    for field, count, marker in (("quality",25,"QUALITY_CHECKS_PASSED"),("portable_metal",24,"METAL_VALIDATION_PASSED")):
        row = report[field]
        assert row["passed"] and row["exit_code"] == 0 and row["code_sha256"] == digest.hexdigest()
        assert row.get("release_tests_passed",row.get("validation_tests_passed")) == count
        assert sha(row["log"]) == row["sha256"] and marker in Path(row["log"]).read_text()
    for row in report["actual_target_cases"]:
        assert row["passed"] and sha(row["log"]) == row["sha256"]
        assert row["success_marker"] in Path(row["log"]).read_text()
    server = report["server"]
    assert server["passed"] and server["code_sha256"] == digest.hexdigest()
    assert server["concurrent_cancellation_survivors"] == 3 and server["partial_codepoint_cases"] == [2,2]
    for name, row in server["reports"].items():
        assert sha(f"results/{name}.json") == row["sha256"]
    for binary, value in report["binary_sha256"].items():
        assert sha(f"target/release/{binary}") == value
    from analyze_round2 import analyze
    assert analyze("results/round2-final-default-raw-256.json",chat=True) == load("results/round2-final-default-statistics.json")
    reference = load("results/target-baseline-256.json")["runs"][0]["tokens"]
    for row in load("results/round2-final-default-raw-256.json")["records"]:
        assert row["generation"]["tokens"] == reference
        assert not any(row[key] for key in ("adaptive_depth","adaptive_vocab","greedy_head","kv_blocks"))
        assert row["qmv"] and row["hc_projection"]
    # A matching old HEAD is insufficient if qualified changes are still local.
    subprocess.check_call(["git","diff","--quiet","HEAD","--",".",":(exclude).agents"])
    unpublished = subprocess.check_output(["git","ls-files","--others","--exclude-standard"],text=True).splitlines()
    assert not [p for p in unpublished if not p.startswith(".agents/")], "unpublished owned artifacts"
    local = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    remote = subprocess.check_output(["gh","api","repos/d9beuD/rust-mlx/commits/main","--jq",".sha"],text=True).strip()
    assert local == remote, "qualified current commit is not public main"
    publication()
    print("ROUND2_FINAL_INTEGRATION_PUBLICATION_VERIFIED")


def publication():
    repository = json.loads(subprocess.check_output([
        "gh", "repo", "view", "d9beuD/rust-mlx", "--json", "nameWithOwner,isPrivate,url"
    ]))
    assert repository["nameWithOwner"] == "d9beuD/rust-mlx"
    assert repository["isPrivate"] is False
    initial = "32e74c1fb6240d19efa18259100fb8ebd56cafa6"
    remote = json.loads(subprocess.check_output(["gh", "api", f"repos/d9beuD/rust-mlx/commits/{initial}"]))
    assert remote["sha"] == initial
    print("ROUND2_PUBLICATION_VERIFIED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("publication", "gpu", "experts", "adaptive", "head", "kv", "final"))
    args = parser.parse_args()
    if args.scope == "final":
        final_integration()
    elif args.scope == "publication":
        publication()
    elif args.scope in ("adaptive", "head", "kv"):
        measured_direction(args.scope)
    else:
        validate_experiment(args.scope)
