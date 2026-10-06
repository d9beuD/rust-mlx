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
    parser.add_argument("scope", choices=("publication", "gpu", "experts"))
    args = parser.parse_args()
    if args.scope == "publication":
        publication()
    else:
        validate_experiment(args.scope)
