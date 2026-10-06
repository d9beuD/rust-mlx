#!/usr/bin/env python3
"""Recompute completed cohorts. Partial progress never marks the scope complete."""
import hashlib
import argparse
import json
import math
import re
from pathlib import Path
import statistics
from run_decode3_checks import core_digest, VALIDATION

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_log(receipt, path=None):
    path = path or ROOT/receipt["log"]
    assert receipt["passed"] and receipt["exit_code"] == 0
    assert digest(path) == (receipt.get("sha256") or receipt["log_sha256"])
    text = path.read_text()
    assert receipt["success_marker"] in text
    return text


def draft_rows(report):
    runtime = report["runtime"]
    assert report["complete"] and runtime["runs"] == 4
    assert runtime["max_tokens"] == runtime["warmup_tokens"] == 256
    assert runtime["batch_size"] == 1 and runtime["mtp"] and runtime["sampler"] == "greedy"
    assert not runtime["prefix_cache"] and runtime["ignore_eos"]
    for path, value in report["source_file_sha256"].items():
        assert digest(ROOT / path) == value, f"changed draft source {path}"
    oracles = {o["prompt"]: o["tokens"] for o in report["oracles"]}
    assert set(oracles) == set(range(4 if runtime["chat"] else 1))
    pairs = {}
    for r in report["records"]:
        g = r["generation"]
        assert r["exact_target_ids"] and g["tokens"] == oracles[r["prompt"]] and len(g["tokens"]) == 256
        assert g["decode_seconds"] > 0 and math.isclose(r["tokens_per_second"], 255 / g["decode_seconds"], rel_tol=1e-12)
        assert sum(g["acceptance"]) <= sum(g["draft_lengths"])
        p = pairs.setdefault((r["prompt"], r["cycle"]), {})
        assert r["candidate"] not in p
        p[r["candidate"]] = r
    assert len(pairs) == 4 * len(oracles)
    result = []
    for prompt in oracles:
        native, candidate, gains, acceptance = [], [], [], []
        for cycle in range(1, 5):
            pair = pairs[prompt, cycle]
            assert set(pair) == {False, True}
            b, c = pair[False], pair[True]
            assert b["prompt_ids"] == c["prompt_ids"]
            native.append(b["tokens_per_second"])
            candidate.append(c["tokens_per_second"])
            gains.append((candidate[-1] / native[-1] - 1) * 100)
            acceptance.append({str(k): [sum(v["generation"]["acceptance"]), sum(v["generation"]["draft_lengths"])] for k,v in pair.items()})
        result.append({"prompt": prompt, "native_samples": native, "candidate_samples": candidate,
                       "native_median": statistics.median(native), "candidate_median": statistics.median(candidate),
                       "paired_gain_samples_percent": gains, "paired_gain_median_percent": statistics.median(gains),
                       "paired_gain_range_percent": [min(gains), max(gains)],
                       "native_cv": statistics.stdev(native)/statistics.mean(native),
                       "candidate_cv": statistics.stdev(candidate)/statistics.mean(candidate),
                       "acceptance": acceptance, "exact_original_target_ids": True})
    return result


def config_rows(report, oracle_paths):
    assert report["complete"] and report["runs"] == 4
    assert report["max_tokens"] == report["warmup_tokens"] == 256
    assert report["batch_size"] == 1 and report["mtp"] and report["sampler"] == "greedy"
    assert not report["prefix_cache"]
    pairs = {}
    costs = {}
    for record in report["records"]:
        path = ROOT / record["report"]
        assert digest(path) == record["report_sha256"]
        r = json.loads(path.read_text())
        command = json.loads(path.with_name(path.stem + "-command.json").read_text())
        assert command["exit_code"] == 0
        assert command["binary_sha256"] == r["environment"]["workload"]["binary_sha256"].split()[0]
        assert r["runtime"]["warmup_tokens"] == 256 and r["runtime"]["ignore_eos"]
        assert r["runtime"]["batch_size"] == 1 and r["runtime"]["sampler"] == "greedy"
        assert r["runtime"]["mtp"] and not r["runtime"]["prefix_cache"]
        assert len(r["runs"]) == 1
        g = r["runs"][0]["generation"]
        assert len(g["tokens"]) == 256 and g["decode_seconds"] > 0
        assert math.isclose(record["rate"], 255/g["decode_seconds"], rel_tol=1e-12)
        oracle_path = oracle_paths[record["oracle_sha256"]]
        oracle = json.loads(oracle_path.read_text())
        assert len(oracle["runs"]) == 1 and not oracle["runtime"]["mtp"]
        assert oracle["prompt_ids"] == r["prompt_ids"]
        assert oracle["runs"][0]["tokens"] == g["tokens"], "native identity mismatch"
        assert bool(r["resident_variant"]) == (record["candidate"] and report["approximate_target"])
        assert oracle["resident_variant"] == r["resident_variant"]
        assert record["own_native_oracle_exact"]
        assert record["exact_original_distribution"] == (not (record["candidate"] and report["approximate_target"]))
        if record["candidate"]:
            assert bool(r["expert_layout"]) == ("--native-gate-up" in report["candidate_flags"])
            assert bool(r["fixed_draft_head"]) == ("--draft-head-bits" in report["candidate_flags"])
        else:
            assert r["expert_layout"] is None and r["fixed_draft_head"] is None
        pair = pairs.setdefault((record["prompt"], record["cycle"]), {})
        assert record["candidate"] not in pair
        pair[record["candidate"]] = record["rate"]
        costs[record["prompt"], record["cycle"], record["candidate"]] = {
            "cold_complete_process_seconds": command["ended_unix"]-command["started_unix"],
            "head_preparation_seconds": (r["fixed_draft_head"] or {}).get("preparation_seconds", 0),
            "expert_layout_preparation_seconds": (r["expert_layout"] or {}).get("preparation_seconds", 0),
            "peak_memory_bytes": r["runs"][0]["peak_memory_bytes"]}
    prompts = sorted({p for p, _ in pairs})
    assert prompts == list(range(4 if report["suite"] == "chat" else 1))
    assert len(pairs) == 4 * len(prompts)
    result = []
    for p in prompts:
        native = [pairs[p,c][False] for c in range(1,5)]
        candidate = [pairs[p,c][True] for c in range(1,5)]
        gains = [(c/b-1)*100 for b,c in zip(native,candidate)]
        saved = next(s for s in report["statistics"] if s["prompt"] == p)
        assert saved["paired_gains_percent"] == gains and saved["paired_median_percent"] == statistics.median(gains)
        result.append({"prompt": p, "native_samples": native, "candidate_samples": candidate,
                       "native_median": statistics.median(native), "candidate_median": statistics.median(candidate),
                       "paired_gain_samples_percent": gains, "paired_gain_median_percent": statistics.median(gains),
                       "paired_gain_range_percent": [min(gains),max(gains)],
                       "native_cv": statistics.stdev(native)/statistics.mean(native),
                       "candidate_cv": statistics.stdev(candidate)/statistics.mean(candidate),
                       "cost_samples": {str(arm): [costs[p,c,arm] for c in range(1,5)] for arm in (False,True)},
                       "cold_wall_scope": "process launch through exit, including model init, full warmup, measured prefill/decode and serialization; model-load-only time is not separated"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="Require all three experiments and current-source qualification.")
    args = parser.parse_args()
    # Prove the identity oracle is not merely copying the recorded boolean.
    sample_paths = sorted((ROOT/"results").glob("decode3-final-draft-*-raw-256.json"))
    drafts = {}
    for path in sorted((ROOT/"results").glob("decode3-final-draft-*-*-256.json")):
        if path.name.endswith("-command.json"):
            continue
        report = json.loads(path.read_text())
        if not report["complete"]:
            continue
        drafts[path.name] = {"sha256": digest(path), "head": report["draft_head"], "statistics": draft_rows(report)}
    if sample_paths:
        fake = json.loads(sample_paths[0].read_text())
        if fake["complete"]:
            fake["records"][0]["generation"]["tokens"][0] += 1
            try:
                draft_rows(fake)
            except AssertionError:
                pass
            else:
                raise AssertionError("target-identity negative control was accepted")
    original = json.loads((ROOT/"results/decode3-quality-original.json").read_text())
    quality = {}
    profiles = [(str(b), f"q{b}-resident", f"q{b}-resident") for b in (4,5,6)]
    if (ROOT / "results/decode3-quality-mixed32.json").exists():
        profiles.append(("mixed32", "mixed32", "mixed32"))
    for name, quality_name, receipt_name in profiles:
        path = ROOT/f"results/decode3-quality-{quality_name}.json"
        r = json.loads(path.read_text())
        assert r["complete"] and r["identity"] == original["identity"] and len(r["records"]) == 28
        predictions = 0
        base_nll = variant_nll = kl = agreement = 0.
        greedy = []
        for b,c in zip(original["records"],r["records"]):
            assert b["document_sha256"] == c["document_sha256"] and b["predictions"] == c["predictions"]
            n = c["predictions"]
            predictions += n
            base_nll += b["nll"]*n
            variant_nll += c["nll"]*n
            kl += c["metrics"]["kl_original_to_variant"]*n
            agreement += c["metrics"]["argmax_agreement"]*n
            if c["continuation"]:
                same = c["continuation"]["tokens"] == b["continuation"]["tokens"]
                assert same == c["continuation"]["comparison"]["identical"]
                greedy.append(same)
        delta = (variant_nll-base_nll)/predictions
        receipt = json.loads((ROOT/f"results/decode3-resident-{receipt_name}.json").read_text())
        quality[name] = {"sha256":digest(path),"predictions":predictions,"original_nll":base_nll/predictions,
                             "variant_nll":variant_nll/predictions,"nll_delta":delta,"perplexity_relative_change":math.expm1(delta),
                             "kl_original_to_variant":kl/predictions,"argmax_agreement":agreement/predictions,
                             "identical_greedy_continuations":sum(greedy),"greedy_continuations":len(greedy),
                             "saved_weight_bytes":sum(v["source_bytes"]-v["candidate_bytes"] for v in receipt["receipts"].values()),
                             "approximate_target":True,"broad_quality_equivalence_established":False}
    oracle_paths = {}
    for directory in (ROOT/".unlazy/decode3-config-oracles", ROOT/"results"):
        pattern = "*.json" if directory.name == "decode3-config-oracles" else "decode3-native-oracle-*.json"
        for path in directory.glob(pattern):
            if not path.name.endswith(".binding.json"):
                oracle_paths[digest(path)] = path
    configurations = {}
    for path in sorted((ROOT/"results").glob("decode3-config-*-*-256.json")):
        report = json.loads(path.read_text())
        if report["complete"]:
            statistics_rows = config_rows(report, oracle_paths)
            oracle_reports = []
            for value in sorted({r["oracle_sha256"] for r in report["records"]}):
                public = ROOT / f"results/decode3-native-oracle-{value}.json"
                if not public.exists():
                    public.write_bytes(oracle_paths[value].read_bytes())
                assert digest(public) == value
                oracle_reports.append(str(public.relative_to(ROOT)))
            configurations[path.name] = {"sha256": digest(path), "approximate_target": report["approximate_target"],
                                         "statistics": statistics_rows, "native_oracles": oracle_reports}
    approximate_paths = [p for p in (ROOT/"results").glob("decode3-config-*-raw-256.json")
                         if json.loads(p.read_text())["approximate_target"]]
    if approximate_paths:
        fake = json.loads(approximate_paths[0].read_text())
        fake["approximate_target"] = False
        for r in fake["records"]:
            r["exact_original_distribution"] = True
        try:
            config_rows(fake, oracle_paths)
        except AssertionError:
            pass
        else:
            raise AssertionError("approximate target mislabeled as original was accepted")
    qualification = {}
    if args.final:
        assert len(drafts) == 20 and len(quality) == 4
        assert all(q["predictions"] == 6239 and q["greedy_continuations"] == 7 for q in quality.values())
        required = {f"decode3-config-{case}-{suite}-256.json" for case in ("combo-full", "combo-mixed32") for suite in ("raw", "chat")}
        required.add("decode3-config-mixed32-raw-256.json")
        assert required <= configurations.keys(), "full configurations incomplete"
        assert all(json.loads((ROOT/"results"/name).read_text())["source_sha256"] == core_digest() for name in required)
        study = json.loads((ROOT/"results/decode3-resident-sensitivity.json").read_text())
        selected = json.loads((ROOT/"results/decode3-resident-mixed32.json").read_text())
        assert study["complete"] and len(study["records"]) == 245
        assert len({r["projection"] for r in study["records"]}) == 245
        assert all(len(r["documents"]) == 7 and r["predictions"] == 441 for r in study["records"])
        assert selected["selection"]["keep_original_projections"] == 32 and len(selected["quantization"]) == 213
        assert selected["selection"]["sensitivity_sha256"] == digest(ROOT/"results/decode3-resident-sensitivity.json")
        for name in ("quality-current.json", "metal-validation.json", "server-qualification.json"):
            path = ROOT/"results"/name
            data = json.loads(path.read_text())
            assert data["passed"] and data["code_sha256"] == core_digest()
            if name != "server-qualification.json":
                log = verified_log(data)
                tests = sum(map(int,re.findall(r"test result: ok\. (\d+) passed",log)))
                assert tests == data["release_tests_passed" if name == "quality-current.json" else "validation_tests_passed"]
                assert tests == (34 if name == "quality-current.json" else 33)
                assert data["runner_sha256"] == digest(ROOT/"scripts/run_decode3_qualification.py")
            qualification[name] = digest(path)
        quality_binary = digest(ROOT/"target/release/resident-evaluate")
        for name in ("original", "q4-resident", "q5-resident", "q6-resident", "mixed32"):
            q = json.loads((ROOT/f"results/decode3-quality-{name}.json").read_text())
            assert q["environment"]["workload"]["binary_sha256"].split()[0] == quality_binary
        checks = sorted([* (ROOT/"results").glob("decode3-current-*-verify-parity-metal-command.json"),
                         * (ROOT/"results").glob("decode3-current-*-rollback-parity-metal-command.json")])
        assert len(checks) == 14
        for path in checks:
            c = json.loads(path.read_text())
            assert c["passed"] and c["exit_code"] == 0 and c["source_sha256"] == core_digest()
            assert c["validation_environment"] == VALIDATION and c["timings_are_diagnostic"]
            assert c["binary_sha256"] == digest(ROOT/c["command"][0])
            verified_log(c, path.with_name(path.name.replace("-command.json", ".log")))
            out = ROOT/"results"/path.name.replace("-command.json", ".json")
            assert digest(out) == c["output_sha256"]
            rows = json.loads(out.read_text())
            assert len(rows) == (14 if "verify-parity" in path.name else 5)
            assert all(r["logit_error"] == 0 and r["state_error"] == 0 for r in rows)
            if "verify-parity" in path.name:
                assert all(r["hidden_error"] == 0 for r in rows)
            if "layout" in path.name or "combo-mixed32" in path.name:
                assert all(r["layout_calls"] > 0 for r in rows)
            qualification[path.name] = digest(path)
        for name in ("full", "code64"):
            path = ROOT/f"results/decode3-current-draft-{name}-metal-command.json"
            c = json.loads(path.read_text())
            assert c["passed"] and c["exit_code"] == 0 and c["code_sha256"] == core_digest()
            assert c["validation_environment"] == VALIDATION
            verified_log(c)
            d = json.loads((ROOT/f"results/decode3-current-draft-{name}-metal.json").read_text())
            oracles = {o["prompt"]:o["tokens"] for o in d["oracles"]}
            assert d["complete"] and all(len(r["generation"]["tokens"]) == 64 and r["generation"]["tokens"] == oracles[r["prompt"]] for r in d["records"])
            qualification[path.name] = digest(path)
        server = json.loads((ROOT/"results/server-qualification.json").read_text())
        assert server["binary_sha256"] == digest(ROOT/"target/release/server")
        assert server["owned_servers_stopped"] and server["concurrent_cancellation_survivors"] == 3
        for name, value in server["reports"].items():
            assert digest(ROOT/f"results/{name}.json") == value["sha256"]
        for name, value in server["scripts"].items():
            assert digest(ROOT/f"scripts/{name}.py") == value
        for item in server["success_logs"]:
            assert item["passed"] and item["exit_code"] == 0
            assert digest(ROOT/item["log"]) == item["sha256"]
            assert item["success_marker"] in (ROOT/item["log"]).read_text()
        # An exact combination that passes the predeclared global gate requires
        # an explicit implementation/default decision before closing this study.
        raw = configurations["decode3-config-combo-full-raw-256.json"]["statistics"][0]
        chat = configurations["decode3-config-combo-full-chat-256.json"]["statistics"]
        eligible = (raw["paired_gain_median_percent"] >= 5 and
                    min(raw["paired_gain_samples_percent"]) > 0 and
                    all(c["paired_gain_median_percent"] >= -2 for c in chat))
        assert not eligible, "exact combined candidate passes speed gate: resolve default selection before final audit"
    result = {"draft_cohorts":drafts,"resident_quality":quality,"complete_configurations":configurations,
              "source_sha256":core_digest(),"analysis_script_sha256":digest(Path(__file__)),
              "scope_complete":args.final,"qualification":qualification,
              "default_promoted":False,"solo100_demonstrated":False,
              "limitation":"No broad target-quality equivalence or general speedup is inferred from a small held-out corpus. Publication is verified separately."}
    (ROOT/f"results/decode3-{'final' if args.final else 'progress'}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("DECODE3_FINAL_EVIDENCE_VERIFIED" if args.final else "DECODE3_PROGRESS_RECOMPUTED")


if __name__ == "__main__":
    main()
