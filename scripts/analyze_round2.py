#!/usr/bin/env python3
"""Describe every measured pair; never fold instrumentation into throughput."""
import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    mean = statistics.mean(values)
    return {"samples": values, "count": len(values), "mean": mean,
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "stddev": statistics.pstdev(values), "cv_percent": 100 * statistics.pstdev(values) / mean if mean else None,
            "p50": statistics.median(values), "p95": ordered[min(len(values)-1, int(.95*len(values)))]}


def analyze(path, chat=False, same_drafts=False):
    d = json.loads(Path(path).read_text())
    runtime = d["runtime"]
    assert runtime["prefix_cache"] is False and runtime["warmup_tokens"] == 256
    assert runtime.get("batch_size", runtime.get("batch")) == 1
    rows = d["records"] if chat else d["runs"]
    flag, cycle, rate = ("candidate", "cycle", "tokens_per_second") if chat else ("candidate_enabled", "run", "decode_tokens_per_second")
    cohorts = {}
    for r in rows:
        key = tuple(r["prompt_ids"]) if chat else tuple(d["prompt_ids"])
        cohorts.setdefault(key, []).append(r)
    summary = []
    for prompt, cohort in cohorts.items():
        assert len(cohort) == 8, "requires four complete alternating pairs"
        native, candidate, gains, depths, vocabulary = [], [], [], {}, {}
        oracle = next(r for r in cohort if not r[flag])["generation"]["tokens"]
        assert len(oracle) == 256
        for i in range(1, 5):
            pair = [r for r in cohort if r[cycle] == i]
            assert len(pair) == 2 and [r[flag] for r in pair] == ([True, False] if i % 2 else [False, True])
            n = next(r for r in pair if not r[flag]); c = next(r for r in pair if r[flag])
            assert n["generation"]["tokens"] == c["generation"]["tokens"] == oracle
            if same_drafts:
                for field in ("draft_tokens", "acceptance", "draft_lengths"):
                    assert n["generation"][field] == c["generation"][field], field
            native.append(n[rate]); candidate.append(c[rate]); gains.append(100*(c[rate]/n[rate]-1))
            for v in c["generation"]["draft_lengths"]: depths[str(v)] = depths.get(str(v), 0)+1
            for v in c["generation"].get("draft_vocab_sizes", []): vocabulary[str(v)] = vocabulary.get(str(v), 0)+1
        summary.append({"prompt_ids": list(prompt), "prompt_length": len(prompt), "native_tok_s": stats(native),
                        "candidate_tok_s": stats(candidate), "paired_gain_percent": stats(gains),
                        "candidate_depth_histogram": depths, "candidate_vocabulary_histogram": vocabulary,
                        "target_ids_exact": True, "drafts_exact": same_drafts})
    return {"source_report": str(path), "instrumentation": False, "cohorts": summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("report"); p.add_argument("output"); p.add_argument("--chat", action="store_true"); p.add_argument("--same-drafts", action="store_true"); a = p.parse_args()
    r = analyze(a.report, a.chat, a.same_drafts)
    Path(a.output).write_text(json.dumps(r, indent=2)+"\n")
    for c in r["cohorts"]:
        print(f"prompt={c['prompt_length']} paired_gain={c['paired_gain_percent']['median']:.6f}% native={c['native_tok_s']['median']:.3f} candidate={c['candidate_tok_s']['median']:.3f}")
