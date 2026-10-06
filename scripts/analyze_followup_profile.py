#!/usr/bin/env python3
"""Own-process GPU timeline accounting, without shader/bandwidth invention."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def summarize(path, pid):
    refs = {}
    intervals = []
    labels = {}
    latencies = []
    rows = 0
    own_rows = 0
    columns = None
    for _, node in ET.iterparse(path, events=["end"]):
        if "id" in node.attrib:
            refs[node.attrib["id"]] = (node.get("fmt", node.text or ""), node.text or "")
        if node.tag == "schema":
            columns = [col.findtext("mnemonic") for col in node.findall("col")]
        if node.tag != "row":
            continue
        rows += 1
        values = []
        for item in node:
            values.append(refs[item.attrib["ref"]] if "ref" in item.attrib else
                          (item.get("fmt", item.text or ""), item.text or ""))
        row = dict(zip(columns, values))
        if str(pid) not in row["process"][0].split("(")[-1].split(")")[0].split():
            node.clear()
            continue
        own_rows += 1
        if row["channel-name"][0] == "Compute" and row["state"][0] == "Active" and row["event-depth"][1] == "0":
            start = int(row["start"][1]) / 1e9
            duration = int(row["duration"][1]) / 1e9
            intervals.append((start, start + duration))
            # Encoder addresses are identifiers, not different shader classes.
            label = row["event-label"][0].split(" (")[0]
            labels[label] = labels.get(label, 0) + 1
            if row["start-latency"][1].isdigit():
                latencies.append(int(row["start-latency"][1]) / 1e9)
        node.clear()
    assert intervals, f"no attributed own Compute/Active/depth0 rows for{pid}"
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    span = merged[-1][1] - merged[0][0]
    union = sum(end - start for start, end in merged)
    return {"exported_rows": rows, "own_rows": own_rows, "own_compute_intervals": len(intervals),
            "own_compute_union_seconds": union, "own_compute_sum_seconds": sum(e - s for s, e in intervals),
            "own_observed_span_seconds": span, "own_compute_union_fraction": union / span,
            "gaps_between_own_compute": {"count": len(gaps), "sum_seconds": sum(gaps),
                                         "median_seconds": statistics.median(gaps) if gaps else 0,
                                         "maximum_seconds": max(gaps, default=0)},
            "submission_latency_median_seconds": statistics.median(latencies) if latencies else None,
            "label_counts": labels, "shader_function_attribution_available": False,
            "dram_bytes_or_bandwidth_measured": False,
            "warning": "Trace instrumentation changes scheduling. Gaps include CPU work, waits and request boundaries; they are not proved removable overhead or raw throughput percentages."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    assert receipt["profiler_exit"] == receipt["inference_exit"] == 0
    assert receipt["all_sixteen_256_token_trajectories_exact"]
    report = {"receipt": str(args.receipt), "source_sha256": receipt["source_sha256"],
              "pid": receipt["pid"], "model": receipt["model"], "instrumented": True,
              "throughput_qualified": False, "requested_seconds": receipt["requested_profile_seconds"],
              "gpu_export_sha256": hashlib.sha256(args.gpu_export.read_bytes()).hexdigest(),
              **summarize(args.gpu_export, receipt["pid"])}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FOLLOWUP_GPU_TIMELINE_ACCOUNTED")


if __name__ == "__main__":
    main()
