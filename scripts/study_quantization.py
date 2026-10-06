#!/usr/bin/env python3
"""Research-only checkpoint inventory and draft-head requantization probe.

Reads safetensors headers without loading the checkpoint. --probe loads only
the LM head and existing MTP oracle inputs; timings are Python component
measurements, never Rust decode throughput or a model-quality evaluation.
"""

import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import struct
import subprocess
import time


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def header(path):
    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(size))


def category(name):
    if name.startswith("vision_tower."):
        return "vision"
    if ".ngram_embedding.shards." in name:
        return "ple_mmap"
    if name.startswith("mtp."):
        return "mtp"
    if ".switch_mlp." in name or ".experts." in name:
        return "target_routed_experts"
    if ".lm_head." in name or name.startswith("lm_head."):
        return "target_head"
    if ".embed_tokens." in name:
        return "target_embedding"
    return "target_resident"


def inventory(model):
    config = json.loads((model / "config.json").read_text())
    quant = config.get("quantization", config.get("quantization_config"))
    tensors, files = {}, {}
    for shard in sorted(model.glob("model-*.safetensors")):
        for name, spec in header(shard).items():
            if name != "__metadata__":
                assert name not in tensors
                tensors[name] = spec
                files[name] = shard.name
    assert tensors
    sizes = collections.Counter()
    for name, spec in tensors.items():
        begin, end = spec["data_offsets"]
        sizes[category(name)] += end - begin
    formats = collections.defaultdict(lambda: {"modules": 0, "bytes": 0, "logical_weights": 0})
    modules = []
    for name, spec in tensors.items():
        if not name.endswith(".scales"):
            continue
        prefix = name.removesuffix(".scales")
        weight = tensors.get(prefix + ".weight")
        if weight is None:
            continue
        q = quant.get(prefix, quant)
        mode, bits, group = q.get("mode", "affine"), q["bits"], q["group_size"]
        assert mode == "affine", "This header inventory assumes affine group metadata"
        logical = math.prod(spec["shape"]) * group
        byte_count = sum(
            tensors[prefix + suffix]["data_offsets"][1]
            - tensors[prefix + suffix]["data_offsets"][0]
            for suffix in (".weight", ".scales", ".biases")
            if prefix + suffix in tensors
        )
        key = f"{category(name)}/affine{bits}/g{group}/{spec['dtype']}"
        row = formats[key]
        row["modules"] += 1
        row["bytes"] += byte_count
        row["logical_weights"] += logical
        modules.append({"prefix": prefix, "category": category(name), "bits": bits,
                        "group_size": group, "scales_dtype": spec["dtype"],
                        "bytes": byte_count, "logical_weights": logical,
                        "weight_shape": weight["shape"]})
    target_modules = [m for m in modules if m["category"].startswith("target_")]
    experts = [m for m in target_modules if m["category"] == "target_routed_experts"]
    assert experts and all(m["weight_shape"][0] == 512 for m in experts)
    resident = [m for m in target_modules if m["category"] == "target_resident"]
    read_bytes = sum(m["bytes"] * 10 / 512 for m in experts) + sum(m["bytes"] for m in resident)
    head_module = next(m for m in modules if m["category"] == "target_head")
    # Byte bounds assume each selected expert is read once, no cache hits,
    # and all resident projections read once. They are NOT measured traffic.
    affine4 = sum(m["logical_weights"] * (0.5 + 4 / m["group_size"]) for m in resident)
    return {
        "tensor_count": len(tensors), "tensor_payload_bytes": sum(sizes.values()),
        "payload_by_category": dict(sizes), "quantized_formats": dict(sorted(formats.items())),
        "head": head_module,
        "byte_model": {
            "scope": "ideal single target position; excludes state/cache/PLE, dispatch and cache reuse",
            "routed_top10_bytes": sum(m["bytes"] * 10 / 512 for m in experts),
            "quantized_resident_bytes": sum(m["bytes"] for m in resident),
            "target_with_full_head_bytes": read_bytes + head_module["bytes"],
            "resident_affine4_same_groups_bf16_metadata_bytes": affine4,
            "resident_only_requantization_byte_saving": sum(m["bytes"] for m in resident) - affine4,
        },
        "modules": modules,
        "header_sha256": hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(),
    }, files


def probe(model, info, files, oracle_path, repetitions):
    import mlx.core as mx

    prefix = info["head"]["prefix"]
    shard = mx.load(str(model / files[prefix + ".weight"]))
    parts = tuple(shard[prefix + suffix] for suffix in (".weight", ".scales", ".biases"))
    del shard
    mx.eval(*parts)
    oracle = mx.load(str(oracle_path))
    x = mx.concatenate([oracle["prefill_mixed"], oracle["decode_0_mixed"],
                        oracle["decode_1_mixed"]], axis=1).reshape(-1, 2560)
    # Explicit singleton calls preserve the original qmv reduction geometry.
    original_bits, original_group = info["head"]["bits"], info["head"]["group_size"]

    def projection(inputs, weights, mode, bits, group):
        return mx.quantized_matmul(inputs, weights[0], weights[1],
                                   weights[2] if mode == "affine" else None,
                                   transpose=True, bits=bits, group_size=group, mode=mode)

    baseline = mx.concatenate([projection(x[i:i + 1], parts, "affine", original_bits,
                                          original_group) for i in range(x.shape[0])])
    mx.eval(baseline)
    oracle_logits = mx.concatenate([oracle["prefill_logits"], oracle["decode_0_logits"],
                                   oracle["decode_1_logits"]], axis=1).reshape(-1, baseline.shape[-1])
    saved_error = mx.max(mx.abs(baseline[-3:].astype(mx.float32) - oracle_logits.astype(mx.float32))).item()
    assert saved_error == 0, f"Head probe baseline differs from saved MTP oracle: {saved_error}"
    baseline_ids = mx.argmax(baseline, axis=-1).tolist()
    variants = []
    for mode, bits, group in [("affine", 4, 64), ("affine", 6, 64),
                              ("mxfp4", 4, 32), ("nvfp4", 4, 16)]:
        chunks = []
        error_sum = weight_sum = 0.0
        started = time.perf_counter()
        for start in range(0, parts[0].shape[0], 8192):
            end = min(start + 8192, parts[0].shape[0])
            dense = mx.dequantize(*(part[start:end] for part in parts),
                                  bits=original_bits, group_size=original_group)
            packed = mx.quantize(dense, bits=bits, group_size=group, mode=mode)
            reconstructed = mx.dequantize(packed[0], packed[1],
                                         packed[2] if mode == "affine" else None,
                                         bits=bits, group_size=group, mode=mode)
            err = mx.sum(mx.square(dense.astype(mx.float32) - reconstructed.astype(mx.float32)))
            norm = mx.sum(mx.square(dense.astype(mx.float32)))
            mx.eval(*packed, err, norm)
            error_sum += err.item()
            weight_sum += norm.item()
            chunks.append(packed)
        candidate = tuple(mx.concatenate([chunk[i] for chunk in chunks], axis=0)
                          for i in range(len(chunks[0])))
        mx.eval(*candidate)
        conversion_seconds = time.perf_counter() - started
        output = mx.concatenate([projection(x[i:i + 1], candidate, mode, bits, group)
                                 for i in range(x.shape[0])])
        mx.eval(output)
        ids = mx.argmax(output, axis=-1).tolist()
        delta = output.astype(mx.float32) - baseline.astype(mx.float32)
        rmse = mx.sqrt(mx.mean(mx.square(delta))).item()
        max_error = mx.max(mx.abs(delta)).item()
        timing = []
        for rows in [1, 4]:
            inp = mx.contiguous(x[:rows])
            for weights, fmt, b, g in [(parts, "affine", original_bits, original_group),
                                        (candidate, mode, bits, group)]:
                mx.eval(projection(inp, weights, fmt, b, g))
            samples = {"original": [], "candidate": []}
            for repetition in range(repetitions):
                order = ["original", "candidate"] if repetition % 2 == 0 else ["candidate", "original"]
                for arm in order:
                    started = time.perf_counter_ns()
                    y = projection(inp, parts if arm == "original" else candidate,
                                   "affine" if arm == "original" else mode,
                                   original_bits if arm == "original" else bits,
                                   original_group if arm == "original" else group)
                    mx.eval(y)
                    samples[arm].append((time.perf_counter_ns() - started) / 1000)
            timing.append({"rows": rows, "samples_us": samples,
                           "medians_us": {arm: statistics.median(v) for arm, v in samples.items()},
                           "cv": {arm: statistics.stdev(v) / statistics.mean(v) for arm, v in samples.items()}})
        variants.append({"mode": mode, "bits": bits, "group_size": group,
                         "bytes": sum(part.nbytes for part in candidate),
                         "conversion_seconds": conversion_seconds,
                         "weight_relative_rmse": math.sqrt(error_sum / weight_sum),
                         "logit_rmse": rmse, "logit_max_abs_error": max_error,
                         "same_head_argmax": sum(a == b for a, b in zip(ids, baseline_ids)),
                         "candidate_ids": ids, "timing": timing})
        del candidate, chunks, output
        mx.clear_cache()
    return {"mlx_version": mx.__version__, "device": mx.device_info(),
            "oracle": str(oracle_path), "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
            "oracle_logits_exact": True, "baseline_ids": baseline_ids,
            "prompt_ids": oracle["input_tokens"].tolist(), "prompt_length": 10,
            "sampler": "greedy head argmax on saved teacher-forced hidden rows",
            "cache": "none; isolated projection; evaluated resident weights",
            "mtp": "saved native MTP prefill + two decode inputs; no target verifier",
            "warmup": "one evaluated projection per arm and row geometry before alternating pairs",
            "input_rows": x.shape[0], "repetitions": repetitions, "variants": variants,
            "limits": "Twelve teacher-forced MTP hidden rows, one prompt. No rollout, target verification, acceptance-rate, quality-suite or end-to-end speed measurement. Requantizes already quantized BF16-dequantized weights."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--oracle", type=pathlib.Path, default=pathlib.Path("results/mtp-oracle.safetensors"))
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    assert args.repetitions >= 2
    info, files = inventory(args.model)
    report = {"kind": "research-only; header byte model and optional Python component probe",
              "model": str(args.model), "git_commit": command("git", "rev-parse", "HEAD"),
              "macos": command("sw_vers", "-productVersion"),
              "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
              "xcode": command("xcodebuild", "-version"),
              "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "process_cpu_names": command("ps", "-Ao", "pcpu,comm", "-r").splitlines()[:12],
              "inventory": info}
    if args.probe:
        report["draft_head_probe"] = probe(args.model, info, files, args.oracle, args.repetitions)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "categories": info["payload_by_category"],
                      "byte_model": info["byte_model"],
                      "probed": args.probe}, indent=2))


if __name__ == "__main__":
    main()
