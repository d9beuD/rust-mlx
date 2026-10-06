#!/usr/bin/env python3
"""Compile tiny feasibility probes; never runs kernels or measures throughput."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    common = "#include <metal_stdlib>\nusing namespace metal;\n"
    signature = "kernel void probe(device float *y [[buffer(0)]], uint tid [[thread_position_in_grid]])"
    sources = {
        "simd": common + signature + " { y[tid] = simd_sum(float(tid)); }\n",
        "inline_gpu_asm": common + signature + ' { asm("nop"); y[tid] = float(tid); }\n',
        "mpp_tensor_ops": common
        + "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        + "using namespace mpp::tensor_ops;\n" + signature
        + " { constexpr auto d = matmul2d_descriptor(16,32,16,false,false,false,"
        + "matmul2d_descriptor::mode::multiply_accumulate);"
        + " [[maybe_unused]] matmul2d<d, metal::execution_simdgroup> op; y[tid] = float(tid); }\n",
    }
    report = {
        "kind": "compiler feasibility, no GPU execution or throughput measurement",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "metal_version": subprocess.check_output(["xcrun", "metal", "--version"], text=True).strip(),
        "cpu_capabilities": subprocess.check_output(
            ["sysctl", "machdep.cpu.brand_string", "hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME2"],
            text=True).strip(),
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="rust-mlx-metal-study-") as directory:
        for name, source in sources.items():
            path = pathlib.Path(directory) / f"{name}.metal"
            path.write_text(source)
            command = ["xcrun", "metal", "-std=metal4.0", "-c", str(path), "-o", str(path.with_suffix(".air"))]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            report["cases"][name] = {
                "source": source,
                "command": [value.replace(directory, "<temporary>") for value in command],
                "exit_code": result.returncode,
                "diagnostics": result.stderr.replace(directory, "<temporary>"),
            }
    assert report["cases"]["simd"]["exit_code"] == 0
    assert report["cases"]["mpp_tensor_ops"]["exit_code"] == 0
    assert "illegal asm statement" in report["cases"]["inline_gpu_asm"]["diagnostics"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"SIMD/MPP compile; inline asm rejected. Report: {args.output}")


if __name__ == "__main__":
    main()
