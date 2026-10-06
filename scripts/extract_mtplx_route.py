#!/usr/bin/env python3
"""Extract reviewed literal MTPLX routing tail without executing upstream code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Path("research/upstream/MTPLX")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    source_path = root / "mtplx/kernels/qwen4_m4_route.py"
    source = source_path.read_text()
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                values[node.targets[0].id] = node.value.value
    args.destination.mkdir(parents=True, exist_ok=True)
    credit = (f"// Adapted from MTPLX by Youssof Altoukhi, Apache-2.0, revision{revision}.\n"
              "// mtplx/kernels/qwen4_m4_route.py; see third-party/MTPLX-{APACHE-2.0,NOTICE}.\n"
              "// Modification: native router/shared projections remain separate; tail accepts1–8 independent rows.\n")
    outputs = {}
    for key, name in [("_TAIL_HEADER", "moe_route_tail.h"), ("_TAIL_SOURCE", "moe_route_tail.metal")]:
        text = credit + values[key]
        (args.destination / name).write_text(text)
        outputs[name] = hashlib.sha256(text.encode()).hexdigest()
    (args.destination / "moe_route_provenance.json").write_text(json.dumps({
        "repository": "https://github.com/youssofal/MTPLX", "commit": revision,
        "source": "mtplx/kernels/qwen4_m4_route.py", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "license": "Apache-2.0", "notice": "third-party/MTPLX-NOTICE", "outputs": outputs,
        "scope": "routing tail only; original router projections, affine formats and weight ownership unchanged"}, indent=2) + "\n")
    print("MTPLX_ROUTE_LITERALS_EXTRACTED")


if __name__ == "__main__":
    main()
