"""Strict research comparison of pinned oMLX's vendored text backend.

Uses the checkpoint's actual 128-shard format through the independent row
adapter. This is not a comparison of oMLX's HTTP scheduler or MTP serving.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import target_loader
from native_bench import generate, command


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--oracle', default='results/target-oracle-mlx32.2.json')
    p.add_argument('--expected', default='results/target-baseline-256.json')
    p.add_argument('--output', default='results/omlx-adapted-baseline.json')
    p.add_argument('--runs', type=int, default=3)
    a = p.parse_args()
    upstream = Path('research/upstream/omlx').resolve()
    vendor = upstream / 'omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp'
    sys.path.insert(0, str(upstream))
    # Avoid loading the unrelated multimodal processor and its optional deps.
    for name in list(sys.modules):
        if name == 'mlx_vlm.models.qwen4_exp' or name.startswith('mlx_vlm.models.qwen4_exp.'):
            del sys.modules[name]
    package = types.ModuleType('mlx_vlm.models.qwen4_exp')
    package.__path__ = [str(vendor)]
    sys.modules[package.__name__] = package
    target_loader.TextConfig = importlib.import_module(package.__name__ + '.config').TextConfig
    target_loader.LanguageModel = importlib.import_module(package.__name__ + '.language').LanguageModel
    report = dict(model=a.model, comparator='pinned oMLX vendored text backend, adapted plain greedy',
                  commit=command('git', '-C', str(upstream), 'rev-parse', 'HEAD'),
                  source_sha256=hashlib.sha256((vendor / 'language.py').read_bytes()).hexdigest(),
                  adapter_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/target_loader.py'),Path('scripts/mapped_embedding.py')]},
                  environment=dict(chip=command('sysctl','-n','machdep.cpu.brand_string'),
                                   macos_build=command('sw_vers','-buildVersion'), mlx=mx.__version__,
                                   thermal=command('pmset','-g','therm'), swap=command('sysctl','vm.swapusage'),
                                   captured_unix_seconds=time.time()),
                  adapters=['same strict official-checkpoint loader and read-only mmap PLE row adapter',
                            'text-only package import; optional vision processor excluded'],
                  excluded=['oMLX HTTP scheduler', 'oMLX MTP backend', 'native PLE resident/SSD management'],
                  runtime=dict(omlx_environment={k:v for k,v in os.environ.items() if k.startswith('OMLX_QWEN4_')}, greedy=True, mtp=False, batch=1, prefix_cache=False, chunk=128,
                               warmup_tokens=256, timing='(generated-1)/decode, fresh cache, no unused final forward'),
                  records=[])
    model, _, config = target_loader.load_target(a.model)
    report['quantization'] = config['quantization']
    oracle = json.loads(Path(a.oracle).read_text())
    prompt = oracle['prompt']
    report['prompt_ids'] = prompt
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache, position_ids=mx.arange(len(prompt))[None]).logits[0, -1]
    mx.eval(logits)
    actual = np.array(logits.astype(mx.float32))
    expected = np.array(oracle['prefill_logits'], dtype=np.float32)
    error = abs(actual - expected)
    report['prefill'] = dict(max_error=float(error.max()), mean_error=float(error.mean()),
                             actual_argmax=int(actual.argmax()), expected_argmax=int(expected.argmax()))
    Path(a.output).write_text(json.dumps(report, indent=2))
    if error.max() != 0:
        print('OMLX_COMPARISON_REJECTED_NUMERICAL_DRIFT', report['prefill'], flush=True)
        return
    expected = json.loads(Path(a.expected).read_text())['runs'][0]['tokens']
    for run in range(a.runs + 1):
        result = generate(model, prompt, len(expected), 128)
        if result['tokens'] != expected:
            report['trajectory_mismatch'] = dict(run=run, tokens=result['tokens'])
            Path(a.output).write_text(json.dumps(report, indent=2))
            print('OMLX_COMPARISON_REJECTED_TRAJECTORY_DRIFT', flush=True)
            return
        print('run', run, 'tps', result['tokens_per_second'], flush=True)
        if run:
            report['records'].append(dict(run=run, generation=result, exact=True))
            report['environment']['peak_memory_bytes'] = mx.get_peak_memory()
            Path(a.output).write_text(json.dumps(report, indent=2))
    print('OMLX_ADAPTED_COMPARISON_PASSED', flush=True)


if __name__ == '__main__':
    main()
