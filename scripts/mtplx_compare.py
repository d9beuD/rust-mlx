"""Research comparator with explicit in-memory checkpoint contract adapters.

Powered by MTPLX by Youssof Altoukhi, https://github.com/youssofal/MTPLX.
The original checkpoint is never modified. This is NOT stock MTPLX serving:
its sidecar-format PLE is replaced by the same read-only row adapter as the
official oracle, and zero-centered norms are shifted in float32 on load, with an explicit
BF16/F32 storage choice. This conversion can change the canonical rounding.
Reject any numerical drift before reporting performance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mapped_embedding import MappedEmbedding


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--oracle', default='results/target-oracle-mlx32.2.json')
    p.add_argument('--expected', default='results/target-baseline-256.json')
    p.add_argument('--output', default='results/mtplx-adapted-baseline.json')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--norm-shift-dtype', choices=['bf16', 'f32'], default='bf16')
    a = p.parse_args()
    upstream = Path('research/upstream/MTPLX').resolve()
    sys.path.insert(0, str(upstream))
    from mtplx.models.qwen4_exp import Model, ModelArgs
    path = Path(a.model)
    mx.set_memory_limit(100 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    config = json.loads((path / 'config.json').read_text())
    text = dict(config['text_config'], ngram_sidecar=True)
    model = Model(ModelArgs.from_dict(dict(config, text_config=text)))
    model.eval()
    for i, layer in enumerate(model.layers):
        if 'ple' in layer:
            prefix = f'language_model.model.layers.{i}.ple.ple_embedding.ngram_embedding'
            layer.ple.ple_embedding.ngram_embedding = MappedEmbedding(path, prefix, config)
    index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    q = config['quantization']
    def predicate(name, module):
        if name + '.scales' not in index:
            return False
        return q.get(name, {k: q[k] for k in ('group_size', 'bits', 'mode')})
    nn.quantize(model, class_predicate=predicate)
    weights = {}
    for shard in sorted(set(index.values())):
        print('loading', shard, flush=True)
        for name, value in mx.load(str(path / shard)).items():
            if name.startswith('language_model.') and '.ngram_embedding.shards.' not in name:
                if name.endswith(Model._HF_NORM_SHIFT_SUFFIXES):
                    value = value.astype(mx.float32) + 1.0
                    if a.norm_shift_dtype == 'bf16':
                        value = value.astype(mx.bfloat16)
                weights[name] = value
    # The checkpoint is already sanitized except MTPLX's PLE conv alias.
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.set_memory_limit(100 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    mx.eval(model.parameters())
    oracle = json.loads(Path(a.oracle).read_text())
    prompt = oracle['prompt']
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    mx.eval(logits)
    actual = np.array(logits.astype(mx.float32))
    expected_logits = np.array(oracle['prefill_logits'], dtype=np.float32)
    error = abs(actual - expected_logits)
    source = upstream / 'mtplx/models/qwen4_exp.py'
    report = dict(model=a.model, comparator='MTPLX adapted text model; plain greedy',
                  attribution='Powered by MTPLX by Youssof Altoukhi',
                  commit=subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  adapter_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/target_loader.py'),Path('scripts/mapped_embedding.py')]},
                  environment=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                                   macos=subprocess.check_output(['sw_vers','-productVersion'],text=True).strip(),
                                   macos_build=subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip(),
                                   thermal=subprocess.check_output(['pmset','-g','therm'],text=True).strip(),
                                   peak_memory_bytes=mx.get_peak_memory(), captured_unix_seconds=time.time()),
                  runtime=dict(mtp=False,batch=1,greedy=True,prefix_cache=False,warmup_tokens=256,
                               norm_shift_dtype=a.norm_shift_dtype,
                               mtplx_environment={k:v for k,v in os.environ.items() if k.startswith('MTPLX_')},
                               timing='(generated-1)/decode; fresh caches, no unused final forward'),
                  dependencies={k: importlib.metadata.version(k) for k in ['mlx', 'mlx-lm', 'transformers']},
                  quantization=q, prompt_ids=prompt,
                  adapters=['read-only 128-shard mmap PLE on demand, no native sidecar prefetch',
                            f'float32 +1 for raw zero-centered norms, stored as {a.norm_shift_dtype}; PLE conv alias'],
                  excluded=['vision', 'MTP sidecar loader', 'native MTPLX HTTP scheduler'],
                  prefill=dict(max_error=float(error.max()), mean_error=float(error.mean()),
                               actual_argmax=int(actual.argmax()), expected_argmax=int(expected_logits.argmax())),
                  records=[])
    Path(a.output).write_text(json.dumps(report, indent=2))
    if error.max() != 0:
        print('MTPLX_COMPARISON_REJECTED_NUMERICAL_DRIFT', report['prefill'], flush=True)
        return
    expected = json.loads(Path(a.expected).read_text())['runs'][0]['tokens']
    for run in range(a.runs + 1):
        cache = model.make_cache()
        started = time.perf_counter()
        logits = model(mx.array([prompt]), cache=cache)[0, -1]
        tokens = [mx.argmax(logits).item()]
        prefill = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(len(expected) - 1):
            logits = model(mx.array([[tokens[-1]]]), cache=cache)[0, -1]
            tokens.append(mx.argmax(logits).item())
        decode = time.perf_counter() - started
        if tokens != expected:
            report['trajectory_mismatch'] = dict(run=run, tokens=tokens)
            Path(a.output).write_text(json.dumps(report, indent=2))
            print('MTPLX_COMPARISON_REJECTED_TRAJECTORY_DRIFT', flush=True)
            return
        print('run', run, 'tps', (len(tokens) - 1) / decode, flush=True)
        if run:
            report['records'].append(dict(run=run, tokens=tokens, prefill_seconds=prefill,
                                          decode_seconds=decode, tokens_per_second=(len(tokens) - 1) / decode))
            Path(a.output).write_text(json.dumps(report, indent=2))
    print('MTPLX_ADAPTED_COMPARISON_PASSED', flush=True)


if __name__ == '__main__':
    main()
