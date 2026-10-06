"""Repeated independent mlx-vlm target baseline; no speculative decoding.

Matches Rust's fresh-cache, greedy (generated-1)/decode convention and verifies
every emitted ID against a saved Rust trajectory. Mapped PLE avoids copying the
106GB checkpoint; its row-by-row reference has not been performance optimized.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from target_loader import load_target


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def generate(model, prompt, count, chunk):
    cache = model.make_cache()
    started = time.perf_counter()
    for offset in range(0, len(prompt), chunk):
        ids = prompt[offset:offset + chunk]
        logits = model(mx.array([ids]), cache=cache,
                       position_ids=mx.arange(offset, offset + len(ids))[None]).logits[0, -1]
        mx.eval(logits)
    tokens = [mx.argmax(logits).item()]
    prefill = time.perf_counter() - started
    latencies = []
    started = time.perf_counter()
    for i in range(count - 1):
        step = time.perf_counter()
        logits = model(mx.array([[tokens[-1]]]), cache=cache,
                       position_ids=mx.array([[len(prompt) + i]])).logits[0, -1]
        tokens.append(mx.argmax(logits).item())
        latencies.append(time.perf_counter() - step)
    decode = time.perf_counter() - started
    return dict(tokens=tokens, prefill_seconds=prefill, decode_seconds=decode,
                tokens_per_second=(count - 1) / decode, latencies=latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--expected', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    reference = json.loads(Path(args.expected).read_text())
    records = reference.get('records', reference.get('runs'))
    first = records[0]
    prompt = first.get('prompt_ids', reference.get('prompt_ids', reference.get('prompt')))
    expected = first.get('tokens', first.get('generation', {}).get('tokens'))
    if not prompt or not expected or len(expected) < 2:
        raise ValueError('expected report must contain prompt IDs and output IDs')
    environment = dict(chip=command('sysctl', '-n', 'machdep.cpu.brand_string'),
                       macos=platform.mac_ver()[0], macos_build=command('sw_vers', '-buildVersion'),
                       mlx=mx.__version__, mlx_vlm=importlib.metadata.version('mlx-vlm'),
                       commit=command('git', 'rev-parse', 'HEAD'),
                       dirty=bool(command('git', 'status', '--porcelain')),
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       thermal=command('pmset', '-g', 'therm'),
                       swap=command('sysctl', 'vm.swapusage'),
                       captured_unix_seconds=time.time())
    model, tokenizer, config = load_target(args.model)
    result = dict(environment=environment, model=args.model, quantization=config['quantization'],
                  prompt_ids=prompt, runtime=dict(mtp=False, greedy=True, batch=1, ignore_eos=True,
                  prefix_cache=False, chunk=args.chunk, warmup_tokens=len(expected),
                  timing='(generated-1)/decode; no unused final forward; fresh caches'), records=[])
    for run in range(args.runs + 1):
        generation = generate(model, prompt, len(expected), args.chunk)
        if generation['tokens'] != expected:
            mismatch = next(i for i, (a, b) in enumerate(zip(generation['tokens'], expected)) if a != b)
            raise AssertionError(f'Native/Rust greedy mismatch at {mismatch}')
        print('run', run, 'tps', generation['tokens_per_second'], 'exact', True, flush=True)
        if run:
            result['records'].append(dict(run=run, generation=generation,
                                          text=tokenizer.decode(generation['tokens']),
                                          peak_memory_bytes=mx.get_peak_memory(), exact=True))
            Path(args.output).write_text(json.dumps(result, indent=2))
    print('NATIVE_BENCH_PASSED', flush=True)


if __name__ == '__main__':
    main()
