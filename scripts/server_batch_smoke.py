"""Real HTTP concurrency, mixed lengths, independent caches and cancellation.

Run against this task's server --no-mtp --batch-size 8, not an unrelated API.
The output is a correctness/HTTP timing report; cached TTFT is not kernel speed.
"""
import concurrent.futures
import json
import time
from pathlib import Path

import requests

BASE = 'http://127.0.0.1:8080'
PROMPTS = [
    'Write a short Rust function that computes Fibonacci numbers.',
    'Explique en français le cache KV.',
    'Write a SQL query for the three most recent orders per customer.',
    'Explain ownership and borrowing in Rust with a short example.',
]


def complete(prompt, stream=False, limit=128, **extra):
    started = time.perf_counter()
    response = requests.post(BASE + '/v1/completions',
        json=dict(prompt=prompt, max_tokens=limit, stream=stream, **extra),
        stream=stream, timeout=240)
    response.raise_for_status()
    if not stream:
        data = response.json()
        assert 'error' not in data, data
        return dict(text=data['choices'][0]['text'], generation=data['rust_mlx'],
                    usage=data['usage'], wall_seconds=time.perf_counter() - started)
    response.encoding = 'utf-8'
    text = ''
    metrics = None
    done = False
    for line in response.iter_lines(decode_unicode=True):
        if not line.startswith('data: '):
            continue
        if line[6:] == '[DONE]':
            done = True
            break
        data = json.loads(line[6:])
        assert 'error' not in data, data
        text += data['choices'][0]['text']
        metrics = data.get('rust_mlx', metrics)
    assert done and metrics is not None
    return dict(text=text, generation=metrics, wall_seconds=time.perf_counter() - started)


health = requests.get(BASE + '/health', timeout=10).json()
assert health['batch_size'] == 8, health
reference = [complete(p) for p in PROMPTS]
# Hit and bypass are explicit, and a hit must preserve exactly the same IDs.
hit = complete(PROMPTS[0])
assert hit['generation']['prefix_cache_hit']
assert hit['generation']['tokens'] == reference[0]['generation']['tokens']
assert hit['usage']['prompt_tokens_details']['cached_tokens'] > 0
bypass = complete(PROMPTS[0], prefix_cache=False)
assert not bypass['generation']['prefix_cache_hit']
assert bypass['generation']['tokens'] == hit['generation']['tokens']
reports = []
for cycle in range(3):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(complete, PROMPTS[i % 4], i % 2 == 0) for i in range(8)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - started
    for i, result in enumerate(results):
        expected = reference[i % 4]
        assert result['text'] == expected['text'], (i, result, expected)
        assert result['generation']['tokens'] == expected['generation']['tokens'], i
    reports.append(dict(cycle=cycle, wall_seconds=wall, results=results,
                        total_output_tokens=sum(len(r['generation']['tokens']) for r in results),
                        exact=True))
# Cancel while three other clients remain active; their cache rows must survive.
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    ongoing = [pool.submit(complete, p) for p in PROMPTS[1:]]
    response = requests.post(BASE + '/v1/completions',
        json=dict(prompt=PROMPTS[0], max_tokens=4000, stream=True), stream=True, timeout=240)
    response.raise_for_status()
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            assert b'"error"' not in line
            break
    assert any(not f.done() for f in ongoing), 'cancellation missed concurrent active clients'
    response.close()
    concurrent_cancellation_results = [f.result() for f in ongoing]
for actual, expected in zip(concurrent_cancellation_results, reference[1:]):
    assert actual['text'] == expected['text']
    assert actual['generation']['tokens'] == expected['generation']['tokens']
probe = complete(PROMPTS[0], limit=1)
assert probe['generation']['tokens'] == reference[0]['generation']['tokens'][:1]
bad = requests.post(BASE + '/v1/completions',
                    json=dict(prompt=PROMPTS[0], max_tokens=8, mtp=True), timeout=20)
assert bad.status_code == 400
Path('results/server-batch-smoke.json').write_text(json.dumps(dict(
    health=health, reference=reference, prefix_hit=hit, prefix_bypass=bypass,
    records=reports, cancellation=True, concurrent_cancellation_results=concurrent_cancellation_results,
    unsupported_mtp_rejected=True,
    timing='HTTP wall, includes scheduling/prefill/stream; prefix hits recorded per request'),
    ensure_ascii=False, indent=2))
print('SERVER_BATCH_SMOKE_PASSED')
