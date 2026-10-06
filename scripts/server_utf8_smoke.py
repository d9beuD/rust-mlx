"""Actual SSE length caps ending within multi-token Unicode characters."""
import json
from pathlib import Path
import requests

BASE = 'http://127.0.0.1:8080'
health = requests.get(BASE + '/health', timeout=10).json()
message_sets = [[dict(role='system', content='Repeat the requested text exactly, with no explanation.'),
                 dict(role='user', content='🫎🫨 café 中文 🪿')],
                [dict(role='user', content='Recopie exactement ce texte, sans aucun commentaire : 🐱')]]
reports = []
incomplete = 0
for prompt_index, messages in enumerate(message_sets):
    for cap in range(1, 33):
        body = dict(messages=messages, enable_thinking=False, max_tokens=cap, mtp=False)
        plain = requests.post(BASE + '/v1/chat/completions', json=body, timeout=180)
        plain.raise_for_status()
        plain = plain.json()
        assert 'error' not in plain, plain
        expected = plain['choices'][0]['message']['content']
        response = requests.post(BASE + '/v1/chat/completions', json=dict(body, stream=True),
                                 stream=True, timeout=180)
        response.raise_for_status()
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
            text += data['choices'][0]['delta'].get('content', '')
            metrics = data.get('rust_mlx', metrics)
        assert done and text == expected, (cap, text, expected)
        assert metrics['tokens'] == plain['rust_mlx']['tokens'], cap
        partial = expected.endswith('\ufffd')
        incomplete += partial
        reports.append(dict(prompt_index=prompt_index, max_tokens=cap, tokens=metrics['tokens'], text=text,
                            final_partial_codepoint=partial, exact=True))
assert incomplete > 0, 'test prompt did not exercise a partial final codepoint'
Path(f'results/server-utf8-batch{health["batch_size"]}.json').write_text(json.dumps(
    dict(health=health, message_sets=message_sets, partial_codepoint_cases=incomplete, records=reports),
    ensure_ascii=False, indent=2))
print('SERVER_UTF8_SMOKE_PASSED', incomplete)
