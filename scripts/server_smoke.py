"""Exercise this task's local server, including plain/MTP parity and SSE."""
import json,requests,time
from pathlib import Path
base='http://127.0.0.1:8080'
model=requests.get(base+'/v1/models').json()['data'][0]['id']
prompt='Write a short Rust function that computes Fibonacci numbers.'
expected=json.loads(Path('results/target-baseline-256.json').read_text())['runs'][0]['tokens']
reports=[]
for enabled in [False,True]:
    s=time.monotonic()
    r=requests.post(base+'/v1/completions',json=dict(model=model,prompt=prompt,max_tokens=64,mtp=enabled),timeout=180)
    r.raise_for_status(); data=r.json()
    assert data['rust_mlx']['tokens']==expected[:64],data
    assert data['usage']['completion_tokens']==64
    reports.append(dict(mtp=enabled,wall_seconds=time.monotonic()-s,generation=data['rust_mlx']))
# The content stream must reconstruct the exact non-streamed UTF-8 decode.
r=requests.post(base+'/v1/completions',json=dict(prompt=prompt,max_tokens=64,stream=True),stream=True,timeout=180)
r.raise_for_status(); text=''; done=False; finish=None
for line in r.iter_lines(decode_unicode=True):
    if not line.startswith('data: '): continue
    value=line[6:]
    if value=='[DONE]': done=True; break
    d=json.loads(value); assert 'error' not in d,d
    text+=d['choices'][0]['text']; finish=d['choices'][0]['finish_reason'] or finish
plain=requests.post(base+'/v1/completions',json=dict(prompt=prompt,max_tokens=64),timeout=180).json()
assert text==plain['choices'][0]['text'] and done and finish=='length'
r=requests.post(base+'/v1/chat/completions',json=dict(messages=[dict(role='user',content='Réponds par un mot : quelle est la capitale de la France ?')],enable_thinking=False,max_tokens=32),timeout=180)
r.raise_for_status(); data=r.json(); assert data['object']=='chat.completion'
reports.append(dict(chat=True,response=data))
for bad in [dict(prompt=prompt,temperature=.7),dict(prompt=prompt,max_tokens=0),dict(prompt=prompt,model='missing'),dict(messages=[dict(role='user',content=[dict(image_url='invalid')])])]:
    route='/v1/chat/completions' if 'messages' in bad else '/v1/completions'
    r=requests.post(base+route,json=bad,timeout=10); assert 400<=r.status_code<500,r.text
# Disconnect early; subsequent request must succeed, proving cancellation clears state.
r=requests.post(base+'/v1/completions',json=dict(prompt=prompt,max_tokens=4000,stream=True),stream=True,timeout=180)
for line in r.iter_lines():
    if line.startswith(b'data: '):
        assert b'"error"' not in line,line
        break
r.close()
r=requests.post(base+'/v1/completions',json=dict(prompt=prompt,max_tokens=1),timeout=180)
r.raise_for_status(); assert r.json()['rust_mlx']['tokens']==expected[:1]
Path('results/server-smoke.json').write_text(json.dumps(dict(reports=reports,sse_exact=True,validation_errors=True,cancellation=True),ensure_ascii=False,indent=2))
print('SERVER_SMOKE_PASSED')
