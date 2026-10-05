"""Reference checkpoint chat strings using Jinja2, with no inference runtime."""
import json
from pathlib import Path
from jinja2 import Environment
path = Path('/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp')
source = json.loads((path/'tokenizer_config.json').read_text())['chat_template']
env = Environment()
def fail(message): raise ValueError(message)
env.globals['raise_exception'] = fail
t = env.from_string(source)
cases = []
for messages in [
    [{'role':'user','content':'Bonjour, explique les emprunts Rust.'}],
    [{'role':'system','content':'Réponds en français.'},{'role':'user','content':'Écris une fonction.'},{'role':'assistant','content':'Voici.','reasoning_content':'Je vérifie.'},{'role':'user','content':'Plus court.'}],
]:
    for thinking in [False, True]:
        for effort in ['low','medium','xhigh']:
            args = dict(messages=messages,enable_thinking=thinking,reasoning_effort=effort,add_generation_prompt=True,tools=[])
            cases.append({**args,'expected':t.render(**args)})
Path('tests/fixtures/chat').mkdir(exist_ok=True)
Path('tests/fixtures/chat/tokenizer_config.json').write_text(json.dumps({'chat_template':source},ensure_ascii=False,indent=2))
Path('tests/fixtures/chat/oracle.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
print('CHAT_ORACLE_SAVED',len(cases))
