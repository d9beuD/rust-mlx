"""Capture real-checkpoint reference layer stages without loading the full trunk."""
import importlib.util,json,sys,types
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
root=Path(importlib.util.find_spec('mlx_vlm').origin).parent;p=types.ModuleType('mlx_vlm');p.__path__=[str(root)];sys.modules['mlx_vlm']=p
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.cache import ArraysCache

from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import Qwen4ExpMTPDraftModel
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
path=Path('/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp');config=json.loads((path/'config.json').read_text());index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map'];q=config['quantization']
m=Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig.from_dict(config));m.eval()
nn.quantize(m,class_predicate=lambda n,l:q.get('mtp.'+n,{k:q[k] for k in ('group_size','bits','mode')}) if 'mtp.'+n+'.scales'in index else False)
embed=nn.Embedding(config['text_config']['vocab_size'],2560);head=nn.Linear(2560,config['text_config']['vocab_size'],bias=False)
modules={'language_model.model.embed_tokens':embed,'language_model.lm_head':head}
for name,mod in list(modules.items()):
    quant=q.get(name,q);modules[name]=nn.QuantizedEmbedding.from_embedding(mod,group_size=quant['group_size'],bits=quant['bits']) if isinstance(mod,nn.Embedding) else nn.QuantizedLinear.from_linear(mod,group_size=quant['group_size'],bits=quant['bits'])
weights={};other={name:{} for name in modules}
for file in sorted(set(index[k] for k in index if k.startswith('mtp.') or any(k.startswith(n+'.') for n in modules))):
    for k,v in mx.load(str(path/file)).items():
        if k.startswith('mtp.'):weights[k[4:]]=v
        for name in modules:
            if k.startswith(name+'.'):other[name][k[len(name)+1:]]=v
m.load_weights(list(weights.items()),strict=True)
for name,mod in modules.items():mod.load_weights(list(other[name].items()),strict=True)
mx.eval(m.parameters());embed=modules['language_model.model.embed_tokens'];head=modules['language_model.lm_head'];m._input_embed=embed;m._lm_head_fn=head;m._cache=m.make_cache()
prompt=json.loads(Path('results/target-trace-prompt.json').read_text());hidden=mx.load('results/target-trace-oracle.safetensors')['47.output'];tokens=mx.array([prompt[1:]+[271]],dtype=mx.int32)
mixed,wide=m._forward_tokens(tokens,hidden,mx.int32);logits=head(mixed[:,-1:]);mx.eval(logits)
data=dict(input_hidden=hidden,input_tokens=tokens,prefill_mixed=mixed,prefill_hidden=wide,prefill_logits=logits)
ids=[]
for i in range(3):
    token=mx.argmax(logits).item();ids.append(token)
    if i<2:
        mixed,wide=m._forward_tokens(mx.array([[token]],dtype=mx.int32),wide[:,-1:],mx.int32);logits=head(mixed);mx.eval(logits);data[f'decode_{i}_mixed']=mixed;data[f'decode_{i}_hidden']=wide;data[f'decode_{i}_logits']=logits
mx.save_safetensors('results/mtp-oracle.safetensors',data);Path('results/mtp-oracle.json').write_text(json.dumps(dict(tokens=ids,mlx=mx.__version__,oracle='mlx-vlm 0.7.6',model=str(path))))
print('MTP_ORACLE_SAVED',ids)
