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
path=Path('/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp');c=json.loads((path/'config.json').read_text());index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.0.'
m=Qwen4ExpDecoderLayer(TextConfig.from_dict(c['text_config']),0)
nn.quantize(m,class_predicate=lambda n,l:c['quantization'].get(prefix+n,{k:c['quantization'][k] for k in ('group_size','bits','mode')}) if prefix+n+'.scales' in index else False)
weights={}
for file in sorted(set(index[k] for k in index if k.startswith(prefix))):
    weights.update({k[len(prefix):]:v for k,v in mx.load(str(path/file)).items() if k.startswith(prefix)})
m.load_weights(list(weights.items()),strict=True)
m.eval()
x=mx.sin(mx.arange(102400,dtype=mx.float32)*0.013).reshape(1,10,10240).astype(mx.bfloat16)
mixed,residual,inj=m.attn_hyper_connection(x)
a=m.linear_attn(mixed,cache=ArraysCache(size=2));h=(residual.reshape(1,10,4,2560)+a[:,:,None,:]*inj[:,:,:,None]).reshape(1,10,10240)
mlpin,_,mlpinj=m.mlp_hyper_connection(h);moe=m.mlp(mlpin)
z=(h.reshape(1,10,4,2560)+moe[:,:,None,:]*mlpinj[:,:,:,None]).reshape(1,10,10240)
stages=dict(input=x,mixed=mixed,inject=inj,gdn=a,after_attention=h,mlp_input=mlpin,mlp_inject=mlpinj,moe=moe,output=z)
from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_kernel
att=m.linear_attn
raw=att.in_proj_qkv(mixed);inp=mx.concatenate([mx.zeros((1,3,10240),dtype=mixed.dtype),raw],1)
conv=nn.silu(att.conv1d(inp));q,k,v=[z.reshape(1,10,h,128) for z,h in zip(mx.split(conv,[2048,4096],-1),[16,16,48])]
q,k=att._normalize_qk(q,k);bb,aa=att._project_gates(mixed);g,beta=_compute_g_beta(att.A_log,aa,bb,att.dt_bias)
y,state=gated_delta_kernel(q,k,v,g,beta,mx.zeros((1,48,128,128),dtype=mx.float32))
stages.update(raw=raw,conv=conv,q=q,k=k,v=v,a=aa,b=bb,g=g,beta=beta,y=y,state=state,norm=att.norm(y,att.in_proj_z(mixed).reshape(1,10,48,128)))
from mlx_vlm.models.qwen4_exp.language import _qwen4_hyper_pointwise
mh=m.mlp_hyper_connection;xn=mh.hc_norm(h);lo=_qwen4_hyper_pointwise(4,2560)[0](mh.input_mix_weight_down(xn));up=mh.input_mix_weight_up(lo)
stages.update(mlp_normed=xn,mlp_lowrank=lo,mlp_projected=up)
mx.eval(stages);mx.save_safetensors('results/target-layer0-oracle.safetensors',stages)
print('TARGET_COMPONENTS_CREATED',[(k,v.dtype) for k,v in stages.items()])
