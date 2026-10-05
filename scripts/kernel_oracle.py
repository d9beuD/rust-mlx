"""Extract small real-shape independent regression vectors; no model weights copied."""
import importlib.util,sys,types
from pathlib import Path
import mlx.core as mx
root=Path(importlib.util.find_spec('mlx_vlm').origin).parent
p=types.ModuleType('mlx_vlm');p.__path__=[str(root)];sys.modules['mlx_vlm']=p
from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_kernel
source=mx.load('results/target-layer0-oracle.safetensors')
q,k,v,g,beta=[source[n][:,-1:] for n in ('q','k','v','g','beta')]
initial=source['state'];y,state=gated_delta_kernel(q,k,v,g,beta,initial)
data={n:z for n,z in zip(('q','k','v','g','beta','initial','y','state'),(q,k,v,g,beta,initial,y,state))}
source=mx.load('results/target-trace-oracle.safetensors');q,k,v=[source[n] for n in ('qrope','krope','values')]
ends=mx.arange(1,11,dtype=mx.int32)[None];ids=mx.arange(512,dtype=mx.int32)[None,None];blocks=mx.where(ids<ends[...,None]//4,ids,-1)
out=qsa_sparse_attention(q,k,v,blocks,ends,scale=256**-0.5,block_size=4)
data.update(qsa_q=q,qsa_k=k,qsa_v=v,qsa_blocks=blocks,qsa_ends=ends,qsa_out=out)
mx.eval(data);mx.save_safetensors('tests/fixtures/native-kernels.safetensors',data)
print('NATIVE_KERNEL_FIXTURES_SAVED',mx.__version__)
