"""Official mlx-vlm text oracle on the user's checkpoint. No production Python dependency."""
import argparse, importlib.util, json, mmap, struct, sys, time, types
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

root=Path(importlib.util.find_spec('mlx_vlm').origin).parent
package=types.ModuleType('mlx_vlm');package.__path__=[str(root)];sys.modules['mlx_vlm']=package
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel

class MappedEmbedding(nn.Module):
    def __init__(self,path,prefix,config):
        super().__init__()
        self.maps=[];self.files=[];self.meta={};self.prefix=prefix;self.q=config['quantization'];self.weight_scale=mx.ones((1,),dtype=mx.bfloat16)
        for shard in sorted(path.glob('model*.safetensors')):
            f=shard.open('rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
            mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);self.files.append(f);self.maps.append(mm)
            for name,value in h.items():
                if name.startswith(prefix):self.meta[name]=(len(self.maps)-1,n+8,value)
        self.offsets=[0]
        for i in range(128):
            key=f'{prefix}.shards.{i}.weight'
            if key not in self.meta:break
            self.offsets.append(self.offsets[-1]+self.meta[key][2]['shape'][0])
    def row(self,name,row):
        i,base,m=self.meta[name];width=m['shape'][1];size=(m['data_offsets'][1]-m['data_offsets'][0])//m['shape'][0]
        offset=base+m['data_offsets'][0]+row*size
        dtype={'U32':np.uint32,'BF16':np.uint16,'F16':np.float16,'F32':np.float32}[m['dtype']]
        a=np.frombuffer(self.maps[i],dtype=dtype,count=width,offset=offset).copy()[None]
        result=mx.array(a)
        return result.view(mx.bfloat16) if m['dtype']=='BF16' else result
    def __call__(self,ids):
        rows=ids.reshape(-1).tolist();results=[]
        for row in rows:
            i=int(np.searchsorted(self.offsets,row,side='right')-1);local=row-self.offsets[i];p=f'{self.prefix}.shards.{i}'
            w=self.row(p+'.weight',local)
            if p+'.scales' in self.meta:
                q=self.q.get(p,self.q);w=mx.dequantize(w,self.row(p+'.scales',local),self.row(p+'.biases',local),q['group_size'],q['bits'])
            results.append(w)
        return (mx.concatenate(results).astype(mx.bfloat16)*self.weight_scale).reshape(*ids.shape,-1)

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tokens',type=int,default=16);p.add_argument('--prompt',default='Write a short Rust function that computes Fibonacci numbers.');p.add_argument('--output',default='results/target-oracle.json');p.add_argument('--prompt-ids');a=p.parse_args()
path=Path(a.model);config=json.loads((path/'config.json').read_text())
mx.set_memory_limit(100*1024**3);mx.set_cache_limit(512*1024**2)
model=LanguageModel(TextConfig.from_dict(config['text_config']));model.eval()
for i,layer in enumerate(model.layers):
    if 'ple' in layer:
        prefix=f'language_model.model.layers.{i}.ple.ple_embedding.ngram_embedding'
        layer.ple.ple_embedding.ngram_embedding=MappedEmbedding(path,prefix,config)
q=config['quantization']
def predicate(name,module):
    prefix='language_model.'+name
    if prefix+'.scales' not in weight_names:return False
    return q.get(prefix,{k:q[k] for k in ('group_size','bits','mode')})
index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map'];weight_names=set(index)
nn.quantize(model,class_predicate=predicate)
weights={}
for filename in sorted(set(index.values())):
    print('loading',filename,flush=True)
    for name,tensor in mx.load(str(path/filename)).items():
        if name.startswith('language_model.') and '.ngram_embedding.shards.' not in name:
            weights[name[len('language_model.'):]]=tensor
model.load_weights(list(weights.items()),strict=True)
mx.eval(model.parameters());print('weights ready',mx.get_active_memory(),flush=True)
from tokenizers import Tokenizer
tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
prompt=json.loads(Path(a.prompt_ids).read_text()) if a.prompt_ids else tokenizer.encode(a.prompt).ids

cache=model.make_cache();ids=mx.array([prompt]);h=model.model.embed_tokens(ids);h=mx.broadcast_to(h[:,:,None,:],(1,len(prompt),4,2560)).reshape(1,len(prompt),10240)
stages={'embedding':h}
for i,l in enumerate(model.layers):
    stages[f'{i}.input']=h
    if 'ple' in l:h=h+l.ple(h,ids,cache[i],None)
    stages[f'{i}.after_ple']=h
    mixed,_,inj=l.attn_hyper_connection(h);stages[f'{i}.mixed']=mixed
    if l.is_linear:branch=l.linear_attn(mixed,cache=cache[i])
    else:branch=l.self_attn(mixed,mask='causal',cache=cache[i],position_ids=mx.arange(len(prompt))[None])
    if i==3:
        att=l.self_attn;qr=att.q_proj(mixed);kr=att.k_proj(mixed);vr=att.v_proj(mixed)
        qq,gg=mx.split(qr.reshape(1,len(prompt),24,512),2,-1)
        stages.update(qraw=qr,kraw=kr,vraw=vr,qnorm=att.q_norm(qq).transpose(0,2,1,3),knorm=att.k_norm(kr.reshape(1,len(prompt),2,256)).transpose(0,2,1,3))
        qq,kk,vv,gg,mask=att._prepare_projected_qkv(qr,kr,vr,model.make_cache()[3],mx.arange(len(prompt))[None],None,'causal')
        oo=mx.fast.scaled_dot_product_attention(qq,kk,vv,scale=256**-0.5,mask='causal')
        stages.update(qrope=qq,krope=kk,values=vv,sdpa=oo,gate=gg,gated=(oo.transpose(0,2,1,3).reshape(1,len(prompt),-1)*mx.sigmoid(gg)))
    stages[f'{i}.branch']=branch
    h=h+(branch[:,:,None,:]*inj[:,:,:,None]).reshape(h.shape);stages[f'{i}.after_attention']=h
    mixed,_,inj=l.mlp_hyper_connection(h);stages[f'{i}.mlp_input']=mixed;branch=l.mlp(mixed);stages[f'{i}.moe']=branch
    h=h+(branch[:,:,None,:]*inj[:,:,:,None]).reshape(h.shape);stages[f'{i}.output']=h
    mx.eval(h);print('layer',i,flush=True)
mx.save_safetensors('results/target-trace-oracle.safetensors',stages)
Path('results/target-trace-prompt.json').write_text(json.dumps(prompt))
print('TARGET_TRACE_SAVED')
