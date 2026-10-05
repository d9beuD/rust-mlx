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

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tokens',type=int,default=16);p.add_argument('--prompt',default='Write a short Rust function that computes Fibonacci numbers.');p.add_argument('--output',default='results/target-oracle.json');p.add_argument('--prompt-ids');p.add_argument('--prefill-chunk',type=int,default=0);a=p.parse_args()
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
cache=model.make_cache();started=time.perf_counter()
chunk=a.prefill_chunk or len(prompt)
for offset in range(0,len(prompt),chunk):
    ids=prompt[offset:offset+chunk]
    out=model(mx.array([ids]),cache=cache,position_ids=mx.arange(offset,offset+len(ids))[None])
    logits=out.logits[0,-1];mx.eval(logits)
prefill=time.perf_counter()-started
initial=logits.astype(mx.float32).tolist();tokens=[];latencies=[]
for i in range(a.tokens):
    token=mx.argmax(logits).item();tokens.append(token);started=time.perf_counter()
    logits=model(mx.array([[token]]),cache=cache,position_ids=mx.array([[len(prompt)+i]])).logits[0,-1];mx.eval(logits)
    latencies.append(time.perf_counter()-started);print('token',i,token,latencies[-1],flush=True)
result=dict(prompt=prompt,tokens=tokens,prefill_logits=initial,text=tokenizer.decode(tokens),prefill_seconds=prefill,decode_seconds=sum(latencies),decode_tokens_per_second=len(latencies)/sum(latencies),latencies=latencies,mlx=mx.__version__,oracle='mlx-vlm 0.7.6',model=str(path),quantization=config['quantization'],mtp=False,temperature=0,batch_size=1,cache='cold prompt',prefill_chunk=a.prefill_chunk,peak_memory=mx.get_peak_memory())
Path(a.output).write_text(json.dumps(result));print(result['text']);print('TARGET_ORACLE_SAVED',result['decode_tokens_per_second'])
