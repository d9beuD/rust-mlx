"""Research-only read-only sharded PLE adapter. Shared row contract, no hash/model logic."""
import json,mmap,struct
import mlx.core as mx
import mlx.nn as nn
import numpy as np

class MappedEmbedding(nn.Module):
    def __init__(self,path,prefix,config):
        super().__init__()
        self.maps=[];self.files=[];self.meta={};self.prefix=prefix;self.q=config['quantization'];self.weight_scale=mx.ones((1,),dtype=mx.bfloat16)
        self._sidecar=None  # Optional upstream prefetch hook; this adapter gathers on demand.
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
