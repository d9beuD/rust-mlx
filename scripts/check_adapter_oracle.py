#!/usr/bin/env python3
"""Nonzero real-shape Python/Rust oracle and actual frozen Q8 head VJP."""
import argparse
import json
from pathlib import Path
import subprocess
import mlx.core as mx
from train_mtp_adapter import sha, correction


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    assert not a.directory.exists();a.directory.mkdir(parents=True)
    report=json.loads(a.collection.read_text());head=mx.load(report['head_path']);q=report['head_quantization']
    h,hc=report['hidden'],report['hc'];mx.random.seed(20261006);records=[]
    for rows in [1,2,3,4,8]:
        mixed=mx.random.normal((1,rows,h)).astype(mx.bfloat16)
        previous=mx.random.normal((1,rows,h*hc)).astype(mx.bfloat16)
        embedding=mx.random.normal((1,rows,h)).astype(mx.bfloat16)
        normalize=lambda x:x.astype(mx.float32)*mx.rsqrt(mx.mean(x.astype(mx.float32)**2,axis=-1,keepdims=True)+1e-6)
        features=mx.concatenate([normalize(x) for x in [mixed,previous,embedding]],axis=-1)
        params={'a':mx.random.normal((32,h*(hc+2)))*.01,'b':mx.random.normal((h,32))*.01}
        # Match Rust's explicit sqrt/divide normalization operation boundaries.
        normalize=lambda x:x.astype(mx.float32)/mx.sqrt(mx.mean(x.astype(mx.float32)**2,axis=-1,keepdims=True)+1e-6)
        features=mx.concatenate([normalize(x) for x in [mixed,previous,embedding]],axis=-1)
        expected=correction(params,features,mixed);mx.eval(expected)
        fixture=a.directory/f'input-{rows}.safetensors';output=a.directory/f'output-{rows}.safetensors'
        mx.save_safetensors(str(fixture),dict(mixed=mixed,previous=previous,embedding=embedding,**params))
        subprocess.run(['target/release/adapter-oracle','--fixture',str(fixture),'--output',str(output)],check=True)
        actual=mx.load(str(output))['actual'];error=float(mx.max(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32))).item());assert error==0,(rows,error)
        records.append(dict(rows=rows,error=error))
    d=mx.load(report['records'][0]['path']);x,m,y=[d[k][:4] for k in ['features','mixed','labels']]
    def loss(p):
        v=mx.quantized_matmul(correction(p,x,m),head['weight'],head['scales'],head['biases'],transpose=True,group_size=q['group_size'],bits=q['bits']).astype(mx.float32)
        if 'bias' in head:v+=head['bias'].astype(mx.float32)
        return mx.mean(mx.logsumexp(v,axis=-1)-mx.take_along_axis(v,y[:,None],axis=-1).squeeze(-1))
    loss_value,gradient=mx.value_and_grad(loss)(params);mx.eval(loss_value,gradient)
    assert all(bool(mx.all(mx.isfinite(v)).item()) and float(mx.sum(mx.abs(v)).item())>0 for v in gradient.values())
    result=dict(complete=True,mlx=mx.__version__,script_sha256=sha(__file__),collection_sha256=sha(a.collection),records=records,head_vjp=dict(loss=float(loss_value.item()),finite_nonzero=True))
    (a.directory/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('ADAPTER_NATIVE_ORACLE_AND_HEAD_VJP_PASSED')


if __name__=='__main__':main()
