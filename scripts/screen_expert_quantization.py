#!/usr/bin/env python3
"""Actual selected expert-down banks and captured verifier activations. Offline study."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import mlx.core as mx


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=100)
    a=p.parse_args();assert not a.output.exists() and a.repetitions>0
    config=json.loads((a.model/'config.json').read_text())
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    fixture_metadata=json.loads((a.fixture/'metadata.json').read_text())
    assert fixture_metadata['complete'] and sha(a.fixture/'activations.safetensors')==fixture_metadata['fixture_sha256']
    fixture=mx.load(str(a.fixture/'activations.safetensors'))
    records=[]
    for layer in [0,23,47]:
        prefix=f'language_model.model.layers.{layer}.mlp.switch_mlp.down_proj'
        tensors={}
        for suffix in ['weight','scales','biases']:
            name=prefix+'.'+suffix
            bank=mx.load(str(a.model/index[name]));tensors[suffix]=bank[name];del bank
        original_q=config.get('quantization',config.get('quantization_config'))
        original_q=original_q.get(prefix,original_q)
        assert original_q['mode']=='affine' and original_q['bits']==4
        original=(tensors['weight'],tensors['scales'],tensors['biases'])
        reconstructed=mx.dequantize(*original,group_size=original_q['group_size'],bits=original_q['bits']);mx.eval(reconstructed)
        for mode,bits,group in [('affine',3,64),('mxfp4',4,32)]:
            started=time.perf_counter()
            candidate=mx.quantize(reconstructed,group_size=group,bits=bits,mode=mode);mx.eval(candidate)
            prepare=time.perf_counter()-started
            for rows in [1,2,3,4,8]:
                x=fixture[f'layer{layer}.routed'][:,:rows]
                ids=fixture[f'layer{layer}.ids'][:,:rows]
                scores=fixture[f'layer{layer}.scores'][:,:rows]
                mx.eval(x,ids,scores)
                def run(weights,qmode,qbits,qgroup):
                    kwargs=dict(rhs_indices=ids,transpose=True,group_size=qgroup,bits=qbits,mode=qmode)
                    y=mx.gather_qmm(x,weights[0],weights[1],weights[2] if len(weights)==3 else None,**kwargs)
                    return mx.sum((mx.squeeze(y,axis=-2)*scores[...,None]).astype(mx.bfloat16),axis=-2).astype(mx.bfloat16)
                try:
                    reference=run(original,'affine',4,original_q['group_size']);changed=run(candidate,mode,bits,group);mx.eval(reference,changed)
                    error=changed.astype(mx.float32)-reference.astype(mx.float32)
                    measurements=dict(max_abs=float(mx.max(mx.abs(error)).item()),relative_l2=float((mx.sqrt(mx.sum(error*error))/mx.maximum(mx.sqrt(mx.sum(reference.astype(mx.float32)**2)),1e-12)).item()))
                    for _ in range(10):mx.eval(run(original,'affine',4,original_q['group_size']),run(candidate,mode,bits,group))
                    times=[]
                    for rep in range(a.repetitions):
                        pair={}
                        for label in (['native','candidate'] if rep%2==0 else ['candidate','native']):
                            start=time.perf_counter();v=run(original,'affine',4,original_q['group_size']) if label=='native' else run(candidate,mode,bits,group);mx.eval(v);pair[label]=time.perf_counter()-start
                        times.append(pair)
                    row=dict(layer=layer,rows=rows,mode=mode,bits=bits,group_size=group,preparation_seconds=prepare,original_bytes=sum(v.nbytes for v in original),candidate_bytes=sum(v.nbytes for v in candidate),quality=measurements,pairs=times,paired_median_gain_percent=statistics.median((v['candidate'] and (v['native']/v['candidate']-1)*100) for v in times),executed=True)
                except Exception as error:
                    row=dict(layer=layer,rows=rows,mode=mode,bits=bits,group_size=group,executed=False,exception=type(error).__name__,message=str(error))
                records.append(row)
                a.output.write_text(json.dumps(dict(complete=False,mlx=mx.__version__,fixture=fixture_metadata,records=records),indent=2)+'\n')
                print('EXPERT_QUANT',layer,rows,mode,row.get('paired_median_gain_percent',row.get('message')),flush=True)
            del candidate;gc.collect()
        del reconstructed,original,tensors;gc.collect()
    result=dict(complete=True,mlx=mx.__version__,model=str(a.model),mixed_quantization=config.get('quantization',config.get('quantization_config')),fixture=fixture_metadata,script_sha256=sha(__file__),records=records,limits='Native Python MLX component cost and captured selected-bank output error; double quantization, not original-target equality or broad quality/performance evidence')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('EXPERT_QUANTIZATION_SCREEN_COMPLETED',flush=True)


if __name__=='__main__':main()
