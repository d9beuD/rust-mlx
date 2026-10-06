#!/usr/bin/env python3
"""Fixed small distillation pilot. Python research oracle, never serving runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',required=True)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    destination=Path(a.destination)
    assert not destination.exists(), 'preserve prior pilot'
    report=json.loads(Path(a.collection).read_text())
    spec=json.loads(Path(a.spec).read_text())
    assert report['complete']
    split={key:[] for key in ['fit','heldout']}
    for row in report['records']:
        assert sha(row['path'])==row['sha256']
        split[row['split']].append((row,mx.load(row['path'])))
    assert len(split['fit'])==7 and len(split['heldout'])==28
    fit_hash={r['document_sha256'] for r,_ in split['fit']}
    assert not any(r['document_sha256'] in fit_hash for r,_ in split['heldout'])
    student=mx.concatenate([v['draft'].reshape(-1,v['draft'].shape[-1]) for _,v in split['fit']])
    teacher=mx.concatenate([v['teacher'].reshape(-1,v['teacher'].shape[-1]) for _,v in split['fit']]).astype(mx.float32)
    probabilities=mx.softmax(teacher,axis=-1)
    bias=mx.zeros((student.shape[-1],),dtype=mx.float32)
    def corrected(x,b):
        return (x+b.astype(mx.bfloat16)).astype(mx.bfloat16).astype(mx.float32)
    def loss(b):
        logits=corrected(student,b)
        logp=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
        return -mx.mean(mx.sum(probabilities*logp,axis=-1)) + .1*mx.sum(b*b)
    vg=mx.value_and_grad(loss)
    values=[]
    started=time.time()
    for step in range(200):
        value,grad=vg(bias)
        bias=mx.clip(bias-.05*grad,-.5,.5)
        mx.eval(value,bias)
        values.append(float(value.item()))
    bias=bias.astype(mx.bfloat16);mx.eval(bias)
    assert bool(mx.all(mx.isfinite(bias)).item()) and float(mx.max(mx.abs(bias)).item())<=.5
    records=[]
    for key,rows in split.items():
        for row,arrays in rows:
            student=arrays['draft'].astype(mx.float32)
            teacher=arrays['teacher'].astype(mx.float32)
            p=mx.softmax(teacher,axis=-1)
            labels=mx.argmax(teacher,axis=-1)
            measurements=[]
            for b in [mx.zeros_like(bias),bias]:
                x=corrected(student.astype(mx.bfloat16),b)
                logp=x-mx.logsumexp(x,axis=-1,keepdims=True)
                logt=teacher-mx.logsumexp(teacher,axis=-1,keepdims=True)
                measurements.append({'teacher_kl':float(mx.mean(mx.sum(p*(logt-logp),axis=-1)).item()),
                    'teacher_argmax_agreement':float(mx.mean(mx.argmax(x,axis=-1)==labels).item())})
            records.append({'split':key,'document_sha256':row['document_sha256'], 'language':row['language'],
                'predictions':16,'baseline':measurements[0],'trained':measurements[1]})
    destination.mkdir(parents=True)
    weights=destination/'bias.safetensors'
    mx.save_safetensors(str(weights),{'bias':bias})
    output={'complete':True,'pilot':spec,'spec_sha256':sha(a.spec),'collection':a.collection,'collection_sha256':sha(a.collection),
        'script_sha256':sha(__file__),'model':report['model'],'config_sha256':report['config_sha256'],
        'tokenizer_sha256':report['tokenizer_sha256'],'corpus_sha256':report['corpus_sha256'],
        'bias_sha256':sha(weights),'bias_shape':list(bias.shape),'bias_dtype':'BF16','bias_max_abs':float(mx.max(mx.abs(bias)).item()),
        'bias_norm':float(mx.sqrt(mx.sum(bias.astype(mx.float32)**2)).item()), 'loss_steps':values,
        'training_seconds':time.time()-started,'records':records,
        'limits':'Fixed tiny additive head-bias pilot. Teacher-forced KL/agreement are not natural draft acceptance or global quality evidence; full greedy MTP remains required.'}
    (destination/'metadata.json').write_text(json.dumps(output,indent=2)+'\n')
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
    print('DRAFT_BIAS_TRAINING_PILOT_COMPLETED')


if __name__=='__main__':
    main()
