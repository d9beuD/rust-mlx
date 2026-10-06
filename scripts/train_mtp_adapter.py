#!/usr/bin/env python3
"""Frozen-head, activation-dependent MTP training. Offline oracle only."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx
import mlx.optimizers as optim


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def correction(params, features, mixed):
    delta = ((features @ params['a'].T) @ params['b'].T).astype(mx.bfloat16)
    return (mixed.astype(mx.bfloat16) + delta).astype(mx.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--rank', type=int, default=32, choices=[32,64])
    p.add_argument('--hours', type=float, default=48)
    p.add_argument('--epochs', type=int, default=100)
    a = p.parse_args()
    assert 0 < a.hours <= 48 and not a.destination.exists()
    report = json.loads(a.collection.read_text())
    assert report['complete']
    split = {s:[r for r in report['records'] if r['split']==s] for s in ['fit','validation','blind']}
    assert [len(split[s]) for s in split] == [128,32,32]
    assert len({r['document_sha256'] for r in report['records']}) == 192
    assert sha(report['head_path']) == report['head_sha256']
    head = mx.load(report['head_path'])
    q = report['head_quantization']
    def logits(mixed):
        y = mx.quantized_matmul(mixed,head['weight'],head['scales'],head['biases'],transpose=True,group_size=q['group_size'],bits=q['bits'])
        return y if 'bias' not in head else y + head['bias']
    def loss(params, x, mixed, labels):
        scores = logits(correction(params,x,mixed)).astype(mx.float32)
        selected = mx.take_along_axis(scores,labels[:,None],axis=-1).squeeze(-1)
        return mx.mean(mx.logsumexp(scores,axis=-1)-selected)
    mx.random.seed(20261006)
    h,hc=report['hidden'],report['hc']
    params={'a':mx.random.normal((a.rank,h*(hc+2)))*.01,'b':mx.zeros((h,a.rank))}
    mx.eval(params)
    optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
    vg=mx.value_and_grad(loss)
    cache=OrderedDict()
    def load(row):
        key=row['path']
        if key not in cache:
            assert sha(key)==row['sha256']
            cache[key]=mx.load(key)
            end=next((i for i,t in enumerate(row['tokens']) if t in [248044,248046]),row['predictions'])
            cache[key]={k:v[:end] for k,v in cache[key].items()}
            if len(cache)>8: cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]
    def metrics(rows, trained):
        total=0; correct=0; nll=0
        for row in rows:
            d=load(row)
            for start in range(0,d['labels'].size,32):
                x,m,y=[d[k][start:start+32] for k in ['features','mixed','labels']]
                s=logits(correction(trained,x,m) if trained is not None else m).astype(mx.float32)
                l=mx.sum(mx.logsumexp(s,axis=-1)-mx.take_along_axis(s,y[:,None],axis=-1).squeeze(-1))
                c=mx.sum(mx.argmax(s,axis=-1)==y)
                mx.eval(l,c);nll+=l.item();correct+=c.item();total+=y.size
        return dict(predictions=total,nll=nll/total,agreement=correct/total)
    a.destination.mkdir(parents=True)
    started=time.monotonic();deadline=started+a.hours*3600
    baseline=metrics(split['validation'],None)
    history=[];best=-1;bad=0;best_epoch=-1;steps=0
    for epoch in range(a.epochs):
        if time.monotonic()>=deadline: break
        sums={k:mx.zeros_like(v) for k,v in params.items()};count=0;loss_sum=0
        # Reproducible rotation avoids always beginning with the same documents.
        rows=split['fit'][epoch%128:]+split['fit'][:epoch%128]
        for row in rows:
            d=load(row)
            for start in range(0,d['labels'].size,32):
                if time.monotonic()>=deadline: break
                x,m,y=[d[k][start:start+32] for k in ['features','mixed','labels']]
                value,grad=vg(params,x,m,y)
                sums={k:sums[k]+grad[k] for k in params};count+=1
                mx.eval(value,sums);loss_sum+=value.item()
                if count==8:
                    grad={k:v/8 for k,v in sums.items()}
                    norm=mx.sqrt(sum(mx.sum(v*v) for v in grad.values()))
                    scale=mx.minimum(1.,1./mx.maximum(norm,1e-12))
                    grad={k:v*scale for k,v in grad.items()}
                    params=optimizer.apply_gradients(grad,params)
                    mx.eval(params,optimizer.state);steps+=1
                    sums={k:mx.zeros_like(v) for k,v in params.items()};count=0
            if time.monotonic()>=deadline: break
        validation=metrics(split['validation'],params)
        history.append(dict(epoch=epoch,steps=steps,validation=validation,training_loss_sum=loss_sum,elapsed_seconds=time.monotonic()-started))
        print('MTP_TRAIN',json.dumps(history[-1]),flush=True)
        if validation['agreement']>best:
            best=validation['agreement'];bad=0;best_epoch=epoch
            mx.save_safetensors(str(a.destination/'adapter.safetensors'),params)
        else: bad+=1
        (a.destination/'progress.json').write_text(json.dumps(dict(complete=False,baseline=baseline,history=history),indent=2)+'\n')
        if bad>=3: break
        # Six-hour pilot: continuation requires measurable validation improvement.
        if time.monotonic()-started>=6*3600 and best<=baseline['agreement']: break
    assert best_epoch>=0, 'no trained checkpoint'
    best_params=mx.load(str(a.destination/'adapter.safetensors'))
    blind_baseline=metrics(split['blind'],None)
    blind_trained=metrics(split['blind'],best_params)
    assert all(bool(mx.all(mx.isfinite(v)).item()) for v in best_params.values())
    index=json.loads((Path(report['model'])/'model.safetensors.index.json').read_text())
    checkpoint_sha256={name:sha(Path(report['model'])/name) for name in sorted(set(index['weight_map'].values()))}
    valid_predictions={name:sum(next((i for i,t in enumerate(r['tokens']) if t in [248044,248046]),r['predictions']) for r in rows) for name,rows in split.items()}
    result=dict(valid_predictions=valid_predictions,termination_policy='discard every sample after first EOS; acquisition raw tokens retained',checkpoint_sha256=checkpoint_sha256,complete=True,kind='mtp-residual-v1',rank=a.rank,hidden=h,hc=hc,config_sha256=report['config_sha256'],tokenizer_sha256=report['tokenizer_sha256'],adapter_sha256=sha(a.destination/'adapter.safetensors'),collection_sha256=sha(a.collection),script_sha256=sha(__file__),head_sha256=report['head_sha256'],baseline_validation=baseline,best_epoch=best_epoch,best_agreement=best,history=history,blind_baseline=blind_baseline,blind_trained=blind_trained,seconds=time.monotonic()-started,stop_reason='validation patience, pilot gate, epoch cap or48h hard deadline',limitations='teacher-state on-policy one-step agreement; natural speculative acceptance and total cost require independent Rust A/B')
    (a.destination/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print('MTP_ADAPTER_TRAINING_COMPLETED',flush=True)


if __name__=='__main__':
    main()
