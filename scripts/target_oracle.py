"""Independent native MLX oracle. Python is research-only."""
import argparse,json,time
from pathlib import Path
import mlx.core as mx
from target_loader import load_target

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tokens',type=int,default=16);p.add_argument('--prompt',default='Write a short Rust function that computes Fibonacci numbers.');p.add_argument('--output',default='results/target-oracle.json');p.add_argument('--prompt-ids');p.add_argument('--prefill-chunk',type=int,default=0);a=p.parse_args()
path=Path(a.model)
model,tokenizer,config=load_target(path)
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
