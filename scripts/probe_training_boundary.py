#!/usr/bin/env python3
"""Small MLX training-boundary probes, never inference throughput."""
import argparse
import hashlib
import json
from pathlib import Path
import mlx.core as mx


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 output=Path(a.output);assert not output.exists()
 records=[]
 kernel=mx.fast.metal_kernel(name='rust_mlx_training_probe',input_names=['x'],output_names=['y'],source='uint i=thread_position_in_grid.x; if(i<4) y[i]=x[i]*2;')
 def custom(x):return mx.sum(kernel(inputs=[x],output_shapes=[(4,)],output_dtypes=[mx.float32],grid=(32,1,1),threadgroup=(32,1,1))[0])
 q,s,b=mx.quantize(mx.ones((4,64)),group_size=64,bits=4)
 def quant_input(x):return mx.sum(mx.quantized_matmul(x,q,s,b,transpose=True,group_size=64,bits=4))
 for name,fn,x in [('custom_metal_vjp',custom,mx.ones((4,))),('affine_quant_input_vjp',quant_input,mx.ones((1,64)))]:
  try:
   g=mx.grad(fn)(x);mx.eval(g)
   records.append({'case':name,'success':True,'shape':list(g.shape),'finite':bool(mx.all(mx.isfinite(g)).item())})
  except Exception as e:
   records.append({'case':name,'success':False,'exception':type(e).__name__,'message':str(e)})
 output.write_text(json.dumps({'mlx':mx.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,'complete':True,'limits':'Toy boundary probes; no full MTP transformer trained. Missing custom-kernel VJP requires derivatives or a native differentiable training graph; it does not prove training impossible.'},indent=2)+'\n')
 print('TRAINING_BOUNDARY_PROBES_COMPLETED')

if __name__=='__main__':main()
