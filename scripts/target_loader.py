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

from mapped_embedding import MappedEmbedding


def load_target(model_path):
    path=Path(model_path);config=json.loads((path/'config.json').read_text())
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
    return model,tokenizer,config
