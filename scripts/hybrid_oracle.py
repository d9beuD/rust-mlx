"""Independent official mlx-vlm oracle and tiny architecture fixtures (research only)."""
import argparse, importlib, importlib.util, json, sys, types
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Import the models without mlx-vlm's unrelated audio/image top-level utilities.
root = Path(importlib.util.find_spec('mlx_vlm').origin).parent
package = types.ModuleType('mlx_vlm'); package.__path__ = [str(root)]
sys.modules['mlx_vlm'] = package
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel

p = argparse.ArgumentParser()
p.add_argument('--output', default='tests/fixtures/hybrid')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
config = dict(model_type='qwen4_exp_text', hidden_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32, hc_count=4,hc_lowrank=32,
    linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=32,linear_value_head_dim=32,
    linear_conv_kernel_dim=4,num_experts=8,num_experts_per_tok=2,
    shared_expert_intermediate_size=64,moe_intermediate_size=64,rms_norm_eps=1e-6,
    vocab_size=256,max_position_embeddings=4096,ple_layer_ids=[2],ple_embed_dim=256,
    ple_conv_kernel_size=4,ngram_size=3,heads_per_ngram=8,ngram_vocab_size_base=37,
    make_ngram_vocab_size_divisible_by=128,split_ngram_parts=4,indexer_n_heads=4,indexer_head_dim=32,
    indexer_budget=16,indexer_compress_ratio=4,eos_token_id=255,tie_word_embeddings=False,
    layer_types=['linear_attention']*3+['full_attention'],output_gate_type='sigmoid',
    rope_parameters=dict(type='default',rope_theta=10000000,partial_rotary_factor=0.25,mrope_section=[1,1,2],mrope_interleaved=True))
mx.random.seed(31)
model = LanguageModel(TextConfig.from_dict(config))
model.eval()
# Keep norms representative: HF zero centered and GDN one centered.
# Quantize all linear layers; tiny n-gram rows stay bf16 to exercise mmap without requiring 32 columns.
nn.quantize(model, group_size=32,bits=4,class_predicate=lambda name,m: isinstance(m,nn.Linear) or type(m).__name__=='SwitchLinear')
weights = dict(tree_flatten(model.parameters()))
weights = {('language_model.'+k): v for k,v in weights.items()}
mx.eval(weights)
mx.save_safetensors(str(out/'model.safetensors'),weights)
config['layer_types']=['linear_attention']*3+['full_attention']
wrapper = dict(model_type='qwen4_exp',text_config=config,quantization=dict(bits=4,group_size=32,mode='affine'))
(out/'config.json').write_text(json.dumps(wrapper))
prompt = [1,17,92,3,145,255,0]
cache = model.make_cache()
result = model(mx.array([prompt]),cache=cache,position_ids=mx.arange(len(prompt))[None])
logits = result.logits[0,-1]
mx.eval(logits)
prefill_logits = logits.astype(mx.float32).tolist()
tokens=[]
for _ in range(24):
    token=mx.argmax(logits).item(); tokens.append(token)
    logits=model(mx.array([[token]]),cache=cache,position_ids=mx.array([[len(prompt)+len(tokens)-1]])).logits[0,-1]
    mx.eval(logits)
(out/'oracle.json').write_text(json.dumps(dict(prompt=prompt,tokens=tokens,prefill_logits=prefill_logits,mlx=mx.__version__,oracle='mlx-vlm 0.7.6')))
print('HYBRID_ORACLE_CREATED',out)
