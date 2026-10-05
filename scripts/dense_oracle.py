"""Research oracle only. The Rust inference executable never invokes Python."""
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.models.cache import KVCache
from mlx.utils import tree_flatten

out = Path("tests/fixtures/dense")
out.mkdir(parents=True, exist_ok=True)
config = dict(model_type="qwen3", hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, rms_norm_eps=1e-6, vocab_size=256, max_position_embeddings=4096,
    tie_word_embeddings=False, rope_theta=1000000.)
mx.random.seed(42)
model = Model(ModelArgs.from_dict(config))
model.eval()
weights = dict(tree_flatten(model.parameters()))
mx.eval(weights)
mx.save_safetensors(str(out / "model.safetensors"), weights)
(out / "config.json").write_text(json.dumps(config))
prompt = [1, 17, 92, 3, 145, 255, 0]
cache = [KVCache() for _ in model.layers]
logits = model(mx.array([prompt]), cache=cache)[0, -1]
mx.eval(logits)
prefill_logits = logits.astype(mx.float32).tolist()
tokens = []
for _ in range(32):
    token = mx.argmax(logits).item()
    tokens.append(token)
    logits = model(mx.array([[token]]), cache=cache)[0, -1]
    mx.eval(logits)
(out / "oracle.json").write_text(json.dumps(dict(prompt=prompt,tokens=tokens,prefill_logits=prefill_logits,mlx=mx.__version__)))
print("DENSE_ORACLE_CREATED", out)
