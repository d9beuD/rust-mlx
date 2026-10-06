"""Extract credited MIT exact short-block affine QMV from mlx-vlm."""
import ast
from pathlib import Path
p=Path('.venv/lib/python3.13/site-packages/mlx_vlm/models/quantized_verifier.py')
module=ast.parse(p.read_text())
header=None;source=None
for node in module.body:
    if isinstance(node,ast.FunctionDef) and node.name=='_target_verify_qlinear_header':
        header=next(n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'using namespace metal' in n.value)
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_TARGET_VERIFY_QMV_SOURCE' for t in node.targets):source=ast.literal_eval(node.value)
assert header and source
credit='// Ported from mlx-vlm 0.7.6 models/quantized_verifier.py, MIT; see third-party/mlx-vlm-MIT.\n'
Path('kernels/verify_qmv.h').write_text(credit+header.replace('__RESULTS_PER_SIMDGROUP__','4'))
Path('kernels/verify_qmv.metal').write_text(credit+source)
# Extend to checkpoint affine6 using MLX0.32.2's exact packing/reduction routines.
native_path=next(Path('target/release/build').glob('mlx-sys-*/out/build/_deps/mlx-src/mlx/backend/metal/kernels/quantized.h'))
native=native_path.read_text()
def branch(text,after):
    start=text.index('else if (bits == 6)',after); opening=text.index('{',start); depth=1; end=opening+1
    while depth:
        depth += (text[end]=='{')-(text[end]=='}');end+=1
    return text[opening+1:end-1].replace('values_per_thread','VALUES_PER_THREAD')
load=branch(native,native.index('inline U load_vector('))
dot=branch(native,native.index('inline U qdot('))
header=header.replace('__RESULTS_PER_SIMDGROUP__','4').replace('(BITS == 5 ? 8 : 32 / BITS)','(BITS == 5 ? 8 : (BITS == 6 ? 4 : 32 / BITS))').replace('(BITS == 5 ? 5 : 32 / 8)','(BITS == 5 ? 5 : (BITS == 6 ? 3 : 32 / 8))')
load_marker='} else if (BITS == 8) {\n        for (int i = 0; i < VALUES_PER_THREAD; i++) {\n          sum'
dot_marker='} else if (BITS == 8) {\n        for (int i = 0; i < VALUES_PER_THREAD; i++) {\n          accum'
assert load_marker in header and dot_marker in header
header=header.replace(load_marker,'} else if (BITS == 6) {'+load+'} else if (BITS == 8) {\n        for (int i = 0; i < VALUES_PER_THREAD; i++) {\n          sum',1)
header=header.replace(dot_marker,'} else if (BITS == 6) {'+dot+'} else if (BITS == 8) {\n        for (int i = 0; i < VALUES_PER_THREAD; i++) {\n          accum',1)
Path('kernels/verify_qmv.h').write_text(credit+'// Affine6 branch from MLX0.32.2 quantized.h, Copyright Apple Inc., MIT.\n'+header)
