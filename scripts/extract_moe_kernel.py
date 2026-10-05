"""Copy and credit the upstream MIT exact affine gate/up Metal specialization."""
import ast
from pathlib import Path
path=Path('.venv/lib/python3.13/site-packages/mlx_vlm/models/fast_ops.py')
values={}
def string(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,str): return node.value
    if isinstance(node,ast.Name): return values.get(node.id)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
        a,b=string(node.left),string(node.right)
        if a is not None and b is not None: return a+b
for node in ast.parse(path.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        value=string(node.value)
        if value is not None: values[node.targets[0].id]=value
for bits in [4,5]:
    header=values['_COMMON_HEADER']+values[f'_AFFINE{bits}_SWITCH_HEADER']
    header=header.replace('__GROUP_SIZE__','GROUP_SIZE').replace('__RESULTS_PER_SIMDGROUP__','4').replace('__NUM_SIMDGROUPS__','2').replace('__PACKS_PER_THREAD__','2')
    source=values['_AFFINE5_SWITCH_GATE_UP_SOURCE']
    if bits==4: source=source.replace('affine5','affine4').replace('* 5 / 8','* 4 / 8')
    credit='// Ported from mlx-vlm 0.7.6 models/fast_ops.py, MIT; see third-party/mlx-vlm-MIT.\n'
    Path(f'kernels/moe_gate_up{bits}.metal').write_text(credit+source)
    Path(f'kernels/moe_gate_up{bits}.h').write_text(credit+header)
