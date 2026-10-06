"""Reproducible restricted Apache-2.0 oMLX / MIT Apple kernel port."""
import ast,re
from pathlib import Path
source=Path('research/upstream/omlx/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_projection.py')
tree=ast.parse(source.read_text());values={}
for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
        for target in node.targets:
            if isinstance(target,ast.Name):values[target.id]=node.value.value
credit='// Adapted from pinned oMLX hc_projection.py (Apache-2.0), with Apple MLX MIT arithmetic.\n// See third-party/omlx-APACHE-2.0, third-party/MLX-MIT and NOTICE.\n'
Path('kernels/hc_projection.h').write_text(credit+values['_HEADER'].strip()+'\n')
s=values['_SOURCE'].strip();s=re.sub(r'\bx\b','xrow',s);s=re.sub(r'\bcombined\b','row_output',s)
prefix='const uint input_row = threadgroup_position_in_grid.z;\nconst device T* xrow = x + (size_t)input_row * K;\ndevice T* row_output = combined + (size_t)input_row * 324;\n'
Path('kernels/hc_projection.metal').write_text(credit+prefix+s+'\n')
