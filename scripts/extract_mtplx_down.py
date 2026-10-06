#!/usr/bin/env python3
"""Extract constant/f-string shader sources without executing upstream Python."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    root=Path('research/upstream/MTPLX');path=root/'mtplx/kernels/qwen4_m4_routed_down.py'
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    values={}
    def literal(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return values[n.id]
        if isinstance(n,ast.JoinedStr):
            return ''.join(str(literal(x.value)) if isinstance(x,ast.FormattedValue) else str(literal(x)) for x in n.values)
        raise ValueError(type(n).__name__)
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:values[n.targets[0].id]=literal(n.value)
            except (ValueError,KeyError):pass
    a.destination.mkdir(parents=True,exist_ok=True)
    credit=f'// MTPLX by Youssof Altoukhi, Apache-2.0, revision{revision}.\n// Source mtplx/kernels/qwen4_m4_routed_down.py; third-party/MTPLX-APACHE-2.0 and NOTICE.\n// Adaptation: affine g64; preserve scalar/reduction casts; bounded1-8-row dispatch.\n'
    header=values['_HEADER'].replace('GROUP_SIZE = 32','GROUP_SIZE = 64')
    source=values['_ROUTED_SOURCE']
    (a.destination/'moe_down_tail.h').write_text(credit+header)
    (a.destination/'moe_down_tail.metal').write_text(credit+source)
    # The same qdot arithmetic, with original linear word addresses translated
    # to an evaluated [E,H/4,K/8,4] lossless allocation. No dequantization.
    ph=header.replace('const thread float* x_thread,','const device uchar* base,\n        const thread float* x_thread,')
    ph=ph.replace('const device ushort* ws = (const device ushort*)w;', '''const size_t logical_word = size_t(w - base) / sizeof(uint);
        const size_t output_row = logical_word / (K / 8);
        const size_t word_in_row = logical_word % (K / 8);
        const size_t packed_word = (output_row / 4) * (K / 8) * 4 + word_in_row * 4 + output_row % 4;
        uint codes = ((const device uint*)base)[packed_word];''')
    ph=ph.replace('(ws[i] & ', '((ushort(codes >> (16 * i))) & ')
    ps=source.replace('                    output_w,','                    output_w,\n                    (const device uchar*)weights,')
    (a.destination/'moe_down_packed.h').write_text(credit+'// Additional modification: lossless four-output uint32 word interleaving.\n'+ph)
    (a.destination/'moe_down_packed.metal').write_text(credit+ps)
    out={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.destination.glob('moe_down_*')}
    (a.destination/'moe_down_provenance.json').write_text(json.dumps({'repository':'https://github.com/youssofal/MTPLX','commit':revision,'source':str(path.relative_to(root)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'license':'Apache-2.0','outputs':out,'changes':['g32->g64','lossless optional four-output code interleaving; scales/biases stay native','short-row geometry validation in Rust']},indent=2)+'\n')
    print('MTPLX_DOWN_SHADERS_EXTRACTED')


if __name__=='__main__':main()
