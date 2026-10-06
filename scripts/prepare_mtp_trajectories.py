#!/usr/bin/env python3
"""Deterministic document-disjoint on-policy prompt specification; texts stay local."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('.unlazy/decode3-corpus'))
    p.add_argument('--destination', type=Path, default=Path('.unlazy/mtp-next'))
    a = p.parse_args()
    a.destination.mkdir(parents=True, exist_ok=True)
    target = a.destination / 'prompts.json'
    assert not target.exists(), 'preserve split'
    tokenizer = Tokenizer.from_file(str(a.model / 'tokenizer.json'))
    old = json.loads((a.source / 'model-corpus.json').read_text())
    excluded = {r['document_sha256'] for v in old.values() for r in v}
    candidates = {}
    for path in sorted(a.source.glob('*-first-rows.json')):
        for row in json.loads(path.read_text())['rows']:
            text = row['row']['text']
            digest = sha(text.encode())
            if digest not in excluded and len(tokenizer.encode(text).ids) >= 64:
                candidates[digest] = (text, path.name, row['row_idx'])
    assert len(candidates) >= 192, f'not enough disjoint documents: {len(candidates)}'
    templates = [
        'Résume et explique en français le passage suivant.',
        'Explain the main ideas and limitations of the following passage in English.',
        'Write a Python function extracting structured facts from this passage. Explain its tests.',
        'Write a Rust function representing information from this passage. Explain its tests.',
        'Design SQL tables and queries for information in this passage.',
        'Give a step-by-step reasoned analysis of the claims in this passage.',
        'Compare two interpretations of this passage and explain their assumptions.',
        'Rédige une réponse pédagogique en français avec des exemples à partir de ce passage.',
    ]
    records = []
    # Hash ordering predates inference; every split has four documents/domain.
    for i, digest in enumerate(sorted(candidates)[:192]):
        text, source, row = candidates[digest]
        excerpt = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False).ids[:128])
        task = templates[i % 8]
        tokens = tokenizer.encode(task + '\n\n' + excerpt + '\n\nResponse:\n', add_special_tokens=False).ids
        records.append(dict(split='fit' if i < 128 else 'validation' if i < 160 else 'blind', domain=i % 8,
                            document_sha256=digest, source=source, row_idx=row, tokens=tokens))
    spec = dict(kind='on-policy-mtp-v1', records=records, max_tokens=256,
                tokenizer_sha256=sha((a.model/'tokenizer.json').read_bytes()),
                excluded_documents=sorted(excluded), limitation='public cached web excerpts plus fixed task templates; not a broad task benchmark')
    target.write_text(json.dumps(spec, ensure_ascii=False) + '\n')
    Path('results/mtp-next-corpus.json').write_text(json.dumps({**spec, 'records': [{k:v for k,v in r.items() if k != 'tokens'} for r in records], 'local_spec_sha256':sha(target.read_bytes())}, indent=2)+'\n')
    print('MTP_TRAJECTORY_SPLITS_PREPARED', len(records))


if __name__ == '__main__':
    main()
