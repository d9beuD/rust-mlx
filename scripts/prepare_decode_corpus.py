"""Small independent public corpus: frequency ranking, calibration, held-out data.

Texts stay in ignored local storage. Public metadata contains hashes/provenance,
not corpus excerpts. Python is a research/data-preparation tool only.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
from tokenizers import Tokenizer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--directory', type=Path, default=Path('.unlazy/decode3-corpus'))
    p.add_argument('--output', type=Path, default=Path('results/decode3-corpus-manifest.json'))
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    tokenizer_path = a.model / 'tokenizer.json'
    t = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha = sha(tokenizer_path.read_bytes())
    configs = [("HuggingFaceFW/fineweb", "sample-10BT"),
               *(('HuggingFaceFW/fineweb-2', language) for language in
                 ['fra_Latn', 'deu_Latn', 'spa_Latn', 'cmn_Hani', 'jpn_Jpan', 'arb_Arab'])]
    jobs = configs

    def fetch(job):
        dataset, config = job
        path = a.directory / f'{config}-first-rows.json'
        params = dict(dataset=dataset, config=config, split='train')
        if not path.exists():
            response = requests.get('https://datasets-server.huggingface.co/first-rows', params=params, timeout=45)
            response.raise_for_status()
            result = response.json()
            assert len(result['rows']) > 0
            path.write_text(json.dumps(result, ensure_ascii=False))
        result = json.loads(path.read_text())
        return job, result, dict(dataset=dataset, config=config,
                                  split='row_idx%4!=3 calibration/ranking, remainder heldout', response_sha256=sha(path.read_bytes()),
                                  endpoint='https://datasets-server.huggingface.co/first-rows',
                                  license='ODC-By1.0; original page rights apply; texts remain local')

    train, heldout, receipts, counts = [], [], [], Counter()
    # Network I/O only; no MLX/GPU concurrency.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for job, response, receipt in pool.map(fetch, jobs):
            dataset, config = job
            documents = []
            for row in response['rows']:
                text = row['row']['text']
                ids = t.encode(text, add_special_tokens=False).ids
                if len(ids) < 64:
                    continue
                record = dict(dataset=dataset, language=config, row_idx=row['row_idx'],
                              document_sha256=sha(text.encode()), tokens=ids[:1024])
                if row['row_idx'] % 4 != 3:
                    # Bound each document's influence in the language mixture.
                    counts.update(ids[:8192])
                    train.append(record)
                else:
                    heldout.append(record)
                documents.append(dict(row_idx=row['row_idx'], sha256=record['document_sha256'],
                                      tokens=len(ids), truncated_cells=row.get('truncated_cells', [])))
            receipts.append(receipt | {'documents': documents})
    assert train and heldout
    assert all(sum(r['language'] == c for r in train) >= 4 and sum(r['language'] == c for r in heldout) >= 4 for _, c in configs)
    assert not ({r['document_sha256'] for r in train} & {r['document_sha256'] for r in heldout})
    # All observed frequencies first. Unseen rows follow by global ID; explicit
    # fallback membership is not a claim that they were frequent in the sample.
    n = json.loads((a.model / 'config.json').read_text())['text_config']['vocab_size']
    assert all(0 <= i < n for i in counts)
    ranked = sorted(range(n), key=lambda i: (-counts[i], i))
    ranks = []
    for size in [32768, 65536, 98304]:
        membership = set(ranked[:size]) | set(range(max(0, n - 1024), n))
        aligned = min(n, ((len(membership) + 127) // 128) * 128)
        for token in ranked[size:]:
            if len(membership) >= aligned:
                break
            membership.add(token)
        rows = sorted(membership)
        path = Path(f'results/decode3-vocab-multilingual-{size}.json')
        value = dict(ids=rows, tokenizer_sha256=tokenizer_sha,
                     source=dict(url='https://huggingface.co/datasets/HuggingFaceFW/fineweb-2',
                                 corpus_manifest=str(a.output), corpus_independent_of_timing_prompts=True,
                                 observed_unique_ids=len(counts), requested_frequency_rows=size,
                                 unseen_fill_rows=sum(counts[i] == 0 for i in rows),
                                 include_special_tail=1024, pad_to_rows_multiple=128, algorithm='count up to8192 tokens/document; rank descending count, ascending ID; fixed membership, no prompt-fit'))
        path.write_text(json.dumps(value, indent=2) + '\n')
        ranks.append(dict(path=str(path),sha256=sha(path.read_bytes()),rows=len(rows)))
    # Selection is fixed before any model quality/performance run. Small, balanced
    # activation/NLL samples; this is a screening corpus, not a general quality suite.
    calibration, evaluation = [], []
    for _, config in configs:
        calibration += [r | {'tokens': r['tokens'][:256]} for r in train if r['language'] == config][:4]
        evaluation += [r | {'tokens': r['tokens'][:256]} for r in heldout if r['language'] == config][:4]
    local = a.directory / 'model-corpus.json'
    local.write_text(json.dumps(dict(calibration=calibration,heldout=evaluation)) + '\n')
    meta = dict(tokenizer_sha256=tokenizer_sha,script_sha256=sha(Path(__file__).read_bytes()),
                documents_train=len(train),documents_heldout=len(heldout),frequency_tokens=sum(counts.values()),
                observed_unique_ids=len(counts),vocab_artifacts=ranks,receipts=receipts,
                local_model_corpus=str(local),local_model_corpus_sha256=sha(local.read_bytes()),
                calibration_documents=len(calibration),heldout_documents=len(evaluation),
                selection='first four qualifying documents/config for model checks; row_idx%4!=3 calibration/ranking versus==3 heldout; max256 tokens; selected before model evaluation; cached preview cells may be truncated',
                limitation='small public-web sample, not a downstream quality certification; zero-frequency membership is explicit')
    a.output.write_text(json.dumps(meta,indent=2)+'\n')
    print('DECODE_CORPUS_PREPARED',len(train),len(heldout),len(counts))


if __name__ == '__main__':
    main()
