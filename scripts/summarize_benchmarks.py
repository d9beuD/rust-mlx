"""Summarize saved paired reports without removing slow samples or cold data.

Reports themselves distinguish excluded warmups and timed repetitions. This
script never turns aggregate batch rates into single-conversation claims.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text())
    groups = defaultdict(list)
    for row in data.get('records', data.get('runs', [])):
        key = str(row.get('batch', row.get('prompt', 'raw')))
        groups[key].append(row)
    result = dict(source=str(path), runtime=data['runtime'], groups=[])
    for key, rows in groups.items():
        modes = defaultdict(list)
        pairs = defaultdict(dict)
        trajectories = set()
        for row in rows:
            candidate = row.get('candidate', row.get('candidate_enabled', False))
            rate = row.get('aggregate_tokens_per_second', row.get('tokens_per_second', row.get('decode_tokens_per_second')))
            if rate is None:
                raise ValueError(f'missing measured throughput in {path}')
            modes[candidate].append(rate)
            pairs[row.get('cycle', row.get('run'))][candidate] = rate
            tokens = row.get('tokens', row.get('generation', {}).get('tokens'))
            if tokens is None:
                raise ValueError(f'missing output trajectory in {path}')
            trajectories.add(json.dumps(tokens, separators=(',', ':')))
        samples = {str(mode).lower(): dict(samples=values, n=len(values),
                    median=statistics.median(values), mean=statistics.mean(values),
                    minimum=min(values), maximum=max(values),
                    standard_deviation=statistics.stdev(values) if len(values)>1 else 0)
                   for mode, values in modes.items()}
        gains = [100*(p[True]/p[False]-1) for p in pairs.values() if set(p)=={False,True}]
        group = dict(workload=key, metric='aggregate batch tok/s' if 'batch' in rows[0] else 'single-conversation tok/s',
                     statistics=samples, identical_output_trajectories=len(trajectories)==1,
                     paired_gain_percent=gains,
                     median_paired_gain_percent=statistics.median(gains) if gains else None)
        if 'batch' in rows[0]:
            group['per_conversation_candidate_median']=samples['true']['median']/rows[0]['batch']
        result['groups'].append(group)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reports', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(dict(method='all saved timed samples; median of within-cycle candidate/reference gains; no outlier removal',
        reports=[summarize(path) for path in a.reports]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
