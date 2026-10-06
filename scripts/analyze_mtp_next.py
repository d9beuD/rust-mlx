#!/usr/bin/env python3
"""Independent trajectory/pair/hash audit, including a corrupted-token negative control."""
import copy
import json
import math
from pathlib import Path
import statistics
from run_decode3_checks import core_digest
from train_mtp_adapter import sha

ROOT=Path(__file__).resolve().parents[1]


def cohort(rows,candidate_key,repetition_key,expected=None):
    repetitions=sorted({r[repetition_key] for r in rows})
    assert repetitions==[1,2,3,4]
    native=[];candidate=[];pairs=[]
    for rep in repetitions:
        n=[r for r in rows if r[repetition_key]==rep and not r[candidate_key]]
        c=[r for r in rows if r[repetition_key]==rep and r[candidate_key]]
        assert len(n)==len(c)==1
        n,c=n[0],c[0]
        if expected is None:expected=n['generation']['tokens']
        assert n['generation']['tokens']==c['generation']['tokens']==expected,'target token mismatch'
        if len(expected)>1:
            nt,ct=[r['generation']['decode_seconds'] for r in [n,c]]
            assert nt>0 and ct>0 and math.isfinite(nt) and math.isfinite(ct)
            pairs.append((nt/ct-1)*100)
            native.append((len(expected)-1)/nt);candidate.append((len(expected)-1)/ct)
    return dict(predictions=len(expected),exact_target_ids=True,paired_gains_percent=pairs,paired_median_gain_percent=statistics.median(pairs) if pairs else None,native_median_tps=statistics.median(native) if native else None,candidate_median_tps=statistics.median(candidate) if candidate else None,native_cv=statistics.stdev(native)/statistics.mean(native) if len(native)>1 else None,candidate_cv=statistics.stdev(candidate)/statistics.mean(candidate) if len(candidate)>1 else None)


def oracle(suite,index):
    path=ROOT/('results/target-baseline-256.json' if suite=='raw' else f'results/decode3-config-combo-full-chat-p{index}-c1-native.json')
    r=json.loads(path.read_text())['runs'][0]
    return r.get('tokens') or r['generation']['tokens']


def main():
    source=core_digest();command_path=ROOT/'results/mtp-next-commands.json'
    commands=json.loads(command_path.read_text())
    assert commands['complete'] and all(r['passed'] and r['exit_code']==0 for r in commands['receipts'])
    for r in commands['receipts']:
        assert r['source_sha256']==source
        path=ROOT/f"results/mtp-next-{r['name']}.log"
        assert sha(path)==r['log_sha256'] and r['marker'] in path.read_text(errors='replace')
    eps=[];adapters=[]
    for suite,count in [('raw',1),('chat',4)]:
        for index in range(count):
            path=ROOT/f'results/mtp-next-epilogue-{suite}-{index}.json'
            d=json.loads(path.read_text());assert d['runtime']['batch_size']==1 and not d['runtime']['prefix_cache']
            c=cohort(d['runs'],'candidate_enabled','run',oracle(suite,index))
            native={r['run']:r for r in d['runs'] if not r['candidate_enabled']}
            candidate={r['run']:r for r in d['runs'] if r['candidate_enabled']}
            assert all(native[i]['generation']['draft_tokens']==candidate[i]['generation']['draft_tokens'] and native[i]['generation']['acceptance']==candidate[i]['generation']['acceptance'] and candidate[i]['epilogue_calls']>0 for i in native)
            eps.append(dict(suite=suite,prompt=index,report_sha256=sha(path),**c))
        path=ROOT/f'results/mtp-next-adapter-{suite}.json'
        d=json.loads(path.read_text());assert d['complete'] and d['batch_size']==1 and not d['prefix_cache']
        for index in range(count):
            rows=[r for r in d['records'] if r['prompt']==index]
            adapters.append(dict(suite=suite,prompt=index,report_sha256=sha(path),**cohort(rows,'candidate','repetition',oracle(suite,index))))
    blind=json.loads((ROOT/'results/mtp-next-blind.json').read_text());assert blind['complete']
    groups=[]
    for i in range(32):groups.append(dict(prompt=i,**cohort([r for r in blind['records'] if r['prompt']==i],'candidate','repetition')))
    timed=[g['paired_median_gain_percent'] for g in groups if g['paired_median_gain_percent'] is not None]
    native=[r for r in blind['records'] if not r['candidate']]
    candidate=[r for r in blind['records'] if r['candidate']]
    count=sum(max(0,len(r['generation']['tokens'])-1) for r in native)
    assert count==sum(max(0,len(r['generation']['tokens'])-1) for r in candidate)
    nt=sum(r['generation']['decode_seconds'] for r in native)
    ct=sum(r['generation']['decode_seconds'] for r in candidate)
    blind_summary=dict(weighted_native_tps=count/nt,weighted_candidate_tps=count/ct,weighted_gain_percent=(nt/ct-1)*100,median_prompt_gain_percent=statistics.median(timed),min_prompt_gain_percent=min(timed),max_prompt_gain_percent=max(timed),timed_prompts=len(timed),prompts_at_least_five_percent=sum(x>=5 for x in timed),all_32_exact=True)
    long_receipt_path=ROOT/'results/mtp-next-adapter-long-command.json'
    lr=json.loads(long_receipt_path.read_text())
    assert lr['passed'] and lr['exit_code']==0 and lr['source_sha256']==source
    assert sha(ROOT/'results/mtp-next-adapter-long.log')==lr['log_sha256']
    long_data=json.loads((ROOT/'results/mtp-next-adapter-long.json').read_text())
    assert long_data['complete'] and long_data['batch_size']==1 and not long_data['prefix_cache']
    long_result=cohort(long_data['records'],'candidate','repetition')
    assert long_result['predictions']==1024
    long_result.update(selection=lr['selection'],forced_length_includes_after_EOS=True,receipt_sha256=sha(long_receipt_path))
    qualification_path=ROOT/'results/mtp-next-current-qualification.json'
    qualification=json.loads(qualification_path.read_text())
    assert qualification['passed'] and qualification['source_sha256']==source
    for r in qualification['receipts']:
        assert r['passed'] and r['exit_code']==0 and r['code_sha256']==source
        log=ROOT/r['log']
        assert sha(log)==r['sha256']
        assert r['success_marker'] is None or r['success_marker'] in log.read_text()
    assert qualification['server']['owned_servers_stopped']
    # A positive real cohort must reject changed IDs even if existing metadata says exact.
    original=json.loads((ROOT/'results/mtp-next-adapter-raw.json').read_text())['records']
    cohort(original,'candidate','repetition',oracle('raw',0))
    bad=copy.deepcopy(original)
    next(r for r in bad if r['candidate'])['generation']['tokens'][0]^=1
    try:cohort(bad,'candidate','repetition',oracle('raw',0))
    except AssertionError:negative=True
    else:raise AssertionError('negative trajectory control did not reject')
    q=json.loads((ROOT/'results/mtp-next-expert-quantization.json').read_text());assert q['complete'] and len(q['records'])==30 and all(r['executed'] for r in q['records'])
    quant=[]
    for mode in ['affine','mxfp4']:
        rows=[r for r in q['records'] if r['mode']==mode]
        quant.append(dict(mode=mode,cases=len(rows),median_component_gain_percent=statistics.median(r['paired_median_gain_percent'] for r in rows),min_gain_percent=min(r['paired_median_gain_percent'] for r in rows),max_gain_percent=max(r['paired_median_gain_percent'] for r in rows),median_output_relative_l2=statistics.median(r['quality']['relative_l2'] for r in rows),qualifies_format_port=all(r['paired_median_gain_percent']>=5 for r in rows)))
    for name in ['epilogue-verifier','epilogue-rollback']:
        rows=json.loads((ROOT/f'results/mtp-next-{name}.json').read_text());assert all(r['logit_error']==0 and r['state_error']==0 and r.get('hidden_error',0)==0 for r in rows)
    collection=json.loads((ROOT/'results/mtp-next-collection.json').read_text());assert collection['complete'] and len(collection['records'])==192
    assert len({r['document_sha256'] for r in collection['records']})==192
    choice=json.loads((ROOT/'results/mtp-next-choice.json').read_text())
    training=json.loads((ROOT/f".unlazy/mtp-next/{choice['adapter']}/metadata.json").read_text())
    assert training['complete'] and training['collection_sha256']==sha(ROOT/'results/mtp-next-collection.json')
    def eligible(rows):
        raw=next(r for r in rows if r['suite']=='raw')['paired_median_gain_percent']
        chat=[r['paired_median_gain_percent'] for r in rows if r['suite']=='chat']
        return raw>=5 and statistics.median(chat)>=5 and min(chat)>=-2
    adapter_eligible=eligible(adapters);epilogue_eligible=eligible(eps)
    result=dict(adapter_qualifies_confirmation=adapter_eligible,epilogue_qualifies_confirmation=epilogue_eligible,complete=True,source_sha256=source,command_receipts_sha256=sha(command_path),choice=choice,training=training,epilogue=eps,adapter=adapters,blind=groups,blind_summary=blind_summary,long_confirmation=long_result,qualification_sha256=sha(qualification_path),quantization=quant,corrupted_token_negative_control=negative,global_promotion=False,limitations='Component gains/errors are not full-model quantized quality; teacher-state agreement is not speculative acceptance. Fresh caches and B1 in timing. Natural validation chooses depth before blind/external tests.')
    (ROOT/'results/mtp-next-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('MTP_NEXT_EVIDENCE_INDEPENDENTLY_VERIFIED')


if __name__=='__main__':main()
