#!/usr/bin/env python3
"""Independent five-direction evidence audit, including corrupted-token control."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
from run_decode3_checks import core_digest
ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def check_trajectory(report,expected):
 assert report['runtime']['batch_size']==1 and report['runtime']['sampler']=='greedy'
 assert report['runtime']['mtp'] and not report['runtime']['prefix_cache']
 assert report['runtime']['warmup_tokens']==256
 assert all(row['draft_depth']==3 and row['generation']['tokens']==expected and len(expected)==256 for row in report['runs'])
 assert report['resident_variant'] is None
 pairs={}
 for r in report['runs']:pairs.setdefault(r['run'],{})[r['candidate_enabled']]=r
 assert set(pairs)=={1,2,3,4} and all(set(p)=={False,True} for p in pairs.values())
 native=[pairs[c][False]['decode_tokens_per_second'] for c in range(1,5)]
 candidate=[pairs[c][True]['decode_tokens_per_second'] for c in range(1,5)]
 gains=[(c/b-1)*100 for b,c in zip(native,candidate)]
 return {'control_rates':native,'candidate_rates':candidate,'control_median':statistics.median(native),'candidate_median':statistics.median(candidate),'paired_gains_percent':gains,'paired_median_gain_percent':statistics.median(gains),'exact_original_ids':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
 cohorts=[]
 cases=[('route-tail','raw'),('route-tail','chat'),('ple-prepare','raw'),('rope-ids','raw'),('config-reuse','raw'),('runtime-prepare','raw'),('runtime-prepare','chat'),('down-tail','raw'),('down-packed','raw'),('draft-bias','raw'),('down-packed-vector','raw'),('down-packed-vector','chat'),('draft-bias','chat')]
 for kernel,suite in cases:
  for i in range(1 if suite=='raw' else 4):
   path=f'results/followup-ab-{kernel}-{suite}-p{i}.json'
   report=read(path);receipt=read(path.replace('.json','-command.json'))
   oracle='results/target-baseline-256.json' if suite=='raw' else f'results/decode3-config-combo-full-chat-p{i}-c1-native.json'
   base=read(oracle);expected=base['runs'][0].get('tokens') or base['runs'][0]['generation']['tokens']
   assert report['prompt_ids']==base['prompt_ids']
   assert receipt['exit_code']==0 and receipt['report_sha256']==sha(path) and receipt['oracle_sha256']==sha(oracle)
   assert len(receipt['source_sha256'])==64 and len(receipt['binary_sha256'])==64
   stats=check_trajectory(report,expected)
   assert all(r['kernel_candidate']==kernel for r in report['runs'])
   cohorts.append({'kernel':kernel,'suite':suite,'prompt':i,'report':path,'source_sha256':receipt['source_sha256'],**stats})
 prototype=read('results/followup-ab-route-tail-raw-p0.json');bad=copy.deepcopy(prototype);bad['runs'][0]['generation']['tokens'][17]^=1
 try:check_trajectory(bad,read('results/target-baseline-256.json')['runs'][0]['tokens'])
 except AssertionError:negative=True
 else:raise AssertionError('corrupted-token control accepted')
 training=read('results/followup-draft-training.json');collection=read('results/followup-draft-collection.json')
 assert training['complete'] and collection['complete'] and training['collection_sha256']==sha('results/followup-draft-collection.json')
 assert training['spec_sha256']==sha('results/followup-draft-spec.json')
 fit=[r for r in collection['records'] if r['split']=='fit'];held=[r for r in collection['records'] if r['split']=='heldout']
 assert len(fit)==7 and len(held)==28
 assert not ({r['document_sha256'] for r in fit}&{r['document_sha256'] for r in held})
 assert not ({tuple(r['prompt_ids']) for r in fit}&{tuple(r['prompt_ids']) for r in held})
 quality=None
 if a.final:
  quality=read('results/followup-current-qualification.json')
  assert quality['passed'] and quality['source_sha256']==core_digest()
  assert all(r['passed'] for r in quality['receipts']) and quality['bias_shader_tokens_exact']
  assert quality['server']['owned_servers_stopped']
  assert quality['quality']['release_tests_passed']>=38 and quality['metal']['validation_tests_passed']>=37
  assert all(sha(r['log'])==r['sha256'] for r in quality['receipts'])
  for prefix in ['followup-current-route','followup-current-runtime','followup-current-down']:
   checks=read(f'results/{prefix}-checks.json');assert checks['complete'] and checks['passed'] and checks['source_sha256']==core_digest()
   assert all(r['passed'] for r in checks['checks'])
 out={'scope':'all five follow-up directions; exact solo greedy depth3; rejected candidates stay off','scope_complete':bool(a.final),'source_sha256':core_digest(),'cohorts':cohorts,'negative_token_control':negative,'training_fit_documents':len(fit),'training_heldout_documents':len(held),'training_heldout_predictions':16*len(held),'training_limits':training['limits'],'quality':quality,'promote_default':False,'solo100_demonstrated':False,'warnings':['Historical experiments bind their own immutable source/binary snapshots; final quality is independently bound to current source.','Component/instrumented rates are excluded from all full-generation gains.','Selected packing allocates1.258GB for3 banks; no all48-bank performance claim.','Tiny teacher-forced bias improvements do not prove full MTP distillation or broad model quality.']}
 path='results/followup-final-summary.json' if a.final else 'results/followup-progress-summary.json'
 (ROOT/path).write_text(json.dumps(out,indent=2)+'\n')
 print('FOLLOWUP_FINAL_EVIDENCE_VERIFIED' if a.final else 'FOLLOWUP_PROGRESS_EVIDENCE_VERIFIED')

if __name__=='__main__':main()
