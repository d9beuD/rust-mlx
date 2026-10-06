#!/usr/bin/env python3
"""Predeclared first blind prompt: longer confirmation after full primary study."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run_decode3_checks import core_digest
from train_mtp_adapter import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    while not json.loads((root/'results/mtp-next-commands.json').read_text()).get('complete'):
        time.sleep(2)
    study=json.loads((root/'results/mtp-next-blind.json').read_text())
    native=[r for r in study['records'] if not r['candidate']];candidate=[r for r in study['records'] if r['candidate']]
    token_count=sum(max(0,len(r['generation']['tokens'])-1) for r in native)
    assert token_count==sum(max(0,len(r['generation']['tokens'])-1) for r in candidate)
    gain=(sum(r['generation']['decode_seconds'] for r in native)/sum(r['generation']['decode_seconds'] for r in candidate)-1)*100
    if gain<5:
        print('LONG_CONFIRMATION_NOT_TRIGGERED',gain);return
    source=core_digest();choice=json.loads((root/'results/mtp-next-choice.json').read_text())
    spec=json.loads((root/'.unlazy/mtp-next/prompts.json').read_text())
    # Fixed document ordering; never pick the fastest measured prompt.
    spec['records']=[next(r for r in spec['records'] if r['split']=='blind')]
    corpus=root/'.unlazy/mtp-next/blind-long.json';corpus.write_text(json.dumps(spec)+'\n')
    command=['target/release/adapter-study','--model',a.model,'--adapter','.unlazy/mtp-next/'+choice['adapter'],'--corpus',str(corpus),'--split','blind','--depths',str(choice['depth']),'--max-tokens','1024','--warmup-tokens','1024','--runs','4','--ignore-eos','--output','results/mtp-next-adapter-long.json']
    log=root/'results/mtp-next-adapter-long.log';assert not log.exists()
    started=time.time()
    with log.open('wb') as stream:process=subprocess.run(command,cwd=root,env=os.environ,stdout=stream,stderr=subprocess.STDOUT)
    passed=process.returncode==0 and 'ADAPTER_TARGET_IDS_AND_DEPTH_STUDY_COMPLETED' in log.read_text()
    receipt=dict(command=command,source_sha256=source,binary_sha256=sha(root/command[0]),corpus_sha256=sha(corpus),started_unix=started,ended_unix=time.time(),exit_code=process.returncode,passed=passed,log_sha256=sha(log),trigger_blind_weighted_gain_percent=gain,selection='first blind document in fixed split order, not fastest prompt',prefix_cache=False,batch_size=1,forced_length_includes_after_EOS=True)
    (root/'results/mtp-next-adapter-long-command.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert passed and source==core_digest()
    print('MTP_NEXT_LONG_CONFIRMATION_COMPLETED')


if __name__=='__main__':main()
