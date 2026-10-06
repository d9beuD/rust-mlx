#!/usr/bin/env python3
"""Serialize the research phases and bind every command to its source/binary/log."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from run_decode3_checks import core_digest, VALIDATION
from train_mtp_adapter import sha

ROOT=Path(__file__).resolve().parents[1]


def rates(report):
    groups={}
    for row in report['records']:
        key=(row['candidate'],row['depth'])
        v=groups.setdefault(key,[0,0.])
        v[0]+=max(0,len(row['generation']['tokens'])-1);v[1]+=row['generation']['decode_seconds']
    return {key:tokens/seconds if seconds else 0. for key,(tokens,seconds) in groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    collection=ROOT/'results/mtp-next-collection.json'
    deadline=time.time()+7200
    while True:
        try:
            if collection.exists() and json.loads(collection.read_text()).get('complete'): break
        except json.JSONDecodeError:
            pass  # Collector flush may be in flight; only a complete report unlocks training.
        assert time.time()<deadline,'collection timed out'
        time.sleep(2)
    snapshot=core_digest();receipts=[]
    environment={**os.environ,'PATH':str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH']}
    assert not any(environment.get(k) for k in VALIDATION)
    assert not any(k.startswith('RUST_MLX_') for k in environment)
    def run(name,command,marker,validation=False):
        log=ROOT/f'results/mtp-next-{name}.log'
        assert not log.exists(),f'preserve {log}'
        assert snapshot==core_digest(),'source changed during phase'
        started=time.time()
        binary=ROOT/command[0]
        executable_sha=sha(binary) if binary.is_file() else None
        workloads=subprocess.check_output(['ps','-axo','pid,comm,%cpu'],text=True)
        with log.open('wb') as stream:
            process=subprocess.run(command,cwd=ROOT,env={**environment,**(VALIDATION if validation else {})},stdout=stream,stderr=subprocess.STDOUT)
        passed=process.returncode==0 and marker in log.read_text(errors='replace')
        script_path=ROOT/command[1] if len(command)>1 else None
        row=dict(script_sha256=sha(script_path) if script_path is not None and script_path.is_file() else None,name=name,command=command,started_unix=started,ended_unix=time.time(),source_sha256=snapshot,binary_sha256=executable_sha,exit_code=process.returncode,marker=marker,passed=passed,log_sha256=sha(log),validation_environment=VALIDATION if validation else {},competing_processes=workloads)
        receipts.append(row)
        (ROOT/'results/mtp-next-commands.json').write_text(json.dumps(dict(complete=False,receipts=receipts),indent=2)+'\n')
        assert passed,f'{name} failed: inspect {log}'
        assert snapshot==core_digest(),'source changed during execution'
        print('MTP_NEXT_PHASE',name,'passed',flush=True)
    if not (ROOT/'.unlazy/mtp-next/experts/metadata.json').exists():
        run('actual-capture',['target/release/expert-collect','--model',a.model,'--destination','.unlazy/mtp-next/experts'],'EXPERT_ACTIVATIONS_COLLECTED',True)
    else:
        fixture=json.loads((ROOT/'.unlazy/mtp-next/experts/metadata.json').read_text())
        assert fixture['complete'] and sha(ROOT/'.unlazy/mtp-next/experts/activations.safetensors')==fixture['fixture_sha256']
        print('Reusing captured native activations, with earlier acquisition provenance retained',flush=True)
    run('epilogue-verifier',['target/release/verify-parity','--model',a.model,'--moe-hc-epilogue','--output','results/mtp-next-epilogue-verifier.json'],'VERIFY_PARITY_PASSED',True)
    run('epilogue-rollback',['target/release/rollback-parity','--model',a.model,'--positions','8','--moe-hc-epilogue','--output','results/mtp-next-epilogue-rollback.json'],'ROLLBACK_PARITY_PASSED',True)
    run('epilogue-components-metal',['target/release/epilogue-bench','--fixture','.unlazy/mtp-next/experts','--output','results/mtp-next-epilogue-components-metal.json','--repetitions','2'],'EPILOGUE_COMPONENT_PARITY_AND_TIMING_COMPLETED',True)
    run('epilogue-components',['target/release/epilogue-bench','--fixture','.unlazy/mtp-next/experts','--output','results/mtp-next-epilogue-components.json'],'EPILOGUE_COMPONENT_PARITY_AND_TIMING_COMPLETED')
    for suite,prompts in [('raw','results/raw-prompt.json'),('chat','results/workload-prompts.json')]:
        values=json.loads((ROOT/prompts).read_text())
        for i,prompt in enumerate(values):
            command=['target/release/mtp-infer','--model',a.model,'--prompt',prompt,'--max-tokens','256','--warmup-tokens','256','--runs','4','--ignore-eos','--ab-kernel','moe-hc-epilogue','--output',f'results/mtp-next-epilogue-{suite}-{i}.json']
            if suite=='chat':command+=['--chat','--no-thinking']
            run(f'epilogue-{suite}-{i}',command,'run 4:')
    run('expert-quantization',['.venv/bin/python','scripts/screen_expert_quantization.py','--model',a.model,'--fixture','.unlazy/mtp-next/experts','--output','results/mtp-next-expert-quantization.json'],'EXPERT_QUANTIZATION_SCREEN_COMPLETED')
    run('train32',['.venv/bin/python','scripts/train_mtp_adapter.py','--collection','results/mtp-next-collection.json','--destination','.unlazy/mtp-next/adapter32','--hours','48'],'MTP_ADAPTER_TRAINING_COMPLETED')
    run('validation32',['target/release/adapter-study','--model',a.model,'--adapter','.unlazy/mtp-next/adapter32','--corpus','.unlazy/mtp-next/prompts.json','--split','validation','--depths','1,2,3,4,5,6,7','--max-tokens','64','--warmup-tokens','64','--output','results/mtp-next-validation32.json'],'ADAPTER_TARGET_IDS_AND_DEPTH_STUDY_COMPLETED')
    report=json.loads((ROOT/'results/mtp-next-validation32.json').read_text());speed=rates(report)
    chosen_adapter='adapter32';chosen_depth=max(range(1,8),key=lambda depth:speed[(True,depth)])
    if speed[(True,3)]>=speed[(False,3)]*1.05:
        remaining=max(0.,48-json.loads((ROOT/'.unlazy/mtp-next/adapter32/metadata.json').read_text())['seconds']/3600)
        if remaining>0:
            run('train64',['.venv/bin/python','scripts/train_mtp_adapter.py','--collection','results/mtp-next-collection.json','--destination','.unlazy/mtp-next/adapter64','--rank','64','--hours',str(remaining)],'MTP_ADAPTER_TRAINING_COMPLETED')
            run('validation64',['target/release/adapter-study','--model',a.model,'--adapter','.unlazy/mtp-next/adapter64','--corpus','.unlazy/mtp-next/prompts.json','--split','validation','--depths','1,2,3,4,5,6,7','--max-tokens','64','--warmup-tokens','64','--output','results/mtp-next-validation64.json'],'ADAPTER_TARGET_IDS_AND_DEPTH_STUDY_COMPLETED')
            other=rates(json.loads((ROOT/'results/mtp-next-validation64.json').read_text()));depth=max(range(1,8),key=lambda d:other[(True,d)])
            if other[(True,depth)]>speed[(True,chosen_depth)]:chosen_adapter='adapter64';chosen_depth=depth
    choice=dict(adapter=chosen_adapter,depth=chosen_depth,criterion='validation total emitted tokens after first / total decode wall, fixed depth; blind and external timing not used for selection',source_sha256=snapshot)
    (ROOT/'results/mtp-next-choice.json').write_text(json.dumps(choice,indent=2)+'\n')
    artifact='.unlazy/mtp-next/'+chosen_adapter
    run('adapter-metal',['target/release/mtp-infer','--model',a.model,'--draft-adapter',artifact,'--ab-kernel','draft-adapter','--draft-depth',str(chosen_depth),'--max-tokens','64','--warmup-tokens','64','--runs','1','--ignore-eos','--output','results/mtp-next-adapter-metal.json'],'run 1:',True)
    run('blind',['target/release/adapter-study','--model',a.model,'--adapter',artifact,'--corpus','.unlazy/mtp-next/prompts.json','--split','blind','--depths',str(chosen_depth),'--max-tokens','256','--warmup-tokens','256','--runs','4','--output','results/mtp-next-blind.json'],'ADAPTER_TARGET_IDS_AND_DEPTH_STUDY_COMPLETED')
    for suite,prompts in [('raw','results/raw-prompt.json'),('chat','results/workload-prompts.json')]:
        command=['target/release/adapter-study','--model',a.model,'--adapter',artifact,'--prompts',prompts,'--depths',str(chosen_depth),'--max-tokens','256','--warmup-tokens','256','--runs','4','--ignore-eos','--output',f'results/mtp-next-adapter-{suite}.json']
        if suite=='chat':command+=['--chat']
        run(f'adapter-{suite}',command,'ADAPTER_TARGET_IDS_AND_DEPTH_STUDY_COMPLETED')
    run('quality',['scripts/check.sh'],'QUALITY_CHECKS_PASSED')
    run('portable-metal',['scripts/validate-metal.sh'],'METAL_VALIDATION_PASSED',True)
    (ROOT/'results/mtp-next-commands.json').write_text(json.dumps(dict(complete=True,choice=choice,receipts=receipts),indent=2)+'\n')
    print('MTP_NEXT_RESEARCH_PHASES_COMPLETED',flush=True)


if __name__=='__main__':main()
