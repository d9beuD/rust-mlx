#!/usr/bin/env python3
"""Frozen-source down-format diagnostic and independent distillation pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from run_decode3_checks import core_digest,VALIDATION

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
 snapshot=core_digest();receipts=[]
 cases=[('vector-format-metal',['target/release/down-bench','--model',a.model,'--output','results/followup-vector-format-metal.json','--allow-inexact'],'DOWN_FORMAT_COMPONENT_STUDY_COMPLETED',True)]
 for name,cmd,marker,validation in cases:
  output=ROOT/cmd[cmd.index('--output')+1];log=ROOT/f'results/followup-{name}-driver.log'
  assert not output.exists(),f'preserve{output}'
  start=time.time()
  with log.open('wb') as f:r=subprocess.run(cmd,cwd=ROOT,env={**os.environ,**(VALIDATION if validation else {})},stdout=f,stderr=subprocess.STDOUT)
  receipt={'command':cmd,'name':name,'started_unix':start,'ended_unix':time.time(),'source_sha256':snapshot,'executable_sha256':sha(ROOT/cmd[0]),'script_sha256':sha(ROOT/cmd[1]) if cmd[0]=='.venv/bin/python' else None,'exit_code':r.returncode,'validation_environment':VALIDATION if validation else {},'marker':marker,'completed':r.returncode==0 and marker in log.read_text(errors='replace'),'report_sha256':sha(output) if output.exists() else None,'log_sha256':sha(log),'timing_qualified':False,'warning':'allow-inexact gathers numerical failures; study completion does not certify the down kernel. Instrumented component rates are diagnostic only.' if name=='vector-format-metal' else 'teacher-forced distillation data/training; not natural MTP throughput'}
  (ROOT/f'results/followup-{name}-command.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
  assert snapshot==core_digest(),'source changed during pilot'
  (ROOT/'results/followup-vector-progress.json').write_text(json.dumps({'source_sha256':snapshot,'receipts':receipts,'complete':len(receipts)==len(cases)},indent=2)+'\n')
  print(name,receipt['completed'],flush=True)
  if not receipt['completed']:raise SystemExit(r.returncode or 1)
 print('FOLLOWUP_PILOT_PROGRAMS_COMPLETED')

if __name__=='__main__':main()
