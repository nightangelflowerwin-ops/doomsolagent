"""Run the complete offline suite 1000 times against frozen source files."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
os.chdir(repo);sys.path.insert(0,str(repo))
def hashes():
    return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('agent','tests') for p in sorted((repo/directory).glob('*.py'))}
frozen=hashes(); ledger=out/'1000-test-runs.jsonl'
if ledger.exists():raise SystemExit('Existing results must be preserved; choose a fresh output directory')
(out/'1000-tested-source-hashes.json').write_text(json.dumps(frozen,indent=2))
started=time.time();total=0
with ledger.open('w',encoding='utf-8') as handle:
    for seed in range(1000):
        if hashes()!=frozen:raise SystemExit('Source changed during verification')
        os.environ['STRESS_SEED']=str(seed)
        stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.discover(str(repo/'tests')))
        total+=result.testsRun
        row=dict(run=seed+1,seed=seed,tests=result.testsRun,failures=len(result.failures),
                 errors=len(result.errors),skipped=len(result.skipped),output=stream.getvalue())
        handle.write(json.dumps(row)+'\n');handle.flush()
        summary=dict(completed_runs=seed+1,requested_runs=1000,test_executions=total,
                     elapsed_seconds=time.time()-started,scope='Offline tests; gameplay requires separate engine evaluation')
        (out/'1000-test-summary.json').write_text(json.dumps(summary,indent=2))
        if not result.wasSuccessful():raise SystemExit('Failure preserved; repair and repeat into a fresh directory')
        if (seed+1)%25==0:print(json.dumps(summary),flush=True)
