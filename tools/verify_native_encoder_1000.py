"""Check all five task encodings 1000 times with a real local Qwen processor."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
from agent.training_dataset import records
root=args.dataset.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
ledger=out/'1000-native-encoder-runs.jsonl'
if ledger.exists():raise SystemExit('Choose a fresh output directory to preserve earlier results')
torch.set_num_threads(2)
processor=AutoProcessor.from_pretrained(str(args.model.resolve()),local_files_only=True)
source=repo/'agent/train_corpus_model.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encode')
scope=dict(processor=processor,torch=torch,Image=Image,json=json)
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
all_rows=sorted((row for split in ('train','validation') for row in records(root,split)),key=lambda row:row['id'])
domains=sorted({row['domain'] for row in all_rows})
selected={domain:[row for row in all_rows if row['domain']==domain][:1000] for domain in domains}
started=time.time()
with ledger.open('w',encoding='utf-8') as handle:
    for run in range(1000):
        checks=[]
        for domain in domains:
            row=dict(selected[domain][run%len(selected[domain])])
            digest=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            if row.get('image'):row['image']=str(root/row['image'])
            inputs,labels,prefix=scope['encode'](row)
            n=prefix['input_ids'].shape[1]
            assert torch.all(labels[:,:n]==-100) and torch.all(labels[:,n:]!=-100)
            checks.append(dict(domain=domain,id=row['id'],input_tokens=inputs['input_ids'].shape[1],
                assistant_tokens=int((labels!=-100).sum()),example_sha256=digest))
        handle.write(json.dumps(dict(run=run+1,checks=checks))+'\n');handle.flush()
        summary=dict(completed_runs=run+1,requested_runs=1000,native_domain_checks=(run+1)*len(domains),
            elapsed_seconds=time.time()-started,scope='Real local processor; no model-forward or gameplay claim')
        (out/'1000-native-encoder-summary.json').write_text(json.dumps(summary,indent=2))
        if (run+1)%25==0:print(json.dumps(summary),flush=True)
