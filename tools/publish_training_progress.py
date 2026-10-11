"""Publish meaningful local training progress under the configured human author."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

FIELDS=('stage','backend','phase_name','phase','cursor','updates','examples_seen',
        'domains_seen','phase_examples','loss','gradient_norm','learning_rate',
        'elapsed_seconds','audit_sha256','updated_at')

def public_report(status):
    report={key:status[key] for key in FIELDS if key in status}
    report['recorded_date']=datetime.now(timezone.utc).date().isoformat()
    report['claim']='Training progress only; no gameplay improvement established.'
    return report

def should_publish(previous,current):
    if previous is None:return True
    if previous.get('updated_at')==current.get('updated_at'):return False
    if previous.get('stage')!=current.get('stage') or previous.get('phase')!=current.get('phase'):return True
    advanced=current.get('updates',0)>previous.get('updates',0)
    return advanced and (previous.get('recorded_date')!=current.get('recorded_date')
        or current.get('examples_seen',0)//10000>previous.get('examples_seen',0)//10000)

def publish(repo,status,repository):
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,capture_output=True,check=True,cwd=repo)
    values=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    token=values['password']
    def api(path,method='GET',data=None):
        request=urllib.request.Request('https://api.github.com/'+path,
            data=json.dumps(data).encode() if data is not None else None,method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)
    identity=api('user')
    email=subprocess.check_output(['git','config','user.email'],cwd=repo,text=True).strip()
    name=subprocess.check_output(['git','config','user.name'],cwd=repo,text=True).strip()
    try:
        verified=api('user/emails')
        associated=any(item['email']==email and item['verified'] for item in verified)
    except urllib.error.HTTPError as error:
        if error.code!=403:raise
        # A repository token may lack email access. Verify the exact configured
        # email against an existing commit already associated with this account.
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        commit=api(f'repos/{repository}/commits/{head}')
        associated=bool(commit.get('author') and commit['author']['login']==identity['login']
            and commit['commit']['author']['email']==email)
    if not associated:
        raise ValueError('Git author email is not associated with the authenticated account')
    endpoint=f'repos/{repository}/contents/training/progress.json'
    try:
        existing=api(endpoint+'?ref=main')
        previous=json.loads(base64.b64decode(existing['content']))
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        existing=None;previous=None
    current=public_report(status)
    if not should_publish(previous,current):return {'published':False,'reason':'No new daily progress or milestone'}
    body={'branch':'main','message':f"Record training progress at {current.get('examples_seen',0)} examples",
        'content':base64.b64encode((json.dumps(current,indent=2,allow_nan=False)+'\n').encode()).decode(),
        'author':{'name':name,'email':email},'committer':{'name':name,'email':email}}
    if existing:body['sha']=existing['sha']
    saved=api(endpoint,'PUT',body)
    commit=api(f"repos/{repository}/commits/{saved['commit']['sha']}")
    if not commit.get('author') or commit['author']['login']!=identity['login']:
        raise ValueError('Published commit author did not resolve to the authenticated account')
    return {'published':True,'sha':saved['commit']['sha'],'author':identity['login'],'url':saved['commit']['html_url']}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--repository',default='nightangelflowerwin-ops/doomsolagent')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    while True:
        try:
            status=json.loads(args.status.read_text())
            print(json.dumps(publish(repo,status,args.repository)),flush=True)
        except Exception as error:
            # Credential values never enter log messages.
            print(json.dumps({'published':False,'error_type':type(error).__name__}),flush=True)
            if not args.watch:raise
        if not args.watch:break
        time.sleep(60)

if __name__=='__main__':main()
