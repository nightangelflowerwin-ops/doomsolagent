"""Publish changed saved CVAT snapshots from this local Docker installation."""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def run(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def fetch(job):
    # Explicit allowlist: no account records, contact information, or local paths.
    code = f'''from cvat.apps.engine.models import Job
from cvat.apps.dataset_manager.task import get_job_data
from django.db import transaction
import json
with transaction.atomic():
 j=Job.objects.select_for_update().get(id={job})
 labels=[dict(id=l.id,name=l.name,type=l.type,color=l.color,attributes=[dict(id=a.id,name=a.name,mutable=a.mutable,input_type=a.input_type,default_value=a.default_value,values=a.values) for a in l.attributespec_set.all()]) for l in j.segment.task.label_set.all()]
 payload=dict(job_id=j.id,task_id=j.segment.task_id,cvat_saved_at=j.updated_date.isoformat(),labels=labels,annotations=get_job_data(j.id))
 print('CVAT_SYNC_JSON='+json.dumps(payload))
'''
    output = run(['wsl','-d','Ubuntu','-u','root','--','docker','exec','cvat_server','python','manage.py','shell','-c',code])
    return json.loads(output.split('CVAT_SYNC_JSON=', 1)[1].strip())


def records(payload):
    result = {}
    for track in payload['annotations']['tracks']:
        result[f"track:{track['id']}"] = {k:v for k,v in track.items() if k != 'shapes'}
        for shape in track['shapes']:
            result[f"track:{track['id']}:frame:{shape['frame']}"] = shape
    for kind in ('shapes','tags','intervals'):
        for item in payload['annotations'].get(kind, []):
            result[f"{kind}:{item['id']}"] = item
    return result


def sync(repo, job):
    # Never overwrite unrelated local changes or force-push divergent history.
    if run(['git','status','--porcelain'],repo).strip():
        raise RuntimeError('Checkout has uncommitted changes; refusing to mix them into sync.')
    pending = int(run(['git','rev-list','--count','origin/main..HEAD'],repo).strip())
    if pending:
        run(['git','push','origin','main'],repo)
    run(['git','pull','--ff-only','origin','main'],repo)
    payload = fetch(job)
    folder = repo / 'annotations/cvat/e1m1-mission/live'
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.json'
    old = json.loads(latest.read_text()) if latest.exists() else None
    stable = {k:v for k,v in payload.items() if k != 'cvat_saved_at'}
    digest = hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    old_stable = {k:v for k,v in old.items() if k != 'cvat_saved_at'} if old else None
    if old_stable == stable:
        return 'No saved annotation changes.'
    before, after = records(old) if old else {}, records(payload)
    event = dict(observed_at=datetime.now(timezone.utc).isoformat(),cvat_saved_at=payload['cvat_saved_at'],sha256=digest,
                 baseline=old is None,added=len(after.keys()-before.keys()),removed=len(before.keys()-after.keys()),
                 changed=sum(before[k]!=after[k] for k in before.keys() & after.keys()),
                 tracks=len(payload['annotations']['tracks']),keyframes=sum(len(t['shapes']) for t in payload['annotations']['tracks']))
    frames=[s['frame'] for t in payload['annotations']['tracks'] for s in t['shapes']]
    event['latest_keyframe']=max(frames) if frames else None
    latest.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    history=folder/'checkpoints.json'
    events=json.loads(history.read_text()) if history.exists() else []
    events.append(event)
    history.write_text(json.dumps(events,indent=2)+'\n',encoding='utf-8')
    run(['git','add',str(latest.relative_to(repo)),str(history.relative_to(repo))],repo)
    run(['git','diff','--cached','--check'],repo)
    run(['git','commit','-m',f"CVAT saved checkpoint: {event['tracks']} tracks, {event['keyframes']} keyframes"],repo)
    run(['git','push','origin','main'],repo)
    return json.dumps(event)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=int,default=3)
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    while True:
        try:
            print(datetime.now(timezone.utc).isoformat(),sync(repo,args.job),flush=True)
        except Exception as exc:
            print(datetime.now(timezone.utc).isoformat(),'Sync error:',str(exc),flush=True)
            if args.once:
                raise SystemExit(1)
        if args.once:
            break
        time.sleep(max(15,args.interval))
