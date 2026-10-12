import argparse,ctypes,datetime,hashlib,json,msvcrt,os,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser(description='Keep local Windows DoomSol training awake and recover an exited trainer.')
parser.add_argument('--task-root',type=Path,required=True,help='Existing local training workspace containing work and outputs folders')
args=parser.parse_args()
ROOT=args.task_root.resolve();OUT=ROOT/'outputs';WORK=ROOT/'work';TRAIN=WORK/'train_faithful_model.py';CHECKPOINTS=WORK/'faithful-training-v6/checkpoints'
LOG=OUT/'Training-Watchdog.json';STOP=OUT/'Stop-Training-Watchdog.flag'
def report(**values):
 values['checked_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-7))).isoformat()
 temporary=LOG.with_suffix('.pending');temporary.write_text(json.dumps(values,indent=2));os.replace(temporary,LOG)
def checkpoint():
 audit=hashlib.sha256((WORK/'faithful-corpus-v6/audit.json').read_bytes()).hexdigest()
 for folder in sorted(CHECKPOINTS.glob('update-*'),reverse=True):
  try:
   state=json.loads((folder/'state.json').read_text())
   required=['optimizer.pt','rng.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json']
   if state['audit_sha256']==audit and all((folder/x).is_file() and (folder/x).stat().st_size>0 for x in required):return folder
  except (OSError,ValueError,KeyError):continue
 raise RuntimeError('No complete compatible checkpoint; preserve the run for inspection')
def active():
 env=dict(os.environ,DOOMSOL_TRAINING_SCRIPT=str(TRAIN))
 command="@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($env:DOOMSOL_TRAINING_SCRIPT) } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
 result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],env=env,capture_output=True,text=True,timeout=45,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
 value=json.loads(result.stdout) if result.stdout.strip() else []
 return value if isinstance(value,list) else [value]
def main():
 lock=(WORK/'training-watchdog.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:return
 last_launch=0;restarts=0;child=None
 try:
  while not STOP.exists():
   if not ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001)):raise OSError('Windows sleep prevention request failed')
   try:
    state=json.loads((OUT/'local-training-status.json').read_text())
    if state.get('stage')=='training_and_imitation_evaluation_complete':
     report(state='completed',automatic_restarts=restarts,sleep_prevention=False);return
    pids=active()
    if pids:
     report(state='monitoring',trainer_pids=pids,automatic_restarts=restarts,training_update=state.get('updates'),sleep_prevention=True)
    elif time.time()-last_launch>=300:
     saved=checkpoint();stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
     with (WORK/f'watchdog-training-{stamp}.log').open('ab') as output,(WORK/f'watchdog-training-{stamp}-errors.log').open('ab') as error:
      child=subprocess.Popen([str(WORK/'training-env/Scripts/python.exe'),'-u',str(TRAIN),'--resume',str(saved),'--threads','7'],cwd=ROOT,stdout=output,stderr=error,creationflags=subprocess.CREATE_NO_WINDOW)
     last_launch=time.time();restarts+=1
     report(state='restarting',launcher_pid=child.pid,resume_checkpoint=saved.name,automatic_restarts=restarts,sleep_prevention=True)
    else:report(state='retry_backoff',automatic_restarts=restarts,sleep_prevention=True)
   except Exception as error:report(state='inspection_error',error_type=type(error).__name__,automatic_restarts=restarts,sleep_prevention=True)
   time.sleep(60)
 finally:
  ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000));lock.close()
if __name__=='__main__':main()
