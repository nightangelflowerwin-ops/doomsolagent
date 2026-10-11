"""CPU LoRA training with real images, full corpus coverage and paired holdout evaluation."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,json,time,random,sqlite3,hashlib,math,contextlib
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor
from peft import LoraConfig,get_peft_model,PeftModel
from PIL import Image
parser=argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--resume', type=Path)
args=parser.parse_args()
BASE=args.dataset.resolve(); OUT=args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
PUBLIC=OUT
torch.set_num_threads(4); torch.manual_seed(7); random.seed(7)
from agent.training_dataset import records, validate
validate(BASE)
db=sqlite3.connect(':memory:')
db.execute('create table examples(id integer primary key, split text, domain text, payload text)')
for split in ('train', 'validation'):
    for row in records(BASE, split, verify_images=False):
        if row.get('image'): row['image']=str(BASE/row['image'])
        db.execute('insert into examples values(?,?,?,?)',(row['id'],split,row['domain'],json.dumps(row)))
db.commit()
audit=json.loads((BASE/'manifest.json').read_text())
started=time.time();progress={'stage':'loading','pid':os.getpid(),'backend':'Transformers/PEFT CPU reference backend; Unsloth published base model','source_audit':'manifest.json','claim':'No gameplay improvement established.'}
def status(**fields):
    progress.update(fields,updated_at=time.time())
    text=json.dumps(progress,indent=2)
    (OUT/'status.json').write_text(text,encoding='utf-8');(PUBLIC/'local-training-status.json').write_text(text,encoding='utf-8')
    print(json.dumps(fields),flush=True)
status()
base_model=args.model.resolve()
model=AutoModelForImageTextToText.from_pretrained(str(base_model),dtype=torch.float32,device_map='cpu',local_files_only=True,attn_implementation='eager')
processor=AutoProcessor.from_pretrained(str(base_model),local_files_only=True)
if args.resume:model=PeftModel.from_pretrained(model,str(args.resume/'adapter'),is_trainable=True)
else:model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',target_modules='all-linear'))
trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
assert any('visual' in n for n in trainable) and any('language' in n for n in trainable)
(OUT/'trainable-parameters.json').write_text(json.dumps({n:list(p.shape) for n,p in trainable.items()},indent=2))
optimizer=torch.optim.AdamW(list(trainable.values()),lr=2e-4,weight_decay=.01)
training_ids=[r[0] for r in db.execute("select id from examples where split='train' order by id")]
visual_ids=[r[0] for r in db.execute("select id from examples where split='train' and domain like '%gameplay_image' order by id")]
# Every eligible training row is consumed once, followed by a single image-only
# consolidation pass so the much larger engine corpus does not define the final
# deployment input. Neither phase has a maximum-step cap.
random.Random(7).shuffle(training_ids);random.Random(8).shuffle(visual_ids)
# Exercise each input schema in the first real accumulation batch while still
# consuming each training example exactly once in this pass.
first_ids=[r[0] for r in db.execute("select min(id) from examples where split='train' group by domain order by domain")]
first_set=set(first_ids);training_ids=first_ids+[identity for identity in training_ids if identity not in first_set]
assert len(training_ids)==len(set(training_ids))
phases=[('full_corpus_epoch',training_ids),('image_consolidation_epoch',visual_ids)]
schedule={'phases':[{'name':n,'examples':len(ids),'passes':1} for n,ids in phases],
 'gradient_accumulation':4,'learning_rate':2e-4,'adapter_rank':8,'adapter_alpha':16,
 'trainable_parameters':sum(p.numel() for p in trainable.values()),'model_revision':'23c69c53358a07516b5827588b3fdb12ae78fd65',
 'eval':'After training: every held-out row, paired base versus adapter; assistant-token loss and deterministic JSON action output.',
 'promotion':'Requires held-out action checks and real gameplay against the current controller on fixed seeds; no automatic promotion.'}
(OUT/'schedule.json').write_text(json.dumps(schedule,indent=2));(PUBLIC/'faithful-training-plan.json').write_text(json.dumps(schedule,indent=2))
audit_hash=hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest()
state={'phase':0,'cursor':0,'updates':0,'examples_seen':0,'domains_seen':{},'audit_sha256':audit_hash}
if args.resume:
    state=json.loads((args.resume/'state.json').read_text())
    assert state['audit_sha256']==audit_hash, 'Corpus changed; do not resume against different examples'
    optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location='cpu',weights_only=True))
    torch.set_rng_state(torch.load(args.resume/'rng.pt',weights_only=True))
def row_for(identity):return json.loads(db.execute('select payload from examples where id=?',(identity,)).fetchone()[0])
def encode(row):
    if row['domain']=='doomsol_gameplay_image':
        with Image.open(row['image']) as im:image=im.convert('RGB').copy()
        prompt='Imitate the recorded DoomSol controls from this gameplay image. Return only JSON: move_x (A=-1,D=1), move_y (W=-1,S=1), look_x (mouse dx/180), look_y (mouse dy/120), axes clipped to [-1,1]; fire, use (Space), map (Tab), escape as 0 or 1. These are recorded actions, not optimal-policy labels.'
        content=[{'type':'text','text':prompt},{'type':'image','image':image}]
    elif row['domain']=='dwasm_hud_grounding':
        with Image.open(row['image']) as im:image=im.convert('RGB').copy()
        prompt='Read the currently displayed Doom health and armor from this rendered gameplay image. Return only JSON with numeric health and armor values. This task is HUD perception, not choosing an action.'
        content=[{'type':'text','text':prompt},{'type':'image','image':image}]
    elif row['domain']=='vizdoom_policy_gameplay_image':
        with Image.open(row['image']) as im:image=im.convert('RGB').copy()
        prompt='Imitate the recorded single-button ViZDoom policy action from this engine gameplay image. Return only JSON with an actions list containing one of: turn left, turn right, move forward, move backward, strafe left, strafe right, attack. This is an older policy demonstration, not an optimal or expert label.'
        content=[{'type':'text','text':prompt},{'type':'image','image':image}]
    elif row['domain']=='engine_observation_and_map_memory':
        prompt='Auxiliary ViZDoom engine and map-memory task. This observation includes privileged engine data. Imitate the recorded controller action list; return only JSON with an actions list. This is not a camera-only task or an optimal-policy label.\n'+json.dumps(row['state'],sort_keys=True)
        content=[{'type':'text','text':prompt}]
    else:
        content=[{'type':'text','text':'Choose the recorded browser candidate controller action (none, aim, fire). Return only JSON with action.\n'+json.dumps(row['state'],sort_keys=True)}]
    user=[{'role':'user','content':content}]
    complete=user+[{'role':'assistant','content':[{'type':'text','text':json.dumps(row['target'],sort_keys=True)}]}]
    inputs=processor.apply_chat_template(complete,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
    prefix=processor.apply_chat_template(user,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False)
    n=prefix['input_ids'].shape[1]
    assert torch.equal(inputs['input_ids'][:,:n],prefix['input_ids']), 'Assistant mask prefix mismatch'
    labels=inputs['input_ids'].clone();labels[:,:n]=-100
    assert (labels!=-100).sum()>0
    if inputs['input_ids'].shape[1]>4096:raise ValueError('Record exceeds validated context; inspect provenance rather than truncate')
    return inputs,labels,prefix
def checkpoint():
    from agent.training_checkpoint import publish_checkpoint
    def writer(folder):
        model.save_pretrained(str(folder/'adapter'));processor.save_pretrained(str(folder/'adapter'))
        torch.save(optimizer.state_dict(),folder/'optimizer.pt');torch.save(torch.get_rng_state(),folder/'rng.pt')
        (folder/'state.json').write_text(json.dumps(state,indent=2))
    folder=publish_checkpoint(OUT/'checkpoints',f"update-{state['updates']:08d}",writer)
    status(checkpoint=str(folder.resolve()))
def evaluation():
    model.eval();metrics={};count=0
    target_path=OUT/'paired-holdout-predictions.jsonl'
    with target_path.open('w',encoding='utf-8') as handle,torch.inference_mode():
        for identity, in db.execute("select id from examples where split='validation' order by id"):
            row=row_for(identity);inputs,labels,prefix=encode(row);result={'id':identity,'domain':row['domain'],'target':row['target'],'provenance':row['provenance']}
            for variant in ('base','trained'):
                context=model.disable_adapter() if variant=='base' else contextlib.nullcontext()
                with context:
                    loss=model(**inputs,labels=labels,use_cache=False).loss.item()
                    generated=model.generate(**prefix,max_new_tokens=256,do_sample=False,use_cache=True)
                    text=processor.decode(generated[0,prefix['input_ids'].shape[1]:],skip_special_tokens=True)
                try:
                    pred=json.loads(text);valid=isinstance(pred,dict) and set(pred)==set(row['target'])
                    if valid and row['domain']=='doomsol_gameplay_image':
                        valid=all(isinstance(pred[k],(int,float)) and not isinstance(pred[k],bool) and math.isfinite(pred[k]) and -1<=pred[k]<=1 for k in ('move_x','move_y','look_x','look_y')) and all(pred[k] in (0,1) for k in ('fire','use','map','escape'))
                    elif valid and 'actions' in row['target']:valid=isinstance(pred['actions'],list) and all(isinstance(a,str) for a in pred['actions'])
                    elif valid and row['domain']=='dwasm_hud_grounding':valid=all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in pred.values())
                    elif valid:valid=pred['action'] in ('none','aim','fire')
                except (ValueError,TypeError):pred=None;valid=False
                key=row['domain']+':'+variant;m=metrics.setdefault(key,{'rows':0,'loss_sum':0.,'valid_json':0,'exact_actions':0,'axis_abs_error_sum':0.,'axis_values':0,'button_confusion':{k:{'tp':0,'fp':0,'fn':0,'tn':0} for k in ('fire','use','map','escape')}})
                m['rows']+=1;m['loss_sum']+=loss;m['valid_json']+=valid;m['exact_actions']+=pred==row['target']
                if valid and row['domain']=='doomsol_gameplay_image':
                    for button in ('fire','use','map','escape'):
                        expected=bool(row['target'][button]);actual=bool(pred[button]);cell='tp' if expected and actual else 'fn' if expected else 'fp' if actual else 'tn';m['button_confusion'][button][cell]+=1
                    for axis in ('move_x','move_y','look_x','look_y'):
                        value=pred.get(axis)
                        if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):m['axis_abs_error_sum']+=abs(value-row['target'][axis]);m['axis_values']+=1
                result[variant]={'loss':loss,'prediction':pred,'raw_output':text}
            handle.write(json.dumps(result)+'\n');handle.flush();count+=1
            if count%10==0:status(stage='full_holdout_evaluation',evaluated_rows=count,metrics=metrics)
    result={'metrics':metrics,'evaluated_rows':count,'training_coverage':state,'model_revision':schedule['model_revision'],'corpus_audit_sha256':audit_hash,'gameplay_validation':'pending; model not promoted'}
    text=json.dumps(result,indent=2);(OUT/'evaluation.json').write_text(text);(PUBLIC/'faithful-training-evaluation.json').write_text(text)
try:
    model.train();seen=Counter(state['domains_seen']);optimizer.zero_grad(set_to_none=True)
    for phase_index,(phase,identities) in enumerate(phases):
        if phase_index<state['phase']:continue
        cursor=state['cursor'] if phase_index==state['phase'] else 0
        while cursor<len(identities):
            batch_ids=identities[cursor:cursor+4];losses=[]
            before_norm=sum(p.detach().float().norm().item() for n,p in trainable.items() if 'lora_B' in n)
            vision_before=sum(p.detach().float().norm().item() for n,p in trainable.items() if 'lora_B' in n and 'visual' in n)
            for identity in batch_ids:
                row=row_for(identity);inputs,labels,_=encode(row)
                loss=model(**inputs,labels=labels,use_cache=False).loss
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no update applied')
                losses.append(loss.item());(loss/len(batch_ids)).backward();seen[row['domain']]+=1
            norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; no update applied')
            # A smooth decay over each complete pass, never a step-budget cutoff.
            fraction=cursor/max(1,len(identities))
            learning_rate=2e-4*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=learning_rate
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            cursor+=len(batch_ids);state.update(phase=phase_index,cursor=cursor,updates=state['updates']+1,examples_seen=state['examples_seen']+len(batch_ids),domains_seen=dict(seen))
            after_norm=sum(p.detach().float().norm().item() for n,p in trainable.items() if 'lora_B' in n)
            vision_after=sum(p.detach().float().norm().item() for n,p in trainable.items() if 'lora_B' in n and 'visual' in n)
            status(stage='training',phase_name=phase,**state,phase_examples=len(identities),loss=sum(losses)/len(losses),gradient_norm=float(norm),learning_rate=learning_rate,adapter_norm_change=after_norm-before_norm,vision_adapter_norm_change=vision_after-vision_before,elapsed_seconds=time.time()-started)
            with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            if state['updates']==1 or state['updates']%100==0 or cursor==len(identities):checkpoint()
        state.update(phase=phase_index+1,cursor=0)
    model.save_pretrained(str(OUT/'adapter'));processor.save_pretrained(str(OUT/'adapter'))
    status(stage='full_holdout_evaluation');evaluation();status(stage='training_and_imitation_evaluation_complete',gameplay_validation='pending; model not promoted')
except BaseException as error:
    status(stage='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),resume_from=(OUT/'latest-checkpoint.txt').read_text() if (OUT/'latest-checkpoint.txt').exists() else None)
    raise
