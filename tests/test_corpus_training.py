"""Exercise the actual training encoder without allocating the base model."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image

def encoder(processor):
    source=Path(__file__).resolve().parents[1]/'agent/train_corpus_model.py'
    tree=ast.parse(source.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='encode')
    scope=dict(processor=processor,torch=torch,Image=Image,json=json)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
    return scope['encode']

class ProcessorStub:
    def __init__(self,mode='normal'): self.mode=mode; self.messages=[]
    def apply_chat_template(self,messages,**kwargs):
        self.messages.append(messages)
        complete=len(messages)==2
        prefix=[1,2,3]
        if self.mode=='mismatch' and complete: prefix=[9,2,3]
        ids=prefix+([4,5] if complete and self.mode!='empty' else [])
        if self.mode=='overflow' and complete: ids=prefix+[4]*4097
        return {'input_ids':torch.tensor([ids])}

class CorpusEncoderTests(unittest.TestCase):
    def test_all_five_domains_mask_only_assistant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frame.png'; Image.new('RGB',(4,4),'red').save(path)
            for domain in ('doomsol_gameplay_image','dwasm_hud_grounding',
                           'vizdoom_policy_gameplay_image','engine_observation_and_map_memory',
                           'browser_candidate_observation'):
                processor=ProcessorStub()
                inputs,labels,prefix=encoder(processor)(dict(domain=domain,image=str(path),state={},target={'test':1}))
                self.assertEqual(labels.tolist(),[[-100,-100,-100,4,5]])
                self.assertTrue(torch.equal(prefix['input_ids'],inputs['input_ids'][:,:3]))
                content=processor.messages[0][0]['content']
                self.assertEqual(any(c['type']=='image' for c in content),domain.endswith('image') or domain=='dwasm_hud_grounding')
                self.assertNotIn('test',content[0]['text'])

    def test_prefix_mismatch_rejected(self):
        with self.assertRaisesRegex(AssertionError,'prefix mismatch'):
            encoder(ProcessorStub('mismatch'))(dict(domain='browser_candidate_observation',state={},target={}))

    def test_empty_assistant_rejected(self):
        with self.assertRaises(AssertionError):
            encoder(ProcessorStub('empty'))(dict(domain='browser_candidate_observation',state={},target={}))

    def test_long_context_rejected_without_truncating(self):
        with self.assertRaisesRegex(ValueError,'rather than truncate'):
            encoder(ProcessorStub('overflow'))(dict(domain='browser_candidate_observation',state={},target={}))
