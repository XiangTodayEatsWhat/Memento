"""Paper-aligned streaming runner. Complete outputs are written atomically; failures are not scores."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import set_seed
from models.arguments_live import LiveMemoryTrainingArguments
from inference import LiveMemInfer

from inference.streaming import process_frame, update_threshold

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eval_json',required=True); p.add_argument('--embed_path',required=True)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--llm',required=True); p.add_argument('--vision',required=True)
    p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int); p.add_argument('--max_frames',type=int)
    p.add_argument('--qms_ratio',type=float,default=None); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); set_seed(a.seed)
    if a.shards<1 or not 0<=a.shard<a.shards:raise ValueError('Invalid shard selection')
    config_path=Path(a.checkpoint,'memento_config.json')
    saved_config=json.loads(config_path.read_text()) if config_path.is_file() else {}
    live_config={k:saved_config[k] for k in ['frame_fps','pt_size','eps','attention_update_ratio','qms_ratio','mem_length','cutoff_len'] if k in saved_config}
    if a.qms_ratio is not None:live_config['qms_ratio']=a.qms_ratio
    assert Path(a.checkpoint,'adapter_config.json').exists(), 'Missing trained adapter'
    dst=Path(a.output); (dst/'infer_results').mkdir(parents=True,exist_ok=True)
    items=json.load(open(a.eval_json)); meta=json.load(open(a.embed_path.rstrip('/')+'_metadata.json'))
    opts=LiveMemoryTrainingArguments(output_dir=str(dst),llm_pretrained=a.llm,vision_pretrained=a.vision,
      resume_from_checkpoint=a.checkpoint,report_to='none',bf16=True,**live_config)
    infer=LiveMemInfer(torch.device('cuda:0'),args=opts,infer_output_path=str(dst))
    checkpoint_sha=hashlib.file_digest(open(Path(a.checkpoint,'adapter_model.safetensors'),'rb'),'sha256').hexdigest()
    code_sha=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).resolve().parents[1].glob('models/**/*.py')))+Path(__file__).read_bytes()+(Path(__file__).resolve().parents[1]/'inference/engine.py').read_bytes()).hexdigest()
    code_sha=hashlib.sha256((code_sha+json.dumps(live_config,sort_keys=True)).encode()).hexdigest()
    completed=0
    for idx,item in enumerate(items):
        if idx%a.shards!=a.shard: continue
        if a.limit is not None and completed>=a.limit: break
        item_sha=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
        path=dst/'infer_results'/f'{idx}.json'
        if path.exists():
            old=json.load(open(path))
            if old.get('complete') and old.get('input_sha256')==item_sha and old.get('checkpoint_sha256')==checkpoint_sha and old.get('code_sha256')==code_sha:
                completed+=1; continue
            raise RuntimeError(f'Existing partial or incompatible result {path}')
        infer.reset(); uid=item['video_uid']
        duration=meta[uid]['duration']
        infer.load_embed(str(Path(a.embed_path,uid+'.pt')),0,duration)
        queries=[x for x in item['conversation'] if x['role']=='user']
        for q in queries: infer.input_query_stream(q['content'],video_time=q['time'])
        n=infer.num_video_frames; cap=min(n,a.max_frames) if a.max_frames is not None else n
        history=[]; state={'consecutive':0,'silent':0,'threshold':0.5}; qi=0
        started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for frame in range(cap):
                t=frame/infer.frame_fps
                query,response=process_frame(infer,t,infer.input_embed_stream,state)
                if query is not None:
                    history.append({'role':'user','content':query,'time':queries[qi]['time']}); qi+=1
                if response:
                    history.append({'role':'assistant','content':response,'time':t})
                if frame%200==0:
                    print(json.dumps({'sample':idx,'frame':frame,'total':cap,'video_time':t,'seconds':round(time.monotonic()-started,2),'responses':sum(x['role']=='assistant' for x in history),'memory_slots':len(infer.long_memory_buffer)}),flush=True)
        result={'test_id':idx,'video_uid':uid,'input_sha256':item_sha,'checkpoint_sha256':checkpoint_sha,'code_sha256':code_sha,'frame_fps':infer.frame_fps,'conversation':history,
          'complete':cap==n,'processed_frames':cap,'expected_frames':n,'checkpoint':str(Path(a.checkpoint).resolve()),
          'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
          'paper_threshold':{'initial':0.5,'raise_after':10,'raise_by':0.1,'reset_after_silent':30}}
        tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); os.replace(tmp,path)
        print(json.dumps({'saved':str(path),'complete':cap==n}),flush=True)
        completed+=1
    print(json.dumps({'finished_shard':a.shard,'completed':completed}),flush=True)
if __name__=='__main__': main()
