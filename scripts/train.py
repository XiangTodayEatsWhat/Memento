"""Launch the documented training recipe with explicit local data/model paths."""
import argparse,json,subprocess,sys
from pathlib import Path
from common import ROOT,save_json,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    p.add_argument('--llm',default='meta-llama/Llama-3.1-8B-Instruct');p.add_argument('--vision',default='google/siglip-large-patch16-384');p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--port',type=int,default=29500)
    p.add_argument('--config',default=str(ROOT/'configs/train.json'));p.add_argument('--max-steps',type=int);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not Path(a.annotations).is_file() or not Path(a.features).is_dir():raise ValueError('Missing annotation file or feature directory')
    config=json.loads(Path(a.config).read_text());output=Path(a.output).resolve()
    if a.max_steps is not None:config['max_steps']=a.max_steps
    gpu_ids=a.gpus.split(',')
    if not gpu_ids or any(not s.isdigit() for s in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):raise ValueError('Invalid GPU list')
    cmd=[sys.executable,'-m','deepspeed.launcher.runner','--include','localhost:'+a.gpus,'--master_port',str(a.port),str(ROOT/'train.py'),'--deepspeed',str(ROOT/'configs/deepspeed/zero2.json')]
    for k,v in dict(config,llm_pretrained=a.llm,vision_pretrained=a.vision,embed_path=str(Path(a.features).resolve()),train_datasets=str(Path(a.annotations).resolve()),output_dir=str(output)).items():
        cmd+=['--'+k,str(v)]
    print(json.dumps(cmd),flush=True)
    if a.dry_run:return
    save_json(output/'run_config.json',{'training':config,'base_model':a.llm,'vision_model':a.vision,'world_size':len(gpu_ids),'annotations_sha256':digest(a.annotations)})
    subprocess.run(cmd,cwd=ROOT,check=True)
    if not (output/'adapter_model.safetensors').is_file():raise RuntimeError('Training finished without adapter weights')
    save_json(output/'training_complete.json',{'complete':True,'weights_sha256':digest(output/'adapter_model.safetensors'),'max_steps_override':a.max_steps})
if __name__=='__main__':main()
