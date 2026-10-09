"""Full-video inference followed by strict temporal metrics and offline judge requests."""
import argparse,os,subprocess,sys,json
from pathlib import Path
from common import ROOT,read_rows,save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--features',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--gpus',default='0');p.add_argument('--llm',default='meta-llama/Llama-3.1-8B-Instruct');p.add_argument('--vision',default='google/siglip-large-patch16-384');a=p.parse_args()
    rows=read_rows(a.annotations);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    gt=out/'annotations.json'
    if gt.exists() and json.loads(gt.read_text())!=rows:raise ValueError('Output folder belongs to another annotation set')
    save_json(gt,rows)
    if not Path(a.features.rstrip('/')+'_metadata.json').is_file():raise FileNotFoundError('Run prepare_features.py or validate_features.py to build metadata')
    gpus=a.gpus.split(',');jobs=[]
    if any(not g.isdigit() for g in gpus) or len(set(gpus))!=len(gpus):raise ValueError('Invalid GPU list')
    try:
        for shard,gpu in enumerate(gpus):
            cmd=[sys.executable,str(ROOT/'benchmark/run_streaming.py'),'--eval_json',str(gt),'--embed_path',str(Path(a.features).resolve()),'--checkpoint',str(Path(a.checkpoint).resolve()),'--output',str(out),'--llm',a.llm,'--vision',a.vision,'--shard',str(shard),'--shards',str(len(gpus))]
            log=open(out/f'shard-{shard}.log','a');jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT),log))
        codes=[p.wait() for p,_ in jobs]
        if any(codes):raise RuntimeError(f'Inference failed: {codes}; inspect shard logs')
    finally:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
            log.close()
    subprocess.run([sys.executable,str(ROOT/'benchmark/evaluate.py'),'--gt',str(gt),'--results',str(out/'infer_results'),'--output',str(out/'metrics')],check=True)
    subprocess.run([sys.executable,str(ROOT/'benchmark/prepare_judge.py'),'--metrics',str(out/'metrics')],check=True)
if __name__=='__main__':main()
