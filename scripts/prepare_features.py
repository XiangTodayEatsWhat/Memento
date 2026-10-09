"""Validate annotations, resample source videos, encode and validate feature coverage."""
import argparse,json,subprocess,sys,os
from pathlib import Path
from common import ROOT,read_rows,save_json,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',nargs='+',required=True);p.add_argument('--videos',required=True);p.add_argument('--work-dir',required=True);p.add_argument('--vision',default='google/siglip-large-patch16-384');p.add_argument('--gpu',default='0');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    rows=[r for file in a.annotations for r in read_rows(file)];uids=sorted({r['video_uid'] for r in rows});src=Path(a.videos).resolve();work=Path(a.work_dir).resolve();raw=work/'videos';raw.mkdir(parents=True,exist_ok=True)
    for uid in uids:
        import uuid;uuid.UUID(uid)
        source=src/(uid+'.mp4')
        if not source.is_file() or not source.stat().st_size:raise FileNotFoundError(str(source))
        target=raw/source.name
        if target.is_symlink():
            if target.resolve()!=source:raise ValueError(f'Source changed for {uid}')
        elif target.exists():raise ValueError(f'Expected managed symlink: {target}')
        else:target.symlink_to(source)
    extras={x.stem for x in raw.glob('*.mp4')}-set(uids)
    if extras:raise ValueError('Work directory contains another split; choose a separate work directory')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable,'-m','data.preprocess.ffmpeg','--video_dir',str(raw),'--frame_fps','2','--frame_resolution','384','--num_workers',str(a.workers)],cwd=ROOT,env=env,check=True)
    sampled=Path(str(raw)+'_2fps_max384');nframes={}
    for uid in uids:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(sampled/(uid+'.mp4'))]))['streams'][0]
        if probe['width']!=384 or probe['height']!=384 or probe['avg_frame_rate']!='2/1':raise ValueError(f'Invalid sampled video: {uid}')
        nframes[uid]=int(probe['nb_frames'])
        if nframes[uid]<=0:raise ValueError('Empty video')
    subprocess.run([sys.executable,'-m','data.preprocess.encode','--video_dir',str(sampled),'--vision_pretrained',a.vision,'--batch_size',str(a.batch_size),'--save_bf16'],cwd=ROOT,env=env,check=True)
    features=Path(str(sampled)+'_2fps_384_1+3x3_'+a.vision.replace('/','--'))
    import torch
    metadata={}
    for uid in uids:
        path=features/(uid+'.pt');x=torch.load(path,map_location='cpu',weights_only=True)
        if x.shape!=(nframes[uid],10,1024) or x.dtype!=torch.bfloat16 or not torch.isfinite(x).all():raise ValueError(f'Invalid features: {uid}')
        metadata[uid]={'duration':(len(x)-1)/2,'path':str(path)}
    negative=[]
    for i,row in enumerate(rows):
        for turn in row['conversation']:
            if turn['time']<0:negative.append(i)
            if turn['time']>=nframes[row['video_uid']]/2:raise ValueError(f'Annotation beyond video: row {i}')
    save_json(str(features)+'_metadata.json',metadata)
    save_json(work/'prepared.json',{'feature_dir':str(features),'videos':len(uids),'annotations_sha256':[digest(x) for x in a.annotations],'rows_with_negative_timestamps':sorted(set(negative)),'frame_fps':2,'frame_tokens':10,'feature_dimension':1024})
    print('FEATURE_DIR='+str(features))
if __name__=='__main__':main()
