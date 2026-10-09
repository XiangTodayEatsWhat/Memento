"""Validate cached feature content and annotation bounds, then rebuild metadata."""
import argparse
from pathlib import Path
from common import read_rows,save_json
p=argparse.ArgumentParser();p.add_argument('--annotations',nargs='+',required=True);p.add_argument('--features',required=True);a=p.parse_args()
import torch
rows=[r for f in a.annotations for r in read_rows(f)];folder=Path(a.features).resolve();meta={}
for uid in sorted({r['video_uid'] for r in rows}):
    path=folder/(uid+'.pt');x=torch.load(path,map_location='cpu',weights_only=True)
    if x.ndim!=3 or tuple(x.shape[1:])!=(10,1024) or len(x)==0 or not torch.isfinite(x).all():raise ValueError(f'Invalid tensor: {uid}')
    meta[uid]={'duration':(len(x)-1)/2,'path':str(path)}
for row in rows:
    if any(t['time']>meta[row['video_uid']]['duration']+0.5 for t in row['conversation']):raise ValueError('Annotation beyond video duration')
save_json(str(folder)+'_metadata.json',meta);print('Validated',len(meta),'videos')
