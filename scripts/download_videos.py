"""Download only the requested split through the official licensed Ego4D CLI."""
import argparse,subprocess,sys
from pathlib import Path
import uuid
def read_uids(path):
    rows=path.read_text().split()
    if len(rows)!=len(set(rows)):raise ValueError("Duplicate UIDs")
    for u in rows:uuid.UUID(u)
    return set(rows)
p=argparse.ArgumentParser();p.add_argument('--uid-dir',required=True);p.add_argument('--split',choices=['train','test','all'],default='all');p.add_argument('--output-dir',required=True);p.add_argument('--aws-profile',default='default');p.add_argument('--dataset',choices=['video_540ss','full_scale'],default='video_540ss');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
uids=Path(a.uid_dir)/f'{a.split}_video_uids.txt';expected=read_uids(uids)
if not expected:raise SystemExit('Refusing an empty UID list')
cmd=['ego4d','--output_directory',str(Path(a.output_dir).resolve()),'--datasets',a.dataset,'--version','v2','--video_uid_file',str(uids),'--aws_profile_name',a.aws_profile]
import shlex;print(shlex.join(cmd),flush=True)
if a.dry_run:sys.exit(0)
subprocess.run(cmd,check=True)
folder=Path(a.output_dir)/'v2'/a.dataset
missing=sorted(u for u in expected if not (folder/(u+'.mp4')).is_file() or (folder/(u+'.mp4')).stat().st_size==0)
print(f'Video files present: {len(expected)-len(missing)}/{len(expected)}. This check does not validate video decoding.')
if missing:print('Missing examples:',missing[:10]);sys.exit(1)
