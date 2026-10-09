import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read_rows(path):
    text=Path(path).read_text()
    rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not isinstance(rows,list) or not rows:raise ValueError('Expected a non-empty annotation array or JSONL')
    return rows

def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
