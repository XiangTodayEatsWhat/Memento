"""Download model or dataset assets from Hugging Face; authentication uses HF_TOKEN/login."""
import argparse
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dataset','model','base','vision'],required=True);p.add_argument('--output',required=True);p.add_argument('--repo');p.add_argument('--revision',default='main');a=p.parse_args()
from huggingface_hub import snapshot_download
repos={'dataset':'liarzone/Memento-54K','model':'liarzone/Memento-8B','base':'meta-llama/Llama-3.1-8B-Instruct','vision':'google/siglip-large-patch16-384'}
print(snapshot_download(repo_id=a.repo or repos[a.kind],repo_type='dataset' if a.kind=='dataset' else 'model',revision=a.revision,local_dir=a.output))
