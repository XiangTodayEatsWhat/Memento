"""Export portable adapter weights, learned connector/attention modules and tokenizer files."""
import argparse,json,shutil,sys
from pathlib import Path
from common import digest,save_json

def export_model(source,output,require_complete=True):
    source=Path(source);output=Path(output)
    if require_complete:
        marker=source/'training_complete.json'
        if not marker.is_file() or not json.loads(marker.read_text()).get('complete'):raise ValueError('A completed training marker is required')
        if json.loads(marker.read_text()).get('max_steps_override') is not None:raise ValueError('Do not export a smoke-test checkpoint as a full model')
        if json.loads(marker.read_text())['weights_sha256']!=digest(source/'adapter_model.safetensors'):raise ValueError('Weights changed after completion')
    from safetensors import safe_open
    with safe_open(source/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        keys=list(f.keys())
    for part in ['lora_','connector','attention_model']:
        if not any(part in k for k in keys):raise ValueError(f'Missing learned component: {part}')
    output.mkdir(parents=True,exist_ok=True)
    names=['adapter_model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja']
    for name in names:
        path=source/name
        if path.exists():shutil.copy2(path,output/name)
    config=json.loads((source/'adapter_config.json').read_text());config['base_model_name_or_path']='meta-llama/Llama-3.1-8B-Instruct';save_json(output/'adapter_config.json',config)
    tokenizer_config=output/'tokenizer_config.json'
    if tokenizer_config.exists():
        value=json.loads(tokenizer_config.read_text());value.pop('name_or_path',None);save_json(tokenizer_config,value)
    save_json(output/'memento_config.json',{'base_model':'meta-llama/Llama-3.1-8B-Instruct','vision_model':'google/siglip-large-patch16-384','frame_fps':2,'pt_size':3,'eps':0.7,'attention_update_ratio':0.2,'qms_ratio':0.5,'mem_length':16,'cutoff_len':16384,'waiting_token':'eos','checkpoint_format':'peft-adapter-with-modules-to-save'})
    run_config=source/'run_config.json'
    training=json.loads(run_config.read_text()).get('training',{}) if run_config.is_file() else {}
    config=json.loads((output/'memento_config.json').read_text())
    for key in ['frame_fps','pt_size','eps','attention_update_ratio','qms_ratio','mem_length','cutoff_len']:
        if key in training:config[key]=training[key]
    save_json(output/'memento_config.json',config)
    save_json(output/'weights_manifest.json',{'files':{p.name:digest(p) for p in sorted(output.iterdir()) if p.name in names+['adapter_config.json','memento_config.json']},'tensor_keys':len(keys),'components':['LoRA','connector','attention_model']})
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(export_model(a.checkpoint,a.output))
if __name__=='__main__':main()
