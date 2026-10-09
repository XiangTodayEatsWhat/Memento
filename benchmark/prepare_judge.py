"""Prepare reviewable GPT-3.5-turbo-0125 requests offline; no network or credentials."""
import json,argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);a=p.parse_args();d=Path(a.metrics)
 spec=json.loads((Path(__file__).parent/'judge_prompt.json').read_text())
 rows=[json.loads(s) for s in (d/'judge_inputs.jsonl').read_text().splitlines() if s.strip()]
 requests=[]
 for i,row in enumerate(rows):
  task=row['task'];assert task in spec['tasks'],task
  system=spec['final_prompt_template'].format(task_name=task,task_focus=spec['tasks'][task])
  user='Question: '+row['question']+chr(10)+'Reference: '+row['reference']+f" (Video Time = {row['reference_time']}s)"+chr(10)
  user+=chr(10).join(f"Candidate {j+1}: {o['content']} (Video Time = {o['time']}s)" for j,o in enumerate(row['candidate_turns']))
  requests.append({'custom_id':f"{row['sample_id']}-{i}",'method':'POST','url':'/v1/chat/completions','body':{'model':'gpt-3.5-turbo-0125','messages':[{'role':'system','content':system},{'role':'user','content':user}]},'local_reference':{'task':task,'time':row['reference_time']}})
 (d/'judge_requests_review.jsonl').write_text(''.join(json.dumps(r)+chr(10) for r in requests))
 print(json.dumps({'offline_judge_requests':len(requests),'model':'gpt-3.5-turbo-0125','network_requests_sent':0}))
if __name__=='__main__':main()
