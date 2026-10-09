"""Offline MementoBench temporal metrics, with strict complete-result coverage."""
import json,argparse,collections
from pathlib import Path

def temporal_metrics(gt,pred,window=5.0):
    assert len(gt)==len(pred), "Prediction coverage mismatch"
    totals=collections.Counter(); hits=collections.Counter(); redundant=0; generated=0; judge=[]
    for idx,(g,p) in enumerate(zip(gt,pred)):
        assert g['video_uid']==p['video_uid'],(idx,'video mismatch')
        refs=[x for x in g['conversation'] if x['role']=='assistant']
        outs=[x for x in p['conversation'] if x['role']=='assistant']
        # Five-second tolerance follows the existing benchmark's abs(delta)<5 convention.
        for o in outs:
            generated+=1
            if not any(abs(o['time']-r['time'])<window for r in refs): redundant+=1
        last_query={}; previous_query=''
        for r in g['conversation']:
            task=r.get('task','unknown')
            if r['role']=='user': last_query[task]=r['content']; previous_query=r['content']; continue
            if r['role']!='assistant': continue
            categories=['all']
            if 'spatial' in task: categories.append('spatial')
            if 'temporal' in task: categories.append('temporal')
            # Long means absolute video time >25 minutes, per Table 3.
            if r['time']>1500: categories.append('long')
            candidates=[o for o in outs if abs(o['time']-r['time'])<window]
            for c in categories:
                totals[c]+=1; hits[c]+=bool(candidates)
            if candidates:
                judge.append({'sample_id':idx,'task':task,'reference_time':r['time'],'question':last_query.get(task,previous_query),'reference':r['content'],'candidates':[o['content'] for o in candidates],'candidate_turns':candidates})
    return {'ground_truth_counts':dict(totals),'matched_counts':dict(hits),'time_recall':{k:hits[k]/v for k,v in totals.items()},'generated_responses':generated,'redundant_responses':redundant,'redundancy':redundant/generated if generated else None,'score_status':'pending_original_GPT-3.5-turbo-0125_judge','window_convention':'abs(pred_time-gt_time)<5_seconds'},judge

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gt',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    gt=json.load(open(a.gt)); pred=[]
    for idx,g in enumerate(gt):
        f=Path(a.results,f'{idx}.json'); assert f.exists(),f'Missing sample {idx}'
        row=json.load(open(f)); assert row.get('complete') is True,f'Partial sample {idx}'
        assert row['test_id']==idx
        pred.append(row)
    fingerprints={(x['checkpoint_sha256'],x['code_sha256']) for x in pred}
    assert len(fingerprints)==1, 'Mixed model weights or inference code'
    import hashlib
    for g,pred_row in zip(gt,pred):
        assert hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest()==pred_row['input_sha256'], 'Input sample changed'
    metrics,judge=temporal_metrics(gt,pred)
    d=Path(a.output);d.mkdir(parents=True,exist_ok=True)
    (d/'temporal_metrics.json').write_text(json.dumps(metrics,indent=2))
    (d/'merged_predictions.json').write_text(json.dumps(pred))
    (d/'judge_inputs.jsonl').write_text(''.join(json.dumps(x)+chr(10) for x in judge))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
