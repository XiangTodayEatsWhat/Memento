"""Run the paper's answer-quality judge with resumable, auditable responses."""
import argparse,hashlib,json,os,re
from pathlib import Path
MODEL='gpt-3.5-turbo-0125'

def request_hash(row):return hashlib.sha256(json.dumps(row['body'],sort_keys=True).encode()).hexdigest()
def parse_score(text):
    text=text.strip()
    if text.startswith('\x60\x60\x60'):
        lines=text.splitlines();text='\n'.join(lines[1:-1])
    value=json.loads(text);score=value['Overall Score']
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not 1<=score<=10:raise ValueError('Judge score must be a number in [1,10]')
    return float(score)
def aggregate(requests,responses,metrics):
    wanted={r['custom_id']:r for r in requests}
    if len(wanted)!=len(requests):raise ValueError('Duplicate request IDs')
    if set(responses)!=set(wanted):raise ValueError('Missing or unexpected judge responses')
    totals={k:0.0 for k in metrics['ground_truth_counts']};counts={k:0 for k in totals}
    for key,r in wanted.items():
        v=responses[key]
        if v['request_sha256']!=request_hash(r) or v['requested_model']!=MODEL or v['returned_model']!=MODEL:raise ValueError('Judge provenance mismatch')
        score=parse_score(v['content']);ref=r['local_reference'];cats=['all']
        if 'spatial' in ref['task']:cats.append('spatial')
        if 'temporal' in ref['task']:cats.append('temporal')
        if ref['time']>1500:cats.append('long')
        for cat in cats:totals[cat]+=score;counts[cat]+=1
    if any(counts[k]!=metrics['matched_counts'].get(k,0) for k in counts):raise ValueError('Judge and temporal metric counts disagree')
    return {'model':MODEL,'judged_reference_counts':counts,'score_matched':{k:totals[k]/counts[k] if counts[k] else None for k in counts},'score_all_zero_for_misses':{k:totals[k]/n if n else None for k,n in metrics['ground_truth_counts'].items()},'complete':True}
def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);p.add_argument('--aggregate-only',action='store_true');p.add_argument('--max-requests',type=int);a=p.parse_args();d=Path(a.metrics)
    requests=[json.loads(x) for x in (d/'judge_requests_review.jsonl').read_text().splitlines() if x.strip()]
    dst=d/'judge_responses';dst.mkdir(exist_ok=True);responses={}
    for r in requests:
        if not re.fullmatch(r'[0-9]+-[0-9]+',r['custom_id']) or r['body']['model']!=MODEL:raise ValueError('Unexpected request ID or model')
        f=dst/(r['custom_id']+'.json')
        if f.exists():
            v=json.loads(f.read_text())
            if v['request_sha256']!=request_hash(r) or v['requested_model']!=MODEL or v['returned_model']!=MODEL:raise ValueError('Cached response provenance mismatch')
            parse_score(v['content']);responses[r['custom_id']]=v
    if not a.aggregate_only:
        pending=[r for r in requests if r['custom_id'] not in responses]
        if a.max_requests is not None:pending=pending[:a.max_requests]
        if pending:
            from openai import OpenAI
            if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Set OPENAI_API_KEY in your environment')
            client=OpenAI(timeout=120,max_retries=2)
            for r in pending:
                response=client.chat.completions.create(**r['body']);content=response.choices[0].message.content or '';parse_score(content)
                if response.model!=MODEL:raise RuntimeError('Provider returned a different model; not comparable to the paper judge')
                v={'request_sha256':request_hash(r),'requested_model':MODEL,'returned_model':response.model,'content':content,'usage':response.usage.model_dump() if response.usage else None}
                f=dst/(r['custom_id']+'.json');temp=f.with_suffix('.tmp');temp.write_text(json.dumps(v));os.replace(temp,f);responses[r['custom_id']]=v
                print('Completed',r['custom_id'],flush=True)
    result=aggregate(requests,responses,json.loads((d/'temporal_metrics.json').read_text()))
    (d/'quality_metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
