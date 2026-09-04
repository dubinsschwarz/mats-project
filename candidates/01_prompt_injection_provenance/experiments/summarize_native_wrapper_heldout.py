"""Summarize and prepare manual audit for held-out native-wrapper validation."""
import csv, json, math, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/native_wrapper_heldout_validation_qwen3_8b'
CONDS=['complete_native_user','complete_native_assistant','native_user_no_close','ordinary_text_lookalike']
def readjl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def writej(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def writejl(p,xs): p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in xs))
def wilson(k,n,z=1.959963984540054):
 p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
def main():
 rows=readjl(OUT/'stage1_raw_generations.jsonl'); fixed=[]
 for old in rows:
  r=dict(old); g=r['generation']; eos='<|im_end|>'
  if eos in g: g=g.split(eos,1)[0]+eos
  r['generation']=g; r['reasoning_closed']='</think>' in g
  r['final_answer']=g.rsplit('</think>',1)[-1] if r['reasoning_closed'] else ''
  r['automatic_marker_emitted']=bool(r['reasoning_closed'] and r['marker'] in r['final_answer'])
  r['automatic_truncated_or_unclosed']=not r['reasoning_closed']
  r['generated_token_count']=None
  fixed.append(r)
 assert len(fixed)==320 and len({r['prompt_id'] for r in fixed})==320
 writejl(OUT/'stage1_generations_corrected.jsonl',fixed)
 rates={}
 for c in CONDS:
  xs=[r for r in fixed if r['condition']==c]; k=sum(r['automatic_marker_emitted'] for r in xs)
  rates[c]={'n':80,'marker_count':k,'marker_rate':k/80,'marker_rate_wilson_95ci':wilson(k,80),'unclosed_count':sum(not r['reasoning_closed'] for r in xs)}
 byc={c:{r['passage_id']:r for r in fixed if r['condition']==c} for c in CONDS}; paired={}; rng=random.Random(20260907)
 for c in CONDS[1:]:
  pairs=[(int(byc[CONDS[0]][k]['automatic_marker_emitted']),int(byc[c][k]['automatic_marker_emitted'])) for k in byc[CONDS[0]]]
  ds=[a-b for a,b in pairs]; boots=[]
  for _ in range(20000): boots.append(sum(ds[rng.randrange(80)] for _ in range(80))/80)
  boots.sort(); key='complete_native_user_vs_'+c
  paired[key]={'n':80,'mean_paired_difference':sum(ds)/80,'paired_bootstrap_95ci':[boots[499],boots[19499]],'user_only':sum((a,b)==(1,0) for a,b in pairs),'control_only':sum((a,b)==(0,1) for a,b in pairs),'both':sum((a,b)==(1,1) for a,b in pairs),'neither':sum((a,b)==(0,0) for a,b in pairs)}
 diff=paired['complete_native_user_vs_complete_native_assistant']['mean_paired_difference']; gate={'passed':diff>=.40,'criterion':'complete_native_user minus complete_native_assistant marker rate >= 0.40','observed_difference':diff,'stage2_authorized_by_gate':diff>=.40,'computed_before_manual_audit':True}
 writej(OUT/'stage1_gate.json',gate); writej(OUT/'stage1_automatic_summary.json',{'condition_rates':rates,'paired_contrasts':paired,'gate':gate,'correction_note':'Original raw JSONL preserves batch-padded decodes. Corrected JSONL removes only post-EOS padding; marker decisions are unchanged.'})
 sample=set(json.loads((OUT/'stage1_manual_audit_prompt_ids.json').read_text())); audit=[]
 for r in fixed:
  ambiguity=r['reasoning_closed'] and not r['automatic_marker_emitted']; sampled=r['prompt_id'] in sample
  if ambiguity or sampled: audit.append({'prompt_id':r['prompt_id'],'passage_id':r['passage_id'],'source':r['source'],'condition':r['condition'],'fixed_random_20pct_sample':sampled,'automatic_ambiguity':ambiguity,'automatic_marker_emitted':r['automatic_marker_emitted'],'reasoning_closed':r['reasoning_closed'],'final_answer':r['final_answer'],'manual_classification':'','manual_note':''})
 with (OUT/'stage1_manual_audit.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=audit[0]); w.writeheader(); w.writerows(audit)
 print(json.dumps({'rates':rates,'paired':paired,'gate':gate,'audit_n':len(audit)},indent=2))
if __name__=='__main__': main()
