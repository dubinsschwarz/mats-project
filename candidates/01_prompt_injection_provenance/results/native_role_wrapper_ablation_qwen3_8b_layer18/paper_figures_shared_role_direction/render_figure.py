import csv, html, json
from pathlib import Path
P=Path(__file__).resolve().parent; R=P.parent
S=json.loads((R/'heldout_shared_role_direction_v0_v5_l1_l4/summary.json').read_text())
E=json.loads((R/'heldout_role_axis_erasure_v0_v5_l1_l4/summary.json').read_text())
b={x['variant']:x for x in S['baselines']}; si={(x['layer_hidden_state_index'],x['direction'],x['condition']):x for x in S['interventions']}; er={(x['layer_hidden_state_index'],x['condition'],x['variant']):x for x in E['per_role']}
base={0:{'compliant':E['baseline']['user_compliant'],'n':20,'truncated':0},5:{'compliant':E['baseline']['assistant_compliant'],'n':20,'truncated':0}}
rows=[]
for L in [1,4]:
 for role,v,d,sign in [('Assistant',5,'assistant_plus_d','+d'),('User',0,'user_minus_d','-d')]:
  for c,disp,z in [('baseline',role+' baseline',b[v]),('shared_direction',role+' '+sign,si[L,d,'natural_direction']),('equal_norm_random',role+' random',si[L,d,'norm_matched_random_control'])]: rows.append({'panel':'A','layer':L,'role':role,'condition':c,'display':disp,'compliant':z['compliant'],'n':z['n'],'truncated':z['truncated']})
bc=[(None,'baseline','Baseline'),(1,'role_axis_erasure','L1 axis erasure'),(1,'norm_matched_random_control','L1 matched random'),(4,'role_axis_erasure','L4 axis erasure'),(4,'norm_matched_random_control','L4 matched random')]
for L,c,disp in bc:
 for role,v in [('User',0),('Assistant',5)]:
  z=base[v] if L is None else er[L,c,v]; rows.append({'panel':'B','layer':'' if L is None else L,'role':role,'condition':c,'display':disp,'compliant':z['compliant'],'n':z['n'],'truncated':z['truncated']})
with (P/'source_data.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
def t(x,y,s,n=15,a='middle',w='normal',fill='#17212b'): return f'<text x="{x}" y="{y}" text-anchor="{a}" font-size="{n}" font-weight="{w}" fill="{fill}">{html.escape(str(s))}</text>'
def bar(x,z,col,w=42):
 q=100*z['compliant']/z['n']; y=545-4.35*q
 return f'<rect x="{x-w/2}" y="{y}" width="{w}" height="{545-y}" rx="2" fill="{col}"/>'+t(x,y-22,f"{z['compliant']}/{z['n']}",13,w='bold')+t(x,y-7,f'{q:.0f}%',12,fill='#4b5563')
x=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="650" viewBox="0 0 1500 650">','<rect width="1500" height="650" fill="white"/>','<style>text{font-family:Arial,Helvetica,sans-serif}</style>',t(750,30,'Causal effects of a shared role-related direction on held-out prompts',22,w='bold')]
for l,r in [(60,710),(790,1450)]:
 for q in range(0,101,20):
  y=545-4.35*q; x += [f'<line x1="{l}" y1="{y}" x2="{r}" y2="{y}" stroke="#d8dde3"/>',t(l-10,y+5,q,12,'end',fill='#4b5563')]
 x += [f'<line x1="{l}" y1="110" x2="{l}" y2="545" stroke="#44515d"/>',f'<line x1="{l}" y1="545" x2="{r}" y2="545" stroke="#44515d"/>']
x += [t(60,70,'A   Single-direction steering',19,'start','bold'),t(790,70,'B   Role-axis erasure',19,'start','bold')]
g=[(1,'Assistant','assistant_plus_d'),(1,'User','user_minus_d'),(4,'Assistant','assistant_plus_d'),(4,'User','user_minus_d')]
for c,(L,role,d) in zip([145,310,475,640],g):
 zs=[b[5 if role=='Assistant' else 0],si[L,d,'natural_direction'],si[L,d,'norm_matched_random_control']]
 for dx,z,col in zip([-48,0,48],zs,['#9aa3ad','#087e8b','#d6b656']): x.append(bar(c+dx,z,col))
 x += [t(c,570,f'L{L} {role}',13,w='bold'),t(c,589,'+d' if role=='Assistant' else '-d',13,fill='#087e8b')]
for xx,col,lab in [(100,'#9aa3ad','Baseline'),(265,'#087e8b','Shared direction'),(475,'#d6b656','Equal-norm random')]: x += [f'<rect x="{xx}" y="607" width="15" height="15" fill="{col}"/>',t(xx+22,620,lab,12,'start')]
for c,(L,cond,disp) in zip([850,985,1120,1255,1390],bc):
 zs=[base[0] if L is None else er[L,cond,0],base[5] if L is None else er[L,cond,5]]
 x += [bar(c-23,zs[0],'#3b82b8',40),bar(c+23,zs[1],'#e07a5f',40)]
 a=disp.split(' ',1); x += [t(c,570,a[0],12,w='bold'),t(c,587,a[1] if len(a)>1 else '',11)]
x += ['<rect x="1150" y="80" width="15" height="15" fill="#3b82b8"/>',t(1172,93,'User',12,'start'),'<rect x="1230" y="80" width="15" height="15" fill="#e07a5f"/>',t(1252,93,'Assistant',12,'start'),t(750,642,'Exact-marker compliance; saved n=20 per bar. Saved truncations remain in denominators.',12,fill='#4b5563'),'</svg>']
(P/'shared_role_direction_steering_erasure.svg').write_text('\n'.join(x))
