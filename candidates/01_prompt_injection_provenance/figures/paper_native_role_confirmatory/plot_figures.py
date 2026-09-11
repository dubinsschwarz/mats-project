"""Create paper figures from saved audited summaries only.

Run: uv run --with matplotlib==3.10.6 python plot_figures.py
"""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[1]; DATA=HERE/"data"; RENDERED=HERE/"rendered"
BEHAVIOR=PROJECT/"results/native_wrapper_heldout_validation_qwen3_8b/summary.json"
PATCHING=PROJECT/"results/native_wrapper_role_token_heldout_replication_qwen3_8b/summary.json"
CONDS=["complete_native_user","complete_native_assistant","native_user_no_close","ordinary_text_lookalike"]
LABELS={"complete_native_user":"Native user","complete_native_assistant":"Native assistant","native_user_no_close":"User, no close","ordinary_text_lookalike":"Text lookalike"}
COLORS={"complete_native_user":"#286F9B","complete_native_assistant":"#D47A36","native_user_no_close":"#69A88D","ordinary_text_lookalike":"#9AA3AC"}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows,fields):
    with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def save(fig,name):
    for ext in ["png","svg","pdf"]: fig.savefig(RENDERED/f"{name}.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",facecolor="white")
    plt.close(fig)
def style(ax):
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color="#DCE1E5",linewidth=.75); ax.set_axisbelow(True); ax.tick_params(colors="#25313B")
def annotate(ax,bars,rows):
    for b,r in zip(bars,rows): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.035,f"{100*r['rate']:.1f}%\n{r['success']}/{r['n']}",ha="center",va="bottom",fontsize=9,color="#18242D",linespacing=1.15)

def behavior_rows(saved):
    panels=[("Marker task","stage1",None),("Arithmetic","stage2","arithmetic"),("String transformation","stage2","string_transformation")]; out=[]
    for title,stage,family in panels:
        source=saved[stage]["condition_rates"] if stage=="stage1" else saved[stage]["rates_by_task_family_and_condition"][family]
        for c in CONDS:
            x=source[c]; n=x["n"]; k=x["marker_count"] if stage=="stage1" else x["task_followed_count"]
            out.append({"panel":title,"condition":c,"condition_label":LABELS[c],"n":n,"success":k,"rate":k/n})
    expected={"Marker task":[57,7,38,7],"Arithmetic":[31,0,17,3],"String transformation":[33,0,20,3]}
    for panel,counts in expected.items():
        got=[r["success"] for r in out if r["panel"]==panel]
        if got!=counts: raise RuntimeError(f"audited behavior count mismatch: {panel}: {got}")
    return out

def figure_a(rows):
    fig,axes=plt.subplots(1,3,figsize=(12.3,4.25),sharey=True); titles=["A  Marker task","B  Arithmetic","C  String transformation"]
    for ax,title,panel in zip(axes,titles,["Marker task","Arithmetic","String transformation"]):
        xs=[r for r in rows if r["panel"]==panel]; bars=ax.bar(np.arange(4),[r["rate"] for r in xs],width=.68,color=[COLORS[r["condition"]] for r in xs],edgecolor="white",linewidth=.8)
        annotate(ax,bars,xs); ax.set_title(title,loc="left",fontweight="bold",fontsize=12,pad=12); ax.set_xticks(np.arange(4),["Native\nuser","Native\nassistant","User,\nno close","Text\nlookalike"],fontsize=9); ax.set_ylim(0,1.12); ax.set_yticks(np.arange(0,1.01,.2),[f"{int(x*100)}%" for x in np.arange(0,1.01,.2)]); style(ax)
    axes[0].set_ylabel("Embedded-task success rate",fontsize=10); fig.suptitle("Native user-role syntax increases instruction following inside tool output",fontsize=14,fontweight="bold",y=1.03)
    fig.text(.5,-.035,"Bars show audited success rates; labels report percentage and successful examples / total.",ha="center",fontsize=9,color="#4D5962")
    fig.tight_layout(w_pad=1.6); save(fig,"figure_A_behavioral_headline")

def patch_rows(saved):
    baselines={r["condition"]:r for r in saved["baseline"]}; out=[]
    dirs=[("complete_native_user->complete_native_assistant",ASSISTANT:="complete_native_assistant","User state → assistant prompt"),("complete_native_assistant->complete_native_user",USER:="complete_native_user","Assistant state → user prompt")]
    for direction,recipient,label in dirs:
        b=baselines[recipient]; out.append({"direction":direction,"direction_label":label,"layer":"baseline","intervention":"baseline_recipient","intervention_label":"Recipient baseline","n":b["n"],"success":b["success"],"rate":b["success"]/b["n"],"truncated":b["truncated"]})
        for layer in ["layer_1","layer_4"]:
            for control,ilabel in [("natural_replacement","Opposite-role state"),("norm_matched_random","Norm-matched random")]:
                r=next(x for x in saved["patches"] if x["site"]==layer and x["direction"]==direction and x["control"]==control)
                out.append({"direction":direction,"direction_label":label,"layer":layer,"intervention":control,"intervention_label":ilabel,"n":r["n"],"success":r["success"],"rate":r["success"]/r["n"],"truncated":r["truncated"]})
    if any(r["n"]!=40 for r in out): raise RuntimeError("patching n mismatch")
    return out

def figure_b(rows):
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.55),sharey=True); directions=["complete_native_user->complete_native_assistant","complete_native_assistant->complete_native_user"]
    categories=[("baseline","baseline_recipient"),("layer_1","natural_replacement"),("layer_1","norm_matched_random"),("layer_4","natural_replacement"),("layer_4","norm_matched_random")]
    colors=["#59636C","#286F9B","#AAB2B9","#286F9B","#AAB2B9"]
    labels=["Recipient\nbaseline","L1\nnatural","L1\nrandom","L4\nnatural","L4\nrandom"]
    for i,(ax,direction) in enumerate(zip(axes,directions)):
        xs=[]
        for layer,control in categories: xs.append(next(r for r in rows if r["direction"]==direction and r["layer"]==layer and r["intervention"]==control))
        bars=ax.bar(np.arange(5),[r["rate"] for r in xs],width=.68,color=colors,edgecolor="white",linewidth=.8); annotate(ax,bars,xs)
        ax.set_title(("A  User state → assistant prompt\n" if i==0 else "B  Assistant state → user prompt\n")+("Expected effect: increase" if i==0 else "Expected effect: suppression"),loc="left",fontweight="bold",fontsize=11,pad=10)
        ax.set_xticks(np.arange(5),labels,fontsize=9); ax.set_ylim(0,1.03); ax.set_yticks(np.arange(0,1.01,.2),[f"{int(x*100)}%" for x in np.arange(0,1.01,.2)]); ax.axhline(xs[0]["rate"],color="#59636C",linewidth=1,linestyle=(0,(3,3)),alpha=.8); style(ax)
    axes[0].set_ylabel("Exact-marker compliance rate",fontsize=10); fig.suptitle("Early role-token state causally transfers held-out behavior",fontsize=14,fontweight="bold",y=1.04)
    fig.text(.5,-.04,"Natural: full residual state from the opposite role at the aligned embedded role token. Random: norm-matched control.",ha="center",fontsize=9,color="#4D5962")
    fig.tight_layout(w_pad=2); save(fig,"figure_B_heldout_role_token_patching")

def main():
    DATA.mkdir(parents=True,exist_ok=True); RENDERED.mkdir(parents=True,exist_ok=True); behavior=json.loads(BEHAVIOR.read_text()); patch=json.loads(PATCHING.read_text())
    a=behavior_rows(behavior); b=patch_rows(patch)
    write_csv(DATA/"figure_A_behavioral_headline.csv",a,["panel","condition","condition_label","n","success","rate"])
    write_csv(DATA/"figure_B_heldout_role_token_patching.csv",b,["direction","direction_label","layer","intervention","intervention_label","n","success","rate","truncated"])
    (DATA/"source_manifest.json").write_text(json.dumps({"behavior":{"path":str(BEHAVIOR),"sha256":sha(BEHAVIOR)},"patching":{"path":str(PATCHING),"sha256":sha(PATCHING)}},indent=2)+"\n")
    figure_a(a); figure_b(b); print(f"Wrote figures to {RENDERED}")
if __name__=="__main__": main()
