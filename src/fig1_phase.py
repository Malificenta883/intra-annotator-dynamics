import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paper_results as pr
from projection import load_units, select_medoid
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig1_phase.png"

def figure_data(source, choose):
    """Inanna's Descent: 3 runs of one reader, projected on text lines (see projection.py)."""
    units_all = load_units("inanna_descent", os.path.dirname(os.path.abspath(DATA)))
    runs = pr.load_runs(DATA, "inanna_descent", source); labs = [r[1] for r in runs]
    S = choose(labs); idx = [labs.index(m) for m in S]
    common = sorted(set.intersection(*[set(m) for m in S]))
    return dict(runs=[i + 1 for i in idx], units=[list(units_all[u]) for u in common],
                labels=[[m[u] for u in common] for m in S],
                bounds=[sorted(runs[i][2] & set(common)) for i in idx], unit_ids=common)

# panel A: the human's 3 runs; panel B: Opus, one typical run per cluster (select_medoid)
D = {"human1": figure_data("human1", lambda L: L), "opus": figure_data("opus", lambda L: select_medoid(L, 3))}
FUN=["preparation","contact","exchange","disruption","negotiation","stabilization","return"]
COL=dict(zip(FUN,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7"]))
TXT={f:("#ffffff" if f in("preparation","stabilization","return") else "#1d1d1b") for f in FUN}
AB=dict(preparation="prep",contact="cont",exchange="exch",disruption="disr",negotiation="nego",stabilization="stab",**{"return":"ret"})
INK="#1d1d1b"; MUTED="#6b6a64"; GRID="#d9d8d2"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})

def segs(d,k):
    units=d["units"]; lab=d["labels"][k]; B=set(d["bounds"][k]); ids=d["unit_ids"]
    out=[]; s=0
    for i in range(1,len(ids)+1):
        if i==len(ids) or lab[i]!=lab[s] or ids[i] in B or ids[i]!=ids[i-1]+1:
            end=units[i][0] if i<len(ids) else units[-1][1]+1
            out.append((units[s][0],end,lab[s])); s=i
    return out

def pieces(d):
    ids=d["unit_ids"]; L=d["labels"]; st=[0]
    for i in range(1,len(ids)):
        if ids[i]!=ids[i-1]+1 or any(L[k][i]!=L[k][i-1] for k in range(3)): st.append(i)
    return st
def onsets(seq): return [(j,seq[j]) for j in range(1,len(seq)) if seq[j]!=seq[j-1]]

def links(d,a,b):
    st=pieces(d); S=[[d["labels"][k][i] for i in st] for k in range(3)]
    x=[d["units"][i][0] for i in st]
    oa={}
    for i,f in [(0,S[a][0])]+onsets(S[a]): oa.setdefault(f,[]).append(i)
    out=[]
    for j,f in onsets(S[b]):
        if f not in oa: continue
        dd=min((i-j for i in oa[f]),key=lambda v:(abs(v),v))
        if abs(dd)<=1: out.append((x[j+dd],x[j],dd))
    return out

fig,axes=plt.subplots(2,1,figsize=(6.6,3.6),dpi=300)
H=0.62; Y=[2,1,0]
for ax,(key,title) in zip(axes,[("human1","A   Human reader: three runs, months apart"),
                                 ("opus","B   Opus: three of ten runs (one typical run per cluster)")]):
    d=D[key]
    for k in range(3):
        for (a,e,f) in segs(d,k):
            ax.add_patch(Rectangle((a,Y[k]-H/2),e-a,H,facecolor=COL[f],edgecolor="white",linewidth=0.9))
            if e-a>=16: ax.text((a+e)/2,Y[k],AB[f],ha="center",va="center",fontsize=5.6,color=TXT[f])
        ax.text(-4,Y[k],f"run {d['runs'][k]}",ha="right",va="center",fontsize=6.5,color=MUTED)
    for (ka,kb) in [(0,1),(1,2)]:
        for xa,xb,dd in links(d,ka,kb):
            ya,yb=Y[ka]-H/2,Y[kb]+H/2
            if dd==0: ax.plot([xa,xb],[ya,yb],color=MUTED,lw=0.6,alpha=0.55,solid_capstyle="round")
            else:     ax.plot([xa,xb],[ya,yb],color=INK,lw=1.1,solid_capstyle="round")
    ax.set_xlim(0,414); ax.set_ylim(-0.55,2.55)
    ax.set_yticks([]); ax.set_title(title,loc="left",fontsize=7.6,color=INK,pad=3,fontweight="semibold")
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x",colors=MUTED,length=2,labelsize=6.5)
axes[1].set_xlabel("line of Inanna's Descent",fontsize=6.8,color=MUTED,labelpad=2)
from matplotlib.lines import Line2D
hand=[Rectangle((0,0),1,1,facecolor=COL[f],edgecolor="none") for f in FUN]
labs=[f"{f} ({AB[f]})" for f in FUN]
fig.legend(hand,labs,loc="lower center",ncol=4,fontsize=5.8,frameon=False,handlelength=1.1,columnspacing=1.2,bbox_to_anchor=(0.5,0.045))
h2=[Line2D([0],[0],color=MUTED,lw=0.8,alpha=0.7),Line2D([0],[0],color=INK,lw=1.2)]
fig.legend(h2,["function starts at the same place in the next run","function starts one piece earlier or later"],loc="lower center",ncol=2,fontsize=5.8,frameon=False,handlelength=1.6,columnspacing=1.5,bbox_to_anchor=(0.5,-0.005))
fig.subplots_adjust(left=0.06,right=0.99,top=0.95,bottom=0.25,hspace=0.55)
fig.savefig(OUT, dpi=300)
for key in ["human1","opus"]:
    d=D[key]; n=[links(d,a,b) for a,b in [(0,1),(1,2)]]
    print(key,"lag0",sum(1 for L in n for t in L if t[2]==0),"lag1",sum(1 for L in n for t in L if t[2]!=0))
