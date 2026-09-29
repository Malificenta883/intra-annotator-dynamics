#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_results.py -- one reproducible pass for the §4 within-human results table.

Computes, per corpus, from the human ('human1') runs:
  * boundary self-consistency  (2-way Cohen's kappa: is a unit a segment start?)
    Units: blocks for inanna_enki, lines elsewhere (see projection.py).
  * label self-consistency     (7-way Cohen's kappa over functions, line-projected)
  * the dissociation gap        (boundary_kappa - label_kappa)
  * boundary placement          (exact / +-1 / +-2 matches vs chance, see boundary_match)
  * B1 concentration            (variance of per-line disagreement vs a rate-matched
                                 relocated-label null) + leave-one-run-out robustness
  * B2 neighbour-licensed flips (event level, majority units only; see b2 docstring)
All chance-corrected / permutation-nulled. kappa is ~category-count-invariant at matched
reproducibility (see kappa_commensurability_sim.py), so boundary(2) vs label(7) is a fair
comparison. Pure stdlib.  Run:  python3 paper_results.py [data_dir]  [--source human1]
"""
import os, sys, json, glob, itertools, random, re
from collections import Counter

random.seed(12345)
N_PERM = 1000
FUNCTIONS = ["preparation","contact","exchange","disruption","negotiation","stabilization","return"]
FSET = set(FUNCTIONS)
CORPORA = ["inanna_enki","inanna_descent","gudea"]

# ---------- loading / projection ----------
# Projection lives in projection.py (shared by all scripts): blocks for inanna_enki,
# lines elsewhere; lines outside the numbered text (lacunae) are never scored.
from pathlib import Path
from projection import load_units, load_run as _load_run

_UNITS = {}
def load_runs(data_dir, corpus, source, granularity="native"):
    root = Path(data_dir).resolve().parent
    key = (corpus, granularity)
    if key not in _UNITS: _UNITS[key] = load_units(corpus, root, granularity)
    pat = os.path.join(data_dir, corpus, f"{source}_run*.json")
    out = []
    for p in sorted(glob.glob(pat), key=lambda s:int(re.search(r"run(\d+)",s).group(1))):
        lab, starts, _ = _load_run(p, _UNITS[key])
        if lab: out.append((os.path.basename(p), lab, starts))
    return out

def kappa(a, b, cats):
    n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    ca=Counter(a); cb=Counter(b)
    pe=sum((ca[c]/n)*(cb[c]/n) for c in cats)
    return (po-pe)/(1-pe) if pe<1 else 1.0

# ---------- boundary vs label ----------
def boundary_label_kappas(runs):
    labs, bnds = [r[1] for r in runs], [r[2] for r in runs]
    lk, bk = [], []
    for i,j in itertools.combinations(range(len(runs)), 2):
        common = sorted(set(labs[i]) & set(labs[j]))   # both layers on the same units
        lk.append(kappa([labs[i][x] for x in common], [labs[j][x] for x in common], FUNCTIONS))
        bk.append(kappa([1 if x in bnds[i] else 0 for x in common],
                        [1 if x in bnds[j] else 0 for x in common], [0,1]))
    return (sum(bk)/len(bk), sum(lk)/len(lk), bk, lk)

# ---------- boundary placement vs chance ----------
def boundary_match(runs):
    """Share of one run's boundaries that fall exactly on a boundary of another run
    (per pair: the run with fewer boundaries vs the other), against chance = the share
    of units that are boundaries in the other run. The first unit is excluded (always a
    boundary). Also reports matches within +-1 and +-2 units (shift vs re-cut)."""
    rows=[]
    for (_,la,ba),(_,lb,bb) in itertools.combinations(runs,2):
        common=set(la)&set(lb)
        if not common: continue
        first=min(common)
        A={u for u in ba if u in common and u!=first}; B={u for u in bb if u in common and u!=first}
        small,big=(A,B) if len(A)<=len(B) else (B,A)
        if not small: continue
        near=lambda d: sum(1 for u in small if any(abs(u-v)<=d for v in big))/len(small)
        rows.append((len(small&big)/len(small), near(1), near(2), len(big)/len(common)))
    if not rows: return None
    m=lambda i: sum(r[i] for r in rows)/len(rows)
    return dict(exact=m(0), within1=m(1), within2=m(2), chance=m(3), pairs=len(rows))

# ---------- B1 concentration ----------
def per_line(labs_list):
    cov={}
    for m in labs_list:
        for ln,f in m.items(): cov.setdefault(ln,[]).append(f)
    return {ln:v for ln,v in cov.items() if len(v)>=2}
def disagree(v): return 1.0 - Counter(v).most_common(1)[0][1]/len(v)
def var(xs):
    if not xs: return 0.0
    m=sum(xs)/len(xs); return sum((x-m)**2 for x in xs)/len(xs)
def b1(labs_list, rng=None):
    rng = rng or random.Random(0)
    ll=per_line(labs_list)
    if len(ll)<4: return None
    lines=sorted(ll); obs=var([disagree(ll[x]) for x in lines])
    shared=[{x:m[x] for x in lines if x in m} for m in labs_list]
    null=[]
    for _ in range(N_PERM):
        cov={x:[] for x in lines}
        for sh in shared:
            vals=list(sh.values()); rng.shuffle(vals)
            for k,v in zip(sh.keys(), vals): cov[k].append(v)
        null.append(var([disagree(cov[x]) for x in lines if len(cov[x])>=2]))
    nm=sum(null)/len(null); p=(sum(1 for z in null if z>=obs)+1)/(len(null)+1)
    return dict(obs=obs, null=nm, gap=obs-nm, ratio=obs/nm if nm else float('nan'), p=p, n_lines=len(lines))
def b1_loo(runs, rng=None):
    rng = rng or random.Random(0)
    labs=[r[1] for r in runs]
    if len(labs)<3: return None
    res=[]
    for drop in range(len(labs)):
        sub=[labs[i] for i in range(len(labs)) if i!=drop]
        r=b1(sub, rng); res.append((drop+1, r["gap"] if r else None, r["p"] if r else None))
    return res

# ---------- B2 neighbour-licensed flips ----------
def b2(labs_list, allowed=None, rng=None):
    """Neighbour-licensed flips, counted only on units with a real majority (ties dropped:
    when all runs disagree there is no majority to depart from).
    Majority units form fragments (consecutive units with the same majority label).
    Event = one run giving one fragment one alternative label (on >= 1 unit); counting
    events, not units, keeps one relabelled 60-line passage from counting 60 times.
    Licensed = the alternative is the majority state of an adjacent fragment.
    Null: each event's alternative is drawn at random from the other states used in the text."""
    rng = rng or random.Random(0)
    ll = per_line(labs_list)
    maj = {}
    for x, v in ll.items():
        t = Counter(v).most_common()
        if len(t) == 1 or t[0][1] > t[1][1]: maj[x] = t[0][0]
    units = sorted(maj)
    if not units: return None
    frag, fid = [], {}
    for x in units:
        if frag and frag[-1][0] == maj[x]: frag[-1][1].append(x)
        else: frag.append((maj[x], [x]))
        fid[x] = len(frag) - 1
    nb = [{frag[j][0] for j in (i-1, i+1) if 0 <= j < len(frag)} for i in range(len(frag))]
    vocab = sorted({f for v in ll.values() for f in v})
    events, line_flags = {}, []
    for r, lab in enumerate(labs_list):
        for x in units:
            if allowed is not None and x not in allowed: continue
            if x in lab and lab[x] != maj[x]:
                i = fid[x]; ok = lab[x] in nb[i]
                line_flags.append(ok)
                events.setdefault((r, i, lab[x]), [0, ok])[0] += 1
    if not events: return None
    ev = list(events.items())
    obs = sum(1 for _, (n, ok) in ev if ok) / len(ev)
    pools = [([f for f in vocab if f != frag[i][0]], nb[i]) for (r, i, alt), _ in ev]
    null = [sum(1 for pool, ls in pools if rng.choice(pool) in ls) / len(pools) for _ in range(N_PERM)]
    nm = sum(null) / len(null); p = (sum(1 for z in null if z >= obs) + 1) / (N_PERM + 1)
    return dict(obs=obs, null=nm, gap=obs-nm, p=p, n_event=len(ev), n_flip=len(line_flags),
                line_share=sum(line_flags)/len(line_flags), n_fragments=len(frag),
                n_units_majority=len(units))

# ---------- main ----------
def main():
    data_dir = sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].startswith("--") else "data"
    source = "human1"
    if "--source" in sys.argv: source = sys.argv[sys.argv.index("--source")+1]
    rows=[]
    for corpus in CORPORA:
        runs=load_runs(data_dir, corpus, source)
        if len(runs)<2:
            print(f"[skip] {corpus}: <2 runs"); continue
        bK,lK,bk,lk = boundary_label_kappas(runs)
        # each test has its own seeded RNG, so tests do not shift each other's numbers
        R=lambda t: random.Random(f"{t}-{corpus}-12345")
        B1=b1([r[1] for r in runs], R("b1")); B2=b2([r[1] for r in runs], rng=R("b2")); LOO=b1_loo(runs, R("loo"))
        BM=boundary_match(runs)
        # B2 robustness: only units more than 2 units away from any boundary in any run
        allb=set().union(*[r[2] for r in runs]); allu=set().union(*[set(r[1]) for r in runs])
        far={u for u in allu if all(abs(u-b)>2 for b in allb)}
        B2F=b2([r[1] for r in runs], far, rng=R("b2far"))
        if BM is not None: BM["B2_far_from_boundaries"]=B2F
        rows.append((corpus,len(runs),bK,lK,bK-lK,B1,B2,LOO,BM))

    print("\n### §4 within-"+source+" results (reproducible)\n")
    print(f"{'corpus':16s}{'runs':>5s}{'bound_k':>9s}{'label_k':>9s}{'gap':>8s}"
          f"{'B1_ratio':>10s}{'B1_p':>7s}{'B2_gap':>8s}{'B2_p':>7s}{'B2_ev':>7s}")
    for c,n,bK,lK,g,B1,B2,LOO,BM in rows:
        b1r=f"{B1['ratio']:.2f}" if B1 else "--"; b1p=f"{B1['p']:.3f}" if B1 else "--"
        b2g=f"{B2['gap']:+.3f}" if B2 else "--"; b2p=f"{B2['p']:.3f}" if B2 else "--"; b2e=f"{B2['n_event']}" if B2 else "--"
        flag=" (provisional)" if n<3 else ""
        print(f"{c:16s}{n:>5d}{bK:>9.3f}{lK:>9.3f}{g:>+8.3f}{b1r:>10s}{b1p:>7s}{b2g:>8s}{b2p:>7s}{b2e:>7s}{flag}")
    print("\nleave-one-run-out on B1 gap (does the concentration survive dropping any single run?):")
    for c,n,bK,lK,g,B1,B2,LOO,BM in rows:
        if LOO: print(f"  {c:16s} " + "  ".join(f"drop_r{d}:gap={gp:+.4f}(p={pp:.3f})" for d,gp,pp in LOO))
        else:   print(f"  {c:16s} n<3, LOO not defined")
    print("\nboundary placement vs chance (share of one run's boundaries on another run's boundaries):")
    for c,n,bK,lK,g,B1,B2,LOO,BM in rows:
        if BM: print(f"  {c:16s} exact={BM['exact']:.1%}  +-1={BM['within1']:.1%}  +-2={BM['within2']:.1%}  chance={BM['chance']:.1%}")
    print("\nB2 on units > 2 units from any boundary (is B2 a side effect of shifted boundaries?):")
    for c,n,bK,lK,g,B1,B2,LOO,BM in rows:
        f=BM.get("B2_far_from_boundaries") if BM else None
        if f: print(f"  {c:16s} gap={f['gap']:+.3f}  p={f['p']:.3f}  events={f['n_event']}")
    print("\nB2 detail (events = one run relabelling one majority fragment):")
    for c,n,bK,lK,g,B1,B2,LOO,BM in rows:
        if B2: print(f"  {c:16s} licensed {B2['obs']:.0%} vs chance {B2['null']:.0%}  events={B2['n_event']}  fragments={B2['n_fragments']}  units_with_majority={B2['n_units_majority']}  (unit-level share {B2['line_share']:.0%})")
    # machine-readable
    out={c:dict(runs=n, boundary_kappa=bK, label_kappa=lK, gap=g,
                B1=B1, B2=B2, B1_loo=LOO, boundary_match=BM) for c,n,bK,lK,g,B1,B2,LOO,BM in rows}
    json.dump(out, open("paper_results.json","w"), indent=2)
    print("\nwrote paper_results.json")

# ---------- raw label agreement between runs (Limitations, point a) ----------
def agreement_levels(data_dir):
    """Mean share of units with the same label over the pairs of a triple:
    the human's 3 runs vs. each model's 3 most different runs (select_far)."""
    from projection import pairwise_agree, select_far
    mean_pair = lambda L: sum(pairwise_agree(a, b) for a, b in itertools.combinations(L, 2)) / 3
    print("raw label agreement between runs (mean over the 3 pairs of a triple):")
    hum, mod = [], []
    for c in CORPORA:
        h = mean_pair([r[1] for r in load_runs(data_dir, c, "human1")]); hum.append(h)
        row = f"  {c:16s} human {h:.2f}"
        for s in ["opus", "gemini", "qwen"]:
            m = mean_pair(select_far([r[1] for r in load_runs(data_dir, c, s)], 3)); mod.append(m)
            row += f"   {s} far-3 {m:.2f}"
        print(row)
    print(f"  range: human {min(hum):.2f}-{max(hum):.2f}, model far-3 triples {min(mod):.2f}-{max(mod):.2f}")

if __name__=="__main__":
    if "--agreement" in sys.argv:
        agreement_levels(sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].startswith("--") else "data")
    else:
        main()
