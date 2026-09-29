#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lag_test.py -- do a reader's runs pass through the same functions with a phase shift?
Separate check; paper_results.py unchanged. Readers: human (own 3 runs); models: 3 of 10 runs,
far and medoid (as in b2_models_matched.py). Units covered by all runs, in text order.

Pieces: maximal stretches of consecutive units with the same label tuple across the runs
(= the text cut at every boundary of every run). seq_r[k] = label of run r on piece k.

Test 1 (as agreed): lagged agreement.
  agree(d) = share of pieces k with seq_a[k] == seq_b[k+d], averaged over ordered pairs (a,b),
  d = 0..3 (ordered pairs cover both directions). Baseline: order of seq_b shuffled (N_PERM).
  Caveat: a run keeps a label over several pieces, so agree(1) is raised by that alone.
  Reported alongside: within-run persistence(d) = share with seq_a[k] == seq_a[k+d].

Test 2 (onsets; removes the persistence effect): an onset of X in run b is a piece j where
  seq_b[j] = X and seq_b[j-1] != X. For every other run a, lag = position of a's nearest onset
  of the same X minus j (negative = a entered X earlier = a leads). Lags binned -3..+3, else
  'none'. Baseline: seq_a shifted round the text by a random offset (N_PERM), which keeps a's
  own sequence and only moves it against b.
Breakdown of |lag| = 1 (observed only): the run that enters X later either
  - "slide"  : also changes label at the earlier piece (its cut there is shared; X arrives one
               segment later over a shared boundary), or
  - "shift"  : keeps its previous label through the earlier piece (its cut is simply placed later).
Cut check for |lag| = 1 (observed only): at the piece where the early run already entered X,
  does the late run have a segment boundary (a segment starts in that unit)? exact unit, and
  within +-2 units. Yes = the cut is shared and the label lags; no = the late run has no cut there.
All-triples mode (--triples): for models, every 3-run combination of the 10 runs (120 triples),
  each analysed exactly like the human's 3 runs (pieces from that triple's own cuts); onset-lag
  counts for lag 0 and |lag|=1 vs the circular-shift baseline with N_TRIPLE_PERM draws per triple.
  Reported: summed obs/exp over triples and the share of triples with obs > exp.
Run:  python3 lag_test.py [data_dir] [--corpus X] [--triples]   writes lag_test[_X][_triples].json
"""
import os, sys, json, random
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "world_model_metrics"))
import paper_results as pr
from projection import select_far as select_diverse_runs   # choice of 3 model runs: see projection.py
from projection import select_medoid as medoid_runs

# the human runs were called "mine" before; the old name stays in the random seeds so that every number matches the paper
SEED_NAME = {"human1": "mine"}

DMAX = 3

def piece_seqs(labs, with_units=False):
    units = sorted(set.intersection(*[set(m) for m in labs]))
    tup = [tuple(m[u] for m in labs) for u in units]
    starts = [i for i in range(len(units))
              if i == 0 or units[i] != units[i - 1] + 1 or tup[i] != tup[i - 1]]
    S = [[tup[i][r] for i in starts] for r in range(len(labs))]
    return (S, [units[i] for i in starts]) if with_units else S

def lag1_cuts(S, pu, bounds):
    """for |lag|==1 matches: does the late run have a boundary where the early run entered X?"""
    K = len(S); c = Counter()
    for a in range(K):
        for b in range(K):
            if a == b: continue
            sa, sb = S[a], S[b]
            oa = {}
            for i, x in [(0, sa[0])] + onsets(sa): oa.setdefault(x, []).append(i)
            ch = {r: {j for j, _ in onsets(S[r])} for r in (a, b)}
            for j, x in onsets(sb):
                if x not in oa: continue
                d = min((i - j for i in oa[x]), key=lambda v: (abs(v), v))
                if abs(d) != 1: continue
                if d == -1: early_piece, late = j - 1, b
                else:       early_piece, late = j, a
                kind = "slide" if early_piece in ch[late] else "prolong"
                u = pu[early_piece]; B = bounds[late]
                c[f"{kind}_cut_exact"] += u in B
                c[f"{kind}_cut_pm2"] += any(abs(u - v) <= 2 for v in B)
                c[f"{kind}_n"] += 1
    return dict(c)

def agree(sa, sb, d):
    n = len(sa) - d
    return sum(sa[k] == sb[k + d] for k in range(n)) / n if n > 0 else float("nan")

def onsets(s):
    return [(j, s[j]) for j in range(1, len(s)) if s[j] != s[j - 1]]

def onset_lags(sa, sb):
    """lag of a's nearest onset of the same label, for every onset in b"""
    oa = {}
    for i, x in [(0, sa[0])] + onsets(sa): oa.setdefault(x, []).append(i)
    out = []
    for j, x in onsets(sb):
        if x not in oa: out.append("none"); continue
        d = min((i - j for i in oa[x]), key=lambda v: (abs(v), v))
        out.append(d if abs(d) <= DMAX else "none")
    return out

BINS = list(range(-DMAX, DMAX + 1)) + ["none"]

def lag1_kind(sa, sb):
    """for |lag|==1 matches: slide vs shift (see docstring)"""
    oa = {}
    for i, x in [(0, sa[0])] + onsets(sa): oa.setdefault(x, []).append(i)
    ch_a = {i for i, _ in onsets(sa)}; ch_b = {j for j, _ in onsets(sb)}
    c = Counter()
    for j, x in onsets(sb):
        if x not in oa: continue
        d = min((i - j for i in oa[x]), key=lambda v: (abs(v), v))
        if abs(d) != 1: continue
        early, late_changes = (j - 1, ch_b) if d == -1 else (j, ch_a)   # d=-1: a early, b late
        c["slide" if early in late_changes else "shift"] += 1
    return c

def analyse(labs, rng, bounds=None):
    S, PU = piece_seqs(labs, with_units=True); K = len(S); P = len(S[0])
    pairs = [(a, b) for a in range(K) for b in range(K) if a != b]
    res = dict(pieces=P, onsets=[len(onsets(s)) for s in S])
    # test 1
    t1 = {}
    for d in range(DMAX + 1):
        obs = sum(agree(S[a], S[b], d) for a, b in pairs) / len(pairs)
        pers = sum(agree(S[a], S[a], d) for a in range(K)) / K if d else 1.0
        null = []
        for _ in range(pr.N_PERM):
            tot = 0
            for a, b in pairs:
                sb = S[b][:]; rng.shuffle(sb); tot += agree(S[a], sb, d)
            null.append(tot / len(pairs))
        m = sum(null) / len(null)
        t1[d] = dict(obs=obs, null=m, ratio=obs / m if m else float("nan"), persistence=pers,
                     p_above=(sum(z >= obs for z in null) + 1) / (len(null) + 1))
    res["lagged_agreement"] = t1
    # test 2
    obs = Counter()
    for a, b in pairs: obs.update(onset_lags(S[a], S[b]))
    null = {k: [] for k in BINS}
    for _ in range(pr.N_PERM):
        c = Counter()
        for a, b in pairs:
            k = rng.randrange(1, P) if P > 1 else 0
            c.update(onset_lags(S[a][k:] + S[a][:k], S[b]))
        for key in BINS: null[key].append(c[key])
    t2 = {}
    for key in BINS:
        o = obs[key]; nl = null[key]; m = sum(nl) / len(nl)
        t2[str(key)] = dict(obs=o, exp=m, p_above=(sum(z >= o for z in nl) + 1) / (len(nl) + 1),
                            p_below=(sum(z <= o for z in nl) + 1) / (len(nl) + 1))
    for name, keys in [("pm1", [-1, 1]), ("pm2_3", [-3, -2, 2, 3])]:
        o = sum(obs[k] for k in keys); nl = [sum(null[k][i] for k in keys) for i in range(pr.N_PERM)]
        m = sum(nl) / len(nl)
        t2[name] = dict(obs=o, exp=m, p_above=(sum(z >= o for z in nl) + 1) / (len(nl) + 1),
                        p_below=(sum(z <= o for z in nl) + 1) / (len(nl) + 1))
    res["onset_lags"] = t2
    res["n_onsets_total"] = sum(obs.values())
    kinds = Counter()
    for a, b in pairs: kinds.update(lag1_kind(S[a], S[b]))
    res["lag1_kinds"] = dict(kinds)
    if bounds is not None: res["lag1_cuts"] = lag1_cuts(S, PU, bounds)
    return res

N_TRIPLE_PERM = 200

def triple_counts(labs, rng):
    S = piece_seqs(labs); K = len(S); P = len(S[0])
    pairs = [(a, b) for a in range(K) for b in range(K) if a != b]
    def cnt(get_a):
        c = Counter()
        for a, b in pairs: c.update(onset_lags(get_a(a), S[b]))
        return c[0], c[-1] + c[1]
    o0, o1 = cnt(lambda a: S[a])
    e0 = e1 = 0.0
    for _ in range(N_TRIPLE_PERM):
        c = Counter()
        for a, b in pairs:
            k = rng.randrange(1, P) if P > 1 else 0
            c.update(onset_lags(S[a][k:] + S[a][:k], S[b]))
        e0 += c[0]; e1 += c[-1] + c[1]
    return o0, e0 / N_TRIPLE_PERM, o1, e1 / N_TRIPLE_PERM

def main_triples(data_dir, only):
    import itertools
    out = {}
    print(f"{'corpus':15s}{'reader':7s}{'triples':>8s} | {'lag0 obs/exp':>14s}{'share>':>8s} | {'±1 obs/exp':>14s}{'share>':>8s} | ±1/exp ratio: min median max")
    for c in pr.CORPORA:
        if only and c != only: continue
        for src in ["human1", "opus", "gemini", "qwen"]:
            labs = [r[1] for r in pr.load_runs(data_dir, c, src)]
            if not labs: continue
            rng = random.Random(f"lagT-{c}-{SEED_NAME.get(src, src)}")
            rows = [triple_counts([labs[i] for i in t], rng) for t in itertools.combinations(range(len(labs)), 3)]
            O0 = sum(r[0] for r in rows); E0 = sum(r[1] for r in rows)
            O1 = sum(r[2] for r in rows); E1 = sum(r[3] for r in rows)
            s0 = sum(r[0] > r[1] for r in rows) / len(rows); s1 = sum(r[2] > r[3] for r in rows) / len(rows)
            rat = sorted(r[2] / r[3] for r in rows if r[3] > 0)
            med = rat[len(rat) // 2] if rat else float("nan")
            out[f"{c}/{src}"] = dict(triples=len(rows), lag0_obs=O0, lag0_exp=E0, lag0_share_above=s0,
                                     pm1_obs=O1, pm1_exp=E1, pm1_share_above=s1,
                                     pm1_ratio_min=rat[0] if rat else None, pm1_ratio_median=med,
                                     pm1_ratio_max=rat[-1] if rat else None,
                                     per_triple=[list(r) for r in rows])
            print(f"{c:15s}{src:7s}{len(rows):>8d} | {O0:>6d}/{E0:<7.1f}{s0:>8.0%} | {O1:>6d}/{E1:<7.1f}{s1:>8.0%} | "
                  f"{rat[0]:.2f} {med:.2f} {rat[-1]:.2f}", flush=True)
    # Table 4, model columns: extremes over all model triples of each myth
    print("\nTable 4 summary (models = all triples of opus, gemini, qwen):")
    print(f"{'corpus':15s}{'human same-place/exp*':>22s}{'human 1-step/same':>19s}{'models: lowest same-place/exp':>32s}{'models: max 1-step/same':>25s}")
    for c in pr.CORPORA:
        if f"{c}/human1" not in out: continue
        h = out[f"{c}/human1"]["per_triple"][0]
        M = [r for s_ in ("opus", "gemini", "qwen") for r in out.get(f"{c}/{s_}", {}).get("per_triple", [])]
        low = min(r[0] / r[1] for r in M if r[1] > 0); top = max(r[2] / max(r[0], 1) for r in M)
        out[f"{c}/summary"] = dict(models_lowest_same_place_ratio=low, models_max_onestep_per_same=top,
                                   human_onestep_per_same=h[2] / max(h[0], 1))
        print(f"{c:15s}{h[0] / h[1]:>22.2f}{h[2] / max(h[0], 1):>19.2f}{low:>32.2f}{top:>25.2f}")
    print("* here with 200 draws, like the models; Table 4 gives the human value from the main mode (1,000 draws).")
    fn = f"lag_test{'_' + only if only else ''}_triples.json"
    json.dump(out, open(fn, "w"), indent=1); print("wrote", fn)

def main():
    if "--triples" in sys.argv:
        data_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(HERE, "..", "data")
        only = sys.argv[sys.argv.index("--corpus") + 1] if "--corpus" in sys.argv else None
        return main_triples(data_dir, only)
    data_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(HERE, "..", "data")
    only = sys.argv[sys.argv.index("--corpus") + 1] if "--corpus" in sys.argv else None
    out = {}
    for c in pr.CORPORA:
        if only and c != only: continue
        print(f"\n== {c}")
        print(f"{'reader':7s}{'sel':7s}{'pcs':>4s} | lagged agreement obs/null (persistence)            | onset lags obs/exp: 0 | ±1 | ±2..3 | none")
        for src in ["human1", "opus", "gemini", "qwen"]:
            runs = pr.load_runs(data_dir, c, src)
            labs = [r[1] for r in runs]
            if not labs: continue
            sels = [("own", labs)] if src == "human1" else \
                   [("far", select_diverse_runs(labs, 3)), ("medoid", medoid_runs(labs, 3))]
            for name, sel in sels:
                bnds = [runs[labs.index(m)][2] for m in sel]
                r = analyse(sel, random.Random(f"lag-{c}-{SEED_NAME.get(src, src)}-{name}"), bnds)
                r["runs"] = [labs.index(m) + 1 for m in sel]
                out[f"{c}/{src}/{name}"] = r
                t1 = r["lagged_agreement"]; t2 = r["onset_lags"]
                la = "  ".join(f"d{d}:{t1[d]['obs']:.2f}/{t1[d]['null']:.2f}({t1[d]['persistence']:.2f})" for d in range(DMAX + 1))
                g = lambda ks: (sum(t2[str(k)]["obs"] for k in ks), sum(t2[str(k)]["exp"] for k in ks))
                z, o1, o23, nn = g([0]), g([-1, 1]), g([-3, -2, 2, 3]), g(["none"])
                print(f"{src:7s}{name:7s}{r['pieces']:>4d} | {la} | {z[0]}/{z[1]:.1f} | {o1[0]}/{o1[1]:.1f} | {o23[0]}/{o23[1]:.1f} | {nn[0]}/{nn[1]:.1f} | ±1 slide/shift {r['lag1_kinds'].get('slide',0)}/{r['lag1_kinds'].get('shift',0)}", flush=True)
    fn = f"lag_test{'_' + only if only else ''}.json"
    json.dump(out, open(fn, "w"), indent=1); print("wrote", fn)

if __name__ == "__main__":
    main()
