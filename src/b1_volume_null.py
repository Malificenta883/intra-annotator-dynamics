#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b1_volume_null.py -- B1 (label concentration) with a baseline that keeps the AMOUNT of
disagreement fixed and only moves WHERE it falls. Separate check; paper_results.py unchanged.

Why: the paper's B1 baseline shuffles each run's labels across units. That also raises the
total amount of disagreement (runs agree only at chance after shuffling), so the variance
ratio mixes "how much" with "where". With 2 runs it measures only "how much".

Logic (K runs, units covered by all K runs, in text order):
  votes v_i   = K - (count of the most common label on unit i)   # dissenting votes, 0..K-1
  V           = sum of v_i                                         # total disagreement, kept fixed
  n_full      = number of units with v_i = K-1                     # K=3: all three runs differ
  With V and N fixed, the variance of d_i = v_i/K depends only on how the votes stack,
  for K=3 exactly: Var = (V + 2*n_full)/(9N) - mean^2. So the test is: do dissenting votes
  of different runs land on the SAME units more often than if the same votes were scattered?

Two baselines (N_PERM draws each), both keep V exactly:
  single : every vote is placed on a random unit (a unit holds at most K-1 votes).
  pieces : disagreement comes in stretches (a run relabels a whole passage). A piece is a
           maximal stretch of consecutive units with the same label tuple and v>0; a piece
           with v votes is split into v one-vote pieces of the same length. Each piece is put
           at a random position; placements that would exceed K-1 votes on a unit are redrawn.
Reported: n_full observed vs expected, variance ratio obs/baseline, one-sided p
(share of baseline draws with variance >= observed), and the old B1 ratio for reference.

Readers: human = its own 3 runs; models = 3 of 10 runs, far and medoid (as in b2_models_matched.py).
Run:  python3 b1_volume_null.py [data_dir]      writes b1_volume_null.json where it is run.
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

def votes(labs):
    K = len(labs)
    units = sorted(set.intersection(*[set(m) for m in labs]))
    tup = [tuple(m[u] for m in labs) for u in units]
    v = [K - Counter(t).most_common(1)[0][1] for t in tup]
    return K, units, tup, v

def var(xs):
    m = sum(xs) / len(xs); return sum((x - m) ** 2 for x in xs) / len(xs)

def pieces_of(tup, v):
    out, i = [], 0
    while i < len(v):
        if v[i] == 0: i += 1; continue
        j = i
        while j + 1 < len(v) and tup[j + 1] == tup[i]: j += 1
        out += [j - i + 1] * v[i]          # v one-vote pieces of this length
        i = j + 1
    return out

def place(lengths, N, cap, rng, tries=200):
    c = [0] * N
    for L in sorted(lengths, reverse=True):      # long pieces first, fewer redraws
        for _ in range(tries):
            s = rng.randrange(N - L + 1)
            if all(c[k] < cap for k in range(s, s + L)): break
        else:
            return None                           # could not fit: drop this draw
        for k in range(s, s + L): c[k] += 1
    return c

def test(labs, rng):
    K, units, tup, v = votes(labs)
    N, V, cap = len(v), sum(v), K - 1
    d = [x / K for x in v]; obs = var(d)
    nfull = sum(1 for x in v if x == cap)
    res = dict(K=K, N=N, V=V, n_full=nfull, var_obs=obs)
    for name, lengths in [("single", [1] * V), ("pieces", pieces_of(tup, v))]:
        vs, nf = [], []
        while len(vs) < pr.N_PERM:
            c = place(lengths, N, cap, rng)
            if c is None: continue
            vs.append(var([x / K for x in c])); nf.append(sum(1 for x in c if x == cap))
        m = sum(vs) / len(vs)
        res[name] = dict(n_full_expected=sum(nf) / len(nf), ratio=obs / m if m else float("nan"),
                         p=(sum(1 for z in vs if z >= obs) + 1) / (len(vs) + 1),
                         n_pieces=len(lengths), mean_piece_len=sum(lengths) / len(lengths) if lengths else 0)
    return res

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(HERE, "..", "data")
    only = sys.argv[sys.argv.index("--corpus") + 1] if "--corpus" in sys.argv else None
    out = {}
    print(f"{'corpus':15s}{'reader':7s}{'sel':7s}{'N':>5s}{'V':>5s}{'all3':>6s} | {'single: exp':>11s}{'ratio':>7s}{'p':>7s} | "
          f"{'pieces: exp':>11s}{'ratio':>7s}{'p':>7s} | {'old B1':>7s}")
    for c in pr.CORPORA:
        if only and c != only: continue
        for src in ["human1", "opus", "gemini", "qwen"]:
            labs = [r[1] for r in pr.load_runs(data_dir, c, src)]
            if not labs: continue
            sels = [("own", labs)] if src == "human1" else \
                   [("far", select_diverse_runs(labs, 3)), ("medoid", medoid_runs(labs, 3))]
            for name, sel in sels:
                r = test(sel, random.Random(f"b1v-{c}-{SEED_NAME.get(src, src)}-{name}"))
                old = pr.b1(sel, random.Random(f"b1-{c}-12345" if src == "human1" else f"b1-{c}-{src}-{name}"))
                r["old_b1_ratio"] = old["ratio"] if old else None
                r["runs"] = [labs.index(m) + 1 for m in sel]
                out[f"{c}/{src}/{name}"] = r
                s, p = r["single"], r["pieces"]
                print(f"{c:15s}{src:7s}{name:7s}{r['N']:>5d}{r['V']:>5d}{r['n_full']:>6d} | {s['n_full_expected']:>11.1f}"
                      f"{s['ratio']:>7.2f}{s['p']:>7.3f} | {p['n_full_expected']:>11.1f}{p['ratio']:>7.2f}{p['p']:>7.3f} | "
                      f"{r['old_b1_ratio']:>7.2f}", flush=True)
    fn = f"b1_volume_null{'_' + only if only else ''}.json"
    json.dump(out, open(fn, "w"), indent=1); print("wrote", fn)

if __name__ == "__main__":
    main()
