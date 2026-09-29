#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_check.py -- does the human-distinctive B1 survive a fairer model-run
selection, and does A's transfer ranking survive equalizing to n=3?

Reuses the metric code from A_transfer.py / B_structuredness.py (same folder).
  python3 robustness_check.py b1 [data]
  python3 robustness_check.py a  [data]
  python3 robustness_check.py a_folds [data]   (per-myth gain + permutation p at n=3)
"""
import os, sys, random
import A_transfer as A
import B_structuredness as B
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import projection as P

random.seed(7)
DATA = None


def medoid_runs(runs, k=3):
    """k seeds by farthest-point, assign to nearest seed, return each cluster's medoid."""
    n = len(runs)
    if n <= k:
        return runs
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - B.pairwise_agree(runs[i], runs[j])
            D[i][j] = D[j][i] = d
    i0, j0 = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                 key=lambda ij: D[ij[0]][ij[1]])
    seeds = [i0, j0]
    while len(seeds) < k:
        seeds.append(max((x for x in range(n) if x not in seeds),
                         key=lambda x: min(D[x][s] for s in seeds)))
    clusters = {s: [] for s in seeds}
    for x in range(n):
        clusters[min(seeds, key=lambda s: D[x][s])].append(x)
    meds = []
    for s, mem in clusters.items():
        meds.append(min(mem, key=lambda x: sum(D[x][y] for y in mem) / len(mem)))
    return [runs[m] for m in meds]


def b1_gap_p(runs, n_perm):
    old = B.N_PERM
    B.N_PERM = n_perm
    r = B.b1_uncertainty_map(runs)
    B.N_PERM = old
    return r


def run_b1(only=None):
    print("B1 ROBUSTNESS -- gap (obs-null) & p under 3 selections; human fixed (<=3 runs)")
    print(f"{'corpus':15s}{'source':9s}{'far gap':>9s}{'p':>6s}"
          f"{'med gap':>9s}{'p':>6s}{'rnd gap':>9s}{'sig/5':>7s}", flush=True)
    for corpus in ([only] if only else B.CORPORA):
        for source in B.SOURCES:
            allr = B.load_runs_projected(DATA, corpus, source)
            if len(allr) < 2:
                continue
            if len(allr) <= 3:                     # human etc.: fixed, one column
                r = b1_gap_p(allr, 300)
                if r:
                    print(f"{corpus:15s}{source:9s}{r['gap']:>+9.4f}{r['p']:>6.3f}"
                          f"{'  (all '+str(len(allr))+' runs, fixed)':>32s}", flush=True)
                continue
            far = b1_gap_p(B.select_diverse_runs(allr, 3), 200)
            med = b1_gap_p(medoid_runs(allr, 3), 200)
            sig = 0
            gaps = []
            NR = 5
            for _ in range(NR):
                rr = b1_gap_p(random.sample(allr, 3), 100)
                gaps.append(rr['gap'])
                sig += 1 if rr['p'] < 0.05 else 0
            rndgap = sum(gaps) / len(gaps)
            print(f"{corpus:15s}{source:9s}{far['gap']:>+9.4f}{far['p']:>6.3f}"
                  f"{med['gap']:>+9.4f}{med['p']:>6.3f}{rndgap:>+9.4f}{sig:>5d}/{NR}",
                  flush=True)


def a_equalized(select):
    summary = {}
    for source in A.SOURCES:
        by_corpus, ok = {}, True
        for c in A.CORPORA:
            # runs are chosen on the shared projection (projection.py), like everywhere else
            units = P.load_units(c, os.path.dirname(os.path.abspath(DATA)))
            projs = [(p, P.load_run(p, units)[0]) for p in A.find_runs(DATA, c, source)]
            projs = [(p, m) for p, m in projs if m]
            if not projs:
                ok = False
                break
            if len(projs) > 3:
                dicts = [m for _, m in projs]
                sel = (P.select_far(dicts, 3) if select == "far"
                       else P.select_medoid(dicts, 3))
                selid = set(id(x) for x in sel)
                chosen = [p for p, m in projs if id(m) in selid]
            else:
                chosen = [p for p, _ in projs]
            seqs = [s for s in (A.load_functions(p) for p in chosen) if len(s) >= 2]
            by_corpus[c] = seqs
        if not ok or any(len(by_corpus[c]) == 0 for c in A.CORPORA):
            continue
        folds = []
        for held in A.CORPORA:
            train = [s for c in A.CORPORA if c != held for s in by_corpus[c]]
            cond, marg = A.fit_grammar(train)
            g, _ = A.eval_gain(by_corpus[held], cond, marg)
            folds.append(g)
        summary[source] = sum(folds) / len(folds)
    return summary


def run_a():
    print("A TRANSFER at equalized n=3 (mean gain, bits/transition):")
    far = a_equalized("far")
    med = a_equalized("med")
    print(f"{'source':9s}{'far-3':>9s}{'medoid-3':>10s}")
    keys = sorted(set(far) | set(med), key=lambda s: -(med.get(s, far.get(s, 0))))
    for s in keys:
        tag = "  <- human" if s == "human1" else ""
        print(f"{s:9s}{far.get(s, float('nan')):>+9.3f}{med.get(s, float('nan')):>+10.3f}{tag}")


def a_equalized_folds(select):
    """Same selection as a_equalized, but per held-out myth: gain and permutation p
    (A.perm_null / A.p_value, as in A_transfer.run). p = share of shuffled-order nulls
    with gain >= observed (one-sided: 'better than shuffled order')."""
    rows = {}
    for source in A.SOURCES:
        by_corpus, ok = {}, True
        for c in A.CORPORA:
            # runs are chosen on the shared projection (projection.py), like everywhere else
            units = P.load_units(c, os.path.dirname(os.path.abspath(DATA)))
            projs = [(p, P.load_run(p, units)[0]) for p in A.find_runs(DATA, c, source)]
            projs = [(p, m) for p, m in projs if m]
            if not projs:
                ok = False
                break
            if len(projs) > 3:
                dicts = [m for _, m in projs]
                sel = (P.select_far(dicts, 3) if select == "far"
                       else P.select_medoid(dicts, 3))
                selid = set(id(x) for x in sel)
                chosen = [p for p, m in projs if id(m) in selid]
            else:
                chosen = [p for p, _ in projs]
            seqs = [s for s in (A.load_functions(p) for p in chosen) if len(s) >= 2]
            by_corpus[c] = seqs
        if not ok or any(len(by_corpus[c]) == 0 for c in A.CORPORA):
            continue
        folds = []
        for held in A.CORPORA:
            train = [s for c in A.CORPORA if c != held for s in by_corpus[c]]
            cond, marg = A.fit_grammar(train)
            g, n = A.eval_gain(by_corpus[held], cond, marg)
            p = A.p_value(g, A.perm_null(by_corpus[held], cond, marg))
            folds.append((held, g, p, n))
        rows[source] = folds
    return rows


def run_a_folds():
    print("A TRANSFER at equalized n=3, per held-out myth (gain bits/transition, permutation p):")
    for select in ("far", "med"):
        print(f"\n[{select}-3]")
        rows = a_equalized_folds(select)
        print(f"{'source':9s}" + "".join(f"{c.split('_')[-1]:>20s}" for c in A.CORPORA) + f"{'mean':>9s}")
        for s, folds in rows.items():
            cells = "".join(f"{g:>+10.3f} (p={p:.3f})" for _, g, p, _ in folds)
            tag = "  <- human" if s == "human1" else ""
            print(f"{s:9s}{cells}{sum(f[1] for f in folds)/3:>+9.3f}{tag}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "b1"
    here = os.path.dirname(os.path.abspath(__file__))
    DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(
        os.path.join(here, "..", "..", "data"))
    print(f"data: {DATA}\n")
    if which == "a":
        run_a()
    elif which == "a_folds":
        run_a_folds()
    else:
        only = sys.argv[3] if len(sys.argv) > 3 else None
        run_b1(only)
