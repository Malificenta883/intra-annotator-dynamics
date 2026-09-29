#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
passage_inspection.py -- where and how a reader's runs disagree on labels, passage by passage.
Separate check; paper_results.py unchanged. Readers: human (own 3 runs); models: 3 of 10 runs,
far and medoid (as in b2_models_matched.py). Units covered by all 3 runs, in text order.

Definitions
  unstable unit : the 3 runs are not unanimous.
  majority      : label held by >= 2 runs; a unit where all 3 differ has none (type D).
  fragment      : maximal stretch of consecutive units with the same majority label.
  passage       : maximal stretch of consecutive unstable units with the same label tuple
                  (run1, run2, run3). A 2-1 passage has one odd run and one alternative label.
  neighbour label (of a fragment, each side): majority label of the nearest majority unit
                  beyond the fragment edge (units without majority are skipped, as in B2).
Types of passages
  A  edge shift     : passage touches an edge of its fragment and the alternative equals the
                      neighbour label on THAT side (= a boundary placed a few units off).
  B  neighbour pull : alternative equals a neighbour label, but not type A (inside the fragment,
                      or at the edge facing the other neighbour).
  C  new label      : alternative is not a neighbour label.
  D  split          : all three runs differ.
  M  mirror swap    : checked first. Two runs use the same two functions X != Y on two nearby
                      pieces in opposite order (run a: X then Y; run b: Y then X). A piece is a
                      maximal stretch of consecutive units with the same label tuple (stable or not);
                      "nearby" = adjacent pieces, or pieces with one piece in between (gap=1).
                      A passage in such a pair gets type M; its A-D type is kept as base_type.
                      (A plain boundary shift is never M: one of the two runs has one label on both.)

Tests (counts are passages, not units; one-sided p both ways: above / below chance)
  1. Edge test   : A passages vs expectation if each 2-1 passage (same length) sat at a random
                   position inside its fragment. Exact (Poisson-binomial).
  2. B vs C      : among non-A 2-1 passages, share whose alternative is a neighbour label vs
                   expectation if the alternative were drawn at random from the other states used
                   in the text (as the B2 null). Exact.
  3. Hard zones  : do different runs deviate in the same zones? For each odd-run mask O_r (units
                   where run r is the odd one in a 2-1 unit), count stretches of O_a that have an
                   O_b unit within W units; summed over ordered pairs a != b. Null: each mask
                   shifted round the text by an independent random offset (keeps lengths and
                   stretch shapes, moves only positions), N_PERM draws.
Run:  python3 passage_inspection.py [data_dir] [--corpus X]
Writes passage_inspection[_X].json and, for the human, passages_human[_X].md (with the text).
"""
import os, sys, json, random
from collections import Counter
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "world_model_metrics"))
import paper_results as pr
from projection import load_units, _UNIT_RE, _NOTE_RE
from projection import select_far as select_diverse_runs   # choice of 3 model runs: see projection.py
from projection import select_medoid as medoid_runs

# the human runs were called "mine" before; the old name stays in the random seeds so that every number matches the paper
SEED_NAME = {"human1": "mine"}

W = 2

def pbinom_tail(ps, k):
    """P(X >= k) and P(X <= k) for a sum of independent Bernoulli(ps)."""
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            new[i] += q * (1 - p); new[i + 1] += q * p
        dist = new
    return sum(dist[k:]), sum(dist[:k + 1])

def analyse(labs, rng):
    K = len(labs)
    units = sorted(set.intersection(*[set(m) for m in labs]))
    N = len(units)
    tup = [tuple(m[u] for m in labs) for u in units]
    maj = []
    for t in tup:
        c = Counter(t).most_common()
        maj.append(c[0][0] if c[0][1] > K // 2 else None)
    adj = [i > 0 and units[i] == units[i - 1] + 1 for i in range(N)]   # consecutive in text

    # fragments
    fid, frags = [0] * N, []
    for i in range(N):
        if i > 0 and adj[i] and maj[i] == maj[i - 1]:
            frags[-1][1] = i
        else:
            frags.append([i, i])
        fid[i] = len(frags) - 1
    def nb_label(i, step):
        j = i + step
        while 0 <= j < N and maj[j] is None: j += step
        return maj[j] if 0 <= j < N else None
    vocab = sorted({l for t in tup for l in t})

    # passages
    passages, i = [], 0
    while i < N:
        if len(set(tup[i])) == 1: i += 1; continue
        j = i
        while j + 1 < N and adj[j + 1] and tup[j + 1] == tup[i]: j += 1
        passages.append((i, j)); i = j + 1

    # pieces (stable and unstable) and mirror swaps between pairs of runs
    pieces, i = [], 0
    while i < N:
        j = i
        while j + 1 < N and adj[j + 1] and tup[j + 1] == tup[i]: j += 1
        pieces.append((i, j)); i = j + 1
    mirrors = []
    for k in range(len(pieces)):
        for gap in (0, 1):
            q = k + 1 + gap
            if q >= len(pieces): continue
            if not all(adj[pieces[m][0]] for m in range(k + 1, q + 1)): continue   # contiguous text
            tk, tq = tup[pieces[k][0]], tup[pieces[q][0]]
            for a in range(K):
                for b in range(a + 1, K):
                    if tk[a] != tq[a] and tk[a] == tq[b] and tq[a] == tk[b]:
                        mirrors.append(dict(runs=(a + 1, b + 1), gap=gap, X=tk[a], Y=tq[a],
                                            p1=pieces[k], p2=pieces[q]))
    in_mirror = {}
    for m in mirrors:
        for pc in (m["p1"], m["p2"]): in_mirror.setdefault(pc, []).append(m)

    rows, edge_p, bc_p = [], [], []
    nA = nBC_lic = 0
    for (s, e) in passages:
        t = tup[s]; L = e - s + 1
        if maj[s] is None:
            rows.append(dict(s=s, e=e, type="D", tup=t)); continue
        M = maj[s]; odd = [r for r in range(K) if t[r] != M][0]; alt = t[odd]
        fs, fe = frags[fid[s]]; F = fe - fs + 1
        left, right = nb_label(fs, -1), nb_label(fe, +1)
        tl, tr = s == fs, e == fe
        isA = (tl and alt == left) or (tr and alt == right)
        # expectation of A if the passage sat at a random position in its fragment
        npos = F - L + 1
        if npos == 1: pA = 1.0 if (alt == left or alt == right) else 0.0
        else: pA = ((alt == left) + (alt == right)) / npos
        edge_p.append(pA); nA += isA
        if isA: typ = "A"
        else:
            lic = alt in (left, right)
            typ = "B" if lic else "C"
            pool = [f for f in vocab if f != M]
            bc_p.append(sum(1 for f in pool if f in (left, right)) / len(pool)); nBC_lic += lic
        rows.append(dict(s=s, e=e, type=typ, tup=t, maj=M, odd_run=odd + 1, alt=alt,
                         left=left, right=right, frag_len=F))
    for r in rows:
        ms = in_mirror.get((r["s"], r["e"]))
        r["base_type"] = r["type"]
        if ms:
            r["type"] = "M"
            r["mirror"] = "; ".join(f"runs {m['runs'][0]}&{m['runs'][1]}: {m['X']}->{m['Y']} vs {m['Y']}->{m['X']}"
                                    + (" (gap 1)" if m["gap"] else "") for m in ms)
    n21 = len(edge_p)
    ea, eb = pbinom_tail(edge_p, nA)
    ba, bb = pbinom_tail(bc_p, nBC_lic) if bc_p else (float("nan"), float("nan"))

    # hard zones
    masks = [[0] * N for _ in range(K)]
    for i in range(N):
        if maj[i] is not None and len(set(tup[i])) > 1:
            masks[[r for r in range(K) if tup[i][r] != maj[i]][0]][i] = 1
    def stretches(m):
        out, i = [], 0
        while i < N:
            if m[i]:
                j = i
                while j + 1 < N and m[j + 1]: j += 1
                out.append((i, j)); i = j + 1
            else: i += 1
        return out
    def zone_stat(ms):
        tot = 0
        for a in range(K):
            for b in range(K):
                if a == b: continue
                mb = ms[b]
                for (s, e) in stretches(ms[a]):
                    if any(mb[k] for k in range(max(0, s - W), min(N, e + W + 1))): tot += 1
        return tot
    obs_z = zone_stat(masks)
    null_z = []
    for _ in range(pr.N_PERM):
        sh = []
        for m in masks:
            k = rng.randrange(N); sh.append(m[k:] + m[:k])
        null_z.append(zone_stat(sh))
    mz = sum(null_z) / len(null_z)
    cnt = Counter(r["type"] for r in rows)
    res = dict(N=N, passages=len(rows), types={t: cnt.get(t, 0) for t in "MABCD"},
               unit_types={t: sum(r["e"] - r["s"] + 1 for r in rows if r["type"] == t) for t in "MABCD"},
               M_from=dict(Counter(r["base_type"] for r in rows if r["type"] == "M")),
               n_mirror_pairs=len(mirrors), n_mirror_adjacent=sum(1 for m in mirrors if m["gap"] == 0),
               edge=dict(n21=n21, A_obs=nA, A_exp=sum(edge_p), p_above=ea, p_below=eb),
               b_vs_c=dict(n=len(bc_p), licensed_obs=nBC_lic, licensed_exp=sum(bc_p), p_above=ba, p_below=bb),
               zones=dict(W=W, obs=obs_z, exp=mz, ratio=obs_z / mz if mz else float("nan"),
                          p_above=(sum(z >= obs_z for z in null_z) + 1) / (len(null_z) + 1),
                          p_below=(sum(z <= obs_z for z in null_z) + 1) / (len(null_z) + 1)))
    return res, rows, units

def unit_text(corpus, root):
    txt = {}
    for raw in (Path(root) / "texts" / f"{corpus}_numbered.txt").read_text(encoding="utf-8-sig").splitlines():
        m = _UNIT_RE.match(raw)
        if m and not _NOTE_RE.match(m.group(3)):
            a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
            txt[(a, b)] = m.group(3).strip()
    return txt

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(HERE, "..", "data")
    root = Path(data_dir).resolve().parent
    only = sys.argv[sys.argv.index("--corpus") + 1] if "--corpus" in sys.argv else None
    out, md = {}, []
    print(f"{'corpus':15s}{'reader':7s}{'sel':7s}{'pass':>5s}{'M':>4s}{'A':>4s}{'B':>4s}{'C':>4s}{'D':>4s} | "
          f"{'A obs/exp':>10s}{'p>':>6s}{'p<':>6s} | {'nb obs/exp':>11s}{'p>':>6s}{'p<':>6s} | {'zones o/e':>10s}{'p>':>6s}{'p<':>6s}")
    for c in pr.CORPORA:
        if only and c != only: continue
        units_all = load_units(c, root); txt = unit_text(c, root)
        for src in ["human1", "opus", "gemini", "qwen"]:
            labs = [r[1] for r in pr.load_runs(data_dir, c, src)]
            if not labs: continue
            sels = [("own", labs)] if src == "human1" else \
                   [("far", select_diverse_runs(labs, 3)), ("medoid", medoid_runs(labs, 3))]
            for name, sel in sels:
                res, rows, units = analyse(sel, random.Random(f"pi-{c}-{SEED_NAME.get(src, src)}-{name}"))
                res["runs"] = [labs.index(m) + 1 for m in sel]
                out[f"{c}/{src}/{name}"] = res
                T, E, B, Z = res["types"], res["edge"], res["b_vs_c"], res["zones"]
                print(f"{c:15s}{src:7s}{name:7s}{res['passages']:>5d}{T['M']:>4d}{T['A']:>4d}{T['B']:>4d}{T['C']:>4d}{T['D']:>4d} | "
                      f"{E['A_obs']:>4d}/{E['A_exp']:<5.1f}{E['p_above']:>6.3f}{E['p_below']:>6.3f} | "
                      f"{B['licensed_obs']:>4d}/{B['licensed_exp']:<6.1f}{B['p_above']:>6.3f}{B['p_below']:>6.3f} | "
                      f"{Z['obs']:>4d}/{Z['exp']:<5.1f}{Z['p_above']:>6.3f}{Z['p_below']:>6.3f}", flush=True)
                if src == "human1":
                    md.append(f"\n## {c}  (human runs; {res['passages']} passages)\n")
                    md.append("| # | type | lines | runs 1 / 2 / 3 | majority | alt (odd run) | neighbours L / R | mirror | text |")
                    md.append("|---|---|---|---|---|---|---|---|---|")
                    for k, r in enumerate(rows, 1):
                        a = units_all[units[r["s"]]][0]; b = units_all[units[r["e"]]][1]
                        text = " ".join(txt.get(units_all[units[u]], "") for u in range(r["s"], r["e"] + 1))
                        text = (text[:220] + " …") if len(text) > 220 else text
                        text = text.replace("|", "/")
                        typ = r["type"] + (f" ({r['base_type']})" if r["type"] == "M" else "")
                        mir = r.get("mirror", "")
                        if r["base_type"] == "D":
                            md.append(f"| {k} | {typ} | {a}–{b} | {' / '.join(r['tup'])} | – | – | – | {mir} | {text} |")
                        else:
                            md.append(f"| {k} | {typ} | {a}–{b} | {' / '.join(r['tup'])} | {r['maj']} | "
                                      f"{r['alt']} (run {r['odd_run']}) | {r['left']} / {r['right']} | {mir} | {text} |")
    suf = "_" + only if only else ""
    json.dump(out, open(f"passage_inspection{suf}.json", "w"), indent=1)
    if md:
        Path(f"passages_human{suf}.md").write_text("# Unstable passages, human runs\n" + "\n".join(md) + "\n", encoding="utf-8")
    print("wrote", f"passage_inspection{suf}.json", f"passages_human{suf}.md" if md else "")

if __name__ == "__main__":
    main()
