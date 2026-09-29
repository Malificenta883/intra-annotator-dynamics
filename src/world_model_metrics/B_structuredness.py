#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_structuredness.py  --  Metric B: structured instability ("world model" candidate #2)

We threw out "stability = model". What is left is HOW a source is unstable.
A model-bearing reader is unstable in a structured way; a model-free one is noisy.

Two tests, both per (corpus, source) with >= 2 runs, both line-projected
(each run's segment labels spread over its covered lines), both with permutation nulls:

B2  LICENSED FLIPS (headline; lots of data -- every disagreeing line is a datum).
    When runs disagree on a line's function, is the alternative label a
    STRUCTURALLY-LICENSED neighbour (a function actually used within +/-W lines)
    or a random other state? Licensed >> chance  =>  even when unsure the reader
    stays inside the local grammar. This operationalizes Hanson's "resistance":
    structured, not random, departure.

B1  UNCERTAINTY MAP (does *where* it is hard reproduce?).
    Is per-line disagreement CONCENTRATED (some lines reliably hard, others easy)
    or spread uniformly (noise)? Concentration is measured by the variance of
    per-line disagreement, compared to a null that relocates the same labels
    across lines (keeps each run's composition, destroys spatial structure).
    Higher-than-null concentration  =>  a stable map of ambiguity = knowing where
    you are unsure.

Human = 'human1'. No presumption the human wins; we print human and every model
side by side. Pure standard library.  Run:  python3 B_structuredness.py [path/to/data]
"""

import os, sys, glob, json, random, math, re
from collections import Counter

random.seed(12345)
N_PERM = 400
W = 3                       # neighbourhood half-window (lines) for "licensed"

FUNCTIONS = ["preparation", "contact", "exchange", "disruption",
             "negotiation", "stabilization", "return"]
FSET = set(FUNCTIONS)

SOURCES = ["human1", "opus", "gemini", "qwen", "qwen_fast"]   # human1 = the human reader
CORPORA = ["gudea", "inanna_enki", "inanna_descent"]


# ---------------------------------------------------------------- loading
def project_lines(path):
    """Return {line_number: function} for one run (segment label spread over its lines)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    segs = d["segments"] if isinstance(d, dict) else d
    segs = sorted(segs, key=lambda s: (s.get("line_start", 0), s.get("line_end", 0)))
    line2f = {}
    for s in segs:
        f = s.get("function")
        if f not in FSET:
            continue
        a, b = s.get("line_start"), s.get("line_end")
        if a is None or b is None:
            continue
        for ln in range(int(a), int(b) + 1):
            line2f[ln] = f            # later segment wins on any overlap
    return line2f


def find_runs(data_dir, corpus, source):
    if source == "qwen_fast":
        pat = os.path.join(data_dir, corpus, "Fast", "qwen_run*.json")
    else:
        pat = os.path.join(data_dir, corpus, f"{source}_run*.json")
    return sorted(glob.glob(pat))


def load_runs_projected(data_dir, corpus, source):
    runs = []
    for p in find_runs(data_dir, corpus, source):
        try:
            m = project_lines(p)
        except Exception as e:
            sys.stderr.write(f"  [skip] {p}: {e}\n")
            continue
        if m:
            runs.append(m)
    return runs


EQUALIZE_K = 3      # for B1/B2: cap models to K diverse runs (human has <=3)


def pairwise_agree(m1, m2):
    sh = set(m1) & set(m2)
    return sum(m1[l] == m2[l] for l in sh) / len(sh) if sh else 0.0


def select_diverse_runs(runs, k=EQUALIZE_K):
    """Farthest-point sampling on distance=1-agreement: maximally separated
    representatives (~one per distinct cluster of readings)."""
    n = len(runs)
    if n <= k:
        return runs
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - pairwise_agree(runs[i], runs[j])
            D[i][j] = D[j][i] = d
    i0, j0 = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                 key=lambda ij: D[ij[0]][ij[1]])
    chosen = [i0, j0]
    while len(chosen) < k:
        nxt = max((x for x in range(n) if x not in chosen),
                  key=lambda x: min(D[x][c] for c in chosen))
        chosen.append(nxt)
    return [runs[c] for c in chosen]


def mean_within_agree(runs):
    ags = [pairwise_agree(runs[i], runs[j])
           for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return sum(ags) / len(ags) if ags else float("nan")


def get_runs(data_dir, corpus, source, equalize=True):
    runs = load_runs_projected(data_dir, corpus, source)
    if equalize and len(runs) > EQUALIZE_K:
        runs = select_diverse_runs(runs)
    return runs


def per_line_labels(runs):
    """{line: [function per run that covers it]} restricted to lines covered by >=2 runs."""
    cov = {}
    for m in runs:
        for ln, f in m.items():
            cov.setdefault(ln, []).append(f)
    return {ln: labs for ln, labs in cov.items() if len(labs) >= 2}


# ---------------------------------------------------------------- B2: licensed flips
def majority_labeling(line_labels):
    return {ln: Counter(labs).most_common(1)[0][0] for ln, labs in line_labels.items()}


def block_neighbor_licensed(maj, lines_sorted):
    """For each line, licensed = functions of the ADJACENT majority-blocks
    (the structural states bordering this region). Neighbours differ from the
    current function by construction, so the licensed set is a subset of {!=m}."""
    # compress the majority sequence into maximal same-function blocks
    blocks = []          # (function, [line indices in lines_sorted])
    for idx, ln in enumerate(lines_sorted):
        f = maj[ln]
        if blocks and blocks[-1][0] == f:
            blocks[-1][1].append(idx)
        else:
            blocks.append((f, [idx]))
    lic = {}
    for bi, (f, idxs) in enumerate(blocks):
        nb = set()
        if bi > 0:
            nb.add(blocks[bi - 1][0])
        if bi < len(blocks) - 1:
            nb.add(blocks[bi + 1][0])
        for idx in idxs:
            lic[lines_sorted[idx]] = nb
    return lic


def b2_licensed_flips(runs):
    line_labels = per_line_labels(runs)
    if not line_labels:
        return None
    maj = majority_labeling(line_labels)
    lines_sorted = sorted(line_labels)
    vocab = sorted({f for labs in line_labels.values() for f in labs})
    lic = block_neighbor_licensed(maj, lines_sorted)

    alts, lic_flags, bars = [], [], []   # bars[i] = the !=m draw pool for line i
    for ln in lines_sorted:
        m = maj[ln]
        lset = lic[ln]
        pool = [f for f in vocab if f != m]      # fair null pool: alternatives only
        if not pool:
            continue
        for f in line_labels[ln]:
            if f != m:
                alts.append(ln)
                lic_flags.append(1 if f in lset else 0)
                bars.append((pool, lset))
    n_flip = len(alts)
    if n_flip == 0:
        return dict(n_flip=0, obs=None, null=None, gap=None, p=None,
                    n_lines=len(lines_sorted))

    obs = sum(lic_flags) / n_flip
    null = []
    for _ in range(N_PERM):
        hit = 0
        for pool, lset in bars:
            if random.choice(pool) in lset:
                hit += 1
        null.append(hit / n_flip)
    null_mean = sum(null) / len(null)
    ge = sum(1 for x in null if x >= obs)
    p = (ge + 1) / (len(null) + 1)
    return dict(n_flip=n_flip, obs=obs, null=null_mean,
                gap=obs - null_mean, p=p, n_lines=len(lines_sorted))


# ---------------------------------------------------------------- B1: uncertainty map
def disagreement(labs):
    return 1.0 - Counter(labs).most_common(1)[0][1] / len(labs)


def variance(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    mu = sum(xs) / n
    return sum((x - mu) ** 2 for x in xs) / n


def b1_uncertainty_map(runs):
    line_labels = per_line_labels(runs)
    if len(line_labels) < 4:
        return None
    lines_sorted = sorted(line_labels)
    obs_dis = [disagreement(line_labels[ln]) for ln in lines_sorted]
    obs_var = variance(obs_dis)

    # null: independently permute each run's line->label mapping across the
    # shared lines (keeps each run's label composition, destroys spatial structure).
    run_maps = []
    for m in runs:
        shared = {ln: m[ln] for ln in lines_sorted if ln in m}
        run_maps.append(shared)
    null = []
    for _ in range(N_PERM):
        cov = {ln: [] for ln in lines_sorted}
        for shared in run_maps:
            keys = list(shared.keys())
            vals = list(shared.values())
            random.shuffle(vals)
            for k, v in zip(keys, vals):
                cov[k].append(v)
        dis = [disagreement(cov[ln]) for ln in lines_sorted if len(cov[ln]) >= 2]
        null.append(variance(dis))
    null_mean = sum(null) / len(null)
    ge = sum(1 for x in null if x >= obs_var)
    p = (ge + 1) / (len(null) + 1)
    return dict(n_lines=len(lines_sorted), obs_var=obs_var,
                null_var=null_mean, gap=obs_var - null_mean, p=p)


# ---------------------------------------------------------------- B3: context-modulated recency
def run_number(path):
    m = re.search(r"run(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_run_full(path):
    """Return ({line: function}, set_of_boundary_lines) for one run."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    segs = d["segments"] if isinstance(d, dict) else d
    segs = sorted(segs, key=lambda s: (s.get("line_start", 0), s.get("line_end", 0)))
    line2f, bnds = {}, set()
    for s in segs:
        f = s.get("function")
        a, b = s.get("line_start"), s.get("line_end")
        if f not in FSET or a is None or b is None:
            continue
        a, b = int(a), int(b)
        for ln in range(a, b + 1):
            line2f[ln] = f
        bnds.add(a)
        bnds.add(b)
    return line2f, bnds


def b3_context_recency(data_dir, corpus, source, half=2):
    """
    Does the LAST run align with the RECENT prior or the DISTANT prior, and is
    that choice modulated by local structure (proximity to a segment boundary)?
      recency_frac  = P(last matches recent | line is informative)   (0.5 = no bias)
      diff          = recency_frac(interior) - recency_frac(boundary)
                      >0  => recency weakens at boundaries = structure modulates resistance
    Informative line = last matches EXACTLY ONE of {recent, distant}.
    Models are the negative control (i.i.d. fresh-session runs -> expect no bias/modulation).
    """
    paths = find_runs(data_dir, corpus, source)
    if len(paths) < 3:
        return None
    runs = sorted(((run_number(p),) + load_run_full(p) for p in paths),
                  key=lambda t: t[0])
    last_n, last_f, last_b = runs[-1]
    rec_f = runs[-2][1]        # recent prior
    dist_f = runs[0][1]        # distant prior (max temporal gap)

    matched_recent, is_boundary = [], []
    for ln, fl in last_f.items():
        if ln in rec_f and ln in dist_f:
            mr = (fl == rec_f[ln])
            md = (fl == dist_f[ln])
            if mr ^ md:                       # exactly one -> informative
                matched_recent.append(1 if mr else 0)
                is_boundary.append(1 if any(abs(ln - e) <= half for e in last_b) else 0)
    n = len(matched_recent)
    if n < 6:
        return dict(n=n, recency=None, diff=None, p=None,
                    order=f"r{runs[0][0]}<r{runs[-2][0]}<r{last_n}")
    recency = sum(matched_recent) / n
    bnd = [m for m, b in zip(matched_recent, is_boundary) if b]
    inte = [m for m, b in zip(matched_recent, is_boundary) if not b]
    fb = sum(bnd) / len(bnd) if bnd else float("nan")
    fi = sum(inte) / len(inte) if inte else float("nan")
    diff = (fi - fb) if (bnd and inte) else None

    p = None
    if diff is not None:
        labels = is_boundary[:]
        null = []
        for _ in range(N_PERM):
            random.shuffle(labels)
            b = [m for m, lab in zip(matched_recent, labels) if lab]
            i = [m for m, lab in zip(matched_recent, labels) if not lab]
            null.append((sum(i) / len(i)) - (sum(b) / len(b)))
        ge = sum(1 for x in null if x >= diff)
        p = (ge + 1) / (len(null) + 1)
    return dict(n=n, recency=recency, fb=fb, fi=fi, diff=diff, p=p,
                order=f"r{runs[0][0]}<r{runs[-2][0]}<r{last_n}")


# ---------------------------------------------------------------- main
def run(data_dir):
    print("=" * 84)
    print("METRIC B  --  structured instability (is the disagreement organized, or noise?)")
    print(f"W={W} lines,  {N_PERM} permutations,  human = 'human1'")
    print("=" * 84)

    # load each (corpus, source) once, EQUALIZED to K diverse runs; keep both results
    print(f"(B1/B2 use up to {EQUALIZE_K} diverse runs per source; 'agree' = mean pairwise "
          f"agreement among the selected runs -- lower = genuinely more varied)")
    b2_rows, b1_rows = [], []
    for corpus in CORPORA:
        for source in SOURCES:
            runs = get_runs(data_dir, corpus, source, equalize=True)
            if len(runs) < 2:
                continue
            wa = mean_within_agree(runs)
            b2_rows.append((corpus, source, len(runs), wa, b2_licensed_flips(runs)))
            b1_rows.append((corpus, source, len(runs), wa, b1_uncertainty_map(runs)))

    print("\nB2  LICENSED FLIPS  (alt label = a bordering-block state vs a random !=majority state)")
    print(f"{'corpus':16s}{'source':10s}{'n':>3s}{'agree':>7s}{'n_flip':>7s}"
          f"{'obs':>8s}{'null':>8s}{'gap':>8s}{'p':>8s}")
    for corpus, source, nr, wa, r in b2_rows:
        if r is None or r["n_flip"] == 0:
            print(f"{corpus:16s}{source:10s}{nr:>3d}{wa:>7.2f}{0:>7d}"
                  f"{'--':>8s}{'--':>8s}{'--':>8s}{'--':>8s}")
            continue
        star = "*" if r["p"] < 0.05 else ""
        print(f"{corpus:16s}{source:10s}{nr:>3d}{wa:>7.2f}{r['n_flip']:>7d}"
              f"{r['obs']:>8.3f}{r['null']:>8.3f}{r['gap']:>+8.3f}{r['p']:>8.3f}{star}")

    print("\nB1  UNCERTAINTY MAP  (variance of per-line disagreement vs relocated-label null)")
    print(f"{'corpus':16s}{'source':10s}{'n':>3s}{'agree':>7s}{'n_lines':>8s}"
          f"{'obs_var':>9s}{'null_var':>9s}{'gap':>8s}{'p':>8s}")
    for corpus, source, nr, wa, r in b1_rows:
        if r is None:
            continue
        star = "*" if r["p"] < 0.05 else ""
        print(f"{corpus:16s}{source:10s}{nr:>3d}{wa:>7.2f}{r['n_lines']:>8d}"
              f"{r['obs_var']:>9.4f}{r['null_var']:>9.4f}{r['gap']:>+8.4f}{r['p']:>8.3f}{star}")

    print("\nB3  CONTEXT-MODULATED RECENCY  (does the last run pick recent vs distant by structure?)")
    print("recency=P(match recent|informative), 0.5=no bias; diff=recency(interior)-recency(boundary)>0")
    print("=> recency weakens at boundaries = structure modulates resistance. Models = i.i.d. control.")
    print(f"{'corpus':16s}{'source':10s}{'order':14s}{'n_inf':>6s}"
          f"{'recency':>9s}{'r_int':>7s}{'r_bnd':>7s}{'diff':>8s}{'p':>8s}")
    for corpus in CORPORA:
        for source in SOURCES:
            r = b3_context_recency(data_dir, corpus, source)
            if r is None:
                continue
            if r["recency"] is None:
                print(f"{corpus:16s}{source:10s}{r['order']:14s}{r['n']:>6d}"
                      f"{'  too few informative lines':s}")
                continue
            diff = r["diff"]
            ds = f"{diff:+.3f}" if diff is not None else "   --"
            ps = f"{r['p']:.3f}" if r["p"] is not None else "   --"
            star = "*" if (r["p"] is not None and r["p"] < 0.05) else ""
            print(f"{corpus:16s}{source:10s}{r['order']:14s}{r['n']:>6d}"
                  f"{r['recency']:>9.3f}{r['fi']:>7.3f}{r['fb']:>7.3f}{ds:>8s}{ps:>7s}{star}")

    print("\n" + "-" * 84)
    print("Reading: B2 gap>0,p<0.05 => flips stay inside the local grammar (structured,")
    print("not noise). B1 gap>0,p<0.05 => a reproducible map of where the text is hard.")
    print("Either one is model-like WITHOUT invoking stability. Compare human vs models.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(os.path.join(here, "..", "..", "data"))
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data
    if not os.path.isdir(data_dir):
        sys.exit(f"data dir not found: {data_dir}")
    print(f"data: {data_dir}")
    run(data_dir)
