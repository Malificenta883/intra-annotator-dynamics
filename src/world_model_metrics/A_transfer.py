#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_transfer.py  --  Metric A: transferable event-grammar ("world model" candidate #1)

Idea (after Hanson & Hanson 1992, train-Game / transfer-Restaurant):
    A reader with an internal model of events carries a PORTABLE grammar of how
    narrative states follow one another (preparation -> contact -> exchange ...),
    independent of the particular myth. A reader without one produces
    corpus-idiosyncratic labels with no shared backbone.

We operationalize the grammar as the transition matrix P(next function | current
function) over consecutive segments. Portability is measured by LEAVE-ONE-CORPUS-OUT
prediction:
    * train the grammar on 2 corpora (all runs pooled),
    * predict the realized transitions of the held-out 3rd corpus,
    * score = mean log2 P(next|cur)  MINUS  the same under a marginal baseline
      P(next) (which already knows "some functions are common").
    gain > 0  =>  sequential structure learned elsewhere transfers here.

Significance per fold: a permutation null that SHUFFLES the order of the held-out
segments (destroys transition structure, keeps composition). Observed gain is
compared against the null gain distribution.

We report per-fold gain (bits/transition), a permutation p-value, the mean gain
across the 3 folds, and finally RANK the sources. No presumption that the human
wins -- the ranking is the finding.

Pure standard library. Run:  python3 A_transfer.py [path/to/data]
"""

import os, sys, glob, json, random, math
from collections import defaultdict, Counter

random.seed(12345)
N_PERM = 1000

FUNCTIONS = ["preparation", "contact", "exchange", "disruption",
             "negotiation", "stabilization", "return"]
FSET = set(FUNCTIONS)

# sources present in ALL three corpora (needed for leave-one-corpus-out)
SOURCES = ["human1", "opus", "gemini", "qwen"]    # human1 = the human reader
CORPORA = ["gudea", "inanna_enki", "inanna_descent"]


# ---------------------------------------------------------------- loading
def load_functions(path):
    """Return the ordered list of segment functions for one run file."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    segs = d["segments"] if isinstance(d, dict) else d
    segs = sorted(segs, key=lambda s: (s.get("line_start", 0), s.get("line_end", 0)))
    out = []
    for s in segs:
        f = s.get("function")
        if f in FSET:
            out.append(f)
    return out


def find_runs(data_dir, corpus, source):
    """List run files for (corpus, source). source 'qwen_fast' -> Fast/ subdir."""
    if source == "qwen_fast":
        pat = os.path.join(data_dir, corpus, "Fast", "qwen_run*.json")
    else:
        pat = os.path.join(data_dir, corpus, f"{source}_run*.json")
    return sorted(glob.glob(pat))


def runs_functions(data_dir, corpus, source):
    """List of function-sequences, one per run."""
    seqs = []
    for p in find_runs(data_dir, corpus, source):
        try:
            seq = load_functions(p)
        except Exception as e:
            sys.stderr.write(f"  [skip] {p}: {e}\n")
            continue
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


# ---------------------------------------------------------------- grammar
def pairs(seq):
    """Consecutive (current, next) function pairs."""
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def fit_grammar(train_seqs, alpha=1.0):
    """Return (logP_cond, logP_marg): dicts giving log2 P(next|cur) and log2 P(next)."""
    trans = defaultdict(Counter)      # cur -> Counter(next)
    marg = Counter()                  # next-token marginal
    for seq in train_seqs:
        for a, b in pairs(seq):
            trans[a][b] += 1
            marg[b] += 1
    k = len(FUNCTIONS)
    logP_cond = {}
    for a in FUNCTIONS:
        tot = sum(trans[a].values()) + alpha * k
        for b in FUNCTIONS:
            logP_cond[(a, b)] = math.log2((trans[a][b] + alpha) / tot)
    totm = sum(marg.values()) + alpha * k
    logP_marg = {b: math.log2((marg[b] + alpha) / totm) for b in FUNCTIONS}
    return logP_cond, logP_marg


def eval_gain(test_seqs, logP_cond, logP_marg):
    """Mean (log2 P(next|cur) - log2 P(next)) over all held-out transitions."""
    tot = 0.0
    n = 0
    for seq in test_seqs:
        for a, b in pairs(seq):
            tot += logP_cond[(a, b)] - logP_marg[b]
            n += 1
    return (tot / n if n else 0.0), n


def perm_null(test_seqs, logP_cond, logP_marg, n_perm=N_PERM):
    """Null: shuffle each held-out sequence's order; recompute gain."""
    null = []
    pool = [list(s) for s in test_seqs]
    for _ in range(n_perm):
        shuffled = []
        for s in pool:
            t = s[:]
            random.shuffle(t)
            shuffled.append(t)
        g, _ = eval_gain(shuffled, logP_cond, logP_marg)
        null.append(g)
    return null


def p_value(obs, null):
    ge = sum(1 for x in null if x >= obs)
    return (ge + 1) / (len(null) + 1)


# ---------------------------------------------------------------- main
def run(data_dir):
    print("=" * 74)
    print("METRIC A  --  transferable event-grammar (leave-one-corpus-out)")
    print("gain = bits/transition the transferred grammar beats the marginal baseline")
    print("=" * 74)

    summary = {}   # source -> list of per-fold gains
    for source in SOURCES:
        by_corpus = {c: runs_functions(data_dir, c, source) for c in CORPORA}
        if any(len(by_corpus[c]) == 0 for c in CORPORA):
            print(f"\n[{source}] missing in some corpus -> skipped")
            continue
        print(f"\n[{source}]  runs per corpus: "
              + ", ".join(f"{c.split('_')[-1]}={len(by_corpus[c])}" for c in CORPORA))
        fold_gains = []
        for held in CORPORA:
            train = [s for c in CORPORA if c != held for s in by_corpus[c]]
            test = by_corpus[held]
            cond, marg = fit_grammar(train)
            gain, npairs = eval_gain(test, cond, marg)
            null = perm_null(test, cond, marg)
            p = p_value(gain, null)
            # in-corpus ceiling (in-sample): grammar fit on the held-out corpus itself
            cond2, marg2 = fit_grammar(test)
            ceil, _ = eval_gain(test, cond2, marg2)
            fold_gains.append(gain)
            star = "*" if p < 0.05 else " "
            print(f"    held-out {held.split('_')[-1]:8s}"
                  f" n_trans={npairs:3d}  gain={gain:+.3f} bits"
                  f"  p_perm={p:.3f}{star}   (in-corpus ceiling={ceil:+.3f})")
        mean_gain = sum(fold_gains) / len(fold_gains)
        summary[source] = mean_gain
        print(f"    --> mean transfer gain = {mean_gain:+.3f} bits/transition")

    print("\n" + "-" * 74)
    print("RANK by mean transfer gain (higher = more portable event-grammar):")
    for i, (src, g) in enumerate(sorted(summary.items(), key=lambda kv: -kv[1]), 1):
        tag = "  <- human" if src == "human1" else ""
        print(f"  {i}. {src:8s} {g:+.3f} bits{tag}")
    print("-" * 74)
    print("Reading: gain>0 and p<0.05 => the grammar learned on OTHER myths predicts")
    print("this myth's transitions above chance = portable structure (world-model-like).")
    print("Stability is NOT used anywhere here -- this is generalization, not repeatability.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(os.path.join(here, "..", "..", "data"))
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data
    if not os.path.isdir(data_dir):
        sys.exit(f"data dir not found: {data_dir}")
    print(f"data: {data_dir}")
    run(data_dir)
