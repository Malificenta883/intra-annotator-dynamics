#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b2_models_matched.py -- B2 (neighbour-licensed flips) for the human vs the three model
families at the SAME number of runs (3).

Human: its own 3 runs.
Models: 3 of their 10 runs, chosen two ways (functions from world_model_metrics/robustness_check.py):
  * far    -- farthest-point: the three most different runs (select_diverse_runs)
  * medoid -- runs grouped into 3 clusters, the most typical run of each cluster (medoid_runs)

B2 is computed with paper_results.b2 exactly as for the paper:
  units without a strict majority are left out; an event = one run giving one majority
  passage one alternative label (counted once per run x passage x label); licensed = the
  alternative is the majority state of an adjacent passage; null = alternative drawn at
  random from the other states used in the text (N = paper_results.N_PERM).

Run:  python3 b2_models_matched.py [data_dir]      (default: ../data)
Writes b2_models_matched.json next to where it is run.
"""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "world_model_metrics"))
import paper_results as pr
from projection import select_far as select_diverse_runs   # choice of 3 model runs: see projection.py
from projection import select_medoid as medoid_runs

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
    out = {}
    print(f"{'corpus':16s}{'reader':8s}{'selection':10s}{'runs':>12s}{'events':>8s}{'licensed':>10s}{'chance':>8s}{'gap':>8s}{'p':>7s}")
    for c in pr.CORPORA:
        for src in ["human1", "opus", "gemini", "qwen"]:
            labs = [r[1] for r in pr.load_runs(data_dir, c, src)]
            if not labs: continue
            sels = [("own", labs)] if src == "human1" else \
                   [("far", select_diverse_runs(labs, 3)), ("medoid", medoid_runs(labs, 3))]
            for name, sel in sels:
                idx = [labs.index(m) + 1 for m in sel]
                # human uses the same seed as paper_results.main, so its row equals Table 3
                seed = f"b2-{c}-12345" if src == "human1" else f"b2-{c}-{src}-{name}"
                r = pr.b2(sel, rng=random.Random(seed))
                out[f"{c}/{src}/{name}"] = dict(runs=idx, **(r or {}))
                if r:
                    print(f"{c:16s}{src:8s}{name:10s}{str(idx):>12s}{r['n_event']:>8d}{r['obs']:>10.0%}"
                          f"{r['null']:>8.0%}{r['gap']:>+8.2f}{r['p']:>7.3f}")
                else:
                    print(f"{c:16s}{src:8s}{name:10s}{str(idx):>12s}   no events")
    json.dump(out, open("b2_models_matched.json", "w"), indent=1)
    print("\nwrote b2_models_matched.json")

if __name__ == "__main__":
    main()
