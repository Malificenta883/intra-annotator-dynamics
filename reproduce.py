#!/usr/bin/env python3
"""
reproduce.py -- run every analysis in the paper and check the numbers.

    python3 reproduce.py                 # everything (a few minutes on a laptop)
    python3 reproduce.py table1 table4   # only some steps
    python3 reproduce.py --list          # show the steps

Each step runs one script from src/, saves its full output in results/,
and prints the paper's numbers next to the numbers it just computed.
Only standard Python is needed; the figure step also needs matplotlib.
"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
MYTHS = ["inanna_enki", "inanna_descent", "gudea"]
MODELS = ["opus", "gemini", "qwen"]
BAD = []


# ---------- helpers ----------
def run(name, script, *args):
    """Run src/<script> with the given arguments inside results/; save the printed output."""
    os.makedirs(OUT, exist_ok=True)
    cmd = [sys.executable, os.path.join(SRC, script), *args]
    res = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8").write(res.stdout + res.stderr)
    if res.returncode != 0:
        sys.exit(f"{script} failed, see results/{name}.txt")
    return res.stdout

def load(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))

def check(label, got, paper, decimals):
    """The paper shows `paper` rounded to `decimals`; is `got` the same after rounding?"""
    ok = abs(got - paper) <= 0.5 * 10 ** -decimals + 1e-9
    if not ok: BAD.append(label)
    sign = "+" if decimals and (paper < 0 or got < 0 or "gap" in label or "gain" in label) else ""
    print(f"  {'ok ' if ok else 'XX '} {label:52s} paper {paper:>{sign}8.{decimals}f}   now {got:>{sign}8.{decimals}f}")


# ---------- steps ----------
def units():
    """§3.4: units per myth; one segment edge falls inside a block (one human run)."""
    out = run("units", "projection.py", ROOT)
    for myth, n in [("inanna_enki", 71), ("inanna_descent", 405), ("gudea", 804)]:
        got = int(re.search(rf"== {myth}: (\d+) units", out).group(1))
        check(f"{myth}: number of units", got, n, 0)
    check("edges inside a block (all runs)", len(re.findall(r"'edges_inside_unit': [1-9]", out)), 1, 0)

def table1():
    """Table 1 (human kappas and gap), §4.1 boundary placement, Table 3 (B2), §7 (c, d)."""
    run("table1", "paper_results.py", DATA)
    r = load("paper_results.json")
    paper = {"inanna_enki": (0.731, 0.291, 0.440, 77.8, 21.1, 0.149, 0.268, 9),
             "inanna_descent": (0.655, 0.203, 0.452, 69.3, 4.8, 0.159, 0.194, 12),
             "gudea": (0.465, 0.168, 0.298, 47.2, 3.2, 0.346, 0.003, 20)}
    for m, (bk, lk, gap, ex, ch, b2g, b2p, ev) in paper.items():
        x = r[m]
        check(f"{m}: boundary kappa", x["boundary_kappa"], bk, 3)
        check(f"{m}: label kappa", x["label_kappa"], lk, 3)
        check(f"{m}: gap", x["gap"], gap, 3)
        check(f"{m}: boundaries on the same unit, %", 100 * x["boundary_match"]["exact"], ex, 1)
        check(f"{m}: same unit by chance, %", 100 * x["boundary_match"]["chance"], ch, 1)
        check(f"{m}: B2 gap", x["B2"]["gap"], b2g, 3)
        check(f"{m}: B2 p", x["B2"]["p"], b2p, 3)
        check(f"{m}: B2 events", x["B2"]["n_event"], ev, 0)

def table2():
    """Table 2: boundary-label gap of each model (all 10 runs)."""
    sys.path.insert(0, SRC)
    import paper_results as pr
    paper = {"opus": (-0.095, 0.160, 0.065), "gemini": (-0.178, 0.032, 0.291), "qwen": (0.084, 0.106, 0.077)}
    lines = []
    for s, gaps in paper.items():
        for m, g in zip(MYTHS, gaps):
            bK, lK, _, _ = pr.boundary_label_kappas(pr.load_runs(DATA, m, s))
            lines.append(f"{m} {s} boundary {bK:.3f} label {lK:.3f} gap {bK - lK:+.3f}")
            check(f"{m}: {s} gap", bK - lK, g, 3)
    open(os.path.join(OUT, "table2.txt"), "w").write("\n".join(lines) + "\n")

def table3_b1():
    """Table 3, B1: concentration against a baseline with the same amount of disagreement."""
    paper = {"inanna_enki": (0.85, 0.85), "inanna_descent": (0.88, 0.88), "gudea": (0.72, 1.00)}
    for m, (ratio, p) in paper.items():
        run(f"table3_b1_{m}", "b1_volume_null.py", DATA, "--corpus", m)
        x = load(f"b1_volume_null_{m}.json")[f"{m}/human1/own"]["pieces"]
        check(f"{m}: B1 ratio", x["ratio"], ratio, 2)
        check(f"{m}: B1 p (above)", x["p"], p, 2)

def b2_models():
    """§4.3: B2 for the models at 3 runs (the text says only: as strong or stronger)."""
    run("b2_models", "b2_models_matched.py", DATA)
    print("  (no single number in the paper; see results/b2_models.txt)")

def table4():
    """Table 4: where two runs start the same function (human: 1,000 draws; models: all triples)."""
    run("table4_human", "lag_test.py", DATA)
    h = load("lag_test.json")
    paper = {"inanna_enki": (20, 7.7, 16, 15.4), "inanna_descent": (8, 6.3, 30, 12.0), "gudea": (14, 8.9, 32, 17.9)}
    for m, (o0, e0, o1, e1) in paper.items():
        t = h[f"{m}/human1/own"]["onset_lags"]
        check(f"{m}: human same place, observed", t["0"]["obs"], o0, 0)
        check(f"{m}: human same place, expected", t["0"]["exp"], e0, 1)
        check(f"{m}: human one step, observed", t["pm1"]["obs"], o1, 0)
        check(f"{m}: human one step, expected", t["pm1"]["exp"], e1, 1)
    paper = {"inanna_enki": (3.8, 0.80, 1.00), "inanna_descent": (2.7, 3.75, 1.86), "gudea": (2.1, 2.29, 1.41)}
    for m, (low, hum, top) in paper.items():
        run(f"table4_triples_{m}", "lag_test.py", DATA, "--corpus", m, "--triples")
        s = load(f"lag_test_{m}_triples.json")[f"{m}/summary"]
        check(f"{m}: models, lowest same-place ratio", s["models_lowest_same_place_ratio"], low, 1)
        check(f"{m}: human, one step per same place", s["human_onestep_per_same"], hum, 2)
        check(f"{m}: models, highest one step per same place", s["models_max_onestep_per_same"], top, 2)

def mirrors():
    """§4.3: on Inanna & Enki, 8 of 15 unstable human passages are mirrored pairs."""
    run("mirrors", "passage_inspection.py", DATA, "--corpus", "inanna_enki")
    x = load("passage_inspection_inanna_enki.json")["inanna_enki/human1/own"]
    check("inanna_enki: unstable passages", x["passages"], 15, 0)
    check("inanna_enki: mirrored passages", x["types"]["M"], 8, 0)

def table5():
    """Table 5: transferable grammar (all runs); §4.4: same order with 3 model runs."""
    out = run("table5", os.path.join("world_model_metrics", "A_transfer.py"), DATA)
    paper = {"human1": (0.043, 0.014, 0.062, 0.040), "opus": (0.441, 0.522, 0.357, 0.440),
             "qwen": (0.258, 0.453, -0.215, 0.165), "gemini": (-0.237, 0.419, 0.213, 0.132)}
    for s, vals in paper.items():
        block = out.split(f"[{s}]")[1].split("-->")[0]
        gains = dict(re.findall(r"held-out (\w+)\s+n_trans=\s*\d+\s+gain=([+-][\d.]+)", block))
        for myth, v in zip(["gudea", "enki", "descent"], vals[:3]):
            check(f"{s}: gain on {myth}", float(gains[myth]), v, 3)
        mean = float(re.search(rf"\[{s}\].*?mean transfer gain = ([+-][\d.]+)", out, re.S).group(1))
        check(f"{s}: mean gain", mean, vals[3], 3)
    out = run("table5_three_runs", os.path.join("world_model_metrics", "robustness_check.py"), "a_folds", DATA)
    for sel in ["far-3", "med-3"]:
        block = out.split(f"[{sel}]")[1].split("[")[0]
        means = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w+)\s.*?([+-][\d.]+)\s*(?:<- human)?$", block, re.M)}
        order = sorted(means, key=lambda k: -means[k])
        ok = order == ["opus", "qwen", "gemini", "human1"]
        if not ok: BAD.append(f"order {sel}")
        print(f"  {'ok ' if ok else 'XX '} order with 3 model runs ({sel}): {' > '.join(order)}")

def agreement():
    """§7 (a): label agreement between runs, human vs. the most different model runs."""
    out = run("agreement", "paper_results.py", DATA, "--agreement")
    lo_h, hi_h, lo_m, hi_m = map(float, re.search(r"human ([\d.]+)-([\d.]+), model far-3 triples ([\d.]+)-([\d.]+)", out).groups())
    check("human, lowest", lo_h, 0.32, 2); check("human, highest", hi_h, 0.38, 2)
    check("models, lowest", lo_m, 0.46, 2); check("models, highest", hi_m, 0.85, 2)

def figure1():
    """Figure 1: three human runs vs. three Opus runs on Inanna's Descent."""
    run("figure1", "fig1_phase.py", DATA, os.path.join(OUT, "fig1_phase.png"))
    print("  wrote results/fig1_phase.png")


STEPS = {"units": units, "table1": table1, "table2": table2, "table3_b1": table3_b1, "b2_models": b2_models,
         "table4": table4, "mirrors": mirrors, "table5": table5, "agreement": agreement, "figure1": figure1}

if __name__ == "__main__":
    if "--list" in sys.argv:
        for k, f in STEPS.items(): print(f"{k:10s} {f.__doc__.strip().splitlines()[0]}")
        sys.exit()
    todo = [a for a in sys.argv[1:] if a in STEPS] or list(STEPS)
    for k in todo:
        print(f"\n== {k}: {STEPS[k].__doc__.strip().splitlines()[0]}")
        STEPS[k]()
    print("\nall numbers match the paper" if not BAD else f"\nMISMATCH in {len(BAD)} numbers: {BAD}")
