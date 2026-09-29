#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projection.py -- the single, shared projection of a reading onto text units.

Every analysis script should load runs through this module, so that all tables
in the paper are computed on the same units with the same rules.

Units
  * gudea, inanna_descent : one unit = one text line
  * inanna_enki           : one unit = one translation block (e.g. "7-16"),
                            because the translation is organised in blocks aligned
                            with the Sumerian original and segments follow block edges.
  Units are read from texts/<corpus>_numbered.txt. Lines that are not in the
  numbered text (lacunae, manuscript notes) are not units and are never scored;
  neither are numbered lines whose text is only "..." (lacunae inside the numbering).

Rules
  * label of a unit    = function of the segment covering most of the unit's lines
                         (ties -> the earlier segment)
  * boundary unit      = the unit that contains a segment's line_start
  * uncovered unit     = absent from the projection (scripts compare runs on the
                         units covered in both)
  * overlaps           = not expected (0% in the current data); check_run() reports them

Usage
  from projection import load_units, load_run, FUNCTIONS
  units  = load_units("inanna_enki", root)          # root = AI_project folder
  labels, bounds, info = load_run(path, units)      # {unit_idx: function}, {unit_idx}
"""
from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path

FUNCTIONS = ["preparation", "contact", "exchange", "disruption",
             "negotiation", "stabilization", "return"]
FSET = set(FUNCTIONS)
CORPORA = ["inanna_enki", "inanna_descent", "gudea"]
BLOCK_CORPORA = {"inanna_enki"}

_UNIT_RE = re.compile(r"^(\d+)(?:\s*[-–]\s*(\d+))?\t(.*)")
_NOTE_RE = re.compile(r"^\s*\(?ms{1,2}\.")          # "ms. adds ...", "mss. add ..."
_DOTS_RE = re.compile(r"^[\s.\u2026()?\[\]]*$")      # line whose text is only "..." = lacuna


# ---------- units ----------
def load_units(corpus: str, root, granularity: str = "native") -> list[tuple[int, int]]:
    """Ordered list of (first_line, last_line) per unit.
    granularity="native": blocks for inanna_enki, lines elsewhere.
    granularity="line"  : every real line is its own unit (for comparison only)."""
    path = Path(root) / "texts" / f"{corpus}_numbered.txt"
    units, seen = [], set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        m = _UNIT_RE.match(raw)
        if not m or _NOTE_RE.match(m.group(3)) or _DOTS_RE.match(m.group(3)):
            continue
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        if (a, b) in seen:
            continue
        seen.add((a, b)); units.append((a, b))
    units.sort()
    if granularity == "line" or corpus not in BLOCK_CORPORA:
        units = [(ln, ln) for a, b in units for ln in range(a, b + 1)]
    return units


def _line_index(units):
    return {ln: i for i, (a, b) in enumerate(units) for ln in range(a, b + 1)}


# ---------- segments ----------
def load_segments(path) -> list[dict]:
    """Segments with int line_start/line_end, lower-cased function, sorted by position.
    Accepts a bare list or {"segments": [...]}. Segments without a valid range are dropped."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    segs = data.get("segments", []) if isinstance(data, dict) else data
    out = []
    for s in segs:
        try:
            a, b = int(s["line_start"]), int(s["line_end"])
        except (KeyError, TypeError, ValueError):
            continue
        seg = dict(s)
        seg["line_start"], seg["line_end"] = a, b
        seg["function"] = str(s.get("function", "")).strip().lower()
        seg["transition_to"] = str(s.get("transition_to", "")).strip().lower()
        out.append(seg)
    return sorted(out, key=lambda s: (s["line_start"], s["line_end"]))


# ---------- projection ----------
def project(segments, units, field: str = "function", valid=FSET):
    """-> ({unit_idx: label}, {unit_idx of each segment start})"""
    idx = _line_index(units)
    votes: dict[int, Counter] = {}
    order: dict[tuple[int, str], int] = {}
    bounds = set()
    for k, s in enumerate(segments):
        lab = s.get(field, "")
        if valid is not None and lab not in valid:
            continue
        a, b = s["line_start"], s["line_end"]
        covered = [idx[ln] for ln in range(a, b + 1) if ln in idx]
        if not covered:
            continue
        bounds.add(covered[0])
        for u in covered:
            votes.setdefault(u, Counter())[lab] += 1
            order.setdefault((u, lab), k)
    labels = {u: max(c, key=lambda l: (c[l], -order[(u, l)])) for u, c in votes.items()}
    return labels, bounds


def check_run(segments, units) -> dict:
    """Diagnostics for the methods section: overlaps, lines outside the text,
    segment edges that fall inside a unit (only possible for block units)."""
    idx = _line_index(units)
    starts = {a for a, b in units}; ends = {b for a, b in units}
    lines = [ln for s in segments for ln in range(s["line_start"], s["line_end"] + 1)]
    return dict(
        segments=len(segments),
        overlap_lines=len(lines) - len(set(lines)),
        lines_outside_text=len({ln for ln in lines if ln not in idx}),
        edges_inside_unit=sum((s["line_start"] in idx and s["line_start"] not in starts) +
                              (s["line_end"] in idx and s["line_end"] not in ends)
                              for s in segments),
        coverage=len({idx[ln] for ln in lines if ln in idx}) / len(units),
    )


def load_run(path, units, field: str = "function"):
    """One call for scripts: -> (labels, bounds, diagnostics)."""
    segs = load_segments(path)
    labels, bounds = project(segs, units, field)
    return labels, bounds, check_run(segs, units)



# ---------- choosing 3 of the 10 model runs ----------
# Used wherever the models are compared with the human at the same number of runs (3).
# Both rules work on projected labels ({unit: label}), so every script picks the same runs.
def pairwise_agree(a: dict, b: dict) -> float:
    """Share of the shared units on which two runs give the same label."""
    shared = set(a) & set(b)
    return sum(a[u] == b[u] for u in shared) / len(shared) if shared else 0.0


def _distances(runs):
    n = len(runs)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = 1.0 - pairwise_agree(runs[i], runs[j])
    return D


def _far_seeds(D, k):
    n = len(D)
    i0, j0 = max(((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda ij: D[ij[0]][ij[1]])
    seeds = [i0, j0]
    while len(seeds) < k:
        seeds.append(max((x for x in range(n) if x not in seeds), key=lambda x: min(D[x][s] for s in seeds)))
    return seeds


def select_far(runs, k=3):
    """'far': the k most different runs (start from the two most different, then add the
    run farthest from those already chosen)."""
    if len(runs) <= k:
        return runs
    return [runs[i] for i in _far_seeds(_distances(runs), k)]


def select_medoid(runs, k=3):
    """'medoid': split the runs into k groups around the 'far' runs and take the most
    typical run of each group (smallest mean distance to its group)."""
    if len(runs) <= k:
        return runs
    D = _distances(runs)
    seeds = _far_seeds(D, k)
    groups = {s: [] for s in seeds}
    for x in range(len(runs)):
        groups[min(seeds, key=lambda s: D[x][s])].append(x)
    return [runs[min(m, key=lambda x: sum(D[x][y] for y in m) / len(m))] for m in groups.values()]

if __name__ == "__main__":
    # quick report: python projection.py [AI_project root]
    import sys, glob
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    for c in CORPORA:
        units = load_units(c, root)
        print(f"== {c}: {len(units)} units")
        for p in sorted(glob.glob(str(root / "data" / c / "*_run*.json"))):
            d = check_run(load_segments(p), units)
            if d["overlap_lines"] or d["edges_inside_unit"] or d["coverage"] < 0.99:
                print(f"   {Path(p).name:18s} {d}")
