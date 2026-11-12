"""External test: score the trained engine on listings from a second, independent source.

Run with model 1.1 artifacts (git checkout aebb3da -- artifacts). Model 2.0 is trained on this source,
so re-running it on 2.0 is no longer an external test.
Run: python analysis/external_test.py   (writes reports/external_test.md)
Uses only listings from the second source; same-unit duplicates of first-source listings are dropped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import ROOT, ValuationEngine  # noqa: E402
from source2 import RAW2, clean_source2, project_key  # noqa: E402

SAMPLE = 2500
eng = ValuationEngine()
adj = eng.meta.get("time_adjustment", 1.0)
comps = eng.comps.copy()
comps["k"] = comps.project.map(project_key)
known = {(k, r) for k, r in zip(comps.k, comps.region)}
same_unit = {(k, r, b, a) for k, r, b, a in zip(comps.k, comps.region, comps.bhk, comps.area)}
name_by_key = comps.drop_duplicates(["k", "region"]).set_index(["k", "region"]).project.to_dict()

d = clean_source2(pd.read_csv(RAW2))
d = d[d.type.isin(["Apartment", "Studio Apartment"])]
d["k"] = d.project.map(project_key)
d = d[[(k, r, b, a) not in same_unit for k, r, b, a in zip(d.k, d.region, d.bhk, d.area)]]
d["known"] = [(k, r) in known for k, r in zip(d.k, d.region)]

rows = []
for grp, sub in d.groupby("known"):
    sub = sub.sample(min(SAMPLE, len(sub)), random_state=1)
    for _, x in sub.iterrows():
        subj = dict(region=x.region, project=name_by_key.get((x.k, x.region), "") if grp else "",
                    bhk=int(x.bhk), area=float(x.area), type=x.type, status="Ready to move", age=x.age)
        v = eng.value(subj)
        rows.append(dict(known=grp, actual=x.price_inr, value=v["value"] / adj, low=v["low"] / adj,
                         high=v["high"] / adj, light=v["light"]))
r = pd.DataFrame(rows)
r["ape"] = (r.value / r.actual - 1).abs()
r["bias"] = r.value / r.actual - 1
r["covered"] = (r.actual >= r.low) & (r.actual <= r.high)


def line(label, s):
    return (f"| {label} | {len(s):,} | {s.ape.median() * 100:.1f}% | {(s.ape <= .10).mean() * 100:.1f}% | "
            f"{s.bias.median() * 100:+.1f}% | {s.covered.mean() * 100:.1f}% |")


out = ["# External test on a second listings source", "",
       "Source: Kaggle 'Mumbai House Price Data (70k entries)' (kevinnadar22). The engine was trained only on "
       "the first source. Listings that look like the same flat in both sources were removed. Values are compared "
       "at the data's own price level (before the 2026 price update).", "",
       "| Group | Flats tested | Median error | Within ±10% | Median bias | 80% range covers |", "|---|---|---|---|---|---|",
       line("Buildings the engine knows", r[r.known]), line("Buildings the engine has never seen", r[~r.known]),
       line("All", r), "", "## By traffic light", "",
       "| Light | Flats | Median error | Within ±10% | Median bias | 80% range covers |", "|---|---|---|---|---|---|"]
for L in ["GREEN", "AMBER", "RED"]:
    s = r[r.light == L]
    if len(s):
        out.append(line(L.title(), s))
(ROOT / "reports" / "external_test.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
print("known share in source 2:", round(d.known.mean() * 100, 1), "% of", len(d))
