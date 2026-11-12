"""Value the 43 hand-collected 99acres listings (November 2026) with the current engine.

Run: python analysis/validate_99acres.py   (writes reports/validation_99acres.md)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import DATA, ROOT, ValuationEngine  # noqa: E402
from source2 import project_key  # noqa: E402

RATIO = {"Carpet": 1.0, "Built-up": 1.20, "Super built-up": 1.35}
eng = ValuationEngine()
eng.comps["status"] = eng.comps.status.astype(str)
c = eng.comps.assign(k=eng.comps.project.map(project_key)).drop_duplicates(["k", "region"])
names = c.set_index(["k", "region"]).project.to_dict()
counts = eng.comps.groupby(["region", "project"]).size().rename("n").reset_index()


def match(name, region):
    """Same rule as the app's search: exact name, else the building in that suburb whose name
    contains every word typed (most listings first)."""
    p = names.get((project_key(name), region))
    if p:
        return p
    words = [w for w in name.lower().split() if len(w) > 1]
    cand = counts[(counts.region == region)
                  & counts.project.str.lower().map(lambda t: all(w in t for w in words))]
    return cand.sort_values("n", ascending=False).project.iloc[0] if len(cand) else ""
v = pd.read_csv(DATA / "validation_99acres.csv")
out = []
for x in v.itertuples():
    proj = match(x.project, x.region)
    model_area = round(x.area / RATIO[x.area_type] * RATIO["Super built-up"], 1)
    r = eng.value(dict(region=x.region, project=proj, bhk=int(x.bhk), area=model_area, type="Apartment",
                       status=x.status, age=x.age))
    out.append(dict(project=x.project, matched=proj, bhk=x.bhk, known=bool(proj), asking=x.asking_lakh, app=r["value"] / 1e5,
                    light=r["light"], inside=r["low"] / 1e5 <= x.asking_lakh <= r["high"] / 1e5))
o = pd.DataFrame(out)
o["err"] = o.app / o.asking - 1


def row(label, s):
    return (f"| {label} | {len(s)} | {s.err.abs().median() * 100:.0f}% | {(s.err.abs() <= .1).mean() * 100:.0f}% | "
            f"{s.err.median() * 100:+.0f}% | {s.inside.mean() * 100:.0f}% |")


lines = ["# 99acres check: 43 live listings collected in November 2026", "",
         f"Model version {eng.meta.get('model_version')}. Asking prices, so part of any gap is negotiation.", "",
         "| Group | Listings | Median error | Within ±10% | Median bias | Asking price inside 80% range |",
         "|---|---|---|---|---|---|", row("Buildings in the data", o[o.known]),
         row("Buildings not in the data (mostly launched after the data)", o[~o.known]), row("All", o)]
(ROOT / "reports" / "validation_99acres.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(o.to_string())
