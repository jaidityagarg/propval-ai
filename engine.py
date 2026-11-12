"""Core valuation engine: data cleaning, features, and valuation logic.

Used by train.py (to build the model) and app.py (to value a property).
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

# Landmarks used for distance features (approximate coordinates).
LANDMARKS = {
    "dist_nariman_point_km": (18.9256, 72.8242),   # old CBD
    "dist_bkc_km": (19.0660, 72.8680),             # new CBD
    "dist_powai_km": (19.1180, 72.9060),           # eastern jobs hub
    "dist_vashi_km": (19.0770, 72.9990),           # Navi Mumbai hub
}

CATEGORICAL = ["type", "status", "age", "region", "zone", "furnished"]
RAIL = ["dist_rail_km", "rail_stations_within_3km"]
NUMERIC = (["bhk", "area", "log_area", "area_per_bhk", "lat", "lon", "loc_precise"] + list(LANDMARKS) + RAIL
           + ["age_years", "bathrooms"])
OPTIONAL = {"age_years": np.nan, "bathrooms": np.nan, "furnished": np.nan, "loc_precise": 0}
FEATURES = NUMERIC + CATEGORICAL

# Friendly names for explanations (groups of model features).
FEATURE_GROUPS = {
    "Location (suburb, map position, distance to job hubs)": ["region", "zone", "lat", "lon", "loc_precise"]
    + list(LANDMARKS),
    "Rail connectivity (nearest local station)": RAIL,
    "Size (area)": ["area", "log_area"],
    "Layout (BHK, area per room)": ["bhk", "area_per_bhk"],
    "Property type": ["type"],
    "Construction status": ["status"],
    "New vs resale, building age": ["age", "age_years"],
    "Bathrooms and furnishing": ["bathrooms", "furnished"],
}

# RBI loan-to-value caps for individual housing loans.
LTV_SLABS = [(30_00_000, 0.90), (75_00_000, 0.80), (float("inf"), 0.75)]

PRICE_UNIT = {"Cr": 1e7, "L": 1e5}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_stations():
    """Mumbai-region suburban railway stations (DataMeet Indian Railways open data)."""
    return pd.read_csv(DATA / "rail_stations.csv")


def market_check(region):
    """Knight Frank India Real Estate H1 2026 price range for the suburb, if the report covers it."""
    kf = pd.read_csv(DATA / "knight_frank_h1_2026.csv")
    r = str(region).lower()
    hit = kf[[r.startswith(m.lower()) for m in kf.region_match]]
    return None if hit.empty else hit.iloc[0].to_dict()


def load_regions():
    return pd.read_csv(DATA / "regions.csv")


_ALIASES = {
    "dombivali east": "Dombivali", "dombivli (west)": "Dombivali",
    "kasaradavali thane west": "Thane West", "taloje": "Taloja",
    "mazgaon": "Mazagaon", "dahisar west": "Dahisar", "dahisar east": "Dahisar",
    "khar west": "Khar", "vikroli east": "Vikhroli", "kanjurmarg east": "Kanjurmarg",
    "nalasopara east": "Nala Sopara", "nalasopara west": "Nala Sopara",
    "ambarnath": "Ambernath West", "badlapur": "Badlapur East",
    "matunga west": "Matunga", "mahim west": "Mahim", "mira road": "Mira Road East",
    "kandivali": "Kandivali West", "mulund": "Mulund West", "borivali": "Borivali West",
    "vile parle west": "Ville Parle West", "vasai west": "Vasai", "vasai east": "Vasai",
    "virar west": "Virar", "virar east": "Virar", "kurla east": "Kurla", "kurla west": "Kurla",
    "goregaon": "Goregaon West", "bandra": "Bandra West", "chembur east": "Chembur",
    "juhu scheme": "Juhu",
}
_SECTOR_HOSTS = ["Kamothe", "Kharghar", "Ulwe", "Vashi", "Nerul", "Seawoods", "Panvel"]


def normalize_region(name, known):
    """Map raw region spellings (e.g. 'Sector 21 Kamothe') to a canonical region."""
    if not isinstance(name, str):
        return None
    s = name.strip()
    low = s.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    if "koparkhairane" in low:
        return "Koper Khairane"
    if low.startswith("sector"):
        for host in _SECTOR_HOSTS:
            if host.lower() in low:
                return host
        return None
    for k in known:
        if k.lower() == low:
            return k
    return None


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw listings file. Returns one row per unique listing."""
    regions = load_regions()
    known = set(regions.region)
    df = raw.copy()
    df = df.drop_duplicates()
    df["price_inr"] = df["price"] * df["price_unit"].map(PRICE_UNIT)
    df["region"] = df["region"].map(lambda r: normalize_region(r, known))
    df = df.dropna(subset=["region", "price_inr"])
    df = df[(df.area >= 150) & (df.area <= 10000)]
    df["psf"] = df.price_inr / df.area
    # Remove listings whose price per sq ft is far from their suburb's typical level
    # (usually typos: price in lakh entered as crore, or total vs per-unit mixups).
    med = df.groupby("region")["psf"].transform("median")
    df = df[(df.psf > med / 3) & (df.psf < med * 3)]
    df = df.merge(regions, on="region", how="left")
    df = df.rename(columns={"locality": "project"})
    for c, v in OPTIONAL.items():
        df[c] = v
    df["source"] = "kaggle_76k"
    df = df.reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for c, v in OPTIONAL.items():
        if c not in out:
            out[c] = v
    for c in ["lat", "lon", "zone"]:
        if c not in out:
            out[c] = np.nan
    # Missing map position or zone: use the suburb's centre.
    cen = out[["region"]].merge(load_regions(), on="region", how="left")
    for c in ["lat", "lon", "zone"]:
        out[c] = out[c].where(out[c].notna(), cen[c])
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    out["log_area"] = np.log(out["area"])
    out["area_per_bhk"] = out["area"] / out["bhk"].clip(lower=1)
    for col, (la, lo) in LANDMARKS.items():
        out[col] = haversine_km(out["lat"], out["lon"], la, lo)
    st = load_stations()
    d = haversine_km(out["lat"].to_numpy()[:, None], out["lon"].to_numpy()[:, None],
                     st["lat"].to_numpy()[None, :], st["lon"].to_numpy()[None, :])
    out["dist_rail_km"] = d.min(axis=1)
    out["rail_stations_within_3km"] = (d <= 3).sum(axis=1)
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out


def set_categories(X: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    return X


def ltv_cap(loan_amount):
    for limit, ratio in LTV_SLABS:
        if loan_amount <= limit:
            return ratio
    return LTV_SLABS[-1][1]


def max_loan_for_value(collateral_value):
    """Largest loan allowed by RBI LTV slabs for a given collateral value."""
    best = 0.0
    for limit, ratio in LTV_SLABS:
        loan = min(collateral_value * ratio, limit)
        best = max(best, loan)
    return best


def inr(x):
    """Format rupees in lakh / crore."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if x >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    return f"₹{x / 1e5:,.1f} L"


class ValuationEngine:
    def __init__(self, artifacts=ARTIFACTS):
        import lightgbm as lgb
        self.meta = json.loads((artifacts / "meta.json").read_text())
        self.model = lgb.Booster(model_file=str(artifacts / "model.txt"))
        self.scale_model = lgb.Booster(model_file=str(artifacts / "scale_model.txt"))
        self.comps = pd.read_csv(artifacts / "comparables.csv.gz", dtype={"status": str, "furnished": str})
        self.regions = load_regions()

    # ---------- comparable sales ----------
    def find_comps(self, subject, k=5):
        pool = self.comps[self.comps.region == subject["region"]]
        if subject.get("project"):
            same = pool[pool.project.str.lower() == subject["project"].strip().lower()]
        else:
            same = pool.iloc[0:0]
        cand = pool.copy()
        cand["score"] = (
            (cand.bhk - subject["bhk"]).abs() * 1.0
            + (np.log(cand.area) - math.log(subject["area"])).abs() * 4.0
            + (cand.status != subject["status"]) * 0.5
            + (cand.age != subject["age"]) * 0.5
            - (cand.project.str.lower() == (subject.get("project") or "").strip().lower()) * 3.0
        )
        top = cand.nsmallest(k, "score").copy()
        top["adjusted_value"] = top.psf * subject["area"] * self.meta.get("time_adjustment", 1.0)
        return top, len(same)

    # ---------- main valuation ----------
    def project_location(self, region, project):
        """Median map position of a known project's listings (None if not precisely known)."""
        if not project or "lat" not in self.comps:
            return None
        p = self.comps[(self.comps.region == region) & (self.comps.project.str.lower() == project.strip().lower())
                       & (self.comps.loc_precise == 1)]
        if p.empty:
            return None
        return float(p.lat.median()), float(p.lon.median())

    def value(self, subject: dict):
        m = self.meta
        subject = dict(subject)
        if "lat" not in subject:
            loc = self.project_location(subject["region"], subject.get("project"))
            if loc:
                subject["lat"], subject["lon"], subject["loc_precise"] = loc[0], loc[1], 1
        row = pd.DataFrame([subject])
        feats = add_features(row)
        X = set_categories(feats[FEATURES], m["categories"])
        log_psf_model = float(self.model.predict(X)[0])
        contrib = self.model.predict(X, pred_contrib=True)[0]

        comps, _ = self.find_comps(subject)
        proj = self.comps.iloc[0:0]
        if subject.get("project"):
            proj = self.comps[(self.comps.region == subject["region"]) &
                              (self.comps.project.str.lower() == subject["project"].strip().lower())]
        n_same_project = len(proj)
        other = proj[proj.area != subject["area"]]
        proj_n = len(other)
        proj_mad = float("nan")
        log_psf = log_psf_model
        tier_key = "no_project_comps"
        if proj_n >= m["min_project_comps"]:
            tier_key = "with_project_comps"
            lp = np.log(other.psf)
            proj_log_psf = float(np.median(lp))
            proj_mad = float(np.median(np.abs(lp - proj_log_psf)))
            w = m["blend_weight"]
            log_psf = w * proj_log_psf + (1 - w) * log_psf_model

        # How uncertain is this valuation? (second model + conformal calibration per band)
        SX = X.copy()
        SX["proj_n"] = proj_n
        SX["proj_mad"] = proj_mad
        sigma = max(float(self.scale_model.predict(SX)[0]), 0.02)
        c = m["conformal"]
        band = int(np.digitize([sigma], c["sigma_edges"])[0])
        h80, h90 = c["q80"][band] * sigma, c["q90"][band] * sigma

        # Out-of-range checks force a low-confidence result.
        warnings = []
        if subject["area"] < m["area_p1"] or subject["area"] > m["area_p99"]:
            warnings.append("Size is outside the range the model has seen often.")
        if subject["type"] not in ("Apartment", "Studio Apartment"):
            warnings.append(f"Few '{subject['type']}' listings in the data; treat as low confidence.")
        region_n = m["region_counts"].get(subject["region"], 0)
        if region_n < 50:
            warnings.append(f"Only {region_n} listings in {subject['region']}; local evidence is thin.")

        time_adj = m.get("time_adjustment", 1.0)
        point = math.exp(log_psf) * subject["area"] * time_adj
        low = math.exp(log_psf - h80) * subject["area"] * time_adj
        high = math.exp(log_psf + h80) * subject["area"] * time_adj
        low90 = math.exp(log_psf - h90) * subject["area"] * time_adj
        half_width = math.exp(h80) - 1

        if warnings:
            light = "RED"
        elif half_width <= m["green_max_halfwidth"]:
            light = "GREEN"
        elif half_width <= m["amber_max_halfwidth"]:
            light = "AMBER"
        else:
            light = "RED"

        # Explanation: group feature contributions (log space) into % effects.
        base = contrib[-1]
        effects = []
        for label, cols in FEATURE_GROUPS.items():
            idx = [FEATURES.index(c) for c in cols]
            val = float(sum(contrib[i] for i in idx))
            effects.append((label, (math.exp(val) - 1) * 100))
        effects.sort(key=lambda t: -abs(t[1]))
        if tier_key == "with_project_comps":
            effects.insert(0, (f"Other listings in the same project ({n_same_project})",
                               (math.exp(log_psf - log_psf_model) - 1) * 100))

        return {
            "value": point, "low": low, "high": high, "low90": low90,
            "psf": point / subject["area"], "half_width_pct": half_width * 100,
            "light": light, "tier": tier_key, "warnings": warnings,
            "baseline_psf": math.exp(base) * time_adj,
            "effects": effects, "comps": comps, "n_same_project": n_same_project,
            "accuracy": m["traffic_lights"][light],
        }

    # ---------- lending view ----------
    @staticmethod
    def lending_view(val, loan_requested=None, agreement_value=None, valuer_value=None):
        notes = []
        candidates = {"AVM lower bound (80% range)": val["low"]}
        if agreement_value:
            candidates["Agreement / documented value"] = agreement_value
        if valuer_value:
            candidates["Valuer report"] = valuer_value
            if valuer_value > val["high"]:
                gap = (valuer_value / val["value"] - 1) * 100
                notes.append(f"Valuer figure is {gap:.0f}% above the AVM estimate and outside its range: "
                             "refer for review (possible over-valuation).")
            elif valuer_value < val["low"]:
                notes.append("Valuer figure is below the AVM range: check for errors or property defects.")
            else:
                notes.append("Valuer figure is consistent with the AVM range.")
        basis_name = min(candidates, key=candidates.get)
        collateral = candidates[basis_name]
        max_loan = max_loan_for_value(collateral)

        light = val["light"]
        route = {
            "GREEN": "Desktop valuation by internal technical valuer; no site visit needed.",
            "AMBER": "One site visit by a valuer; AVM used as a cross-check.",
            "RED": "Full physical valuation; AVM figure is indicative only.",
        }[light]
        reg = []
        if loan_requested:
            if loan_requested >= 75_00_000:
                reg.append("Loan ≥ ₹75 L: RBI requires two independent valuation reports; use the lower. "
                           "AVM acts as a third, independent challenger.")
            elif loan_requested >= 50_00_000:
                reg.append("Loan ₹50–75 L: RBI requires two valuation reports (at least one independent); "
                           "use the lower. AVM acts as a challenger.")
            else:
                reg.append("Loan below ₹50 L: one valuation report needed under RBI rules; "
                           "the traffic light decides whether it can be a desktop review.")
            cap = ltv_cap(loan_requested)
            implied_ltv = loan_requested / collateral
            ok = implied_ltv <= cap
            notes.append(f"Requested loan gives LTV of {implied_ltv * 100:.0f}% against the {cap * 100:.0f}% RBI cap: "
                         + ("within limit." if ok else "ABOVE limit: reduce the loan or refer."))
        return {"collateral": collateral, "basis": basis_name, "max_loan": max_loan,
                "route": route, "regulatory": reg, "notes": notes, "candidates": candidates}
