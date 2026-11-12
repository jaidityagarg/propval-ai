"""Second listings source: Kaggle 'Mumbai House Price Data (70k entries)' by kevinnadar22.

Adds building-level map coordinates, bathrooms, furnishing and building age.
clean_source2() returns rows in the same shape as engine.clean() so both can be merged.
"""
import re

import numpy as np
import pandas as pd

from engine import DATA, haversine_km, load_regions, normalize_region

RAW2 = DATA / "Mumbai_House_Prices_2.csv"
MMR_BOX = (18.80, 19.80, 72.70, 73.40)  # lat min, lat max, lon min, lon max


def project_key(name):
    """Lower-case letters and digits only, used to match the same project across sources."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def tidy_title(t):
    """'Origin Origin Wisteria Square' -> 'Origin Wisteria Square' (builder name repeated)."""
    w = str(t).split()
    if len(w) >= 2 and w[0].lower() == w[1].lower():
        w = w[1:]
    return " ".join(w).strip()


def assign_region(locality, lat, lon, regions, known):
    """Map a locality to one of our suburbs; use the map position to choose East/West."""
    r = normalize_region(locality, known)
    if r:
        return r
    low = str(locality).strip().lower()
    cands = regions[regions.region.str.lower().str.startswith(low + " ")]
    if len(cands) == 0:
        cands = regions[regions.region.str.lower().str.contains(re.escape(low))]
    if len(cands) == 0:
        cands = regions
        d = haversine_km(lat, lon, cands.lat.to_numpy(), cands.lon.to_numpy())
        return cands.region.iloc[int(np.argmin(d))] if d.min() <= 2.0 else None
    d = haversine_km(lat, lon, cands.lat.to_numpy(), cands.lon.to_numpy())
    return cands.region.iloc[int(np.argmin(d))]


def clean_source2(raw: pd.DataFrame, report=None) -> pd.DataFrame:
    report = report if report is not None else {}
    regions = load_regions()
    known = set(regions.region)
    df = raw.copy()
    report["raw_rows"] = len(df)
    df = df.drop_duplicates()
    report["after_exact_dedupe"] = len(df)
    la0, la1, lo0, lo1 = MMR_BOX
    df = df[df.latitude.between(la0, la1) & df.longitude.between(lo0, lo1)]
    report["after_map_box"] = len(df)
    df["region"] = [assign_region(l, a, o, regions, known) for l, a, o in zip(df.locality, df.latitude, df.longitude)]
    df = df.dropna(subset=["region"])
    report["after_suburb_mapping"] = len(df)
    # A building more than 6 km from its suburb's centre has a wrong map position: use the centre instead.
    cen = df[["region"]].merge(regions, on="region", how="left")
    far = haversine_km(df.latitude.to_numpy(), df.longitude.to_numpy(), cen.lat.to_numpy(), cen.lon.to_numpy()) > 6
    report["map_positions_replaced"] = int(far.sum())
    df["lat"] = np.where(far, cen.lat.to_numpy(), df.latitude.to_numpy())
    df["lon"] = np.where(far, cen.lon.to_numpy(), df.longitude.to_numpy())
    df["loc_precise"] = (~far).astype(int)
    out = pd.DataFrame({
        "project": df.title.map(tidy_title).replace("", np.nan),
        "region": df.region, "bhk": df.bedroom_num, "area": df.area,
        "price_inr": df.price.astype(float),
        "type": df.property_type.replace({"Independent Floor": "Independent House"}),
        "status": np.nan,                       # not recorded in this source
        "age": np.where(df.age == 0, "New", "Resale"),
        "age_years": df.age.astype(float),
        "bathrooms": df.bathroom_num.astype(float),
        "furnished": df.furnished,
        "lat": df.lat, "lon": df.lon, "loc_precise": df.loc_precise, "source": "kaggle_70k",
    })
    out = out.dropna(subset=["project"])
    report["after_blank_names"] = len(out)
    out = out[(out.area >= 150) & (out.area <= 10000) & (out.bhk >= 1)]
    report["after_size_checks"] = len(out)
    out["psf"] = out.price_inr / out.area
    med = out.groupby("region")["psf"].transform("median")
    out = out[(out.psf > med / 3) & (out.psf < med * 3)]
    report["after_price_outliers"] = len(out)
    out = out.merge(regions[["region", "zone"]], on="region", how="left")
    return out.reset_index(drop=True)


def combine(src1: pd.DataFrame, src2: pd.DataFrame, report=None):
    """Merge both sources into one training table.

    1. Give first-source projects the map position of the same project in the second source.
    2. Drop second-source rows that repeat a first-source flat (same project, suburb, BHK and area).
    3. Scale second-source prices to the first source's price level (measured on shared projects).
    """
    report = report if report is not None else {}
    a, b = src1.copy(), src2.copy()
    a["k"], b["k"] = a.project.map(project_key), b.project.map(project_key)
    pos = (b[b.loc_precise == 1].groupby(["k", "region"])[["lat", "lon"]].median()
           .rename(columns={"lat": "plat", "lon": "plon"}).reset_index())
    a = a.merge(pos, on=["k", "region"], how="left")
    hit = a.plat.notna()
    a.loc[hit, "lat"], a.loc[hit, "lon"], a.loc[hit, "loc_precise"] = a.plat[hit], a.plon[hit], 1
    a = a.drop(columns=["plat", "plon"])
    report["source1_rows_given_building_position"] = int(hit.sum())
    seen = set(zip(a.k, a.region, a.bhk, a.area))
    rep = np.array([(k, r, x, y) in seen for k, r, x, y in zip(b.k, b.region, b.bhk, b.area)])
    report["source2_repeats_removed"] = int(rep.sum())
    b = b[~rep]
    # Price level: same project and BHK, median price per sq ft ratio (source 2 / source 1).
    ga = a.groupby(["k", "region", "bhk"]).psf.median()
    gb = b.groupby(["k", "region", "bhk"]).psf.median()
    j = pd.concat([ga, gb], axis=1, join="inner")
    level = float((j.iloc[:, 1] / j.iloc[:, 0]).median())
    report["source2_price_level_vs_source1"] = round(level, 3)
    b["price_inr"] = b.price_inr / level
    b["psf"] = b.psf / level
    # Use the first source's project spelling when the same project appears in both.
    names = a.drop_duplicates(["k", "region"]).set_index(["k", "region"]).project
    b["project"] = [names.get((k, r), p) for k, r, p in zip(b.k, b.region, b.project)]
    out = pd.concat([a, b], ignore_index=True).drop(columns=["k"])
    report["combined_rows"] = len(out)
    return out, level
