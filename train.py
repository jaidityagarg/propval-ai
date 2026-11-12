"""Train and test the valuation model.

Run:  python train.py
Reads data/Mumbai_House_Prices.csv, writes artifacts/ (model, metadata, comparables)
and reports/accuracy.md.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from engine import ARTIFACTS, CATEGORICAL, DATA, FEATURES, ROOT, add_features, clean, set_categories

RAW = DATA / "Mumbai_House_Prices.csv"
SEED = 42
PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=10, verbose=-1, seed=SEED)
ROUNDS = 1200
SCALE_PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                    verbose=-1, seed=SEED)
MIN_PROJECT_COMPS = 3
GREEN_MAX, AMBER_MAX = 0.15, 0.25  # 80%-range half-width limits for each light (a lender policy setting)
TIME_ADJUSTMENT = 1.159  # RBI All-India House Price Index: 100 (2022-23 base) -> 115.9 (Q4 2025-26)
MODEL_VERSION = "1.1"
TIME_ADJUSTMENT_NOTE = "Updated from c.2022 to 2026 prices using the RBI All-India House Price Index (2022-23 = 100; Q4 2025-26 = 115.9)"


def scale_features(X, proj_n, proj_mad):
    SX = X.copy()
    SX["proj_n"] = proj_n
    SX["proj_mad"] = proj_mad
    return SX


def metrics(pred, actual):
    ape = np.abs(pred / actual - 1)
    return {"n": int(len(ape)), "median_abs_pct_error": round(float(np.median(ape) * 100), 1),
            "within_10pct": round(float((ape <= 0.10).mean() * 100), 1),
            "within_20pct": round(float((ape <= 0.20).mean() * 100), 1)}


def main():
    raw = pd.read_csv(RAW)
    df = clean(raw)
    # Same unit posted more than once with tiny differences: keep one.
    df = df.drop_duplicates(subset=["project", "region", "bhk", "area", "price_inr"]).reset_index(drop=True)
    df = add_features(df)
    df["y"] = np.log(df.psf)
    cats = {c: [str(v) for v in df[c].astype(str).unique()] for c in CATEGORICAL}
    X = set_categories(df[FEATURES], cats)
    groups = df["project"].str.lower().str.strip() + "|" + df["region"].astype(str)
    print(f"Clean listings: {len(df):,} in {df.region.nunique()} suburbs, {groups.nunique():,} projects")

    # ---- 1. Out-of-fold predictions, grouping by project, so every test flat is in a
    #         project the model has never seen (the hardest, most honest test).
    oof = np.zeros(len(df))
    baseline = np.zeros(len(df))
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(X, df.y, groups)):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], df.y.iloc[tr]), ROUNDS)
        oof[te] = m.predict(X.iloc[te])
        reg_med = df.iloc[tr].groupby("region", observed=True)["y"].median()
        baseline[te] = df.iloc[te]["region"].astype(str).map(reg_med.rename(index=str)).fillna(df.y.iloc[tr].median())
        print(f"  fold {fold + 1} done")

    # ---- 2. Same-project evidence: median price/sqft of OTHER listings in the same project,
    #         excluding listings with the same area (likely the same unit re-posted).
    proj_log = np.full(len(df), np.nan)
    proj_n = np.zeros(len(df), dtype=int)
    proj_mad = np.full(len(df), np.nan)
    for _, idx in df.groupby(groups).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub = df.loc[idx, ["area", "y"]].to_numpy()
        for j, i in enumerate(idx):
            mask = sub[:, 0] != sub[j, 0]
            if mask.sum() >= 1:
                proj_log[i] = np.median(sub[mask, 1])
                proj_n[i] = mask.sum()
                proj_mad[i] = np.median(np.abs(sub[mask, 1] - proj_log[i])) if mask.sum() >= 2 else np.nan
    has_comps = proj_n >= MIN_PROJECT_COMPS

    # ---- 3. Split projects into calibration (choose settings) and test (report) halves.
    rng = np.random.default_rng(SEED)
    uniq = groups.unique()
    calib_groups = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    is_cal = groups.isin(calib_groups).to_numpy()

    best_w, best_err = 0.0, 1e9
    for w in np.arange(0, 1.01, 0.05):
        sel = is_cal & has_comps
        blend = w * proj_log[sel] + (1 - w) * oof[sel]
        err = np.median(np.abs(np.exp(blend - df.y.to_numpy()[sel]) - 1))
        if err < best_err:
            best_w, best_err = float(w), err
    final_log = oof.copy()
    final_log[has_comps] = best_w * proj_log[has_comps] + (1 - best_w) * oof[has_comps]

    y = df.y.to_numpy()
    resid = np.abs(final_log - y)

    # Locally adaptive ranges: a second model predicts how uncertain each valuation is
    # (e.g. projects whose listings disagree a lot get wider ranges). Fitted on one part of
    # the calibration projects, the range multiplier is set on the other part (conformal).
    SX = scale_features(X, proj_n, proj_mad)
    fit_groups = set(rng.choice(np.array(sorted(calib_groups)), size=len(calib_groups) // 2, replace=False))
    fit_part = is_cal & groups.isin(fit_groups).to_numpy()
    conf_part = is_cal & ~fit_part
    scale_model = lgb.train(SCALE_PARAMS, lgb.Dataset(SX[fit_part], resid[fit_part]), 400)
    sigma = np.clip(scale_model.predict(SX), 0.02, None)
    score = resid / sigma
    # Calibrate the range separately within 5 bands of predicted uncertainty, so each band's
    # 80% range really covers ~80% of flats (conformal prediction, per band).
    edges = np.quantile(sigma[conf_part], [0.2, 0.4, 0.6, 0.8]).tolist()
    band = np.digitize(sigma, edges)
    conformal = {"sigma_edges": edges, "q80": [], "q90": []}
    for bnd in range(5):
        cal = conf_part & (band == bnd)
        n = cal.sum()
        conformal["q80"].append(float(np.quantile(score[cal], min(1, np.ceil((n + 1) * 0.80) / n))))
        conformal["q90"].append(float(np.quantile(score[cal], min(1, np.ceil((n + 1) * 0.90) / n))))
    q80_all = np.array(conformal["q80"])[band]
    half80 = q80_all * sigma
    hw_all = np.exp(half80) - 1

    actual_price = np.exp(y) * df.area.to_numpy()
    tp = np.exp(final_log) * df.area.to_numpy()
    te_all_mask = ~is_cal
    accuracy = {}
    for name, sel in [("with_project_comps", has_comps), ("no_project_comps", ~has_comps)]:
        te = sel & te_all_mask
        acc = metrics(tp[te], actual_price[te])
        acc["range_coverage_80"] = round(float((resid[te] <= half80[te]).mean() * 100), 1)
        acc["range_halfwidth_pct_median"] = round(float(np.median(hw_all[te]) * 100), 1)
        accuracy[name] = acc
    lights = {}
    for light, lo, hi in [("GREEN", 0, GREEN_MAX), ("AMBER", GREEN_MAX, AMBER_MAX), ("RED", AMBER_MAX, 99)]:
        m_ = te_all_mask & (hw_all > lo) & (hw_all <= hi)
        lights[light] = metrics(tp[m_], actual_price[m_]) if m_.sum() else {"n": 0}
        lights[light]["share_pct"] = round(float(m_.sum() / te_all_mask.sum() * 100), 1)
        if m_.sum():
            lights[light]["range_coverage_80"] = round(float((resid[m_] <= half80[m_]).mean() * 100), 1)
    overall_cov = round(float((resid[te_all_mask] <= half80[te_all_mask]).mean() * 100), 1)

    te_all = te_all_mask
    base_price = np.exp(baseline[te_all]) * df.area.to_numpy()[te_all]
    model_only = np.exp(oof[te_all]) * df.area.to_numpy()[te_all]
    blended = tp[te_all]
    summary = {
        "baseline_suburb_median_rate": metrics(base_price, actual_price[te_all]),
        "ml_model_only": metrics(model_only, actual_price[te_all]),
        "ml_model_plus_project_comparables": metrics(blended, actual_price[te_all]),
    }
    summary["ml_model_plus_project_comparables"]["range_coverage_80"] = overall_cov

    # ---- 4. Feature-group ablation: how much does location data add?
    abl = {}
    for label, drop in [("without location & distance features",
                         ["region", "zone", "lat", "lon", "dist_nariman_point_km", "dist_bkc_km",
                          "dist_powai_km", "dist_vashi_km", "dist_rail_km", "rail_stations_within_3km"]),
                        ("without rail-station features", ["dist_rail_km", "rail_stations_within_3km"]),
                        ("without distance-to-hub features",
                         ["dist_nariman_point_km", "dist_bkc_km", "dist_powai_km", "dist_vashi_km"])]:
        keep = [f for f in FEATURES if f not in drop]
        pred = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(X, df.y, groups):
            m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr][keep], df.y.iloc[tr]), 600)
            pred[te] = m.predict(X.iloc[te][keep])
        abl[label] = metrics(np.exp(pred[te_all]) * df.area.to_numpy()[te_all], actual_price[te_all])
        print(f"  ablation '{label}' done")

    # ---- 5. Final model on all data.
    final = lgb.train(PARAMS, lgb.Dataset(X, df.y), ROUNDS)
    ARTIFACTS.mkdir(exist_ok=True)
    final.save_model(str(ARTIFACTS / "model.txt"))
    scale_model.save_model(str(ARTIFACTS / "scale_model.txt"))
    comps_cols = ["project", "region", "bhk", "area", "price_inr", "psf", "type", "status", "age"]
    out = df[comps_cols].copy()
    for c in ["region", "type", "status", "age"]:
        out[c] = out[c].astype(str)
    out.to_csv(ARTIFACTS / "comparables.csv.gz", index=False, compression="gzip")

    meta = {
        "categories": cats, "conformal": conformal, "accuracy": accuracy, "summary": summary,
        "ablation": abl, "traffic_lights": lights, "blend_weight": best_w, "min_project_comps": MIN_PROJECT_COMPS,
        "region_counts": df.region.astype(str).value_counts().to_dict(),
        "area_p1": float(df.area.quantile(0.01)), "area_p99": float(df.area.quantile(0.99)),
        "green_max_halfwidth": GREEN_MAX, "amber_max_halfwidth": AMBER_MAX,
        "model_version": MODEL_VERSION, "trained_on": __import__("datetime").date.today().isoformat(), "time_adjustment": TIME_ADJUSTMENT, "time_adjustment_note": TIME_ADJUSTMENT_NOTE, "n_listings": int(len(df)),
        "n_projects": int(groups.nunique()), "n_regions": int(df.region.nunique()),
        "share_with_project_comps": round(float(has_comps.mean() * 100), 1),
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=1))
    write_report(meta)
    print(json.dumps({"summary": summary, "lights": lights, "accuracy": accuracy, "blend_weight": best_w, "ablation": abl}, indent=1))


def write_report(meta):
    s, a = meta["summary"], meta["accuracy"]
    lines = [
        "# Accuracy report", "",
        f"Data: {meta['n_listings']:,} cleaned listings, {meta['n_projects']:,} projects, "
        f"{meta['n_regions']} suburbs of the Mumbai Metropolitan Region.", "",
        "Method: 5-fold cross-validation grouped by project, so every test flat belongs to a project the "
        "model never saw in training. Settings were chosen on half the projects (calibration) and all numbers "
        "below are on the other half (test).", "",
        "## Engine vs simple baseline (test projects)", "",
        "| Method | Median error | Within ±10% | Within ±20% |", "|---|---|---|---|",
    ]
    names = {"baseline_suburb_median_rate": "Baseline: suburb median rate × area",
             "ml_model_only": "ML model only",
             "ml_model_plus_project_comparables": "ML model + same-project comparables (full engine)"}
    for k, v in s.items():
        lines.append(f"| {names[k]} | {v['median_abs_pct_error']}% | {v['within_10pct']}% | {v['within_20pct']}% |")
    lines += ["", "## Traffic light: does confidence mean anything? (test projects)", "",
              "| Light | Share of flats | Median error | Within ±10% | Within ±20% | 80% range actually covers |",
              "|---|---|---|---|---|---|"]
    for light, v in meta["traffic_lights"].items():
        if v.get("n"):
            lines.append(f"| {light} | {v['share_pct']}% | {v['median_abs_pct_error']}% | {v['within_10pct']}% | "
                         f"{v['within_20pct']}% | {v['range_coverage_80']}% |")
    lines += ["", "## By evidence available (test projects)", "",
              "| Situation | Median error | Within ±10% | Typical 80% range (±) | Range actually covers |",
              "|---|---|---|---|---|"]
    for k, label in [("with_project_comps", "3+ other listings in same project"),
                     ("no_project_comps", "New / unknown project")]:
        v = a[k]
        lines.append(f"| {label} | {v['median_abs_pct_error']}% | {v['within_10pct']}% | "
                     f"{v['range_halfwidth_pct_median']}% | {v['range_coverage_80']}% |")
    lines += ["", "## What the location data adds (ablation, ML model only)", "",
              "| Model | Median error | Within ±10% |", "|---|---|---|",
              f"| Full model | {s['ml_model_only']['median_abs_pct_error']}% | {s['ml_model_only']['within_10pct']}% |"]
    for k, v in meta["ablation"].items():
        lines.append(f"| {k} | {v['median_abs_pct_error']}% | {v['within_10pct']}% |")
    lines += ["", "Caveat: these are listing (asking) prices from about 2022, not registered sale prices. "
              "The same test method should be rerun on registered transactions (IGR) and the lender's own "
              "valuation reports before production use."]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "accuracy.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
