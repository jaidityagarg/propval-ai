"""PropVal AI: digital property valuation prototype (Streamlit app).

Run locally:  streamlit run app.py
"""
import datetime as dt
import html

import pandas as pd
import streamlit as st

from engine import ROOT, ValuationEngine, inr, load_regions

st.set_page_config(page_title="PropVal AI – Digital Property Valuation", page_icon="🏠", layout="wide")

LIGHT_STYLE = {
    "GREEN": ("#1b873f", "🟢 GREEN – high confidence"),
    "AMBER": ("#b7791f", "🟠 AMBER – medium confidence"),
    "RED": ("#c53030", "🔴 RED – low confidence"),
}


def artifacts_version():
    """Changes whenever the model files change, so a redeploy never keeps a stale model in memory."""
    files = sorted((ROOT / "artifacts").glob("*"))
    return tuple((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in files) + (
        (ROOT / "artifacts" / "meta.json").read_text().__hash__(),)


@st.cache_resource
def _load_engine(version):
    return ValuationEngine()


def get_engine():
    return _load_engine(artifacts_version())


@st.cache_data
def suburb_table():
    eng = get_engine()
    reg = load_regions()
    stats = eng.comps.groupby("region").agg(listings=("psf", "size"), median_rate=("psf", "median")).reset_index()
    return reg.merge(stats, on="region")


@st.cache_data
def project_index():
    idx = (get_engine().comps.groupby(["project", "region"]).size().rename("n").reset_index())
    idx["key"] = idx.project.str.lower()
    return idx


def search_projects(query, limit=10):
    """Find buildings by name. Returns (exact, similar): exact = names containing every word
    of the query; similar = names sharing some words, offered only as suggestions."""
    words = [w for w in query.lower().replace(",", " ").split() if len(w) > 1]
    idx = project_index().copy()
    if not words:
        return idx.iloc[0:0], idx.iloc[0:0]
    idx["score"] = sum(idx.key.str.contains(w, regex=False).astype(int) for w in words)
    exact = idx[idx.score == len(words)].sort_values("n", ascending=False).head(limit)
    similar = idx[(idx.score > 0) & (idx.score < len(words))].sort_values(["score", "n"], ascending=False)
    return exact, similar.head(limit)


# Typical Mumbai area ratios relative to carpet area (approximate; builder loading varies 20-40%).
AREA_RATIO = {"Carpet": 1.0, "Built-up": 1.20, "Super built-up": 1.35}


# Basis of the areas in the training listings. Checked in the data: new and resale listings in the same
# project and BHK quote almost identical areas (median ratio 1.02), and the median 1BHK is ~610 sq ft,
# well above typical Mumbai 1BHK carpet areas, so listings are treated as super built-up throughout.
DATA_AREA_BASIS = "Super built-up"


def to_model_area(area, area_type, status=None, age=None):
    """Convert the entered area to carpet, then to the basis the training listings used."""
    carpet = area / AREA_RATIO[area_type]
    return carpet, round(carpet * AREA_RATIO[DATA_AREA_BASIS], 1)


eng = get_engine()
meta = eng.meta
subs = suburb_table().sort_values("listings", ascending=False)

st.title("🏠 PropVal AI")
st.caption("AI-powered digital property valuation for home loans · Prototype for the Hero Housing Fintech "
           "Challenge, Problem Statement 1")

tab_value, tab_acc, tab_how = st.tabs(["Value a property", "How accurate is it?", "How it works"])

# ------------------------------------------------------------------ VALUE
with tab_value:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Property details")
        NEW = "— New / not in list —"
        query = st.text_input("🔍 Search building anywhere in Mumbai (optional)",
                              placeholder="e.g. raheja vistas, lodha, hiranandani")
        picked = None
        if query.strip():
            exact, similar = search_projects(query)
            NONE = "— None of these: choose the suburb below —"
            if not exact.empty:
                hits = exact
                labels = [f"{r.project} — {r.region} ({r.n} listings)" for r in hits.itertuples()]
                choice = st.selectbox("Matching buildings", labels + [NONE])
            else:
                hits = similar
                st.warning(f"'{query.strip()}' is not in the data (it may be newer than 2022). "
                           "Choose its real suburb below; the building will be valued as a new project.")
                labels = [f"{r.project} — {r.region} ({r.n} listings)" for r in hits.itertuples()]
                choice = st.selectbox("Similar names (only pick one if it is the same building)",
                                      [NONE] + labels) if labels else NONE
            if choice != NONE:
                picked = hits.iloc[labels.index(choice)]
        if picked is not None:
            region, project = picked.region, picked.project
            st.info(f"Using **{project}**, {region}")
        else:
            region = st.selectbox("Suburb", subs.region.tolist(), index=0)
            projects = (eng.comps[eng.comps.region == region].project.value_counts().index.tolist())
            project = st.selectbox("Building / project", [NEW] + projects,
                                   help="Pick the project if it appears. Other listings in the same project "
                                        "make the valuation much more confident.")
        c1, c2 = st.columns(2)
        bhk = c1.number_input("BHK", 1, 10, 2)
        area = c2.number_input("Area (sq ft)", 150, 10000, 650, step=10)
        area_type = st.radio("This area is", list(AREA_RATIO), horizontal=True,
                             help="Carpet area is what RERA and sale agreements use. Super built-up includes "
                                  "walls, lobbies and common areas (builder 'loading').")
        ptype = st.selectbox("Property type", ["Apartment", "Studio Apartment", "Villa", "Independent House",
                                               "Penthouse"])
        status = st.selectbox("Construction status", ["Ready to move", "Under Construction"])
        age = st.selectbox("New or resale", ["Resale", "New", "Unknown"])

        st.subheader("Loan details (optional)")
        loan_l = st.number_input("Loan requested (₹ lakh)", 0.0, 10000.0, 0.0, step=5.0)
        agree_l = st.number_input("Agreement / documented value (₹ lakh)", 0.0, 50000.0, 0.0, step=5.0)
        valuer_l = st.number_input("Valuer's reported value (₹ lakh)", 0.0, 50000.0, 0.0, step=5.0,
                                   help="Enter a valuer's figure to cross-check it against the AVM.")
        go = st.button("Value this property", type="primary", width="stretch")

    with right:
        if not go and "last" not in st.session_state:
            st.info("Enter the property details and press **Value this property**.")
            m = subs.copy()
            m["size"] = 200 + m.listings.clip(upper=3000) / 3
            m["color"] = m.median_rate.rank(pct=True).map(lambda r: [int(255 * r), int(80 + 100 * (1 - r)), 60, 160])
            st.caption("Suburbs in the data (bigger dot = more listings, redder = higher price per sq ft)")
            st.map(m, latitude="lat", longitude="lon", size="size", color="color", zoom=9)
        else:
            carpet, model_area = to_model_area(area, area_type, status, age)
            subject = dict(region=region, project="" if project == NEW else project, bhk=int(bhk),
                           area=float(model_area), type=ptype, status=status, age=age,
                           carpet=carpet, area_text=f"{area:,.0f} sq ft {area_type.lower()}")
            if go:
                st.session_state.last = subject
            subject = dict(st.session_state.last)
            carpet = subject.pop("carpet")
            area_text = subject.pop("area_text")
            v = eng.value(subject)
            lend = eng.lending_view(v, loan_l * 1e5 or None, agree_l * 1e5 or None, valuer_l * 1e5 or None)
            color, label = LIGHT_STYLE[v["light"]]

            st.markdown(
                f"""<div style="border-left:8px solid {color};padding:12px 18px;background:rgba(0,0,0,0.03);
                border-radius:6px">
                <div style="font-size:15px;color:#666">Estimated market value</div>
                <div style="font-size:38px;font-weight:700">{inr(v['value'])}</div>
                <div style="font-size:17px">80% range: <b>{inr(v['low'])} – {inr(v['high'])}</b>
                &nbsp;(±{v['half_width_pct']:.0f}%) · ₹{v['value'] / carpet:,.0f} per sq ft carpet</div>
                <div style="font-size:19px;margin-top:8px;color:{color};font-weight:700">{label}</div>
                <div style="font-size:15px">{lend['route']}</div></div>""", unsafe_allow_html=True)
            for w in v["warnings"]:
                st.warning(w)
            st.caption(f"📐 {area_text} → about {carpet:,.0f} sq ft carpet, "
                       f"{carpet * AREA_RATIO['Super built-up']:,.0f} sq ft super built-up (typical Mumbai ratios). "
                       f"Compared with listings on a {DATA_AREA_BASIS.lower()} basis.")
            st.caption("📈 " + meta.get("time_adjustment_note", "Prices are c.2022 listing prices."))
            acc = v["accuracy"]
            st.caption(f"In testing, {v['light']} valuations had a median error of {acc['median_abs_pct_error']}% "
                       f"and {acc['within_10pct']}% were within ±10% of the actual price.")

            st.subheader("Why this value?")
            st.write(f"Starting point: average rate across the region, ₹{v['baseline_psf']:,.0f} per sq ft. "
                     "Each factor below moves the price up or down:")
            eff = pd.DataFrame(v["effects"], columns=["Factor", "Effect on price (%)"])
            eff = eff[eff["Effect on price (%)"].abs() >= 0.5]
            import altair as alt
            eff["Direction"] = eff["Effect on price (%)"].map(lambda x: "Raises price" if x > 0 else "Lowers price")
            chart = alt.Chart(eff).mark_bar().encode(
                x=alt.X("Effect on price (%):Q"),
                y=alt.Y("Factor:N", sort=None, axis=alt.Axis(labelLimit=320, title=None)),
                color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises price", "Lowers price"],
                                                                 range=["#2f855a", "#c53030"])),
                tooltip=["Factor", alt.Tooltip("Effect on price (%):Q", format="+.1f")],
            ).properties(height=45 * len(eff) + 30)
            st.altair_chart(chart, width="stretch")

            st.subheader("Comparable listings")
            comps = v["comps"][["project", "bhk", "area", "price_inr", "psf", "status", "age", "adjusted_value"]].copy()
            comps.columns = ["Project", "BHK", "Area (sq ft)", "Listed price (2022)", "₹ per sq ft (2022)", "Status",
                             "New/Resale", "Implied value for this flat (today)"]
            comps["Listed price (2022)"] = comps["Listed price (2022)"].map(inr)
            comps["Implied value for this flat (today)"] = comps["Implied value for this flat (today)"].map(inr)
            comps["₹ per sq ft (2022)"] = comps["₹ per sq ft (2022)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(comps, hide_index=True, width="stretch")
            st.caption(f"{v['n_same_project']} listings found in the same project." if v["n_same_project"]
                       else "No listings from the same project; comparables are similar flats in the same suburb.")

            st.subheader("Lending decision support")
            c1, c2 = st.columns(2)
            c1.metric("Collateral value used", inr(lend["collateral"]), help="Lowest of the figures available")
            c2.metric("Maximum loan under RBI LTV caps", inr(lend["max_loan"]))
            st.write(f"Collateral basis: **{lend['basis']}** (the lowest of: "
                     + ", ".join(f"{k} {inr(x)}" for k, x in lend["candidates"].items()) + ")")
            for n in lend["regulatory"] + lend["notes"]:
                st.write("• " + n)

            # Downloadable report
            rows = "".join(f"<tr><td>{html.escape(f)}</td><td>{e:+.1f}%</td></tr>" for f, e in v["effects"])
            notes = "".join(f"<li>{html.escape(n)}</li>" for n in lend["regulatory"] + lend["notes"] + v["warnings"])
            report = f"""<html><head><meta charset="utf-8"><title>Valuation report</title></head>
<body style="font-family:Arial;max-width:800px;margin:auto">
<h2>PropVal AI – Automated valuation report</h2>
<p>Generated {dt.datetime.now():%d %b %Y %H:%M} · Model trained on {meta['n_listings']:,} listings</p>
<h3>Property</h3><p>{bhk} BHK {html.escape(ptype)}, {html.escape(area_text)} (≈ {carpet:,.0f} sq ft carpet), {html.escape(subject['project'] or 'project not specified')},
{html.escape(region)} · {html.escape(status)} · {html.escape(age)}</p>
<h3>Valuation</h3><p><b>{inr(v['value'])}</b> (80% range {inr(v['low'])} – {inr(v['high'])}; 90% lower bound {inr(v['low90'])})<br>
Confidence: <b>{v['light']}</b> – {html.escape(lend['route'])}</p>
<h3>Main factors</h3><table border=1 cellpadding=4>{rows}</table>
<h3>Comparables</h3>{comps.to_html(index=False)}
<h3>Lending</h3><p>Collateral value used: {inr(lend['collateral'])} ({html.escape(lend['basis'])}).
Maximum loan under RBI LTV caps: {inr(lend['max_loan'])}.</p><ul>{notes}</ul>
<p style="color:#888">Prototype. Based on listing prices (c. 2022), not registered sale prices. {html.escape(meta.get('time_adjustment_note', ''))}. Not a substitute for a
valuation by a registered valuer.</p></body></html>"""
            st.download_button("⬇ Download valuation report", report, file_name="valuation_report.html",
                               mime="text/html")

# ------------------------------------------------------------------ ACCURACY
with tab_acc:
    st.markdown((ROOT / "reports" / "accuracy.md").read_text())

# ------------------------------------------------------------------ HOW
with tab_how:
    st.markdown(f"""
### How PropVal AI values a property
1. **Data**: {meta['n_listings']:,} cleaned listings across {meta['n_regions']} suburbs and {meta['n_projects']:,}
   projects in the Mumbai Metropolitan Region, plus suburb coordinates, distances to the main job hubs
   (Nariman Point, BKC, Powai, Vashi) and to the nearest of 112 suburban railway stations (DataMeet open data).
2. **ML model**: a gradient-boosting model (LightGBM) predicts the price per sq ft from location, size, layout,
   type, construction status and new/resale.
3. **Comparables**: if the same project has other listings, their prices are blended in
   ({meta['blend_weight']:.0%} weight), the way a valuer uses recent sales in the same building.
4. **Confidence range**: a second model predicts how uncertain each valuation is, and *conformal prediction*
   calibrates the range so that an 80% range really contains the true price about 80% of the time.
5. **Traffic light**: range within ±{meta['green_max_halfwidth']:.0%} → GREEN (desktop valuation, no site visit);
   within ±{meta['amber_max_halfwidth']:.0%} → AMBER (one site visit); wider → RED (full physical valuation).
   These limits are policy settings a lender's board would choose.
6. **Lending**: collateral value = the lowest of the AVM lower bound, agreement value and valuer figure. The maximum
   loan follows RBI loan-to-value caps (90% up to ₹30 L, 80% up to ₹75 L, 75% above). Valuer figures outside the AVM
   range are flagged. RBI rules on how many valuation reports are needed (≥ ₹50 L: two) are shown for every loan.

**Price update**: {meta.get('time_adjustment_note', '')}. Source: RBI press release, 29 May 2026.

**Limitations**: listing (asking) prices from about 2022, updated with a single all-India index (city-level indices for Mumbai and Thane are on RBI's DBIE portal and would be more precise), not registered sale prices; suburb-level coordinates;
Mumbai region only. The production roadmap replaces listings with registered transaction data (IGR) and the
lender's own valuation reports, adds satellite and infrastructure features, and extends city by city.
""")
