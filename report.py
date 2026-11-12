"""Branded PDF valuation report (ReportLab, BSD licence; DejaVu fonts, free licence)."""
import datetime as dt
import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from engine import inr

ASSETS = Path(__file__).parent / "assets"
pdfmetrics.registerFont(TTFont("DejaVu", str(ASSETS / "fonts" / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(ASSETS / "fonts" / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

TEAL, DARK, GREY, LIGHT = colors.HexColor("#0F7C80"), colors.HexColor("#0B3C49"), colors.HexColor("#5E7B85"), colors.HexColor("#EEF4F4")
LIGHT_COLOR = {"GREEN": colors.HexColor("#2E9B5A"), "AMBER": colors.HexColor("#E39A2D"), "RED": colors.HexColor("#C8423B")}
LIGHT_WORD = {"GREEN": "GREEN – high confidence", "AMBER": "AMBER – medium confidence", "RED": "RED – low confidence"}

base = ParagraphStyle("base", fontName="DejaVu", fontSize=9, leading=12.5, textColor=colors.HexColor("#1F2A30"))
small = ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=GREY)
h2 = ParagraphStyle("h2", parent=base, fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=DARK, spaceBefore=9, spaceAfter=4)
cell = ParagraphStyle("cell", parent=base, fontSize=8, leading=10.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DejaVu-Bold", textColor=colors.white)
right = ParagraphStyle("right", parent=small, alignment=TA_RIGHT)


def P(text, style=base):
    return Paragraph(escape(str(text)), style)


def grid(rows, widths, header=True):
    data = [[P(c, cellb if header and i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7D3D2")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), DARK))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(v, lend, prop, comps, meta, when=None):
    """v: engine.value() result; lend: lending_view(); prop: dict of property text fields;
    comps: DataFrame of comparables already formatted for display; meta: model metadata."""
    when = when or dt.datetime.now()
    ref = f"PV-{when:%Y%m%d-%H%M%S}"
    buf = io.BytesIO()
    W = A4[0] - 36 * mm

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(TEAL)
        canvas.rect(0, A4[1] - 6 * mm, A4[0], 6 * mm, stroke=0, fill=1)
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(18 * mm, 10 * mm, f"PropVal AI · Report {ref} · Model {meta.get('model_version', '')}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title="PropVal AI valuation report", author="PropVal AI")
    s = []
    logo = Image(str(ASSETS / "logo.png"), width=62 * mm, height=62 * mm * 160 / 651)
    head = Table([[logo, Paragraph(f"<b>Automated valuation report</b><br/>Report {ref}<br/>"
                                   f"{when:%d %b %Y, %H:%M}", ParagraphStyle("r", parent=base, alignment=TA_RIGHT,
                                                                             textColor=DARK))]],
                 colWidths=[W * 0.55, W * 0.45])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (0, 0), 0),
                              ("RIGHTPADDING", (-1, 0), (-1, 0), 0)]))
    s += [head, Spacer(1, 4 * mm)]

    # Property
    s.append(Paragraph("Property", h2))
    s.append(grid([["Building / project", prop["project"] or "Not specified (valued as a new project)"],
                   ["Suburb", prop["region"]],
                   ["Flat", f"{prop['bhk']} BHK {prop['type']} · {prop['status']} · {prop['age']}"],
                   ["Area", prop["area_text"] if prop["area_text"].endswith("carpet")
                    else f"{prop['area_text']} (about {prop['carpet']:,.0f} sq ft carpet)"]],
                  [W * 0.28, W * 0.72], header=False))

    # Valuation box
    lc = LIGHT_COLOR[v["light"]]
    big = ParagraphStyle("big", parent=base, fontName="DejaVu-Bold", fontSize=22, leading=27, textColor=DARK)
    box = Table([[Paragraph("Estimated market value", small)],
                 [Paragraph(escape(inr(v["value"])), big)],
                 [P(f"80% range {inr(v['low'])} – {inr(v['high'])} (±{v['half_width_pct']:.0f}%) · "
                    f"90% lower bound {inr(v['low90'])} · ₹{v['value'] / prop['carpet']:,.0f} per sq ft carpet")],
                 [Paragraph(f"<font color='{lc.hexval().replace('0x', '#')}'><b>{LIGHT_WORD[v['light']]}</b></font>"
                            f" – {escape(lend['route'])}", base)]],
                colWidths=[W - 4])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 5, lc),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
    s += [Paragraph("Valuation", h2), box]
    acc = v["accuracy"]
    s.append(Spacer(1, 2 * mm))
    s.append(P(f"In testing, {v['light'].lower()} valuations had a median error of {acc['median_abs_pct_error']}% and "
               f"{acc['within_10pct']}% were within ±10% of the actual price.", small))
    for w in v["warnings"]:
        s.append(P("⚠ " + w, small))
    if prop.get("market_check"):
        s.append(P("Market check: " + prop["market_check"], small))

    # Factors
    rows = [["Factor", "Effect on price"]] + [[f, f"{e:+.1f}%"] for f, e in v["effects"] if abs(e) >= 0.5]
    s += [Paragraph("Why this value?", h2),
          P(f"Starting point: average rate across the region, ₹{v['baseline_psf']:,.0f} per sq ft. "
            "Each factor moves the price up or down.", small), Spacer(1, 1.5 * mm),
          grid(rows, [W * 0.75, W * 0.25])]

    # Comparables
    heads = ["Project", "BHK", "Sq ft", "Listed (2022 level)", "₹ per sq ft (2022 level)", "Status", "New or resale",
             "Value for this flat today"]
    crow = [heads] + comps.astype(str).values.tolist()
    s += [Paragraph("Comparable listings", h2),
          grid(crow, [W * x for x in (0.24, 0.07, 0.08, 0.12, 0.12, 0.13, 0.10, 0.14)])]

    # Lending
    lrows = [["Collateral value used", f"{inr(lend['collateral'])} ({lend['basis']})"],
             ["Maximum loan under RBI loan-to-value limits", inr(lend["max_loan"])]]
    lrows += [[k, inr(x)] for k, x in lend["candidates"].items()]
    lend_block = [Paragraph("Lending decision support", h2), grid(lrows, [W * 0.5, W * 0.5], header=False),
                  Spacer(1, 1.5 * mm)] + [P("• " + n) for n in lend["regulatory"] + lend["notes"]]
    s.append(KeepTogether(lend_block))

    # Audit and disclaimer
    s += [Paragraph("Audit details", h2),
          P(f"Model version {meta.get('model_version', '')}, trained {meta.get('trained_on', '')} on "
            f"{meta['n_listings']:,} listings. Settings: green within ±{meta['green_max_halfwidth']:.0%}, amber within "
            f"±{meta['amber_max_halfwidth']:.0%}, price index factor {meta.get('time_adjustment', 1.0)}. "
            f"{meta.get('time_adjustment_note', '')}.", small),
          Spacer(1, 6 * mm),
          grid([["Reviewed by (credit officer / valuer)", "Signature", "Date"], ["", "", ""]],
               [W * 0.45, W * 0.33, W * 0.22]),
          Spacer(1, 4 * mm),
          P("Prototype for the Hero Housing Fintech Challenge. Based on listing (asking) prices from two public "
            "Kaggle datasets (about 2022–2024), not registered sale prices. Not a substitute for a valuation by a "
            "registered valuer.", small)]
    doc.build(s, onFirstPage=frame, onLaterPages=frame)
    return buf.getvalue(), ref
