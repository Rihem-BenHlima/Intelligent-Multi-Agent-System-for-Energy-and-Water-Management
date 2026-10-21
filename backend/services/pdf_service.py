"""
Renders generated reports into downloadable PDF documents with styling and charts.
"""
import json
import re
from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable,
    Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.doughnut import Doughnut


def _fmt_compact(v: float) -> str:
    """Short, readable number for chart labels."""
    v = float(v)
    if abs(v) >= 100_000:
        return f"{v / 1000:,.0f}k"
    if abs(v) >= 10_000:
        return f"{v / 1000:,.1f}k"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"

_SFMDATA_RE = re.compile(r"\s*<!--SFMDATA\s*(\{.*?\})\s*-->\s*$", re.DOTALL)
_CHART_PALETTE = [colors.HexColor(c) for c in ("#2563eb", "#f59e0b", "#ef4444", "#10b981")]

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
# Fallback regex for numeric headings
_NUM_HEADING_RE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$", re.MULTILINE)

DOMAIN_ACCENT = {"energy": colors.HexColor("#2563eb"), "water": colors.HexColor("#0891b2")}
DOMAIN_TITLE = {"energy": "Electricity Report", "water": "Water Report"}
SOURCE_LABEL = {"weekly_cron": "Weekly automated report", "manual": "Manual report"}

# Dashboard KPI card colours: (accent / left-bar, background tint)
_STATUS_COLORS = {
    "normal": (colors.HexColor("#16a34a"), colors.HexColor("#f0fdf4")),
    "watch":  (colors.HexColor("#d97706"), colors.HexColor("#fffbeb")),
    "alert":  (colors.HexColor("#dc2626"), colors.HexColor("#fef2f2")),
}


def _extract_sfmdata(raw: str) -> tuple[str, dict | None]:
    """Pull the trailing `<!--SFMDATA {...}-->` chart payload off the report
    text (report_agent appends it). Returns (clean_text, data_or_None)."""
    m = _SFMDATA_RE.search(raw or "")
    if not m:
        return raw, None
    clean = raw[:m.start()].rstrip()
    try:
        return clean, json.loads(m.group(1))
    except (ValueError, TypeError):
        return clean, None


def _split_sections(raw: str) -> list[tuple[str, str]]:
    """Split report text into (title, body) pairs on markdown headings."""
    matches = list(_HEADING_RE.finditer(raw))
    if not matches:
        matches = list(_NUM_HEADING_RE.finditer(raw))
    if not matches:
        return [("Report", raw)]
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections.append((m.group(1).strip(), raw[start:end].strip()))
    return sections


def _inline_bold(text: str) -> str:
    """Convert **bold** / _italic_ to reportlab tags; escape XML-unsafe chars."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<i>\1</i>", text)
    return text


def _body_flowables(body: str, styles) -> list:
    flowables = []
    bullet_items: list[str] = []

    def _flush_bullets():
        if bullet_items:
            flowables.append(ListFlowable(
                [ListItem(Paragraph(_inline_bold(b), styles["Body"])) for b in bullet_items],
                bulletType="bullet", start="•", leftIndent=14,
            ))
            bullet_items.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet_match:
            bullet_items.append(bullet_match.group(1))
            continue
        _flush_bullets()
        flowables.append(Paragraph(_inline_bold(stripped), styles["Body"]))
        flowables.append(Spacer(1, 3))
    _flush_bullets()
    return flowables


def _kpi_cards(kpi_rows: list, styles) -> list:
    """`(label, value, interp, status)` tuples → colour-coded cards."""
    label_style = ParagraphStyle("KpiLabel", parent=styles["Body"], fontSize=7.5,
                                 textColor=colors.HexColor("#64748b"), leading=9,
                                 spaceAfter=1)
    interp_style = ParagraphStyle("KpiInterp", parent=styles["Body"], fontSize=8.5,
                                  textColor=colors.HexColor("#475569"), leading=11,
                                  spaceBefore=2)
    out = []
    for label, value, interp, status in kpi_rows:
        accent, bg = _STATUS_COLORS.get(status, _STATUS_COLORS["normal"])
        value_style = ParagraphStyle("KpiValue", parent=styles["Body"], fontSize=13,
                                     leading=16, textColor=accent)
        cell = [
            Paragraph(label.upper(), label_style),
            Paragraph(f"<b>{_inline_bold(value)}</b>", value_style),
            Paragraph(_inline_bold(interp), interp_style),
        ]
        card = Table([[cell]], colWidths=[styles["_content_width"]])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("LINEBEFORE", (0, 0), (-1, -1), 3, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        out.append(card)
        out.append(Spacer(1, 5))
    return out


_AXIS_GREY = colors.HexColor("#94a3b8")
_LABEL_GREY = colors.HexColor("#475569")
_TITLE_GREY = colors.HexColor("#64748b")


def _bar_chart(daily_7: list, unit: str, accent) -> Drawing:
    vals = [float(v) for _, v in daily_7]
    total = sum(vals)
    d = Drawing(238, 140)
    d.add(String(2, 128, f"Daily use, last {len(daily_7)} days ({unit})",
                 fontSize=8, fillColor=_TITLE_GREY))
    d.add(String(2, 117, f"{_fmt_compact(total)} {unit} total  ·  {_fmt_compact(total / len(vals))} {unit}/day avg",
                 fontSize=6.8, fillColor=_AXIS_GREY))
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 24, 16, 205, 84
    bc.data = [vals]
    bc.categoryAxis.categoryNames = [str(lbl) for lbl, _ in daily_7]
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.fillColor = _LABEL_GREY
    bc.valueAxis.labels.fontSize = 6.5
    bc.valueAxis.labels.fillColor = _AXIS_GREY
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(vals) * 1.25 if any(vals) else 1
    bc.bars[0].fillColor = accent
    bc.bars[0].strokeColor = None
    bc.barWidth = 7
    bc.groupSpacing = 4
    # real value on top of every bar
    bc.barLabels.fontSize = 6.5
    bc.barLabels.fillColor = colors.HexColor("#334155")
    bc.barLabels.boxAnchor = "s"
    bc.barLabels.dy = 3
    bc.barLabelFormat = _fmt_compact
    d.add(bc)
    return d


def _donut(tod_week: list, unit: str) -> Drawing:
    pairs = [(str(n), max(float(v), 0.0)) for n, v in tod_week]
    total = sum(v for _, v in pairs) or 1.0
    d = Drawing(238, 140)
    d.add(String(2, 128, f"Usage by time of day ({unit})", fontSize=8, fillColor=_TITLE_GREY))
    dn = Doughnut()
    dn.x, dn.y, dn.width, dn.height = 6, 12, 96, 96
    dn.data = [v for _, v in pairs]
    dn.innerRadiusFraction = 0.55
    for i in range(len(pairs)):
        dn.slices[i].fillColor = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        dn.slices[i].strokeWidth = 0.75
        dn.slices[i].strokeColor = colors.white
    d.add(dn)
    # legend: swatch · name · value · share
    for i, (name, v) in enumerate(pairs):
        y = 96 - i * 16
        d.add(Rect(112, y, 7, 7, fillColor=_CHART_PALETTE[i % len(_CHART_PALETTE)], strokeColor=None))
        d.add(String(123, y + 1, name, fontSize=6.8, fillColor=_LABEL_GREY))
        d.add(String(236, y + 1, f"{_fmt_compact(v)} {unit}  ·  {v / total * 100:.0f}%",
                     fontSize=6.8, fillColor=colors.HexColor("#334155"), textAnchor="end"))
    return d


def _charts_flowable(cd: dict, accent) -> list:
    """cd = {"unit", "daily_7", "tod_week"} for one domain."""
    if not cd:
        return []
    unit = cd.get("unit", "")
    daily_7 = cd.get("daily_7") or []
    tod_week = cd.get("tod_week") or []
    left = _bar_chart(daily_7, unit, accent) if any(v for _, v in daily_7) else None
    right = _donut(tod_week, unit) if sum(max(v, 0) for _, v in tod_week) > 0 else None
    if not left and not right:
        return []
    row = [left or Spacer(1, 1), right or Spacer(1, 1)]
    t = Table([row], colWidths=[85 * mm, 85 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    return [Spacer(1, 4), t, Spacer(1, 2)]


def _summary_flowables(body: str, styles, chart_data: dict | None, accent) -> list:
    """Render `## Summary` — verdict paragraph(s), then KPI cards, then the
    bar + doughnut charts. Falls back to plain body if the shape is off."""
    verdict_lines, kpi_rows = [], []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^[-*]\s+(.+)$", s)
        if m and "|" in m.group(1):
            parts = [p.strip() for p in m.group(1).split("|")]
            if len(parts) >= 3:
                status = parts[3].lower() if len(parts) > 3 else "normal"
                kpi_rows.append((parts[0], parts[1], parts[2], status))
                continue
        verdict_lines.append(s)

    if not kpi_rows:
        return _body_flowables(body, styles)

    out = []
    for vl in verdict_lines:
        out.append(Paragraph(_inline_bold(vl), styles["Body"]))
        out.append(Spacer(1, 3))
    out.append(Spacer(1, 4))
    out.extend(_kpi_cards(kpi_rows, styles))
    out.extend(_charts_flowable(chart_data or {}, accent))
    return out


def generate_report_pdf(report_text: str, domain: str, generated_at: datetime | None = None,
                        source: str | None = None) -> bytes:
    """Render `report_text` (as produced by report_agent) into PDF bytes."""
    generated_at = generated_at or datetime.now(timezone.utc)
    accent = DOMAIN_ACCENT.get(domain, colors.HexColor("#334155"))
    report_title = DOMAIN_TITLE.get(domain, f"{domain.capitalize()} Report")

    report_text, sfmdata = _extract_sfmdata(report_text)
    chart_data = ((sfmdata or {}).get("domains") or {}).get(domain)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
        title=f"SFMConnect — {report_title}",
    )

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("ReportTitle", parent=base["Title"], textColor=accent, fontSize=20, spaceAfter=2),
        "Subtitle": ParagraphStyle("ReportSubtitle", parent=base["Normal"], textColor=colors.HexColor("#64748b"), fontSize=9, spaceAfter=8),
        "Meta": ParagraphStyle("ReportMeta", parent=base["Normal"], textColor=colors.HexColor("#94a3b8"), fontSize=8.5),
        "Heading": ParagraphStyle("SectionHeading", parent=base["Heading2"], textColor=accent, fontSize=13, spaceBefore=14, spaceAfter=6),
        "Body": ParagraphStyle("Body", parent=base["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#1e293b")),
        # A4 (210mm) minus the 20mm left + 20mm right margins.
        "_content_width": 170 * mm,
    }

    source_label = SOURCE_LABEL.get(source or "", "").strip()
    subtitle = "SFMConnect" + (f"  ·  {source_label}" if source_label else "")

    story = [
        Paragraph(report_title, styles["Title"]),
        Paragraph(subtitle, styles["Subtitle"]),
        HRFlowable(width="100%", color=colors.HexColor("#e2e8f0"), thickness=1),
        Spacer(1, 4),
        Paragraph(f"Generated {generated_at.strftime('%d %b %Y, %H:%M UTC')}", styles["Meta"]),
        Spacer(1, 10),
    ]

    for title, body in _split_sections(report_text):
        story.append(Paragraph(title, styles["Heading"]))
        if title.strip().lower() == "summary":
            story.extend(_summary_flowables(body, styles, chart_data, accent))
        else:
            story.extend(_body_flowables(body, styles))

    doc.build(story)
    return buf.getvalue()
