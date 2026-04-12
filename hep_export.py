"""
hep_export.py - Export search results to PDF or Word document.
Called from hep_search.py /export route.
PDF: ReportLab (pure Python, no system dependencies)
Word: python-docx
"""

import io
import re
from datetime import datetime


def _page_ref(c: dict) -> str:
    if c.get("page_start") and c.get("page_end"):
        return f"pp. {c['page_start']}–{c['page_end']}"
    if c.get("page_start"):
        return f"p. {c['page_start']}"
    return ""


def _paragraphs(answer: str) -> list[str]:
    paras = []
    for block in answer.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r'^#{1,3}\s+', '', block)
        block = re.sub(r'\*\*(.+?)\*\*', r'\1', block)
        block = re.sub(r'\*(.+?)\*', r'\1', block)
        paras.append(block)
    return paras


def _strip_citations(text: str) -> str:
    return re.sub(r'\[\d+\]', '', text).strip()


def export_pdf(query: str, answer: str, citations: list, contradictions: list) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf = io.BytesIO()

    DEEP_PURPLE  = colors.HexColor("#280F2E")
    MID_PURPLE   = colors.HexColor("#8B388D")
    LIGHT_PURPLE = colors.HexColor("#C5A8C4")
    GOLD         = colors.HexColor("#C89040")
    AMBER        = colors.HexColor("#B06020")
    DARK_GREY    = colors.HexColor("#333333")
    MID_GREY     = colors.HexColor("#666666")
    LIGHT_GREY   = colors.HexColor("#999999")
    BLACK        = colors.HexColor("#1A1A1A")

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title="HEP Research Library Export",
        author="Hope Education Project",
    )

    def style(name, **kwargs):
        defaults = dict(fontName="Helvetica", fontSize=10, leading=15,
                        textColor=BLACK, spaceAfter=6)
        defaults.update(kwargs)
        return ParagraphStyle(name, **defaults)

    s_org    = style("org",   fontSize=7.5, textColor=MID_PURPLE,
                     fontName="Helvetica-Bold", spaceAfter=2, leading=10)
    s_date   = style("date",  fontSize=7.5, textColor=MID_GREY, spaceAfter=10, leading=10)
    s_label  = style("label", fontSize=7.5, textColor=MID_PURPLE,
                     fontName="Helvetica-Bold", spaceAfter=4, leading=10)
    s_query  = style("query", fontSize=13,  textColor=BLACK,
                     fontName="Helvetica-Bold", spaceAfter=16, leading=18)
    s_body   = style("body",  fontSize=10,  textColor=BLACK, spaceAfter=8, leading=15)
    s_src_title = style("src_title", fontSize=9.5, textColor=BLACK,
                        fontName="Helvetica-Bold", spaceAfter=2, leading=13)
    s_src_meta  = style("src_meta",  fontSize=8.5, textColor=MID_GREY,
                        spaceAfter=2, leading=12)
    s_excerpt   = style("excerpt",   fontSize=8.5, textColor=MID_GREY,
                        fontName="Helvetica-Oblique", spaceAfter=6, leading=12, leftIndent=10)
    s_contra_label = style("clabel", fontSize=7.5, textColor=AMBER,
                           fontName="Helvetica-Bold", spaceAfter=4, leading=10)
    s_contra_topic = style("ctopic", fontSize=9.5, textColor=AMBER,
                           fontName="Helvetica-Bold", spaceAfter=3, leading=13)
    s_contra_side  = style("cside",  fontSize=9,   textColor=DARK_GREY,
                           spaceAfter=3, leading=13, leftIndent=12)
    s_footer = style("footer", fontSize=7.5, textColor=LIGHT_GREY,
                     alignment=TA_CENTER, leading=10)

    date_str = datetime.now().strftime("%-d %B %Y")
    story = []

    story.append(Paragraph("HOPE EDUCATION PROJECT  \u00b7  RESEARCH LIBRARY", s_org))
    story.append(Paragraph(f"Research Export  \u00b7  {date_str}", s_date))
    story.append(HRFlowable(width="100%", thickness=2, color=MID_PURPLE, spaceAfter=14))

    story.append(Paragraph("RESEARCH QUESTION", s_label))
    story.append(Paragraph(query, s_query))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_PURPLE, spaceAfter=12))

    story.append(Paragraph("ANSWER", s_label))
    for para_text in _paragraphs(answer):
        story.append(Paragraph(_strip_citations(para_text), s_body))
    story.append(Spacer(1, 8))

    if contradictions:
        story.append(HRFlowable(width="100%", thickness=0.5, color=AMBER, spaceAfter=8))
        story.append(Paragraph("CONFLICTING EVIDENCE DETECTED", s_contra_label))
        for ct in contradictions:
            story.append(Paragraph(ct.get("topic", ""), s_contra_topic))
            for side in ct.get("sides", []):
                story.append(Paragraph(
                    f"[{side['index']}]  {side.get('claim', '')}", s_contra_side))
        story.append(Spacer(1, 8))

    story.append(HRFlowable(width="100%", thickness=1.5, color=LIGHT_PURPLE, spaceAfter=10))
    story.append(Paragraph(f"SOURCES  \u00b7  {len(citations)} DOCUMENTS", s_label))

    for c in citations:
        strength  = c.get("evidence_strength", 1)
        badge_txt = f"  [{strength} source{'s' if strength != 1 else ''}]"
        story.append(Paragraph(
            f"<font color='#8B388D'>[{c['index']}]</font>  {c.get('title','Unknown')}{badge_txt}",
            s_src_title))
        meta_parts = [p for p in [
            c.get("authors") or "", c.get("year") or "",
            _page_ref(c), c.get("category") or "",
        ] if p]
        if meta_parts:
            story.append(Paragraph("  \u00b7  ".join(meta_parts), s_src_meta))
        excerpt = (c.get("excerpt") or "")[:250]
        if excerpt:
            story.append(Paragraph(
                excerpt + ("\u2026" if len(c.get("excerpt","")) > 250 else ""),
                s_excerpt))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_PURPLE, spaceAfter=6))
    story.append(Paragraph(
        f"Hope Education Project  \u00b7  hopeeducationproject.org  \u00b7  {date_str}",
        s_footer))

    doc.build(story)
    return buf.getvalue()


def export_docx(query: str, answer: str, citations: list, contradictions: list) -> bytes:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    HEP_PURPLE = RGBColor(0x8B, 0x38, 0x8D)
    MID_GREY   = RGBColor(0x66, 0x66, 0x66)
    AMBER      = RGBColor(0xC8, 0x78, 0x30)
    BLACK      = RGBColor(0x1A, 0x1A, 0x1A)

    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(3.0)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    def _add_rule(colour_hex="8B388D", thickness=18):
        p   = doc.paragraphs[-1]._element
        pPr = p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), str(thickness))
        bottom.set(qn("w:space"), "4")
        bottom.set(qn("w:color"), colour_hex)
        pBdr.append(bottom)
        pPr.append(pBdr)

    def _add_label(text, colour=None):
        if colour is None:
            colour = HEP_PURPLE
        p = doc.add_paragraph()
        r = p.add_run(text.upper())
        r.font.size = Pt(7.5); r.font.bold = True
        r.font.color.rgb = colour; r.font.name = "Calibri"
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after  = Pt(4)

    p = doc.add_paragraph()
    r = p.add_run("HOPE EDUCATION PROJECT  \u00b7  RESEARCH LIBRARY")
    r.font.size = Pt(8); r.font.bold = True
    r.font.color.rgb = HEP_PURPLE; r.font.name = "Calibri"
    p.paragraph_format.space_after = Pt(2)

    p2 = doc.add_paragraph()
    r2 = p2.add_run(f"Research Export  \u00b7  {datetime.now().strftime('%-d %B %Y')}")
    r2.font.size = Pt(8); r2.font.color.rgb = MID_GREY; r2.font.name = "Calibri"
    p2.paragraph_format.space_after = Pt(6)
    _add_rule("8B388D", 24)

    _add_label("Research Question")
    q = doc.add_paragraph()
    qr = q.add_run(query)
    qr.font.size = Pt(13); qr.font.bold = True
    qr.font.color.rgb = BLACK; qr.font.name = "Calibri"
    q.paragraph_format.space_after = Pt(16)
    _add_rule("D0C0D0", 6)

    _add_label("Answer")
    for para_text in _paragraphs(answer):
        p = doc.add_paragraph()
        r = p.add_run(_strip_citations(para_text))
        r.font.size = Pt(10.5); r.font.color.rgb = BLACK; r.font.name = "Calibri"
        p.paragraph_format.space_after = Pt(8)

    if contradictions:
        _add_label("Conflicting Evidence Detected", AMBER)
        for ct in contradictions:
            tp = doc.add_paragraph()
            tr = tp.add_run(ct.get("topic", ""))
            tr.font.bold = True; tr.font.size = Pt(9.5)
            tr.font.color.rgb = AMBER; tr.font.name = "Calibri"
            tp.paragraph_format.space_after = Pt(2)
            for side in ct.get("sides", []):
                sp = doc.add_paragraph(style="List Bullet")
                nr = sp.add_run(f"[{side['index']}] ")
                nr.font.bold = True; nr.font.color.rgb = HEP_PURPLE
                nr.font.name = "Calibri"; nr.font.size = Pt(9.5)
                cr = sp.add_run(side.get("claim", ""))
                cr.font.size = Pt(9.5); cr.font.color.rgb = BLACK; cr.font.name = "Calibri"
                sp.paragraph_format.space_after = Pt(2)

    _add_label("Sources", MID_GREY)
    _add_rule("D0C0D0", 12)

    for c in citations:
        p = doc.add_paragraph()
        p.paragraph_format.space_after  = Pt(2)
        p.paragraph_format.space_before = Pt(10)
        nr = p.add_run(f"[{c['index']}]  ")
        nr.font.bold = True; nr.font.color.rgb = HEP_PURPLE
        nr.font.size = Pt(9); nr.font.name = "Calibri"
        tr = p.add_run(c.get("title", "Unknown"))
        tr.font.bold = True; tr.font.size = Pt(9.5)
        tr.font.color.rgb = BLACK; tr.font.name = "Calibri"
        strength = c.get("evidence_strength", 1)
        er = p.add_run(f"  [{strength} source{'s' if strength != 1 else ''}]")
        er.font.size = Pt(7.5); er.font.color.rgb = HEP_PURPLE; er.font.name = "Calibri"

        meta_parts = [pt for pt in [
            c.get("authors") or "", c.get("year") or "",
            _page_ref(c), c.get("category") or "",
        ] if pt]
        if meta_parts:
            mp = doc.add_paragraph()
            mr = mp.add_run("  \u00b7  ".join(meta_parts))
            mr.font.size = Pt(8.5); mr.font.color.rgb = MID_GREY; mr.font.name = "Calibri"
            mp.paragraph_format.space_before = Pt(0)
            mp.paragraph_format.space_after  = Pt(2)

        excerpt = (c.get("excerpt") or "")[:250]
        if excerpt:
            ep = doc.add_paragraph()
            er2 = ep.add_run(excerpt + ("\u2026" if len(c.get("excerpt","")) > 250 else ""))
            er2.font.size = Pt(8.5); er2.font.italic = True
            er2.font.color.rgb = MID_GREY; er2.font.name = "Calibri"
            ep.paragraph_format.space_before = Pt(0)
            ep.paragraph_format.space_after  = Pt(2)
            ep.paragraph_format.left_indent  = Inches(0.2)

    _add_rule("D0C0D0", 6)
    fp = doc.add_paragraph()
    fr = fp.add_run(
        f"Hope Education Project  \u00b7  hopeeducationproject.org  \u00b7  {datetime.now().strftime('%-d %B %Y')}"
    )
    fr.font.size = Pt(7.5); fr.font.color.rgb = MID_GREY; fr.font.name = "Calibri"
    fp.paragraph_format.space_before = Pt(8)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
