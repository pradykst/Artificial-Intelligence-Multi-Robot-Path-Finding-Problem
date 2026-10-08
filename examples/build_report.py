"""Build the assignment PDF from editable Markdown and committed image assets.

Optional dependency: pip install -r requirements-report.txt
"""
import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font face="ReportSans">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" color="#185a87">\1</a>', text)


def build(source, output):
    # Vera is distributed with ReportLab; no host-specific font is required.
    import reportlab
    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, filename in (("ReportSans", "Vera.ttf"), ("ReportBold", "VeraBd.ttf"), ("ReportItalic", "VeraIt.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportBold", italic="ReportItalic", boldItalic="ReportBold")
    styles = {
        "body": ParagraphStyle("body", fontName="ReportSans", fontSize=9.7, leading=13.1, spaceAfter=6),
        "h1": ParagraphStyle("h1", fontName="ReportBold", fontSize=17, leading=21, spaceAfter=9),
        "h2": ParagraphStyle("h2", fontName="ReportBold", fontSize=12, leading=16, spaceBefore=5, spaceAfter=6, keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName="ReportBold", fontSize=10.2, leading=14, spaceBefore=4, spaceAfter=5, keepWithNext=True),
        "caption": ParagraphStyle("caption", fontName="ReportSans", fontSize=8.5, leading=11.2, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName="ReportSans", fontSize=8.3, leading=10.7),
        "code": ParagraphStyle("code", fontName="ReportSans", fontSize=8, leading=10.5, spaceAfter=4),
    }
    margin = 35
    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=margin, rightMargin=margin,
                          topMargin=margin, bottomMargin=margin,
                          title="Single-Agent and Multi-Agent Pathfinding Using A*, Conflict-Guided Coordination, and Decentralized Negotiation",
                          author="Mohd. Kaif; Pradyumna Kaushal; Ujjwal Sinha")

    def decorate(canvas, document):
        size = landscape(A4) if document.pageTemplate.id == "landscape" else A4
        canvas.setPageSize(size)
        canvas.setFont("ReportSans", 7.5)
        canvas.setFillColor(colors.HexColor("#53616d"))
        canvas.drawString(margin, 19, "CSMI17 Assignment 2 | Single-robot and multi-robot pathfinding")
        canvas.drawRightString(size[0] - margin, 19, str(document.page))

    for name, size in (("portrait", A4), ("landscape", landscape(A4))):
        doc.addPageTemplates(PageTemplate(name, [Frame(margin, margin, size[0]-2*margin, size[1]-2*margin,
                                                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                                         onPage=decorate, pagesize=size))
    width = A4[0] - 2*margin
    image_height = 410
    table_widths = None
    story = []
    lines = source.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("<!-- page:"):
            mode = line.split(":", 1)[1].split("-->")[0].strip()
            if mode not in ("portrait", "landscape"):
                raise ValueError(f"Unknown page template: {mode}")
            story.extend([NextPageTemplate(mode), PageBreak()])
            width = (landscape(A4) if mode == "landscape" else A4)[0] - 2*margin
        elif line.startswith("<!-- image-height:"):
            image_height = float(line.split(":", 1)[1].split("-->")[0].strip())
            if not 0 < image_height <= 700:
                raise ValueError("Image height must be positive and at most 700 points.")
        elif line.startswith("<!-- table-widths:"):
            table_widths = [float(value) for value in line.split(":", 1)[1].split("-->")[0].split(",")]
            if any(value <= 0 for value in table_widths) or abs(sum(table_widths)-100) > .01:
                raise ValueError("Table widths must be positive percentages summing to 100.")
        elif line.startswith("<!--"):
            continue
        elif line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(inline(line[depth:].strip()), styles[f"h{min(depth, 3)}"]))
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(f"Invalid image: {line}")
            asset = (source.parent / match[2]).resolve()
            img = Image(str(asset))
            scale = min(width / img.imageWidth, image_height / img.imageHeight)
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            group = [img, Spacer(1, 5)]
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].startswith("Figure "):
                caption = []
                while i < len(lines) and lines[i].strip():
                    caption.append(lines[i].strip())
                    i += 1
                group.append(Paragraph(inline(" ".join(caption)), styles["caption"]))
            story.append(KeepTogether(group))
            image_height = 410
        elif line.startswith("|"):
            table_lines = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            cells = [row for row in cells if not all(re.fullmatch(r":?-+:?", c) for c in row)]
            n = len(cells[0])
            widths = [width / n] * n
            if n >= 5:
                widths = [width * .21] + [width * .79 / (n-1)] * (n-1)
            if table_widths is not None:
                if len(table_widths) != n:
                    raise ValueError("Table width count must match its columns.")
                widths = [width * value / 100 for value in table_widths]
                table_widths = None
            data = [[Paragraph(inline(c), styles["cell"]) for c in row] for row in cells]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eef3")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fb")]),
                ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#8a9eaf")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1),
                ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ]))
            story.extend([table, Spacer(1, 7)])
        elif line.startswith("```"):
            while i < len(lines) and not lines[i].startswith("```"):
                story.append(Paragraph(escape(lines[i]), styles["code"]))
                i += 1
            i += 1
        else:
            paragraph = [line]
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "![", "<!--", "```")):
                paragraph.append(lines[i].strip())
                i += 1
            text = " ".join(paragraph)
            style = "caption" if text.startswith(("Figure ", "Table ")) else "body"
            story.append(Paragraph(inline(text), styles[style]))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/Assignment_2_Final_Report.pdf")
    args = parser.parse_args()
    build(args.source, args.output)
