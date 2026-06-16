"""Render REPORT.md to a Chinese-friendly PDF with embedded screenshots."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "REPORT.md"
OUTPUT_PDF = ROOT / "reports" / "REPORT.pdf"

WIDTH = 1240
HEIGHT = 1754
MARGIN = 86

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
MONO_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/CascadiaMono.ttf",
    "C:/Windows/Fonts/cour.ttf",
]


def pick_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE_FONT = pick_font(FONT_CANDIDATES, 34)
H2_FONT = pick_font(FONT_CANDIDATES, 27)
BODY_FONT = pick_font(FONT_CANDIDATES, 21)
CODE_FONT = pick_font(MONO_CANDIDATES, 18)
TABLE_FONT = pick_font(FONT_CANDIDATES, 18)


class PdfCanvas:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.image = Image.new("RGB", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.y = MARGIN

    def new_page(self) -> None:
        self.pages.append(self.image)
        self.image = Image.new("RGB", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.y = MARGIN

    def ensure_space(self, height: int) -> None:
        if self.y + height > HEIGHT - MARGIN:
            self.new_page()

    def text_width(self, text: str, font: ImageFont.ImageFont) -> int:
        bbox = self.draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def wrap(self, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        text = text.strip()
        if not text:
            return [""]
        lines: list[str] = []
        current = ""
        for char in text:
            trial = current + char
            if self.text_width(trial, font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
        return lines

    def paragraph(self, text: str, font: ImageFont.ImageFont = BODY_FONT, indent: int = 0, gap: int = 10) -> None:
        lines = self.wrap(text, font, WIDTH - 2 * MARGIN - indent)
        self.ensure_space(len(lines) * 31 + gap)
        for line in lines:
            self.draw.text((MARGIN + indent, self.y), line, font=font, fill="#1f2933")
            self.y += 31
        self.y += gap

    def code_block(self, lines: list[str]) -> None:
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(self.wrap(line, CODE_FONT, WIDTH - 2 * MARGIN - 34))
        height = len(wrapped) * 27 + 28
        self.ensure_space(height + 12)
        self.draw.rounded_rectangle(
            (MARGIN, self.y, WIDTH - MARGIN, self.y + height),
            radius=8,
            fill="#f4f6f8",
            outline="#d0d7de",
        )
        yy = self.y + 14
        for line in wrapped:
            self.draw.text((MARGIN + 17, yy), line, font=CODE_FONT, fill="#111827")
            yy += 27
        self.y += height + 16

    def table(self, lines: list[str]) -> None:
        rows: list[list[str]] = []
        for line in lines:
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue
            rows.append([cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")])
        if not rows:
            return

        columns = max(len(row) for row in rows)
        column_width = (WIDTH - 2 * MARGIN) / columns
        row_height = 42
        self.ensure_space(row_height * len(rows) + 20)
        for row_index, row in enumerate(rows):
            fill = "#eef2f7" if row_index == 0 else ("#fafafa" if row_index % 2 else "white")
            x = MARGIN
            for column_index in range(columns):
                self.draw.rectangle(
                    (x, self.y, x + column_width, self.y + row_height),
                    fill=fill,
                    outline="#d0d7de",
                )
                cell = row[column_index] if column_index < len(row) else ""
                cell_lines = self.wrap(cell, TABLE_FONT, int(column_width - 18))[:2]
                yy = self.y + 10
                for cell_line in cell_lines:
                    self.draw.text((x + 9, yy), cell_line, font=TABLE_FONT, fill="#1f2933")
                    yy += 21
                x += column_width
            self.y += row_height
        self.y += 18

    def screenshot(self, relative_path: str) -> None:
        path = (ROOT / "reports" / relative_path).resolve()
        if not path.exists():
            return
        screenshot = Image.open(path).convert("RGB")
        max_width = WIDTH - 2 * MARGIN
        scale = min(max_width / screenshot.width, 0.92)
        resized = screenshot.resize(
            (int(screenshot.width * scale), int(screenshot.height * scale)),
            Image.LANCZOS,
        )
        self.ensure_space(resized.height + 24)
        self.draw.rectangle(
            (MARGIN, self.y, MARGIN + resized.width, self.y + resized.height),
            outline="#d0d7de",
        )
        self.image.paste(resized, (MARGIN, self.y))
        self.y += resized.height + 24

    def save(self, path: Path) -> None:
        if self.y > MARGIN or not self.pages:
            self.pages.append(self.image)
        self.pages[0].save(path, save_all=True, append_images=self.pages[1:], resolution=150)


def render() -> None:
    canvas = PdfCanvas()
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            if in_code:
                canvas.code_block(code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not line.strip():
            canvas.y += 10
            if canvas.y > HEIGHT - MARGIN:
                canvas.new_page()
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            canvas.table(table_lines)
            continue
        image_match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line.strip())
        if image_match:
            canvas.screenshot(image_match.group(1))
        elif line.startswith("# "):
            canvas.ensure_space(58)
            canvas.draw.text((MARGIN, canvas.y), line[2:].strip(), font=TITLE_FONT, fill="#0f172a")
            canvas.y += 58
        elif line.startswith("## "):
            canvas.ensure_space(48)
            canvas.draw.text((MARGIN, canvas.y), line[3:].strip(), font=H2_FONT, fill="#0f172a")
            canvas.y += 44
        elif line.startswith("- "):
            canvas.paragraph("• " + line[2:].strip().replace("**", "").replace("`", ""), indent=10, gap=4)
        else:
            canvas.paragraph(line.strip().replace("**", "").replace("`", ""))
        index += 1

    canvas.save(OUTPUT_PDF)
    print(f"Rendered {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    render()
