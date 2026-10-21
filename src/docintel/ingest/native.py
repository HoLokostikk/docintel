from __future__ import annotations

import hashlib
import time
from pathlib import Path
from statistics import median

import fitz

from docintel.ingest.models import (
    BBox,
    Block,
    BlockType,
    Cell,
    Document,
    Page,
    ParseQuality,
    Table,
)
from docintel.ingest.quality import LARGE_IMAGE_AREA_RATIO, char_density, is_scanned

MARGIN_RATIO = 0.07


HEADING_SIZE_RATIO = 1.15


class NativeParser:
    name = "native/pymupdf"

    def parse(self, path: Path) -> Document:
        started = time.perf_counter()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        pages: list[Page] = []
        warnings: list[str] = []

        with fitz.open(path) as pdf:
            for page_no, fitz_page in enumerate(pdf, start=1):
                pages.append(self._parse_page(fitz_page, page_no, warnings))

        total_chars = sum(page.char_count for page in pages)
        total_area = sum(page.width * page.height for page in pages)

        return Document(
            source_path=str(path),
            content_hash=content_hash,
            parser=self.name,
            parse_seconds=time.perf_counter() - started,
            pages=pages,
            quality=ParseQuality(
                char_density=char_density(total_chars, total_area),
                scanned_pages=sum(1 for page in pages if page.is_scanned),
                warnings=warnings,
            ),
        )

    def _parse_page(self, fitz_page: fitz.Page, page_no: int, warnings: list[str]) -> Page:
        rect = fitz_page.rect
        tables = self._extract_tables(fitz_page, page_no)

        raw_blocks: list[tuple[str, tuple[float, float, float, float], float]] = []
        char_count = 0

        for raw in fitz_page.get_text("dict")["blocks"]:
            if raw.get("type") != 0:  # 0 is a text block, 1 is an image
                continue

            spans = [span for line in raw["lines"] for span in line["spans"]]
            line_texts = [
                "".join(span["text"] for span in line["spans"]).strip() for line in raw["lines"]
            ]
            text = " ".join(part for part in line_texts if part)
            if not text:
                continue

            char_count += len(text)
            x0, y0, x1, y1 = raw["bbox"]
            font_size = max((float(span["size"]) for span in spans), default=0.0)
            raw_blocks.append((text, (float(x0), float(y0), float(x1), float(y1)), font_size))

        body_size = median([size for _, _, size in raw_blocks]) if raw_blocks else 0.0

        blocks = [
            Block(
                page_no=page_no,
                type=self._classify_block(bbox, font_size, body_size, rect.height, tables),
                text=text,
                bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
            )
            for text, bbox, font_size in raw_blocks
        ]

        scanned = is_scanned(char_count, self._has_large_image(fitz_page))
        if scanned:
            warnings.append(f"page {page_no}: scanned, text layer is empty")

        return Page(
            number=page_no,
            width=rect.width,
            height=rect.height,
            char_count=char_count,
            is_scanned=scanned,
            blocks=blocks,
            tables=tables,
        )

    def _classify_block(
        self,
        bbox: tuple[float, float, float, float],
        font_size: float,
        body_size: float,
        page_height: float,
        tables: list[Table],
    ) -> BlockType:
        """Assign a coarse role to a text block.

        Order matters: a block that sits inside a detected table is table
        content regardless of where it is on the page.
        """
        if any(self._center_inside(bbox, table.bbox) for table in tables):
            return BlockType.TABLE

        _, y0, _, y1 = bbox
        if y1 < page_height * MARGIN_RATIO:
            return BlockType.HEADER
        if y0 > page_height * (1.0 - MARGIN_RATIO):
            return BlockType.FOOTER

        if body_size > 0.0 and font_size >= body_size * HEADING_SIZE_RATIO:
            return BlockType.HEADING

        return BlockType.TEXT

    @staticmethod
    def _center_inside(bbox: tuple[float, float, float, float], area: BBox) -> bool:
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        return area.x0 <= center_x <= area.x1 and area.y0 <= center_y <= area.y1

    def _extract_tables(self, fitz_page: fitz.Page, page_no: int) -> list[Table]:
        tables: list[Table] = []

        for found in fitz_page.find_tables().tables:
            rows = found.extract()
            if not rows:
                continue

            cells = [
                Cell(row=row_idx, col=col_idx, text=(value or "").strip())
                for row_idx, row in enumerate(rows)
                for col_idx, value in enumerate(row)
            ]

            x0, y0, x1, y1 = found.bbox
            tables.append(
                Table(
                    page_no=page_no,
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    n_rows=len(rows),
                    n_cols=max(len(row) for row in rows),
                    cells=cells,
                )
            )

        return tables

    def _has_large_image(self, fitz_page: fitz.Page) -> bool:
        page_area = fitz_page.rect.width * fitz_page.rect.height
        if page_area <= 0:
            return False
        for image in fitz_page.get_images(full=True):
            xref = image[0]
            for image_rect in fitz_page.get_image_rects(xref):
                if image_rect.width * image_rect.height > LARGE_IMAGE_AREA_RATIO * page_area:
                    return True
        return False
