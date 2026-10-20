from __future__ import annotations

import hashlib
import time
from pathlib import Path

import fitz

from docintel.ingest.models import (
    BBox,
    Block,
    BlockType,
    Document,
    Page,
    ParseQuality,
)
from docintel.ingest.quality import LARGE_IMAGE_AREA_RATIO, char_density, is_scanned


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
        blocks: list[Block] = []
        char_count = 0

        for raw in fitz_page.get_text("dict")["blocks"]:
            if raw.get("type") != 0:  # 0 is a text block, 1 is an image
                continue

            text = "".join(span["text"] for line in raw["lines"] for span in line["spans"]).strip()
            if not text:
                continue

            char_count += len(text)
            x0, y0, x1, y1 = raw["bbox"]
            blocks.append(
                Block(
                    page_no=page_no,
                    type=BlockType.TEXT,
                    text=text,
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                )
            )

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
            tables=[],
        )

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
