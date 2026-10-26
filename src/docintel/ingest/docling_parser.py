from __future__ import annotations

import hashlib
import time
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import TextItem
from docling_core.types.doc.labels import DocItemLabel

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
from docintel.ingest.quality import char_density

LABEL_TO_BLOCK_TYPE = {
    DocItemLabel.PAGE_HEADER: BlockType.HEADER,
    DocItemLabel.PAGE_FOOTER: BlockType.FOOTER,
    DocItemLabel.SECTION_HEADER: BlockType.HEADING,
    DocItemLabel.TITLE: BlockType.HEADING,
    DocItemLabel.TABLE: BlockType.TABLE,
    DocItemLabel.TEXT: BlockType.TEXT,
    DocItemLabel.PARAGRAPH: BlockType.TEXT,
    DocItemLabel.LIST_ITEM: BlockType.TEXT,
    DocItemLabel.CAPTION: BlockType.TEXT,
    DocItemLabel.FOOTNOTE: BlockType.TEXT,
}

FULL_RANGE = (1, 10**9)


def _to_bbox(box: BoundingBox, page_height: float) -> BBox:
    top_left = box.to_top_left_origin(page_height)
    return BBox(x0=top_left.l, y0=top_left.t, x1=top_left.r, y1=top_left.b)


class DoclingParser:
    def __init__(self, page_range: tuple[int, int] | None = None) -> None:
        self._range = page_range or FULL_RANGE
        self._converter = DocumentConverter()

    @property
    def name(self) -> str:
        if self._range == FULL_RANGE:
            return "docling"
        return f"docling[{self._range[0]}-{self._range[1]}]"

    def parse(self, path: Path) -> Document:
        started = time.perf_counter()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        converted: DoclingDocument = self._converter.convert(path, page_range=self._range).document

        heights = {no: item.size.height for no, item in converted.pages.items()}
        widths = {no: item.size.width for no, item in converted.pages.items()}

        blocks: dict[int, list[Block]] = {}
        tables: dict[int, list[Table]] = {}
        chars: dict[int, int] = {}

        for text_item in converted.texts:
            page_no, box = self._placement(text_item, heights)
            if page_no is None or box is None:
                continue
            text = text_item.text.strip()
            if not text:
                continue
            chars[page_no] = chars.get(page_no, 0) + len(text)
            blocks.setdefault(page_no, []).append(
                Block(
                    page_no=page_no,
                    type=LABEL_TO_BLOCK_TYPE.get(text_item.label, BlockType.OTHER),
                    text=text,
                    bbox=box,
                )
            )

        for table_item in converted.tables:
            page_no, box = self._placement(table_item, heights)
            if page_no is None or box is None:
                continue
            tables.setdefault(page_no, []).append(self._table(table_item, page_no, box))

        pages = [
            Page(
                number=no,
                width=widths[no],
                height=heights[no],
                char_count=chars.get(no, 0),
                is_scanned=False,
                blocks=blocks.get(no, []),
                tables=tables.get(no, []),
            )
            for no in sorted(converted.pages)
        ]

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
                scanned_pages=0,
                warnings=[
                    "docling OCRs scans transparently; scan detection stays with the native parser"
                ],
            ),
        )

    def _placement(
        self, item: TextItem | TableItem, heights: dict[int, float]
    ) -> tuple[int | None, BBox | None]:
        if not item.prov:
            return None, None
        provenance = item.prov[0]
        height = heights.get(provenance.page_no)
        if height is None:
            return None, None
        return provenance.page_no, _to_bbox(provenance.bbox, height)

    def _table(self, item: TableItem, page_no: int, box: BBox) -> Table:
        data = item.data
        cells = [
            Cell(
                row=cell.start_row_offset_idx,
                col=cell.start_col_offset_idx,
                row_span=cell.row_span,
                col_span=cell.col_span,
                text=cell.text.strip(),
            )
            for cell in data.table_cells
        ]
        return Table(
            page_no=page_no,
            bbox=box,
            n_rows=data.num_rows,
            n_cols=data.num_cols,
            cells=cells,
        )
