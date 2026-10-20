from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class BlockType(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Block(BaseModel):
    page_no: int
    type: BlockType
    text: str
    bbox: BBox


class Cell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str


class Table(BaseModel):
    page_no: int
    bbox: BBox
    n_rows: int
    n_cols: int
    cells: list[Cell]


class Page(BaseModel):
    number: int
    width: float
    height: float
    char_count: int
    is_scanned: bool
    blocks: list[Block] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class ParseQuality(BaseModel):
    char_density: float
    scanned_pages: int
    warnings: list[str] = Field(default_factory=list)


class Document(BaseModel):
    source_path: str
    content_hash: str
    parser: str
    parse_seconds: float
    pages: list[Page]
    quality: ParseQuality
