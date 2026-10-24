from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from docintel.eval.normalize import normalize_cell, parse_number
from docintel.ingest.golden import GOLD_DIR, INDEX_PATH, table_to_rows
from docintel.ingest.models import BBox, Document, Table

MIN_IOU = 0.3


@dataclass(frozen=True)
class GoldEntry:
    gold_id: str
    source_file: str
    page: int
    bbox: BBox
    kind: str


@dataclass(frozen=True)
class GridComparison:
    gold_shape: tuple[int, int]
    pred_shape: tuple[int, int]
    structure_match: bool
    cell_accuracy: float | None
    numeric_accuracy: float | None
    value_recall: float


@dataclass(frozen=True)
class TableScore:
    gold_id: str
    kind: str
    matched: bool
    comparison: GridComparison | None


def load_index() -> list[GoldEntry]:
    if not INDEX_PATH.exists():
        return []
    with INDEX_PATH.open(encoding="utf-8", newline="") as handle:
        return [
            GoldEntry(
                gold_id=row["gold_id"],
                source_file=row["source_file"],
                page=int(row["page"]),
                bbox=BBox(
                    x0=float(row["x0"]),
                    y0=float(row["y0"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                ),
                kind=row.get("kind") or "other",
            )
            for row in csv.DictReader(handle)
        ]


def load_gold_grid(gold_id: str) -> list[list[str]]:
    path = Path(GOLD_DIR) / f"{gold_id}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def iou(left: BBox, right: BBox) -> float:
    overlap_x = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    overlap_y = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    intersection = overlap_x * overlap_y
    if intersection <= 0.0:
        return 0.0

    area_left = (left.x1 - left.x0) * (left.y1 - left.y0)
    area_right = (right.x1 - right.x0) * (right.y1 - right.y0)
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


def find_match(entry: GoldEntry, doc: Document) -> Table | None:
    candidates = [table for page in doc.pages if page.number == entry.page for table in page.tables]
    if not candidates:
        return None

    best = max(candidates, key=lambda table: iou(entry.bbox, table.bbox))
    return best if iou(entry.bbox, best.bbox) >= MIN_IOU else None


def _shape(grid: list[list[str]]) -> tuple[int, int]:
    return len(grid), max((len(row) for row in grid), default=0)


def _value_counts(grid: list[list[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in grid:
        for cell in row:
            value = normalize_cell(cell)
            if value:
                counts[value] += 1
    return counts


def compare_grids(gold: list[list[str]], pred: list[list[str]]) -> GridComparison:
    gold_shape = _shape(gold)
    pred_shape = _shape(pred)
    structure_match = gold_shape == pred_shape

    cell_accuracy: float | None = None
    numeric_accuracy: float | None = None

    if structure_match:
        total = matched = numeric_total = numeric_matched = 0
        for gold_row, pred_row in zip(gold, pred, strict=False):
            for gold_cell, pred_cell in zip(gold_row, pred_row, strict=False):
                total += 1
                same = normalize_cell(gold_cell) == normalize_cell(pred_cell)
                matched += int(same)
                if parse_number(gold_cell) is not None:
                    numeric_total += 1
                    numeric_matched += int(same)
        cell_accuracy = matched / total if total else 0.0
        numeric_accuracy = numeric_matched / numeric_total if numeric_total else None

    gold_values = _value_counts(gold)
    pred_values = _value_counts(pred)
    gold_total = sum(gold_values.values())
    overlap = sum((gold_values & pred_values).values())

    return GridComparison(
        gold_shape=gold_shape,
        pred_shape=pred_shape,
        structure_match=structure_match,
        cell_accuracy=cell_accuracy,
        numeric_accuracy=numeric_accuracy,
        value_recall=overlap / gold_total if gold_total else 0.0,
    )


def score_table(entry: GoldEntry, doc: Document) -> TableScore:
    match = find_match(entry, doc)
    if match is None:
        return TableScore(entry.gold_id, entry.kind, matched=False, comparison=None)

    comparison = compare_grids(load_gold_grid(entry.gold_id), table_to_rows(match))
    return TableScore(entry.gold_id, entry.kind, matched=True, comparison=comparison)
