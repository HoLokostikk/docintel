from __future__ import annotations

import argparse
import csv
from pathlib import Path

import fitz

from docintel.ingest.cache import parse_cached
from docintel.ingest.models import Document, Table
from docintel.ingest.native import NativeParser

GOLDEN_DIR = Path("data/golden")
PRED_DIR = GOLDEN_DIR / "pred"
GOLD_DIR = GOLDEN_DIR / "gold"
PAGES_DIR = GOLDEN_DIR / "pages"
INDEX_PATH = GOLDEN_DIR / "index.csv"

INDEX_FIELDS = [
    "gold_id",
    "source_file",
    "page",
    "x0",
    "y0",
    "x1",
    "y1",
    "n_rows",
    "n_cols",
    "kind",
]

RENDER_DPI = 120


def table_to_rows(table: Table) -> list[list[str]]:
    grid = [["" for _ in range(table.n_cols)] for _ in range(table.n_rows)]
    for cell in table.cells:
        if 0 <= cell.row < table.n_rows and 0 <= cell.col < table.n_cols:
            grid[cell.row][cell.col] = cell.text
    return grid


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def render_page(source: Path, page_no: int, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as pdf:
        pdf[page_no - 1].get_pixmap(dpi=RENDER_DPI).save(target)


def append_index(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not INDEX_PATH.exists()
    with INDEX_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def export_page(doc: Document, source: Path, page_no: int) -> list[str]:
    page = next((p for p in doc.pages if p.number == page_no), None)
    if page is None:
        raise ValueError(f"{source.name} has no page {page_no}")

    stem = source.stem.replace(" ", "-")
    index_rows: list[dict[str, str]] = []
    created: list[str] = []

    for position, table in enumerate(page.tables, start=1):
        gold_id = f"{stem}_p{page_no:04d}_t{position}"
        write_csv(PRED_DIR / f"{gold_id}.csv", table_to_rows(table))
        index_rows.append(
            {
                "gold_id": gold_id,
                "source_file": source.name,
                "page": str(page_no),
                "x0": f"{table.bbox.x0:.1f}",
                "y0": f"{table.bbox.y0:.1f}",
                "x1": f"{table.bbox.x1:.1f}",
                "y1": f"{table.bbox.y1:.1f}",
                "n_rows": str(table.n_rows),
                "n_cols": str(table.n_cols),
                "kind": "",
            }
        )
        created.append(gold_id)

    render_page(source, page_no, PAGES_DIR / f"{stem}_p{page_no:04d}.png")
    append_index(index_rows)
    return created


def check_golden() -> int:
    """Validate the golden set against the index. Returns the number of problems."""
    if not INDEX_PATH.exists():
        print("index.csv is missing")
        return 1

    with INDEX_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    problems = 0
    for row in rows:
        gold_id = row["gold_id"]
        gold_file = GOLD_DIR / f"{gold_id}.csv"

        if not gold_file.exists():
            print(f"  MISSING   {gold_id}")
            problems += 1
            continue

        with gold_file.open(encoding="utf-8", newline="") as handle:
            grid = list(csv.reader(handle))

        if not grid:
            print(f"  EMPTY     {gold_id}")
            problems += 1
            continue

        widths = {len(line) for line in grid}
        kind = row.get("kind") or ""
        status = "ok"
        if len(widths) > 1:
            status = "RAGGED"
            problems += 1
        elif not kind:
            status = "NO-KIND"
            problems += 1

        print(
            f"  {status:9s} {gold_id:24s} gold {len(grid)}x{max(widths)}"
            f"   pred {row['n_rows']}x{row['n_cols']}   kind={kind or '?'}"
        )

    print(f"\n{len(rows)} entries, {problems} problems")
    return problems


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Export detected tables as golden-set candidates for correction."
    )
    arg_parser.add_argument("source", type=Path, nargs="?", help="path to the PDF")
    arg_parser.add_argument(
        "pages", type=int, nargs="*", help="page numbers; omit to list pages with tables"
    )
    arg_parser.add_argument(
        "--check", action="store_true", help="validate the golden set and exit"
    )
    args = arg_parser.parse_args()

    if args.check:
        raise SystemExit(1 if check_golden() else 0)
    if args.source is None:
        arg_parser.error("source is required unless --check is given")

    doc = parse_cached(NativeParser(), args.source)

    if not args.pages:
        pages_with_tables = [p for p in doc.pages if p.tables]
        print(f"{len(pages_with_tables)} pages carry tables:")
        for page in pages_with_tables:
            shapes = ", ".join(f"{t.n_rows}x{t.n_cols}" for t in page.tables)
            print(f"  page {page.number:4d}: {shapes}")
        return

    for page_no in args.pages:
        ids = export_page(doc, args.source, page_no)
        print(f"page {page_no}: {len(ids)} tables -> {', '.join(ids) or '(none)'}")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
