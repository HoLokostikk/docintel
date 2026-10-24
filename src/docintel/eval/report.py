from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from docintel.eval.normalize import normalize_cell
from docintel.eval.tables import (
    TableScore,
    find_match,
    load_gold_grid,
    load_index,
    score_table,
)
from docintel.ingest.base import Parser
from docintel.ingest.cache import parse_cached
from docintel.ingest.golden import table_to_rows
from docintel.ingest.models import Document
from docintel.ingest.native import NativeParser

SAMPLES_DIR = Path("data/samples")

PARSER_NAMES = ("native", "docling")


def build_parser(name: str, page_range: tuple[int, int] | None) -> Parser:
    if name == "native":
        return NativeParser()
    if name == "docling":
        from docintel.ingest.docling_parser import DoclingParser

        return DoclingParser(page_range=page_range)
    raise ValueError(f"unknown parser: {name}")


def run(parser: Parser) -> list[TableScore]:
    entries = load_index()
    documents: dict[str, Document] = {}
    scores: list[TableScore] = []

    for entry in entries:
        source = SAMPLES_DIR / entry.source_file
        if not source.exists():
            print(f"  missing source file: {entry.source_file}")
            continue
        if entry.source_file not in documents:
            documents[entry.source_file] = parse_cached(parser, source)
        scores.append(score_table(entry, documents[entry.source_file]))

    return scores


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _mean_or_na(values: Sequence[float]) -> str:
    return "n/a" if not values else f"{mean(values):.2f}"


def print_rows(scores: Sequence[TableScore]) -> None:
    header = (
        f"{'gold_id':26} {'kind':10} {'gold':>7} {'pred':>7} "
        f"{'struct':>7} {'cell':>6} {'numeric':>8} {'recall':>7}"
    )
    print(header)
    print("-" * len(header))

    for score in scores:
        if score.comparison is None:
            print(f"{score.gold_id:26} {score.kind:10} {'':>7} {'no match':>7}")
            continue

        comparison = score.comparison
        gold_shape = f"{comparison.gold_shape[0]}x{comparison.gold_shape[1]}"
        pred_shape = f"{comparison.pred_shape[0]}x{comparison.pred_shape[1]}"
        print(
            f"{score.gold_id:26} {score.kind:10} {gold_shape:>7} {pred_shape:>7} "
            f"{'yes' if comparison.structure_match else 'no':>7} "
            f"{_optional(comparison.cell_accuracy):>6} "
            f"{_optional(comparison.numeric_accuracy):>8} "
            f"{comparison.value_recall:>7.2f}"
        )


def summarise(scores: Sequence[TableScore], label: str) -> None:
    if not scores:
        return

    structure = [
        1.0 if score.comparison and score.comparison.structure_match else 0.0 for score in scores
    ]
    cells = [
        score.comparison.cell_accuracy
        for score in scores
        if score.comparison is not None and score.comparison.cell_accuracy is not None
    ]
    numerics = [
        score.comparison.numeric_accuracy
        for score in scores
        if score.comparison is not None and score.comparison.numeric_accuracy is not None
    ]
    recalls = [
        score.comparison.value_recall if score.comparison is not None else 0.0 for score in scores
    ]

    print(f"\n{label} — {len(scores)} tables")
    print(f"  structure_match   {mean(structure):.2f}")
    print(
        f"  cell_accuracy     {_mean_or_na(cells):>4}   (on {len(cells)} with matching structure)"
    )
    print(f"  numeric_accuracy  {_mean_or_na(numerics):>4}   (on {len(numerics)})")
    print(f"  value_recall      {mean(recalls):.2f}")


def print_diff(parser: Parser, gold_id: str) -> None:
    entry = next((e for e in load_index() if e.gold_id == gold_id), None)
    if entry is None:
        print(f"unknown gold_id: {gold_id}")
        return

    document = parse_cached(parser, SAMPLES_DIR / entry.source_file)
    match = find_match(entry, document)
    if match is None:
        print(f"{gold_id}: no predicted table overlaps the gold box")
        return

    gold = load_gold_grid(gold_id)
    pred = table_to_rows(match)
    print(f"{gold_id}: gold {len(gold)} rows, pred {len(pred)} rows\n")

    mismatches = 0
    for row_idx, (gold_row, pred_row) in enumerate(zip(gold, pred, strict=False)):
        for col_idx, (gold_cell, pred_cell) in enumerate(zip(gold_row, pred_row, strict=False)):
            if normalize_cell(gold_cell) == normalize_cell(pred_cell):
                continue
            mismatches += 1
            print(f"  r{row_idx:<3} c{col_idx}")
            print(f"      gold {gold_cell!r}")
            print(f"      pred {pred_cell!r}")

    print(f"\n{mismatches} mismatched cells")


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Score a parser against the golden set.")
    arg_parser.add_argument("--diff", metavar="GOLD_ID", help="show per-cell mismatches")
    arg_parser.add_argument(
        "--parser", choices=PARSER_NAMES, default="native", help="which parser to score"
    )
    args = arg_parser.parse_args()

    entries = load_index()
    pages = [entry.page for entry in entries]
    page_range = (min(pages), max(pages)) if pages else None
    parser = build_parser(args.parser, page_range)
    print(f"parser: {parser.name}\n")
    if args.diff:
        print_diff(parser, args.diff)
        return

    scores = run(parser)
    if not scores:
        print("no scored tables — is data/golden/index.csv populated?")
        return

    print_rows(scores)
    summarise(scores, "all")
    summarise([s for s in scores if s.kind == "financial"], "financial only")
