from __future__ import annotations

import argparse
import random
from decimal import Decimal

from docintel.eval.normalize import parse_number
from docintel.eval.oracle import check_table
from docintel.eval.tables import load_gold_grid, load_index

MUTATIONS = ("scale", "digits", "sign", "offby")


def numeric_cells(grid: list[list[str]]) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row, line in enumerate(grid)
        for col, cell in enumerate(line)
        if parse_number(cell) is not None
    ]


def mutate(grid: list[list[str]], kind: str, cell: tuple[int, int]) -> list[list[str]]:
    row, col = cell
    mutated = [line[:] for line in grid]
    value = parse_number(grid[row][col])
    if value is None:
        return mutated

    if kind == "scale":
        new = value * 10
    elif kind == "digits":
        digits = str(abs(value)).replace(".", "")[::-1].lstrip("0") or "1"
        new = Decimal(digits)
    elif kind == "sign":
        new = -value
    else:
        new = value + 1

    mutated[row][col] = str(new)
    return mutated


def describe_tables() -> None:
    header = (
        f"{'gold_id':26} {'kind':10} {'cols':>5} {'relations':>10} "
        f"{'coverage':>9} {'violations':>11}"
    )
    print(header)
    print("-" * len(header))

    for entry in load_index():
        report = check_table(load_gold_grid(entry.gold_id))
        print(
            f"{entry.gold_id:26} {entry.kind:10} "
            f"{len(report.numeric_columns):>5} {len(report.relations):>10} "
            f"{report.coverage:>9.2f} {len(report.violations):>11}"
        )


def measure(trials: int, seed: int) -> None:
    rng = random.Random(seed)
    grids = [(entry.gold_id, load_gold_grid(entry.gold_id)) for entry in load_index()]

    false_positives = sum(1 for _, grid in grids if not check_table(grid).consistent)
    print(f"\nfalse positives on clean tables: {false_positives}/{len(grids)}")

    print(f"\ndetection over {trials} trials per type per table")
    for kind in MUTATIONS:
        attempted = detected = 0
        for _, grid in grids:
            cells = numeric_cells(grid)
            if not cells:
                continue
            for _ in range(trials):
                cell = rng.choice(cells)
                attempted += 1
                if not check_table(mutate(grid, kind, cell)).consistent:
                    detected += 1
        rate = detected / attempted if attempted else 0.0
        print(f"  {kind:8} {rate:.2f}   ({detected}/{attempted})")

    covered = [check_table(grid).coverage for _, grid in grids]
    mean_coverage = sum(covered) / len(covered) if covered else 0.0
    print(f"\nmean coverage across tables: {mean_coverage:.2f}")
    print("detection is bounded by coverage: a cell in no relation cannot be checked")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arithmetic oracle over the golden set.")
    parser.add_argument("--trials", type=int, default=25, help="mutations per type per table")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    describe_tables()
    measure(args.trials, args.seed)
