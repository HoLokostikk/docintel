from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from docintel.eval.normalize import parse_number

MIN_SUM_TERMS = 2
"""A one-term 'sum' is just equality, which carries no information."""

DISCOVERY_SLACK = Decimal("1")
"""Per-term slack used when *searching* for candidate relations.

Discovery is deliberately permissive: a column with a corrupted value should
still surface the relation, so that judging it strictly can expose the
contradiction. Judging uses exact equality, not this slack.
"""

MIN_NUMERIC_RATIO = 0.5
"""A column counts as numeric when at least this share of its cells parse."""


@dataclass(frozen=True)
class Relation:
    kind: str  # "sum" or "diff"
    start: int
    end: int
    total: int

    def rows(self) -> range:
        return range(self.start, self.total + 1)


@dataclass(frozen=True)
class Violation:
    relation: Relation
    holds_in: tuple[int, ...]
    breaks_in: tuple[int, ...]
    residuals: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class OracleReport:
    relations: tuple[Relation, ...]
    violations: tuple[Violation, ...]
    numeric_columns: tuple[int, ...]
    coverage: float

    @property
    def consistent(self) -> bool:
        return not self.violations


def _matrix(grid: list[list[str]]) -> list[list[Decimal | None]]:
    return [[parse_number(cell) for cell in row] for row in grid]


def _columns(matrix: list[list[Decimal | None]]) -> dict[int, list[Decimal | None]]:
    if not matrix:
        return {}
    width = max(len(row) for row in matrix)
    result: dict[int, list[Decimal | None]] = {}
    for index in range(width):
        column = [row[index] if index < len(row) else None for row in matrix]
        parsed = sum(value is not None for value in column)
        if parsed >= max(2, int(MIN_NUMERIC_RATIO * len(column))):
            result[index] = column
    return result


def _sum(values: list[Decimal]) -> Decimal:
    total = Decimal(0)
    for value in values:
        total += value
    return total


def _unit(values: list[Decimal]) -> Decimal:
    places = 0
    for value in values:
        exponent = value.as_tuple().exponent
        if isinstance(exponent, int) and exponent < 0:
            places = max(places, -exponent)
    return Decimal(1).scaleb(-places)


def _dense(column: list[Decimal | None], start: int, stop: int) -> list[Decimal] | None:
    values: list[Decimal] = []
    for index in range(start, stop):
        value = column[index]
        if value is None:
            return None
        values.append(value)
    return values


def _residual(column: list[Decimal | None], relation: Relation) -> Decimal | None:
    values: list[Decimal] = []
    for index in relation.rows():
        value = column[index]
        if value is None:
            return None
        values.append(value)
    if relation.kind == "sum":
        return _sum(values[:-1]) - values[-1]
    return values[0] - _sum(values[1:-1]) - values[-1]


def _candidates(column: list[Decimal | None]) -> set[Relation]:
    found: set[Relation] = set()

    for total_index in range(len(column)):
        total_value = column[total_index]
        if total_value is None:
            continue

        for start in range(0, total_index - MIN_SUM_TERMS + 1):
            block = _dense(column, start, total_index)
            if block is None or len(block) < MIN_SUM_TERMS:
                continue
            block_sum = _sum(block)
            if block_sum == 0:
                continue
            slack = _unit([*block, total_value]) * DISCOVERY_SLACK * len(block)
            if abs(block_sum - total_value) <= slack:
                found.add(Relation("sum", start, total_index - 1, total_index))

        for start in range(0, total_index - MIN_SUM_TERMS + 1):
            head = column[start]
            block = _dense(column, start + 1, total_index)
            if head is None or block is None or not block:
                continue
            slack = _unit([*block, head, total_value]) * DISCOVERY_SLACK * (len(block) + 1)
            if abs(head - _sum(block) - total_value) <= slack:
                found.add(Relation("diff", start, total_index - 1, total_index))
    return found


def check_table(grid: list[list[str]]) -> OracleReport:
    columns = _columns(_matrix(grid))
    if not columns:
        return OracleReport((), (), (), 0.0)

    candidates: set[Relation] = set()
    for column in columns.values():
        candidates |= _candidates(column)

    violations: list[Violation] = []
    structural: list[Relation] = []

    for relation in sorted(candidates, key=lambda item: (item.total, item.start)):
        residuals = {index: _residual(column, relation) for index, column in columns.items()}
        holds = tuple(i for i, value in residuals.items() if value == 0)
        breaks = tuple(i for i, value in residuals.items() if value is not None and value != 0)

        if holds and breaks:
            violations.append(
                Violation(
                    relation=relation,
                    holds_in=holds,
                    breaks_in=breaks,
                    residuals=tuple((i, str(residuals[i])) for i in breaks),
                )
            )
        if len(holds) >= 2:
            structural.append(relation)

    covered = {
        (row, index)
        for relation in structural
        for index, column in columns.items()
        if _residual(column, relation) is not None
        for row in relation.rows()
    }
    numeric = {
        (row, index)
        for index, column in columns.items()
        for row in range(len(column))
        if column[row] is not None
    }

    return OracleReport(
        relations=tuple(structural),
        violations=tuple(violations),
        numeric_columns=tuple(sorted(columns)),
        coverage=len(covered) / len(numeric) if numeric else 0.0,
    )
