from docintel.eval.oracle import check_table

BERKSHIRE = [
    ["MidAmerican (89.8% owned)", "Earnings (in millions)", ""],
    ["", "2012", "2011"],
    ["U.K. utilities", "$429", "$469"],
    ["Iowa utility", "236", "279"],
    ["Western utilities", "737", "771"],
    ["Pipelines", "383", "388"],
    ["HomeServices", "82", "39"],
    ["Other (net)", "91", "36"],
    ["Operating earnings before interest and taxes", "1958", "1982"],
    ["Interest", "314", "336"],
    ["Income taxes", "172", "315"],
    ["Net earnings", "$1,472", "$1,331"],
    ["Earnings applicable to Berkshire", "$1,323", "$1,204"],
]


def _corrupt(row: int, col: int, value: str) -> list[list[str]]:
    grid = [line[:] for line in BERKSHIRE]
    grid[row][col] = value
    return grid


def test_clean_table_has_no_violations() -> None:
    report = check_table(BERKSHIRE)
    assert report.consistent
    assert report.numeric_columns == (1, 2)


def test_discovers_both_structural_relations() -> None:
    report = check_table(BERKSHIRE)
    kinds = {(relation.kind, relation.start, relation.total) for relation in report.relations}
    assert ("sum", 2, 8) in kinds
    assert ("diff", 8, 11) in kinds


def test_coverage_is_reported() -> None:
    assert 0.8 <= check_table(BERKSHIRE).coverage <= 0.9


def test_detects_a_scaled_value() -> None:
    assert not check_table(_corrupt(4, 1, "7370")).consistent


def test_detects_an_off_by_one_error() -> None:
    assert not check_table(_corrupt(4, 1, "738")).consistent


def test_misses_corruption_that_stays_self_consistent() -> None:

    grid = [line[:] for line in BERKSHIRE]
    grid[2][1] = "529"
    grid[8][1] = "2058"
    grid[11][1] = "$1,572"
    assert check_table(grid).consistent
