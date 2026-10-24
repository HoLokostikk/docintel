from docintel.eval.tables import compare_grids, iou
from docintel.ingest.models import BBox

GOLD = [
    ["MidAmerican (89.8% owned)", "Earnings (in millions)", ""],
    ["", "2012", "2011"],
    ["U.K. utilities", "$429", "$469"],
    ["Iowa utility", "236", "279"],
]


def test_identical_grids_score_perfectly() -> None:
    result = compare_grids(GOLD, GOLD)
    assert result.structure_match
    assert result.cell_accuracy == 1.0
    assert result.value_recall == 1.0


def test_equivalent_number_rendering_still_matches() -> None:
    pred = [row[:] for row in GOLD]
    pred[2][2] = "469.00"
    result = compare_grids(GOLD, pred)
    assert result.cell_accuracy == 1.0


def test_split_currency_column_breaks_structure_but_keeps_recall() -> None:
    pred = [
        ["MidAmerican(898%owned)", "Earnings(inmillions)", "", "", ""],
        ["", "", "2012", "", "2011"],
        ["UK utilities", "$", "429", "$", "469"],
        ["Iowa utility", "", "236", "", "279"],
    ]
    result = compare_grids(GOLD, pred)
    assert not result.structure_match
    assert result.cell_accuracy is None
    assert result.value_recall > 0.5


def test_iou_rewards_overlap() -> None:
    box = BBox(x0=0, y0=0, x1=10, y1=10)
    assert iou(box, box) == 1.0
    assert iou(box, BBox(x0=20, y0=20, x1=30, y1=30)) == 0.0
    assert 0.3 < iou(box, BBox(x0=5, y0=0, x1=15, y1=10)) < 0.4
