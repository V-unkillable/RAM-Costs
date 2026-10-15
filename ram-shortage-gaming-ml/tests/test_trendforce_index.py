import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from trendforce_index import build_index

DATA = Path(__file__).resolve().parent.parent / "data" / "trendforce_dram_public_figures.csv"


def test_quarters_and_order():
    df = build_index(DATA)
    assert list(df["quarter"]) == ["1Q26", "2Q26", "3Q26", "4Q26"]


def test_uses_latest_figure_for_1q26():
    df = build_index(DATA)           # June survey result (93-98) replaces the February forecast (90-95)
    assert df.loc[0, "low_pct_qoq"] == 93 and df.loc[0, "high_pct_qoq"] == 98


def test_chained_index_values():
    df = build_index(DATA)
    assert list(df["index_4Q25_100"]) == [195.5, 313.8, 362.4, 407.7]
    assert abs(df.loc[3, "index_low"] - 379.0) < 0.6 and abs(df.loc[3, "index_high"] - 437.9) < 0.6
