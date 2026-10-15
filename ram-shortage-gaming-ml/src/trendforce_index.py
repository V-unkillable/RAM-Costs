"""Chain TrendForce quarter-over-quarter DRAM contract-price figures into an index (Eq. 1 in the paper)."""
import pandas as pd


def build_index(path):
    """Return one row per quarter with the chained index (4Q25 = 100) and its low/high paths."""
    tf = pd.read_csv(path)
    conv = (tf[tf["segment"] == "Conventional DRAM"]
            .sort_values("published").groupby("quarter").tail(1).copy())   # latest figure per quarter
    conv["order"] = conv["quarter"].str[-2:] + conv["quarter"].str[0]        # sorts 1Q26 < 2Q26 < ...
    conv = conv.sort_values("order")
    conv["mid_pct"] = (conv["low_pct_qoq"] + conv["high_pct_qoq"]) / 2
    conv["index_4Q25_100"] = (100 * (1 + conv["mid_pct"] / 100).cumprod()).round(1)
    conv["index_low"] = (100 * (1 + conv["low_pct_qoq"] / 100).cumprod()).round(1)
    conv["index_high"] = (100 * (1 + conv["high_pct_qoq"] / 100).cumprod()).round(1)
    return conv[["quarter", "low_pct_qoq", "high_pct_qoq", "mid_pct", "index_4Q25_100",
                 "index_low", "index_high", "figure_type"]].reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path
    print(build_index(Path(__file__).resolve().parent.parent / "data" / "trendforce_dram_public_figures.csv").to_string(index=False))
