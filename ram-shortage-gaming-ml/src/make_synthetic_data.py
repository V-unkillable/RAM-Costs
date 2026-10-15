"""Create a SYNTHETIC panel for checking that the pipeline runs. Values are invented; never report results from it."""
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
months = pd.period_range("2024-01", "2026-09", freq="M")
cats = {"gpu_midrange": (400, 0.35), "gpu_highend": (1200, 0.30), "ddr5_kit": (110, 0.90), "ddr4_kit": (70, 0.80),
        "console": (500, 0.12), "handheld_pc": (650, 0.25), "prebuilt_pc": (1500, 0.20)}
t = np.arange(len(months))
contract = 100 + np.where(t >= 17, np.maximum(t - 17, 0) ** 1.6 * 3.2, 0) + rng.normal(0, 2, len(t))
spot = contract * 1.05 + rng.normal(0, 4, len(t))
inventory = np.clip(9 - 0.28 * np.maximum(t - 14, 0) + rng.normal(0, 0.4, len(t)), 1.5, 10)
hbm = np.clip(8 + 0.9 * t / (len(t) / 10) + rng.normal(0, 0.5, len(t)), 5, 30)
rows = []
for c, (base, s) in cats.items():
    price_prev, dem = base, 100.0
    for i, m in enumerate(months):
        target = base * (1 + s * (contract[max(i - 2, 0)] - 100) / 100) + rng.normal(0, base * 0.01)
        price = 0.6 * price_prev + 0.4 * target
        change = (price / price_prev - 1) * 100
        search = max(5, 60 - 0.6 * (price / base - 1) * 100 + rng.normal(0, 4))
        dem = max(20, dem * (1 - 0.35 * change / 100 + rng.normal(0, 0.02)))
        rows.append(dict(month=str(m), category=c, retail_price=round(price, 2), msrp=base, price_to_msrp=round(price / base, 3),
                         dram_contract_price=round(contract[i], 2), dram_spot_price=round(spot[i], 2),
                         inventory_weeks=round(inventory[i], 2), hbm_share_pct=round(hbm[i], 2),
                         search_interest=round(search, 1), demand_index=round(dem, 2), tariff_flag=int(i >= 20)))
        price_prev = price
out = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "dram_gaming_panel_SYNTHETIC_TEST.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
