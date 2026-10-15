"""Framework from the paper: TrendForce index + 15-classifier comparison.
Needs dram_gaming_panel.csv (monthly rows per category) - NOT supplied, because the data could not be collected.
The classifier part refuses to run on fewer than 100 rows."""
import numpy as np, pandas as pd, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"
OUT.mkdir(exist_ok=True)
from trendforce_index import build_index

# ---------- 0. Real public data: TrendForce QoQ contract-price figures (Eq. 1 in the paper) ----------
print(build_index(DATA / "trendforce_dram_public_figures.csv")[["quarter", "mid_pct", "index_4Q25_100", "figure_type"]].to_string(index=False))

# ---------- 1. Load panel, guard, label, lag ----------
df = pd.read_csv(DATA / "dram_gaming_panel.csv", parse_dates=["month"])
df = df.sort_values(["category", "month"]).reset_index(drop=True)
if len(df) < 100:   # small-sample guard
    raise SystemExit(f"Only {len(df)} rows in the panel; need monthly data for several categories before ranking models.")

df["price_change_next"] = df.groupby("category")["retail_price"].transform(lambda s: s.pct_change().shift(-1) * 100)
df["demand_change_next"] = df.groupby("category")["demand_index"].transform(lambda s: s.pct_change().shift(-1) * 100)
df["price_band"] = pd.cut(df["price_change_next"], [-np.inf, 2, 8, np.inf], labels=[0, 1, 2])
df["demand_band"] = pd.cut(df["demand_change_next"], [-np.inf, -5, 5, np.inf], labels=[0, 1, 2])

raw_targets = ["price_change_next", "demand_change_next", "price_band", "demand_band"]
base_cols = [c for c in df.columns if c not in ["month", "category"] + raw_targets]
for c in base_cols:
    for L in (1, 3, 6):
        df[f"{c}_lag{L}"] = df.groupby("category")[c].shift(L)
df = pd.concat([df, pd.get_dummies(df["category"], prefix="cat", dtype=float)], axis=1)

# ---------- 2. Models ----------
MODELS = {
    "AdaBoost": lambda: AdaBoostClassifier(random_state=SEED),
    "CatBoost": lambda: CatBoostClassifier(verbose=0, random_seed=SEED),
    "KNN": lambda: KNeighborsClassifier(),
    "LDA": lambda: LinearDiscriminantAnalysis(),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LightGBM": lambda: LGBMClassifier(random_state=SEED, verbose=-1, class_weight="balanced"),
    "Linear SVM": lambda: LinearSVC(class_weight="balanced", random_state=SEED),
    "MLP": lambda: MLPClassifier(max_iter=1000, random_state=SEED),
    "Naive Bayes": lambda: GaussianNB(),
    "QDA": lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
    "XGBoost": lambda: XGBClassifier(random_state=SEED, eval_metric="mlogloss"),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=SEED),
    "Kernel SVM": lambda: SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(random_state=SEED, class_weight="balanced"),
}

def fit_predict(make, Xtr, ytr, Xte):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", make())])
    pipe.fit(Xtr, ytr)          # imputer and scaler fitted on training split only
    return pipe.predict(Xte)

# ---------- 3. Run both targets ----------
results = {}
for target in ["price_band", "demand_band"]:
    d = df.dropna(subset=[target]).sort_values("month").reset_index(drop=True)
    y = d[target].astype(int)
    X = d.drop(columns=["month", "category"] + raw_targets)
    rows = []
    for name, make in MODELS.items():
        row = {"Algorithm": name}
        for test_size in (0.2, 0.3, 0.4):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
            row[f"Acc {int((1-test_size)*100)}:{int(test_size*100)}"] = round(
                accuracy_score(yte, fit_predict(make, Xtr, ytr, Xte)) * 100, 2)
        months = np.sort(d["month"].unique())
        cut = months[int(len(months) * 0.8)]            # chronological hold-out: last 20% of months
        tr, te = d["month"] < cut, d["month"] >= cut
        pred = fit_predict(make, X[tr], y[tr], X[te])
        row["Macro-F1 chrono"] = round(f1_score(y[te], pred, average="macro") * 100, 2)
        row["Severe-class recall chrono"] = round(
            recall_score(y[te], pred, labels=[2], average=None, zero_division=0)[0] * 100, 2)
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("Macro-F1 chrono", ascending=False)
    results[target] = res
    res.to_csv(OUT / f"results_{target}.csv", index=False)
    print(f"\n=== {target} ===")
    print(res.to_markdown(index=False))

# ---------- 4. Charts for the report (Figs. 6-8) ----------
plt.rcParams.update({"font.family": "serif", "font.size": 7})
algos = list(MODELS)
acc_cols = ["Acc 80:20", "Acc 70:30", "Acc 60:40"]
GREY, HATCH = ["#333333", "#bbbbbb", "#eeeeee", "#888888"], ["", "//", "..", "xx"]

def grouped(ax, res, cols, labels):
    r = res.set_index("Algorithm").loc[algos]
    x, w = np.arange(len(algos)), 0.8 / len(cols)
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, r[c], w, label=lab, color=GREY[i],
               hatch=HATCH[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(algos, rotation=60, ha="right"); ax.set_ylim(0, 105)

# Fig. 6: accuracy of each classifier at the three random splits (price band)
fig, ax = plt.subplots(figsize=(4.8, 2.6))
grouped(ax, results["price_band"], acc_cols, ["80:20", "70:30", "60:40"])
ax.set_ylabel("Accuracy (%)"); ax.legend(ncol=3, fontsize=6); fig.tight_layout()
fig.savefig(OUT / "fig_per_split_price_band.png", dpi=300); plt.close(fig)

# Fig. 7: final chart, all executions (both targets; 3 random splits + chronological macro-F1)
fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.4), sharex=True)
for ax, t, title in zip(axes, ["price_band", "demand_band"], ["Price band", "Demand band"]):
    grouped(ax, results[t], acc_cols + ["Macro-F1 chrono"], ["Acc 80:20", "Acc 70:30", "Acc 60:40", "Macro-F1 (chronological)"])
    ax.set_ylabel(title + " (%)")
axes[0].legend(ncol=4, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout()
fig.savefig(OUT / "fig_final_all_executions.png", dpi=300); plt.close(fig)

# Fig. 8: best result of this work vs the five best reviewed papers (reported accuracies)
papers = [("[1] CatBoost", 99.738), ("[6] BFS-GSRF", 99.0), ("[5] Random Forest", 98.11),
          ("[2] Decision Tree", 98.02), ("[4] Random Forest", 98.0)]
best_acc, best_algo, best_target = max((res[acc_cols].max().max(),
                                        res.loc[res[acc_cols].max(axis=1).idxmax(), "Algorithm"], t)
                                       for t, res in results.items())
print(f"BEST: {best_algo} on {best_target}, accuracy {best_acc:.2f}%")
names = [n for n, _ in papers] + [f"This work: {best_algo}"]
vals = [v for _, v in papers] + [best_acc]
fig, ax = plt.subplots(figsize=(4.8, 2.4))
ax.barh(range(len(vals)), vals[::-1], color=["#333333"] + ["#cccccc"] * 5, edgecolor="black", linewidth=0.4)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(names[::-1]); ax.set_xlim(0, 112)
ax.set_xlabel("Accuracy (%)"); fig.tight_layout(); fig.savefig(OUT / "fig_comparison_best_papers.png", dpi=300)
