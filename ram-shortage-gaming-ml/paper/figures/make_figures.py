"""Regenerates the figures used in the paper (run from this folder: python make_figures.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":7.5,"axes.spines.top":False,"axes.spines.right":False,"hatch.linewidth":0.6})
W = 4.6  # inches ~ 11.7 cm
DARK, LIGHT = "#333333", "#d0d0d0"

# Fig 1: base study accuracies (80:20) from Table II of [1]
d = [("LightGBM",99.738),("CatBoost",99.73),("Decision Tree",99.70),("Random Forest",99.639),("XGBoost",99.52),
     ("Gradient Boosting",99.16),("KNN",98.18),("MLP",95.99),("Kernel SVM",95.36),("Linear SVM",91.96),
     ("Logistic Reg.",89.52),("QDA",87.86),("LDA",87.44),("Naive Bayes",86.32),("AdaBoost",82.05)]
fig, ax = plt.subplots(figsize=(W,3.0))
y = np.arange(len(d))[::-1]
for yi,(n,v) in zip(y,d):
    ax.barh(yi, v, color=DARK if v>=99 else LIGHT, edgecolor="black", linewidth=0.4, hatch="" if v>=99 else "//")
    ax.text(v+0.8, yi, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_yticks(y); ax.set_yticklabels([n for n,_ in d]); ax.set_xlim(0,112); ax.set_xlabel("Accuracy (%), 80:20 split")
fig.tight_layout(); fig.savefig("fig1.png", dpi=300); plt.close(fig)

# Fig 2: DRAM index
qoq = [("1Q26",93,98,"survey"),("2Q26",58,63,"forecast"),("3Q26",13,18,"forecast"),("4Q26",10,15,"forecast")]
lo=hi=mid=100; rows=[("4Q25",100,100,100,"base")]
for q,l,h,k in qoq:
    lo*=1+l/100; hi*=1+h/100; mid*=1+(l+h)/200; rows.append((q,lo,hi,mid,k))
fig, ax = plt.subplots(figsize=(W,2.7))
for i,(q,l,h,m,k) in enumerate(rows):
    ax.bar(i, m, width=0.6, color=DARK if k=="survey" else LIGHT, edgecolor="black", linewidth=0.5, hatch="" if k!="forecast" else "//")
    if h>l: ax.errorbar(i, m, yerr=[[m-l],[h-m]], color="black", capsize=3, linewidth=0.8)
    ax.text(i, h+8, f"{m:.0f}", ha="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r[0]}\n({r[4]})" for r in rows])
ax.set_ylabel("Index (4Q25 = 100)"); ax.set_ylim(0,480)
fig.tight_layout(); fig.savefig("fig2.png", dpi=300); plt.close(fig)
print([(r[0],round(r[1]),round(r[2]),round(r[3])) for r in rows])

# Fig 3: console price rises
c = [("PS5 disc (2 Apr)",18.2),("PS5 digital (2 Apr)",20.0),("Xbox Series X disc (1 Aug)",23.1),
     ("Xbox Series X digital (1 Aug)",25.0),("Switch 2 (1 Sep)",11.1)]
fig, ax = plt.subplots(figsize=(W,2.2))
y = np.arange(len(c))[::-1]
for yi,(n,v) in zip(y,c):
    ax.barh(yi, v, color=LIGHT, edgecolor="black", linewidth=0.5, hatch="..")
    ax.text(v+0.5, yi, f"+{v:.1f}%", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([n for n,_ in c]); ax.set_xlim(0,31); ax.set_xlabel("US list-price increase, 2026 (%)")
fig.tight_layout(); fig.savefig("fig3.png", dpi=300); plt.close(fig)

# Fig 4: August 2026 Circana YoY
m = [("Avg. price\npaid",(541/476-1)*100),("Hardware\nunits",-15.0),("Hardware\nspending",-3.0)]
print(m[0])
fig, ax = plt.subplots(figsize=(W*0.8,2.2))
for i,(n,v) in enumerate(m):
    ax.bar(i, v, width=0.5, color=DARK if v>0 else LIGHT, edgecolor="black", linewidth=0.5, hatch="" if v>0 else "//")
    ax.text(i, v+(0.8 if v>0 else -1.8), f"{v:+.1f}%", ha="center", fontsize=7.5)
ax.axhline(0,color="black",linewidth=0.6); ax.set_xticks(range(3)); ax.set_xticklabels([n for n,_ in m]); ax.set_ylim(-20,19)
ax.set_ylabel("Change vs. Aug 2025 (%)")
fig.tight_layout(); fig.savefig("fig4.png", dpi=300); plt.close(fig)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":7.5,"axes.spines.top":False,"axes.spines.right":False})
W=4.6
# Fig 1: five best papers (reported accuracies)
p=[("[1] CatBoost",99.738),("[6] BFS-GSRF",99.0),("[5] Random Forest",98.11),("[2] Decision Tree",98.02),("[4] Random Forest",98.0)]
fig,ax=plt.subplots(figsize=(W,2.1))
for i,(n,v) in enumerate(p[::-1]):
    ax.barh(i,v,color="#d0d0d0",edgecolor="black",linewidth=0.5,hatch="//")
    ax.text(v+0.05,i,f"{v:.2f}",va="center",fontsize=7)
ax.set_yticks(range(5)); ax.set_yticklabels([n for n,_ in p[::-1]]); ax.set_xlim(96,100.4); ax.set_xlabel("Reported accuracy (%) - axis starts at 96")
fig.tight_layout(); fig.savefig("fig1.png",dpi=300); plt.close(fig)

# Flowchart
fig,ax=plt.subplots(figsize=(W,6.3)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
cx=0.40; bw=0.58; bh=0.052
nodes=[("Start","term"),("Collect public data\n(TrendForce, console prices, Circana)","box"),
 ("Clean data\n(duplicates, missing values)","box"),("Build features and labels\n(lags 1, 3, 6; price and demand bands)","box"),
 ("Enough rows? (at least 100)","dec"),("Split data\n(random 60:40, 70:30, 80:20; chronological)","box"),
 ("Fit scaler on training data only;\nscale train and test","box"),("Train 15 classifiers","box"),
 ("Evaluate: accuracy, macro-F1,\nsevere-class recall","box"),("Rank models and select the best","box"),
 ("Compare best result with\nfive best reviewed papers","box"),("End","term")]
ys=[0.955-i*0.0835 for i in range(len(nodes))]
for (t,k),y in zip(nodes,ys):
    if k=="box":
        ax.add_patch(FancyBboxPatch((cx-bw/2,y-bh/2),bw,bh,boxstyle="round,pad=0,rounding_size=0.008",fc="white",ec="black",lw=0.8))
    elif k=="term":
        ax.add_patch(FancyBboxPatch((cx-0.12,y-0.022),0.24,0.044,boxstyle="round,pad=0,rounding_size=0.022",fc="#e6e6e6",ec="black",lw=0.8))
    else:
        ax.add_patch(Polygon([(cx-bw/2,y),(cx,y+0.04),(cx+bw/2,y),(cx,y-0.04)],closed=True,fc="#e6e6e6",ec="black",lw=0.8))
    ax.text(cx,y,t,ha="center",va="center",fontsize=6.2)
for i in range(len(nodes)-1):
    y0=ys[i]-(0.04 if nodes[i][1]=="dec" else (0.022 if nodes[i][1]=="term" else bh/2)); y1=ys[i+1]+(0.022 if nodes[i+1][1]=="term" else bh/2)
    ax.annotate("",xy=(cx,y1),xytext=(cx,y0),arrowprops=dict(arrowstyle="-|>",lw=0.8,color="black"))
yd=ys[4]
ax.text(cx+0.015,yd-0.052,"Yes",fontsize=6.5)
ax.annotate("",xy=(0.76,yd),xytext=(cx+bw/2,yd),arrowprops=dict(arrowstyle="-|>",lw=0.8,color="black"))
ax.text(cx+bw/2+0.01,yd+0.012,"No",fontsize=6.5)
ax.add_patch(FancyBboxPatch((0.76,yd-0.04),0.23,0.08,boxstyle="round,pad=0,rounding_size=0.008",fc="#e6e6e6",ec="black",lw=0.8))
ax.text(0.875,yd,"Stop: report\ndata limits",ha="center",va="center",fontsize=6.5)
fig.savefig("flow.png",dpi=300); plt.close(fig)
print("ok")
