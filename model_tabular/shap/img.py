import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"

IMP_CSV = "tabpfn_shap_importance_raw.csv"
SV_CSV  = "tabpfn_shap_values.csv"
TOP_K = 10
BAR_DECIMALS = 4
OUT_FIG = "shap_summary.png"
OUT_PDF = "shap_summary.pdf"
OUT_SVG = "shap_summary.svg"
DPI = 600

imp = pd.read_csv(IMP_CSV)
sv  = pd.read_csv(SV_CSV)

top_feats = imp.nlargest(TOP_K, "mean_abs_shap")["feature"].tolist()
order = top_feats[::-1]

imp_left = imp.set_index("feature").loc[order].reset_index()

sv_right = sv[sv["feature"].isin(order)].copy()
sv_right["feature"] = pd.Categorical(sv_right["feature"], categories=order, ordered=True)

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 2], wspace=0.04)

ax1 = fig.add_subplot(gs[0, 0])

bars = ax1.barh(imp_left["feature"], imp_left["mean_abs_shap"], color="tab:orange")

max_v = float(imp_left["mean_abs_shap"].max())
right_lim = max_v * 1.12 if max_v > 0 else 0.03
left_lim = 0.02
if right_lim <= left_lim:
    right_lim = left_lim + 0.01
ax1.set_xlim(left_lim, right_lim+0.02)

pad = max_v * 0.01 if max_v > 0 else 0.001
for y, v in enumerate(imp_left["mean_abs_shap"]):
    x_txt = v + pad
    x_txt = max(x_txt, left_lim + pad)
    x_txt = min(x_txt, right_lim - pad)
    ax1.text(x_txt, y, f"{v:.{BAR_DECIMALS}f}", va="center", ha="left", clip_on=True)

ax1.set_xlabel("Mean(|SHAP|)")
ax1.invert_yaxis()
ax1.grid(axis="x", linestyle="--", alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])

rng = np.random.default_rng(0)
y_positions = {feat: i for i, feat in enumerate(order)}

cbar_norm = Normalize(0, 1)
cbar_map = plt.cm.ScalarMappable(cmap="coolwarm", norm=cbar_norm)

first_scatter = None
for feat in order:
    sub = sv_right[sv_right["feature"] == feat]
    if sub.empty:
        continue

    vmin, vmax = sub["value"].min(), sub["value"].max()
    val_norm = (sub["value"] - vmin) / (vmax - vmin + 1e-12)

    y0 = y_positions[feat]
    jitter = (rng.random(len(sub)) - 0.5) * 0.25

    sc = ax2.scatter(
        sub["shap"].values,
        np.full(len(sub), y0) + jitter,
        c=val_norm.values,
        cmap="coolwarm",
        s=14,
        alpha=0.85,
        edgecolors="none",
        vmin=0, vmax=1
    )
    if first_scatter is None:
        first_scatter = sc

ax2.set_yticks(range(len(order)))
ax2.tick_params(axis="y", which="both", labelleft=False)

ax2.set_xlabel("SHAP value (impact on model output)")
ax2.axvline(0, color="k", linewidth=1)
ax2.grid(axis="x", linestyle="--", alpha=0.3)

cb = fig.colorbar(cbar_map, ax=ax2, pad=0.01)
cb.set_label("Feature value")

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
plt.savefig(OUT_SVG, bbox_inches="tight", transparent=True)
plt.savefig(OUT_FIG, dpi=DPI, bbox_inches="tight")
plt.show()

print(f"Saved to: {OUT_PDF}, {OUT_SVG}, {OUT_FIG}")
