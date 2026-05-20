#File 2:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

np.random.seed(7)
n = 1000

# --- Simulate SHAP values ---
sdr        = np.random.beta(2, 5, n)
age        = np.random.normal(67, 9, n)
bmi        = np.random.normal(27, 5, n)
sex        = np.random.binomial(1, 0.35, n).astype(float)
ethnicity  = np.random.choice([0, 1, 2, 3], n).astype(float)
power      = np.random.normal(58, 10, n)
skull_vol  = np.random.normal(145, 25, n)
tremor     = np.random.normal(18, 4, n)

shap_vals = {
    "SDR Ratio":        -0.55 * (sdr - sdr.mean()) / sdr.std()         + np.random.normal(0, 0.08, n),
    "Age":               0.38 * (age - age.mean()) / age.std()         + np.random.normal(0, 0.07, n),
    "BMI":               0.28 * (bmi - bmi.mean()) / bmi.std()         + np.random.normal(0, 0.06, n),
    "Sonication Power":  0.22 * (power - power.mean()) / power.std()   + np.random.normal(0, 0.05, n),
    "Skull Volume":      0.15 * (skull_vol - skull_vol.mean()) / skull_vol.std() + np.random.normal(0, 0.04, n),
    "Baseline Tremor":   0.12 * (tremor - tremor.mean()) / tremor.std()+ np.random.normal(0, 0.04, n),
    "Sex":               np.random.normal(0, 0.07, n),
    "Ethnicity":         np.random.normal(0, 0.05, n),
}

# Mean absolute SHAP values, sorted ascending for plot
mean_abs = {f: np.mean(np.abs(v)) for f, v in shap_vals.items()}
order = sorted(mean_abs, key=mean_abs.get)  # ascending = top is most important

labels = order
values = [mean_abs[f] for f in order]

# --- Color: map importance magnitude to SHAP's default red-pink palette ---
norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
cmap = cm.get_cmap("RdPu")
colors = [cmap(norm(v)) for v in values]

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)

# Value labels at end of each bar
for bar, val in zip(bars, values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=10, color="#333333")

# Colorbar to indicate importance magnitude
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
cbar.set_label("Mean |SHAP Value|", fontsize=10)
cbar.set_ticks([min(values), max(values)])
cbar.set_ticklabels(["Lower\nImportance", "Higher\nImportance"], fontsize=8)

# Formatting
ax.set_xlabel("Mean |SHAP Value| (average impact on energy output)", fontsize=11)
ax.set_title(
    "SHAP Feature Importance — MRgFUS Energy Output\n"
    "Clinical & Demographic Features  |  Simulated Data",
    fontsize=13, fontweight="bold", pad=12
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
ax.set_xlim(0, max(values) * 1.18)
ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("shap_bar_mrgfus.png", dpi=180, bbox_inches="tight")
plt.show()

