#File 3:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

np.random.seed(7)
n = 1000

# --- Simulate feature values ---
sdr       = np.random.beta(2, 5, n)
age       = np.random.normal(67, 9, n)
bmi       = np.random.normal(27, 5, n)
sex       = np.random.binomial(1, 0.35, n).astype(float)
ethnicity = np.random.choice([0, 1, 2, 3], n).astype(float)
power     = np.random.normal(58, 10, n)
skull_vol = np.random.normal(145, 25, n)
tremor    = np.random.normal(18, 4, n)

# --- Simulate mean SHAP values (signed, not absolute) ---
# Negative = feature decreases energy output on average
# Positive = feature increases energy output on average
mean_shap = {
    "SDR Ratio":        -0.42,   # higher SDR -> less energy needed (negative)
    "Age":               0.31,   # older patients -> more energy required
    "BMI":               0.22,   # higher BMI -> more energy required
    "Sonication Power":  0.18,   # directly increases output
    "Skull Volume":     -0.13,   # larger skull -> attenuates energy (negative)
    "Baseline Tremor":   0.10,   # worse tremor -> more energy used
    "Sex":              -0.07,   # slight negative association
    "Ethnicity":         0.04,   # minimal association
}

# Add small noise per feature to simulate real variation across patients
shap_vals = {f: v + np.random.normal(0, 0.015) for f, v in mean_shap.items()}

# Sort by absolute value ascending (so largest bar is on top)
order = sorted(shap_vals, key=lambda f: abs(shap_vals[f]))
labels = order
values = [shap_vals[f] for f in order]

# --- Colors: red for positive, blue for negative (classic SHAP style) ---
colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)

# Value labels at end of each bar
for bar, val in zip(bars, values):
    offset = 0.005 if val >= 0 else -0.005
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=10, color="#222222")

# Zero reference line
ax.axvline(0, color="black", linewidth=1.0, linestyle="-", zorder=3)

# Legend
pos_patch = plt.matplotlib.patches.Patch(color="#d73027", label="Increases energy output")
neg_patch = plt.matplotlib.patches.Patch(color="#4575b4", label="Decreases energy output")
ax.legend(handles=[pos_patch, neg_patch], fontsize=10,
          loc="lower right", framealpha=0.85)

# Formatting
ax.set_xlabel("Mean SHAP Value (average impact on predicted energy output)", fontsize=11)
ax.set_title(
    "SHAP Feature Importance — MRgFUS Energy Output Prediction\n"
    "Clinical & Demographic Features  |  Simulated Data",
    fontsize=13, fontweight="bold", pad=12
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Extend xlim slightly to fit labels
ax.set_xlim(min(values) * 1.25, max(values) * 1.25)

plt.tight_layout()
plt.savefig("shap_signed_bar_mrgfus.png", dpi=180, bbox_inches="tight")
plt.show()
