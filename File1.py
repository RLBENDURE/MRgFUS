#File 1:

import numpy as np
import matplotlib.pyplot as plt
import shap

# Set seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Define features
feature_names = [
    "Skull Density Ratio", "Age", "BMI",
    "Treatment Duration", "Sex", "Race", "Target Volume",
    "Number of Sonicatons", "Baseline Tremor Score"
]

# Simulate feature values (normalized 0-1 for color mapping)
feature_values = {
    "Skull Density Ratio": np.random.beta(2, 5, n_samples),
    "Age": np.random.normal(68, 8, n_samples),
    "BMI": np.random.normal(27, 4, n_samples),
    "Treatment Duration": np.random.normal(120, 20, n_samples),
    "Sex": np.random.binomial(1, 0.35, n_samples),
    "Race": np.random.choice([0, 1, 2, 3], n_samples),
    "Target Volume": np.random.normal(150, 30, n_samples),
    "Number of Sonicatons": np.random.randint(10, 30, n_samples),
    "Baseline Tremor Score": np.random.normal(18, 4, n_samples)
}

# Simulate SHAP values (with realistic importance ordering)
shap_data = {
    "Skull Density Ratio": np.random.normal(0, 0.45, n_samples),
    "Age":                  np.random.normal(0, 0.30, n_samples),
    "BMI":                  np.random.normal(0, 0.22, n_samples),
    "Treatment Duration":   np.random.normal(0, 0.16, n_samples),
    "Sex":                  np.random.normal(0, 0.10, n_samples),
    "Race":                 np.random.normal(0, 0.08, n_samples),
    "Target Volume":        np.random.normal(0, 0.07, n_samples),
    "Number of Sonicatons": np.random.normal(0, 0.06, n_samples),
    "Baseline Tremor Score":np.random.normal(0, 0.05, n_samples)
}

# Order features by mean absolute SHAP value
order = sorted(feature_names, key=lambda f: np.mean(np.abs(shap_data[f])))

# Build arrays in sorted order
shap_matrix = np.column_stack([shap_data[f] for f in order])
value_matrix = np.column_stack([feature_values[f] for f in order])

# Normalize feature values to [0,1] for color mapping
value_norm = (value_matrix - value_matrix.min(axis=0)) / (
    value_matrix.max(axis=0) - value_matrix.min(axis=0) + 1e-8
)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.cm.RdBu_r

for i, feature in enumerate(order):
    col_idx = i
    sv = shap_matrix[:, col_idx]
    fv = value_norm[:, col_idx]
    colors = cmap(fv)
    # Add jitter for visibility
    y_jitter = i + np.random.uniform(-0.2, 0.2, n_samples)
    ax.scatter(sv, y_jitter, c=colors, alpha=0.7, s=20, linewidths=0)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=11)
ax.set_xlabel("SHAP Value (impact on predicted energy output)", fontsize=11)
ax.set_title("SHAP Summary Plot — MRgFUS Energy Output Prediction\n(Random Forest, Simulated Data)", fontsize=12)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Feature Value (low → high)", fontsize=10)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Low", "High"])

plt.tight_layout()
plt.savefig("shap_summary_mrgfus.png", dpi=150)
plt.show()
