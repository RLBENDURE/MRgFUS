#File 4:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  MODULE: MRgFUS Energy Sonication Estimator
#          & Clinical Interpretation Tool
#  Institution: University of Florida
#  Purpose:     Surgical decision support
#  Data:        Simulated — not for clinical use
# ─────────────────────────────────────────────

# ── 1. SHAP-based feature weights (from simulated model) ──────────────────────
SHAP_WEIGHTS = {
    "sdr_ratio":       -0.42,   # Higher SDR = less energy needed
    "age":              0.31,   # Older patients need more energy
    "bmi":              0.22,   # Higher BMI = more energy needed
    "sonication_power": 0.18,   # Direct positive contribution
    "skull_volume":    -0.13,   # Larger skull attenuates energy
    "baseline_tremor":  0.10,   # Worse tremor = more energy
    "sex":             -0.07,   # Female = slight reduction
    "ethnicity":        0.04,   # Minimal effect
}

# ── Population reference means (for centering inputs) ─────────────────────────
POP_MEANS = {
    "sdr_ratio":        0.45,
    "age":             67.0,
    "bmi":             27.0,
    "sonication_power": 58.0,
    "skull_volume":    145.0,
    "baseline_tremor":  18.0,
    "sex":              0.35,
    "ethnicity":        1.5,
}

POP_STDS = {
    "sdr_ratio":        0.12,
    "age":              9.0,
    "bmi":              5.0,
    "sonication_power": 10.0,
    "skull_volume":     25.0,
    "baseline_tremor":   4.0,
    "sex":               0.48,
    "ethnicity":         1.1,
}

BASE_ENERGY_J = 580.0   # Baseline joules (population mean)

# ── 2. Core estimator ──────────────────────────────────────────────────────────
def estimate_energy(
    sdr_ratio: float,
    age: float,
    bmi: float,
    sonication_power: float,
    skull_volume: float,
    baseline_tremor: float,
    sex: int,           # 0 = Male, 1 = Female
    ethnicity: int,     # 0=White, 1=Black, 2=Hispanic, 3=Asian/Other
) -> dict:
    """
    Estimates predicted sonication energy output (Joules) for MRgFUS
    treatment of essential tremor, with SHAP-based feature contributions.

    Returns a results dictionary for use in the clinical interpreter.
    """
    inputs = {
        "sdr_ratio":        sdr_ratio,
        "age":              age,
        "bmi":              bmi,
        "sonication_power": sonication_power,
        "skull_volume":     skull_volume,
        "baseline_tremor":  baseline_tremor,
        "sex":              float(sex),
        "ethnicity":        float(ethnicity),
    }

    # Standardize inputs relative to population
    contributions = {}
    total_adjustment = 0.0
    for feature, value in inputs.items():
        z = (value - POP_MEANS[feature]) / POP_STDS[feature]
        contribution_j = SHAP_WEIGHTS[feature] * z * BASE_ENERGY_J * 0.15
        contributions[feature] = round(contribution_j, 2)
        total_adjustment += contribution_j

    predicted_energy = round(BASE_ENERGY_J + total_adjustment, 1)

    # Risk stratification
    if predicted_energy < 480:
        risk_tier = "LOW"
        risk_color = "#2ecc71"
    elif predicted_energy < 650:
        risk_tier = "MODERATE"
        risk_color = "#f39c12"
    else:
        risk_tier = "HIGH"
        risk_color = "#e74c3c"

    return {
        "predicted_energy_j":  predicted_energy,
        "feature_contributions": contributions,
        "risk_tier":           risk_tier,
        "risk_color":          risk_color,
        "inputs":              inputs,
    }


# ── 3. Clinical interpretation text ───────────────────────────────────────────
def interpret(results: dict) -> str:
    """
    Generates a plain-language clinical summary for the surgeon
    based on the energy estimate and feature contributions.
    """
    e      = results["predicted_energy_j"]
    tier   = results["risk_tier"]
    contribs = results["feature_contributions"]
    inputs   = results["inputs"]

    # Identify top drivers
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_3 = sorted_contribs[:3]

    # Sex/ethnicity labels
    sex_label       = "Female" if inputs["sex"] == 1 else "Male"
    ethnicity_map   = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    ethnicity_label = ethnicity_map.get(int(inputs["ethnicity"]), "Unknown")

    lines = [
        "=" * 60,
        "  MRgFUS CLINICAL ENERGY ESTIMATION REPORT",
        "  ⚠  SIMULATED DATA — NOT FOR CLINICAL USE",
        "=" * 60,
        f"  Patient Profile:",
        f"    Age:              {inputs['age']:.0f} yrs",
        f"    Sex:              {sex_label}",
        f"    BMI:              {inputs['bmi']:.1f} kg/m²",
        f"    Ethnicity:        {ethnicity_label}",
        f"    SDR Ratio:        {inputs['sdr_ratio']:.3f}",
        f"    Skull Volume:     {inputs['skull_volume']:.0f} mm³",
        f"    Baseline Tremor:  {inputs['baseline_tremor']:.1f} (CRST score)",
        f"    Sonication Power: {inputs['sonication_power']:.0f} W",
        "-" * 60,
        f"  ▶  Predicted Energy Output:  {e:.1f} J",
        f"  ▶  Risk Tier:                {tier}",
        "-" * 60,
        "  Top Contributing Factors:",
    ]

    direction_map = {
        "sdr_ratio":        ("SDR Ratio",        "lower SDR reduces acoustic efficiency"),
        "age":              ("Age",               "older age increases energy demand"),
        "bmi":              ("BMI",               "higher BMI increases path attenuation"),
        "sonication_power": ("Sonication Power",  "higher power directly raises output"),
        "skull_volume":     ("Skull Volume",      "larger skull increases energy loss"),
        "baseline_tremor":  ("Baseline Tremor",   "more severe tremor increases dosing"),
        "sex":              ("Sex",               "female sex slightly reduces demand"),
        "ethnicity":        ("Ethnicity",         "minimal modeled effect"),
    }

    for feat, val in top_3:
        fname, explanation = direction_map[feat]
        direction = "↑ increases" if val > 0 else "↓ decreases"
        lines.append(f"    • {fname:<20} {direction} output by {abs(val):.1f} J")
        lines.append(f"      ({explanation})")

    lines += [
        "-" * 60,
        "  Clinical Guidance:",
    ]

    if tier == "LOW":
        lines.append("    Standard protocol applicable. SDR and patient")
        lines.append("    anatomy are favorable. Monitor thermal dose closely.")
    elif tier == "MODERATE":
        lines.append("    Consider incremental power stepping. Review SDR")
        lines.append("    and skull density imaging before proceeding.")
    else:
        lines.append("    ⚠ Elevated energy demand detected. Verify SDR,")
        lines.append("    skull density, and consider adjusted targeting.")
        lines.append("    Multidisciplinary review recommended.")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── 4. SHAP bar chart visualization ───────────────────────────────────────────
def plot_shap_contributions(results: dict, save_path: str = None):
    """
    Plots a signed SHAP contribution bar chart for the individual patient,
    sorted by absolute contribution magnitude.
    """
    contribs = results["feature_contributions"]
    risk_color = results["risk_color"]

    label_map = {
        "sdr_ratio":        "SDR Ratio",
        "age":              "Age",
        "bmi":              "BMI",
        "sonication_power": "Sonication Power",
        "skull_volume":     "Skull Volume",
        "baseline_tremor":  "Baseline Tremor",
        "sex":              "Sex",
        "ethnicity":        "Ethnicity",
    }

    order  = sorted(contribs, key=lambda f: abs(contribs[f]))
    labels = [label_map[f] for f in order]
    values = [contribs[f] for f in order]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    # ── Left: SHAP bar chart ──
    ax = axes[0]
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", linewidth=1.0)

    for bar, val in zip(bars, values):
        offset = 0.3 if val >= 0 else -0.3
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f} J", va="center", ha=ha, fontsize=10)

    pos_patch = mpatches.Patch(color="#d73027", label="↑ Increases energy output")
    neg_patch = mpatches.Patch(color="#4575b4", label="↓ Decreases energy output")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10,
              loc="lower right", framealpha=0.9)

    ax.set_xlabel("SHAP Contribution to Predicted Energy Output (J)", fontsize=11)
    ax.set_title("Feature Contributions — Individual Patient\n(MRgFUS Energy Prediction)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # ── Right: Risk gauge panel ──
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Risk tier box
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.1, 0.55), 0.8, 0.35,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=risk_color,
        facecolor=risk_color + "33"  # transparent fill
    ))
    ax2.text(0.5, 0.775, "Risk Tier", ha="center", va="center",
             fontsize=12, color="#333333")
    ax2.text(0.5, 0.65, results["risk_tier"], ha="center", va="center",
             fontsize=22, fontweight="bold", color=risk_color)

    # Predicted energy box
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.1, 0.10), 0.8, 0.35,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor="#555555",
        facecolor="#f5f5f5"
    ))
    ax2.text(0.5, 0.375, "Predicted Energy", ha="center", va="center",
             fontsize=11, color="#333333")
    ax2.text(0.5, 0.22, f"{results['predicted_energy_j']:.1f} J",
             ha="center", va="center",
             fontsize=22, fontweight="bold", color="#222222")

    ax2.set_title("Clinical Summary", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ── 5. Main clinical interface ─────────────────────────────────────────────────
def run_clinical_tool(
    sdr_ratio,
    age,
    bmi,
    sonication_power,
    skull_volume,
    baseline_tremor,
    sex,
    ethnicity,
    save_plot: str = None
):
    """
    Full pipeline: estimate energy, print clinical report, display SHAP plot.
    """
    results = estimate_energy(
        sdr_ratio=sdr_ratio,
        age=age,
        bmi=bmi,
        sonication_power=sonication_power,
        skull_volume=skull_volume,
        baseline_tremor=baseline_tremor,
        sex=sex,
        ethnicity=ethnicity,
    )

    report = interpret(results)
    print(report)
    plot_shap_contributions(results, save_path=save_plot)
    return results


# ══════════════════════════════════════════════════════
#  EXAMPLE PATIENT — run this block to test the module
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    run_clinical_tool(
        sdr_ratio        = 0.38,   # Below average SDR — poorer skull quality
        age              = 72,     # Older patient
        bmi              = 31.2,   # Overweight
        sonication_power = 62,     # Slightly above average power
        skull_volume     = 138,    # Slightly smaller skull
        baseline_tremor  = 22,     # Moderately severe tremor
        sex              = 0,      # Male
        ethnicity        = 0,      # White
        save_plot        = "patient_shap_report.png"
    )
