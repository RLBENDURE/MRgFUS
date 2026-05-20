#File 5:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  MODULE: MRgFUS Energy Sonication Estimator
#          & Clinical Interpretation Tool
#  Institution: University of Florida
#  Purpose:     Surgical decision support
#  Data:        Simulated — not for clinical use
# ─────────────────────────────────────────────

SHAP_WEIGHTS = {
    "sdr_ratio":       -0.42,
    "age":              0.31,
    "bmi":              0.22,
    "sonication_power": 0.18,
    "skull_volume":    -0.13,
    "baseline_tremor":  0.10,
    "sex":             -0.07,
    "ethnicity":        0.04,
}

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

BASE_ENERGY_J = 10000


# ── 1. Core estimator ──────────────────────────────────────────────────────────
def estimate_energy(
    sdr_ratio, age, bmi, sonication_power,
    skull_volume, baseline_tremor, sex, ethnicity
) -> dict:

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

    contributions = {}
    total_adjustment = 0.0
    for feature, value in inputs.items():
        z = (value - POP_MEANS[feature]) / POP_STDS[feature]
        contribution_j = SHAP_WEIGHTS[feature] * z * BASE_ENERGY_J * 0.15
        contributions[feature] = round(contribution_j, 2)
        total_adjustment += contribution_j

    predicted_energy = round(BASE_ENERGY_J + total_adjustment, 1)

    if predicted_energy < 480:
        risk_tier, risk_color = "LOW",      "#2ecc71"
    elif predicted_energy < 650:
        risk_tier, risk_color = "MODERATE", "#f39c12"
    else:
        risk_tier, risk_color = "HIGH",     "#e74c3c"

    return {
        "predicted_energy_j":    predicted_energy,
        "feature_contributions": contributions,
        "risk_tier":             risk_tier,
        "risk_color":            risk_color,
        "inputs":                inputs,
    }


# ── 2. Clinical interpretation text ───────────────────────────────────────────
def interpret(results: dict) -> str:
    e        = results["predicted_energy_j"]
    tier     = results["risk_tier"]
    contribs = results["feature_contributions"]
    inputs   = results["inputs"]

    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_3 = sorted_contribs[:3]

    sex_label       = "Female" if inputs["sex"] == 1 else "Male"
    ethnicity_map   = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    ethnicity_label = ethnicity_map.get(int(inputs["ethnicity"]), "Unknown")

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

    lines = [
        "=" * 60,
        "  MRgFUS CLINICAL ENERGY ESTIMATION REPORT",
        "  ⚠  SIMULATED DATA — NOT FOR CLINICAL USE",
        "=" * 60,
        f"  Patient:  Age {inputs['age']:.0f} | {sex_label} | BMI {inputs['bmi']:.1f} | {ethnicity_label}",
        f"  SDR: {inputs['sdr_ratio']:.3f} | Skull Vol: {inputs['skull_volume']:.0f} mm³ | Tremor: {inputs['baseline_tremor']:.1f}",
        "-" * 60,
        f"  ▶  Predicted Energy:  {e:.1f} J     Risk Tier: {tier}",
        "-" * 60,
        "  Top Contributing Factors:",
    ]

    for feat, val in top_3:
        fname, explanation = direction_map[feat]
        direction = "↑ increases" if val > 0 else "↓ decreases"
        lines.append(f"    • {fname:<20} {direction} output by {abs(val):.1f} J")
        lines.append(f"      ({explanation})")

    lines += ["-" * 60, "  Clinical Guidance:"]
    if tier == "LOW":
        lines += ["    Standard protocol applicable. Monitor thermal dose."]
    elif tier == "MODERATE":
        lines += ["    Consider incremental power stepping. Review SDR."]
    else:
        lines += ["    ⚠ Elevated demand. Verify SDR & skull density.",
                  "    Multidisciplinary review recommended."]
    lines.append("=" * 60)
    return "\n".join(lines)


# ── 3. Full figure with patient profile panel ──────────────────────────────────
def plot_shap_contributions(results: dict, save_path: str = None):

    contribs   = results["feature_contributions"]
    risk_color = results["risk_color"]
    inputs     = results["inputs"]

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

    sex_label       = "Female" if inputs["sex"] == 1 else "Male"
    ethnicity_map   = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    ethnicity_label = ethnicity_map.get(int(inputs["ethnicity"]), "Unknown")

    order  = sorted(contribs, key=lambda f: abs(contribs[f]))
    labels = [label_map[f] for f in order]
    values = [contribs[f] for f in order]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

    # ── Layout: 3 columns ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.1, 2.4, 1.0],
        wspace=0.35
    )

    ax_profile = fig.add_subplot(gs[0])   # Patient profile
    ax_shap    = fig.add_subplot(gs[1])   # SHAP bar chart
    ax_summary = fig.add_subplot(gs[2])   # Risk / energy summary

    # ── Panel 1: Patient Profile ───────────────────────────────────────────────
    ax_profile.set_xlim(0, 1)
    ax_profile.set_ylim(0, 1)
    ax_profile.axis("off")

    # Header banner
    ax_profile.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.88), 1.0, 0.12,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#2c3e50"
    ))
    ax_profile.text(0.5, 0.944, "PATIENT PROFILE",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

    # Profile fields
    profile_rows = [
        ("Age",              f"{inputs['age']:.0f} yrs"),
        ("Sex",              sex_label),
        ("BMI",              f"{inputs['bmi']:.1f} kg/m²"),
        ("Ethnicity",        ethnicity_label),
        ("SDR Ratio",        f"{inputs['sdr_ratio']:.3f}"),
        ("Skull Volume",     f"{inputs['skull_volume']:.0f} mm³"),
        ("Baseline Tremor",  f"{inputs['baseline_tremor']:.1f} (CRST)"),
        ("Sonication Power", f"{inputs['sonication_power']:.0f} W"),
    ]

    y_start = 0.82
    row_h   = 0.095

    for i, (field, value) in enumerate(profile_rows):
        y = y_start - i * row_h
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        ax_profile.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.04), 1.0, row_h,
            boxstyle="square,pad=0.0",
            linewidth=0, facecolor=bg
        ))
        ax_profile.text(0.05, y + 0.005, field,
                        ha="left", va="center",
                        fontsize=9.5, color="#555555")
        ax_profile.text(0.95, y + 0.005, value,
                        ha="right", va="center",
                        fontsize=9.5, fontweight="bold", color="#222222")

    # Disclaimer
    ax_profile.text(0.5, 0.01,
                    "⚠ Simulated data — not for clinical use",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#e74c3c", style="italic")

    ax_profile.set_title("", fontsize=1)

    # ── Panel 2: SHAP Bar Chart ────────────────────────────────────────────────
    bars = ax_shap.barh(labels, values, color=colors, edgecolor="none", height=0.6)
    ax_shap.axvline(0, color="black", linewidth=1.0)

    for bar, val in zip(bars, values):
        offset = 0.3 if val >= 0 else -0.3
        ha     = "left" if val >= 0 else "right"
        ax_shap.text(val + offset,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:+.1f} J",
                     va="center", ha=ha, fontsize=10)

    pos_patch = mpatches.Patch(color="#d73027", label="↑ Increases energy output")
    neg_patch = mpatches.Patch(color="#4575b4", label="↓ Decreases energy output")
    ax_shap.legend(handles=[pos_patch, neg_patch],
                   fontsize=10, loc="lower right", framealpha=0.9)

    ax_shap.set_xlabel("SHAP Contribution to Predicted Energy Output (J)", fontsize=11)
    ax_shap.set_title(
        "Feature Contributions — Individual Patient\nMRgFUS Energy Output Prediction",
        fontsize=12, fontweight="bold"
    )
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    ax_shap.spines["left"].set_visible(False)
    ax_shap.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_shap.set_axisbelow(True)
    xlim = max(abs(min(values)), abs(max(values))) * 1.4
    ax_shap.set_xlim(-xlim, xlim)

    # ── Panel 3: Risk & Energy Summary ────────────────────────────────────────
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis("off")

    # Header
    ax_summary.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.88), 0.90, 0.12,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#2c3e50"
    ))
    ax_summary.text(0.5, 0.944, "CLINICAL SUMMARY",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

    # Predicted energy box
    ax_summary.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.60), 0.90, 0.24,
        boxstyle="round,pad=0.03",
        linewidth=2, edgecolor="#555555", facecolor="#f5f5f5"
    ))
    ax_summary.text(0.5, 0.80, "Predicted Energy",
                    ha="center", va="center", fontsize=10, color="#555555")
    ax_summary.text(0.5, 0.67, f"{results['predicted_energy_j']:.1f} J",
                    ha="center", va="center",
                    fontsize=24, fontweight="bold", color="#222222")

    # Risk tier box
    ax_summary.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.32), 0.90, 0.24,
        boxstyle="round,pad=0.03",
        linewidth=2.5, edgecolor=risk_color,
        facecolor=risk_color + "28"
    ))
    ax_summary.text(0.5, 0.52, "Risk Tier",
                    ha="center", va="center", fontsize=10, color="#555555")
    ax_summary.text(0.5, 0.39, results["risk_tier"],
                    ha="center", va="center",
                    fontsize=24, fontweight="bold", color=risk_color)

    # Guidance blurb
    guidance = {
        "LOW":      "Standard protocol\napplicable.\nMonitor thermal\ndose closely.",
        "MODERATE": "Incremental power\nstepping advised.\nReview SDR before\nproceeding.",
        "HIGH":     "⚠ Elevated demand.\nVerify SDR & skull\ndensity. MDT review\nrecommended.",
    }
    ax_summary.text(0.5, 0.16, guidance[results["risk_tier"]],
                    ha="center", va="center",
                    fontsize=8.5, color="#333333",
                    linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#fffde7",
                              edgecolor="#cccccc",
                              linewidth=1))

    # ── Super title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "MRgFUS Clinical Energy Estimation & Interpretability Tool  |  University of Florida",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ── 4. Main pipeline ───────────────────────────────────────────────────────────
def run_clinical_tool(
    sdr_ratio, age, bmi, sonication_power,
    skull_volume, baseline_tremor, sex, ethnicity,
    save_plot: str = None
):
    results = estimate_energy(
        sdr_ratio=sdr_ratio, age=age, bmi=bmi,
        sonication_power=sonication_power,
        skull_volume=skull_volume,
        baseline_tremor=baseline_tremor,
        sex=sex, ethnicity=ethnicity,
    )
    print(interpret(results))
    plot_shap_contributions(results, save_path=save_plot)
    return results


# ══════════════════════════════════════════════════════
#  EXAMPLE PATIENT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    run_clinical_tool(
        sdr_ratio        = 0.38,
        age              = 72,
        bmi              = 31.2,
        sonication_power = 62,
        skull_volume     = 138,
        baseline_tremor  = 22,
        sex              = 0,
        ethnicity        = 0,
        save_plot        = "mrgfus_clinical_report.png"
    )


