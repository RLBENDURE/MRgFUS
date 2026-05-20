#File 8:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE: MRgFUS Per-Pass Energy Sonication Estimator
#          & Clinical Interpretation Tool
#  Institution: University of Florida
#  Purpose:     Surgical decision support
#  Data:        Simulated — not for clinical use
#
#  SDR Kurtosis Reference:
#  Higher SDR kurtosis (>-0.32) correlates with superior lesion formation,
#  lower peak sonication temps, and 43-62% sustained tremor improvement
#  at 12+ months. Kurtosis is a stronger prognostic factor than mean SDR
#  or SDR standard deviation alone.
# ─────────────────────────────────────────────────────────────────────────────

SHAP_WEIGHTS = {
    "sdr_ratio":        -0.42,
    "sdr_kurtosis":     -0.35,   # Higher kurtosis = less energy needed
    "age":               0.31,
    "bmi":               0.22,
    "sonication_power":  0.18,
    "skull_volume":     -0.13,
    "baseline_tremor":   0.10,
    "sex":              -0.07,
    "ethnicity":         0.04,
}

POP_MEANS = {
    "sdr_ratio":         0.45,
    "sdr_kurtosis":     -0.20,   # Population mean kurtosis (near cutoff)
    "age":              67.0,
    "bmi":              27.0,
    "sonication_power": 58.0,
    "skull_volume":     145.0,
    "baseline_tremor":  18.0,
    "sex":               0.35,
    "ethnicity":         1.5,
}

POP_STDS = {
    "sdr_ratio":         0.12,
    "sdr_kurtosis":      0.45,   # Reasonable spread around cutoff
    "age":               9.0,
    "bmi":               5.0,
    "sonication_power":  10.0,
    "skull_volume":      25.0,
    "baseline_tremor":   4.0,
    "sex":               0.48,
    "ethnicity":         1.1,
}

BASE_ENERGY_J     = 10000
ABLATION_TEMP_C   = 55.0
BASELINE_TEMP_C   = 37.0
COOLING_RATE      = 0.18
HEATING_RATE_BASE = 0.22

# SDR kurtosis clinical cutoff
SDR_KURTOSIS_CUTOFF = -0.32


# ── 1. SDR kurtosis interpreter ────────────────────────────────────────────────
def interpret_sdr_kurtosis(kurtosis: float) -> dict:
    """
    Classifies SDR kurtosis and returns clinical interpretation.

    Cutoff > -0.32 identifies favorable candidates.
    Higher kurtosis = sharper, tighter SDR distribution = better skull
    homogeneity = more efficient energy delivery and superior outcomes.
    """
    favorable = kurtosis > SDR_KURTOSIS_CUTOFF

    if kurtosis > 0.50:
        tier        = "EXCELLENT"
        color       = "#27ae60"
        description = (
            "Highly peaked SDR distribution. Skull highly homogeneous.\n"
            "Expect efficient sonication, lower peak temps,\n"
            "and strong long-term tremor improvement (>55%)."
        )
        energy_modifier = -0.12    # 12% energy reduction expected
        tremor_prognosis = "55–62% CRST improvement at 12 months"

    elif kurtosis > SDR_KURTOSIS_CUTOFF:
        tier        = "FAVORABLE"
        color       = "#2ecc71"
        description = (
            "Above-threshold kurtosis. Good skull homogeneity.\n"
            "Suitable candidate even if mean SDR is below 0.40.\n"
            "Expected tremor improvement: 44–55%."
        )
        energy_modifier = -0.06
        tremor_prognosis = "44–55% CRST improvement at 12 months"

    elif kurtosis > -0.60:
        tier        = "BORDERLINE"
        color       = "#f39c12"
        description = (
            "Below cutoff but marginal. Broader SDR distribution\n"
            "with more low-SDR tail elements. Consider additional\n"
            "skull CT analysis before proceeding."
        )
        energy_modifier = 0.04
        tremor_prognosis = "30–44% CRST improvement at 12 months"

    else:
        tier        = "UNFAVORABLE"
        color       = "#e74c3c"
        description = (
            "Low kurtosis: flat, broad SDR distribution with\n"
            "significant low-SDR tail. Higher energy demand,\n"
            "elevated thermal risk. MDT review recommended."
        )
        energy_modifier = 0.10
        tremor_prognosis = "<30% CRST improvement at 12 months"

    return {
        "kurtosis":          kurtosis,
        "tier":              tier,
        "color":             color,
        "favorable":         favorable,
        "description":       description,
        "energy_modifier":   energy_modifier,
        "tremor_prognosis":  tremor_prognosis,
    }


# ── 2. Base energy estimator ───────────────────────────────────────────────────
def estimate_base_energy(
    sdr_ratio, sdr_kurtosis, age, bmi, sonication_power,
    skull_volume, baseline_tremor, sex, ethnicity
) -> dict:

    inputs = {
        "sdr_ratio":        sdr_ratio,
        "sdr_kurtosis":     sdr_kurtosis,
        "age":              age,
        "bmi":              bmi,
        "sdr_ratio":        sdr_ratio,
        "sonication_power": sonication_power,
        "skull_volume":     skull_volume,
        "baseline_tremor":  baseline_tremor,
        "sex":              float(sex),
        "ethnicity":        float(ethnicity),
    }

    # Re-insert sdr_kurtosis explicitly after dict (avoids duplicate key issue)
    inputs["sdr_kurtosis"] = sdr_kurtosis

    kurtosis_info = interpret_sdr_kurtosis(sdr_kurtosis)

    contributions    = {}
    total_adjustment = 0.0
    for feature, value in inputs.items():
        z = (value - POP_MEANS[feature]) / POP_STDS[feature]
        contribution_j = SHAP_WEIGHTS[feature] * z * BASE_ENERGY_J * 0.15
        contributions[feature] = round(contribution_j, 2)
        total_adjustment += contribution_j

    # Apply kurtosis energy modifier on top of SHAP adjustment
    kurtosis_adjustment = BASE_ENERGY_J * kurtosis_info["energy_modifier"]
    predicted_energy    = round(BASE_ENERGY_J + total_adjustment + kurtosis_adjustment, 1)

    return {
        "base_energy_j":    predicted_energy,
        "contributions":    contributions,
        "inputs":           inputs,
        "kurtosis_info":    kurtosis_info,
    }


# ── 3. Per-pass thermal simulation ────────────────────────────────────────────
def simulate_passes(
    base_result:       dict,
    n_passes:          int   = 5,
    ablation_duration: float = 20.0,
    cooling_time:      float = 40.0,
    power_ramp:        float = 0.05,
    targeting_adjust:  float = 0.03,
    seed:              int   = 42,
) -> dict:

    np.random.seed(seed)

    base_energy      = base_result["base_energy_j"]
    sonication_power = base_result["inputs"]["sonication_power"]
    kurtosis_info    = base_result["kurtosis_info"]
    heating_rate     = HEATING_RATE_BASE / (1 + 0.1 * base_result["inputs"]["bmi"] / 27)

    # Higher kurtosis slightly reduces peak temperature achieved
    # (more homogeneous skull = more efficient, less thermal scatter)
    temp_modifier = -kurtosis_info["energy_modifier"] * 8.0

    passes           = []
    cumulative_e     = 0.0
    cumulative_cem43 = 0.0
    all_time         = []
    all_temp         = []
    global_t         = 0.0

    for p in range(1, n_passes + 1):

        pass_noise  = np.random.normal(0, 0.025)
        pass_energy = round(
            base_energy * (1 + power_ramp * (p - 1))
                        * (1 - targeting_adjust * (p - 1))
                        + pass_noise * base_energy,
            1
        )
        pass_energy = max(pass_energy, 200.0)

        # Heating phase
        start_temp   = BASELINE_TEMP_C + np.random.normal(0, 0.5)
        temp_to_rise = ABLATION_TEMP_C - start_temp + temp_modifier
        heat_duration = temp_to_rise / (heating_rate * sonication_power / 10)
        heat_duration = max(heat_duration, 5.0)

        heat_t    = np.linspace(0, heat_duration, int(heat_duration * 5))
        heat_temp = start_temp + temp_to_rise * (1 - np.exp(-heat_t / (heat_duration * 0.45)))
        heat_temp += np.random.normal(0, 0.3, len(heat_t))

        # Ablation hold phase
        hold_t    = np.linspace(0, ablation_duration, int(ablation_duration * 5))
        hold_temp = ABLATION_TEMP_C + np.random.normal(0, 0.8, len(hold_t))

        # Cooling phase
        cool_t    = np.linspace(0, cooling_time, int(cooling_time * 5))
        cool_temp = ABLATION_TEMP_C * np.exp(-COOLING_RATE * cool_t / ABLATION_TEMP_C * 10)
        cool_temp = np.clip(cool_temp, BASELINE_TEMP_C, ABLATION_TEMP_C)
        cool_temp += np.random.normal(0, 0.3, len(cool_t))

        pass_temp = np.concatenate([heat_temp, hold_temp, cool_temp])
        pass_time = np.linspace(
            0, heat_duration + ablation_duration + cooling_time, len(pass_temp)
        )

        all_time.extend(global_t + pass_time)
        all_temp.extend(pass_temp)
        global_t += heat_duration + ablation_duration + cooling_time

        # CEM43
        dt_s  = (heat_duration + ablation_duration + cooling_time) / len(pass_temp)
        cem43 = 0.0
        for t_val in pass_temp:
            R = 0.5 if t_val >= 43 else 0.25
            cem43 += dt_s / 60 * (R ** (43 - t_val))
        cumulative_cem43 += cem43
        cumulative_e     += pass_energy

        passes.append({
            "pass_number":         p,
            "pass_energy_j":       pass_energy,
            "cumulative_energy_j": round(cumulative_e, 1),
            "heat_duration_s":     round(heat_duration, 1),
            "ablation_hold_s":     ablation_duration,
            "cooling_time_s":      cooling_time,
            "total_pass_time_s":   round(heat_duration + ablation_duration + cooling_time, 1),
            "peak_temp_c":         round(float(np.max(pass_temp)), 1),
            "ablation_achieved":   float(np.max(pass_temp)) >= ABLATION_TEMP_C,
            "cem43":               round(cem43, 2),
            "cumulative_cem43":    round(cumulative_cem43, 2),
        })

    return {
        "passes":            passes,
        "n_passes":          n_passes,
        "ablation_duration": ablation_duration,
        "cooling_time":      cooling_time,
        "total_energy_j":    round(cumulative_e, 1),
        "total_cem43":       round(cumulative_cem43, 2),
        "total_time_s":      round(global_t, 1),
        "time_series":       np.array(all_time),
        "temp_series":       np.array(all_temp),
        "base_result":       base_result,
    }


# ── 4. Clinical report ─────────────────────────────────────────────────────────
def interpret_passes(sim: dict) -> str:
    passes        = sim["passes"]
    inputs        = sim["base_result"]["inputs"]
    kurtosis_info = sim["base_result"]["kurtosis_info"]
    sex_label     = "Female" if inputs["sex"] == 1 else "Male"
    eth_map       = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    eth_label     = eth_map.get(int(inputs["ethnicity"]), "Unknown")

    lines = [
        "=" * 68,
        "  MRgFUS PER-PASS ENERGY & THERMAL DOSE REPORT",
        "  ⚠  SIMULATED DATA — NOT FOR CLINICAL USE",
        "=" * 68,
        f"  Patient:  Age {inputs['age']:.0f} | {sex_label} | "
        f"BMI {inputs['bmi']:.1f} | {eth_label}",
        f"  SDR: {inputs['sdr_ratio']:.3f} | "
        f"SDR Kurtosis: {inputs['sdr_kurtosis']:.3f} | "
        f"Skull Vol: {inputs['skull_volume']:.0f} mm³",
        f"  Ablation Hold: {sim['ablation_duration']:.0f} s  |  "
        f"Cooling Window: {sim['cooling_time']:.0f} s",
        "-" * 68,
        f"  SDR Kurtosis Assessment:  {kurtosis_info['tier']}  "
        f"(cutoff > {SDR_KURTOSIS_CUTOFF})",
        f"  Prognosis:  {kurtosis_info['tremor_prognosis']}",
        "-" * 68,
        f"  {'Pass':<6} {'Energy (J)':<13} {'Cumul. (J)':<13} "
        f"{'Peak °C':<10} {'CEM43':<10} {'Ablation'}",
        "-" * 68,
    ]

    for p in passes:
        ablated = "✔ Yes" if p["ablation_achieved"] else "✘ No"
        lines.append(
            f"  {p['pass_number']:<6} "
            f"{p['pass_energy_j']:<13.1f} "
            f"{p['cumulative_energy_j']:<13.1f} "
            f"{p['peak_temp_c']:<10.1f} "
            f"{p['cem43']:<10.2f} "
            f"{ablated}"
        )

    lines += [
        "-" * 68,
        f"  Total Energy Delivered:   {sim['total_energy_j']:.1f} J",
        f"  Total Thermal Dose:       {sim['total_cem43']:.2f} CEM43",
        f"  Total Procedure Time:     {sim['total_time_s'] / 60:.1f} min",
        "=" * 68,
    ]
    return "\n".join(lines)


# ── 5. Full figure ─────────────────────────────────────────────────────────────
def plot_per_pass(sim: dict, save_path: str = None):

    passes        = sim["passes"]
    time_series   = sim["time_series"] / 60
    temp_series   = sim["temp_series"]
    inputs        = sim["base_result"]["inputs"]
    contribs      = sim["base_result"]["contributions"]
    kurtosis_info = sim["base_result"]["kurtosis_info"]

    sex_label = "Female" if inputs["sex"] == 1 else "Male"
    eth_map   = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    eth_label = eth_map.get(int(inputs["ethnicity"]), "Unknown")

    label_map = {
        "sdr_ratio":        "SDR Ratio",
        "sdr_kurtosis":     "SDR Kurtosis",
        "age":              "Age",
        "bmi":              "BMI",
        "sonication_power": "Sonication Power",
        "skull_volume":     "Skull Volume",
        "baseline_tremor":  "Baseline Tremor",
        "sex":              "Sex",
        "ethnicity":        "Ethnicity",
    }

    shap_order  = sorted(contribs, key=lambda f: abs(contribs[f]))
    shap_labels = [label_map[f] for f in shap_order]
    shap_values = [contribs[f] for f in shap_order]
    shap_colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_values]

    pass_nums   = [p["pass_number"]         for p in passes]
    pass_energy = [p["pass_energy_j"]       for p in passes]
    pass_cum    = [p["cumulative_energy_j"] for p in passes]
    pass_cem    = [p["cem43"]               for p in passes]

    # ── Layout: 4 rows x 3 cols ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 17))
    gs  = gridspec.GridSpec(
        4, 3,
        hspace=0.55, wspace=0.38,
        height_ratios=[1.1, 1.0, 1.0, 0.85]
    )

    ax_profile  = fig.add_subplot(gs[:, 0])     # Left col: patient profile
    ax_temp     = fig.add_subplot(gs[0, 1:])    # Row 0: temperature curve
    ax_kurtosis = fig.add_subplot(gs[1, 1])     # Row 1 left: kurtosis panel
    ax_energy   = fig.add_subplot(gs[1, 2])     # Row 1 right: per-pass energy
    ax_cum      = fig.add_subplot(gs[2, 1])     # Row 2 left: cumulative energy
    ax_cem      = fig.add_subplot(gs[2, 2])     # Row 2 right: CEM43
    ax_shap     = fig.add_subplot(gs[3, 1:])    # Row 3: SHAP full width

    # ══ Patient Profile Panel ═════════════════════════════════════════════════
    ax_profile.set_xlim(0, 1)
    ax_profile.set_ylim(0, 1)
    ax_profile.axis("off")

    ax_profile.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.91), 1.0, 0.09,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#2c3e50"
    ))
    ax_profile.text(0.5, 0.955, "PATIENT PROFILE",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

    # Kurtosis tier badge inline in profile
    kt_color = kurtosis_info["color"]
    kt_tier  = kurtosis_info["tier"]

    profile_rows = [
        ("Age",              f"{inputs['age']:.0f} yrs"),
        ("Sex",              sex_label),
        ("BMI",              f"{inputs['bmi']:.1f} kg/m²"),
        ("Ethnicity",        eth_label),
        ("SDR Ratio",        f"{inputs['sdr_ratio']:.3f}"),
        ("SDR Kurtosis",     f"{inputs['sdr_kurtosis']:.3f}  [{kt_tier}]"),
        ("Skull Volume",     f"{inputs['skull_volume']:.0f} mm³"),
        ("Baseline Tremor",  f"{inputs['baseline_tremor']:.1f} CRST"),
        ("Sonication Power", f"{inputs['sonication_power']:.0f} W"),
        ("No. of Passes",    str(sim["n_passes"])),
        ("Ablation Hold",    f"{sim['ablation_duration']:.0f} s"),
        ("Cooling Window",   f"{sim['cooling_time']:.0f} s"),
        ("Total Energy",     f"{sim['total_energy_j']:.1f} J"),
        ("Total CEM43",      f"{sim['total_cem43']:.2f}"),
        ("Procedure Time",   f"{sim['total_time_s']/60:.1f} min"),
        ("Tremor Prognosis", kurtosis_info["tremor_prognosis"]),
    ]

    y_start = 0.88
    row_h   = 0.054

    for i, (field, value) in enumerate(profile_rows):
        y  = y_start - i * row_h
        bg = "#f0f4f8" if i % 2 == 0 else "white"

        # Highlight kurtosis row with tier color
        if field == "SDR Kurtosis":
            bg = kt_color + "22"

        ax_profile.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.028), 1.0, row_h,
            boxstyle="square,pad=0.0",
            linewidth=0, facecolor=bg
        ))
        ax_profile.text(0.04, y + 0.001, field,
                        ha="left", va="center",
                        fontsize=8.5, color="#555555")

        # Color-code kurtosis value
        val_color = kt_color if field == "SDR Kurtosis" else "#222222"
        ax_profile.text(0.96, y + 0.001, value,
                        ha="right", va="center",
                        fontsize=8.5, fontweight="bold", color=val_color)

    ax_profile.text(0.5, 0.01,
                    "⚠ Simulated data — not for clinical use",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#e74c3c", style="italic")

    # ══ Temperature Time-Series ════════════════════════════════════════════════
    ax_temp.plot(time_series, temp_series,
                 color="#e74c3c", linewidth=1.4, alpha=0.85, zorder=3)
    ax_temp.axhline(ABLATION_TEMP_C, color="#c0392b", linewidth=1.2,
                    linestyle="--", label=f"Ablation threshold ({ABLATION_TEMP_C}°C)")
    ax_temp.axhline(BASELINE_TEMP_C, color="#3498db", linewidth=1.0,
                    linestyle=":", label=f"Body temp ({BASELINE_TEMP_C}°C)")
    ax_temp.fill_between(time_series, ABLATION_TEMP_C, temp_series,
                         where=(temp_series >= ABLATION_TEMP_C),
                         color="#e74c3c", alpha=0.18, label="Ablation zone")

    t_cursor = 0.0
    for p in passes:
        t_cursor += p["total_pass_time_s"] / 60
        ax_temp.axvline(t_cursor, color="#888888", linewidth=0.8,
                        linestyle="--", alpha=0.6)
        ax_temp.text(
            t_cursor - p["total_pass_time_s"] / 120,
            ax_temp.get_ylim()[1] if ax_temp.get_ylim()[1] != 1 else 75,
            f"P{p['pass_number']}",
            ha="center", va="top", fontsize=8, color="#555555"
        )

    ax_temp.set_xlabel("Time (min)", fontsize=10)
    ax_temp.set_ylabel("Tissue Temperature (°C)", fontsize=10)
    ax_temp.set_title("Tissue Temperature Curve — All Passes",
                      fontsize=11, fontweight="bold")
    ax_temp.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)
    ax_temp.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_temp.set_axisbelow(True)

    # ══ SDR Kurtosis Visual Panel ══════════════════════════════════════════════
    ax_kurtosis.set_xlim(0, 1)
    ax_kurtosis.set_ylim(0, 1)
    ax_kurtosis.axis("off")
    ax_kurtosis.set_title("SDR Kurtosis Assessment", fontsize=11, fontweight="bold")

    # Simulated SDR histogram shape based on kurtosis tier
    x = np.linspace(0, 1, 300)
    k = inputs["sdr_kurtosis"]

    # Gaussian with width shaped by kurtosis
    sigma  = max(0.06, 0.18 - k * 0.04)
    mu     = inputs["sdr_ratio"]
    y_dist = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y_dist /= y_dist.max()

    # Inset axes for the histogram shape
    ax_hist = ax_kurtosis.inset_axes([0.05, 0.42, 0.90, 0.50])
    ax_hist.fill_between(x, y_dist, alpha=0.4, color=kt_color)
    ax_hist.plot(x, y_dist, color=kt_color, linewidth=2.0)
    ax_hist.axvline(SDR_KURTOSIS_CUTOFF + 0.32 + mu - 0.45,
                    color="#888888", linewidth=1.0,
                    linestyle="--", alpha=0.7)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("SDR Value", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.tick_params(labelsize=7)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.set_title("SDR Distribution Shape", fontsize=8.5, pad=3)

    # Kurtosis value and tier text
    ax_kurtosis.text(0.5, 0.36,
                     f"Kurtosis = {k:.3f}",
                     ha="center", va="center",
                     fontsize=13, fontweight="bold", color=kt_color)
    ax_kurtosis.text(0.5, 0.26,
                     f"Tier:  {kt_tier}",
                     ha="center", va="center",
                     fontsize=11, color=kt_color)
    ax_kurtosis.text(0.5, 0.13,
                     kurtosis_info["tremor_prognosis"],
                     ha="center", va="center",
                     fontsize=8.5, color="#333333",
                     style="italic")
    ax_kurtosis.text(0.5, 0.04,
                     f"Cutoff threshold:  > {SDR_KURTOSIS_CUTOFF}",
                     ha="center", va="center",
                     fontsize=8, color="#888888")

    # ══ Per-Pass Energy ════════════════════════════════════════════════════════
    bar_colors = ["#2ecc71" if p["ablation_achieved"] else "#e74c3c" for p in passes]
    ax_energy.bar(pass_nums, pass_energy, color=bar_colors,
                  edgecolor="none", width=0.6)
    for pn, pe in zip(pass_nums, pass_energy):
        ax_energy.text(pn, pe + 8, f"{pe:.0f}J",
                       ha="center", va="bottom", fontsize=8.5)

    achieved_patch = mpatches.Patch(color="#2ecc71", label="Ablation achieved")
    missed_patch   = mpatches.Patch(color="#e74c3c", label="Ablation not reached")
    ax_energy.legend(handles=[achieved_patch, missed_patch],
                     fontsize=8, loc="upper left", framealpha=0.85)
    ax_energy.set_xlabel("Pass Number", fontsize=10)
    ax_energy.set_ylabel("Energy (J)", fontsize=10)
    ax_energy.set_title("Energy per Pass", fontsize=11, fontweight="bold")
    ax_energy.set_xticks(pass_nums)
    ax_energy.spines["top"].set_visible(False)
    ax_energy.spines["right"].set_visible(False)
    ax_energy.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_energy.set_axisbelow(True)

    # ══ Cumulative Energy ══════════════════════════════════════════════════════
    ax_cum.plot(pass_nums, pass_cum, color="#8e44ad",
                linewidth=2.2, marker="o", markersize=7, zorder=3)
    for pn, pc in zip(pass_nums, pass_cum):
        ax_cum.text(pn, pc + 25, f"{pc:.0f}J",
                    ha="center", va="bottom", fontsize=8.5, color="#8e44ad")
    ax_cum.fill_between(pass_nums, pass_cum, alpha=0.12, color="#8e44ad")
    ax_cum.set_xlabel("Pass Number", fontsize=10)
    ax_cum.set_ylabel("Cumulative Energy (J)", fontsize=10)
    ax_cum.set_title("Cumulative Energy Delivered", fontsize=11, fontweight="bold")
    ax_cum.set_xticks(pass_nums)
    ax_cum.spines["top"].set_visible(False)
    ax_cum.spines["right"].set_visible(False)
    ax_cum.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_cum.set_axisbelow(True)

    # ══ CEM43 ══════════════════════════════════════════════════════════════════
    ax_cem.bar(pass_nums, pass_cem, color="#e67e22",
               edgecolor="none", width=0.6)
    ax_cem.axhline(240, color="#c0392b", linewidth=1.1,
                   linestyle="--", label="Safety limit (240 CEM43)")
    for pn, pc in zip(pass_nums, pass_cem):
        ax_cem.text(pn, pc + 1, f"{pc:.1f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax_cem.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_cem.set_xlabel("Pass Number", fontsize=10)
    ax_cem.set_ylabel("CEM43 (min)", fontsize=10)
    ax_cem.set_title("Thermal Dose per Pass (CEM43)", fontsize=11, fontweight="bold")
    ax_cem.set_xticks(pass_nums)
    ax_cem.spines["top"].set_visible(False)
    ax_cem.spines["right"].set_visible(False)
    ax_cem.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_cem.set_axisbelow(True)

    # ══ SHAP Bar Chart (full width bottom) ════════════════════════════════════
    bars = ax_shap.barh(shap_labels, shap_values, color=shap_colors,
                        edgecolor="none", height=0.6)
    ax_shap.axvline(0, color="black", linewidth=1.0)
    for i, (label, val) in enumerate(zip(shap_labels, shap_values)):
        offset = 0.15 if val >= 0 else -0.15
        ha     = "left" if val >= 0 else "right"
        ax_shap.text(val + offset, i, f"{val:+.1f}J",
                     va="center", ha=ha, fontsize=9)

    # Highlight SDR kurtosis bar label
    for i, label in enumerate(shap_labels):
        if label == "SDR Kurtosis":
            ax_shap.get_yticklabels()[i].set_color(kt_color)
            ax_shap.get_yticklabels()[i].set_fontweight("bold")

    pos_patch = mpatches.Patch(color="#d73027", label="↑ Increases energy output")
    neg_patch = mpatches.Patch(color="#4575b4", label="↓ Decreases energy output")
    ax_shap.legend(handles=[pos_patch, neg_patch],
                   fontsize=9, loc="lower right", framealpha=0.9)
    ax_shap.set_xlabel("SHAP Contribution to Predicted Energy Output (J)", fontsize=10)
    ax_shap.set_title("Feature Contributions (SHAP) — All Features",
                      fontsize=11, fontweight="bold")
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    ax_shap.spines["left"].set_visible(False)
    ax_shap.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax_shap.set_axisbelow(True)

    # ══ Super title ════════════════════════════════════════════════════════════
    fig.suptitle(
        "MRgFUS Per-Pass Energy & Thermal Ablation Tool  |  University of Florida\n"
        "⚠  Simulated Data — Not for Clinical Use",
        fontsize=13, fontweight="bold", y=1.01, color="#2c3e50"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ── 6. Main pipeline ───────────────────────────────────────────────────────────
def run_per_pass_tool(
    sdr_ratio, sdr_kurtosis, age, bmi, sonication_power,
    skull_volume, baseline_tremor, sex, ethnicity,
    n_passes:          int   = 5,
    ablation_duration: float = 20.0,
    cooling_time:      float = 40.0,
    power_ramp:        float = 0.05,
    targeting_adjust:  float = 0.03,
    save_plot:         str   = None,
):
    """
    Full pipeline: estimate base energy → simulate passes → report → plot.

    Parameters
    ----------
    sdr_kurtosis : float
        Kurtosis of the patient's SDR histogram.
        Cutoff > -0.32 identifies favorable MRgFUS candidates.
        Higher kurtosis = sharper distribution = better skull homogeneity
        = lower energy demand and superior long-term tremor outcomes.
    ablation_duration : float
        Seconds tissue is held at or above ablation temperature per pass.
    cooling_time : float
        Seconds of cooling between passes.
    """
    base = estimate_base_energy(
        sdr_ratio=sdr_ratio, sdr_kurtosis=sdr_kurtosis,
        age=age, bmi=bmi, sonication_power=sonication_power,
        skull_volume=skull_volume, baseline_tremor=baseline_tremor,
        sex=sex, ethnicity=ethnicity,
    )
    sim = simulate_passes(
        base_result       = base,
        n_passes          = n_passes,
        ablation_duration = ablation_duration,
        cooling_time      = cooling_time,
        power_ramp        = power_ramp,
        targeting_adjust  = targeting_adjust,
    )
    print(interpret_passes(sim))
    plot_per_pass(sim, save_path=save_plot)
    return sim


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE PATIENTS — demonstrating kurtosis tier differences
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n--- PATIENT A: Favorable kurtosis (above cutoff) ---")
    run_per_pass_tool(
        sdr_ratio         = 0.38,    # Below standard mean SDR cutoff
        sdr_kurtosis      = 0.12,    # But favorable kurtosis — good candidate
        age               = 72,
        bmi               = 31.2,
        sonication_power  = 62,
        skull_volume      = 138,
        baseline_tremor   = 22,
        sex               = 0,
        ethnicity         = 0,
        n_passes          = 5,
        ablation_duration = 20.0,
        cooling_time      = 40.0,
        save_plot         = "patient_A_favorable_kurtosis.png"
    )

    print("\n--- PATIENT B: Unfavorable kurtosis (below cutoff) ---")
    run_per_pass_tool(
        sdr_ratio         = 0.41,    # Mean SDR within standard range
        sdr_kurtosis      = -0.85,   # But unfavorable kurtosis
        age               = 68,
        bmi               = 27.5,
        sonication_power  = 58,
        skull_volume      = 150,
        baseline_tremor   = 19,
        sex               = 1,
        ethnicity         = 1,
        n_passes          = 5,
        ablation_duration = 20.0,
        cooling_time      = 40.0,
        save_plot         = "patient_B_unfavorable_kurtosis.png"
    )

