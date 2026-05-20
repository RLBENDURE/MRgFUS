#File 6:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE: MRgFUS Per-Pass Energy Sonication Estimator
#          & Clinical Interpretation Tool
#  Institution: University of Florida
#  Purpose:     Surgical decision support
#  Data:        Simulated — not for clinical use
# ─────────────────────────────────────────────────────────────────────────────

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

BASE_ENERGY_J     = 10000
ABLATION_TEMP_C   = 55.0    # Threshold for thermal ablation (°C)
BASELINE_TEMP_C   = 37.0    # Body temperature
COOLING_RATE      = 0.18    # °C/sec cooling after sonication ends
HEATING_RATE_BASE = 0.22    # °C/J heating rate (base)


# ── 1. Base energy estimator ───────────────────────────────────────────────────
def estimate_base_energy(
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
    return {"base_energy_j": predicted_energy,
            "contributions": contributions,
            "inputs": inputs}


# ── 2. Per-pass thermal simulation ────────────────────────────────────────────
def simulate_passes(
    base_result:       dict,
    n_passes:          int   = 5,
    ablation_duration: float = 20.0,   # seconds to hold ablation temp
    cooling_time:      float = 40.0,   # seconds of cooling between passes
    power_ramp:        float = -0.1,   # fractional power increase per pass
                                       # to account for tissue changes
    targeting_adjust:  float = 0.03,   # fractional energy correction per pass
                                       # simulating improved targeting accuracy
    seed:              int   = 42,
) -> dict:
    """
    Simulates n_passes of sonication for a single patient.

    Each pass:
      - Ramps energy slightly (tissue changes + improved targeting)
      - Simulates a temperature curve: heating → ablation hold → cooling
      - Tracks cumulative energy, thermal dose, and time-at-temperature

    Returns per-pass data and full temperature time-series.
    """
    np.random.seed(seed)

    base_energy     = base_result["base_energy_j"]
    sonication_power = base_result["inputs"]["sonication_power"]
    heating_rate    = HEATING_RATE_BASE / (1 + 0.1 * base_result["inputs"]["bmi"] / 27)

    passes          = []
    cumulative_e    = 0.0
    cumulative_cem43 = 0.0     # Cumulative Equivalent Minutes at 43°C (thermal dose)
    all_time        = []
    all_temp        = []
    global_t        = 0.0

    for p in range(1, n_passes + 1):

        # Energy for this pass: increases slightly each pass
        pass_noise  = np.random.normal(0, 0.025)
        pass_energy = round(
            base_energy * (1 + power_ramp * (p - 1))
                        * (1 - targeting_adjust * (p - 1))
                        + pass_noise * base_energy,
            1
        )
        pass_energy = max(pass_energy, 200.0)   # floor

        # ── Heating phase ──────────────────────────────────────────────────
        # Time to reach ablation temp from current tissue temp
        start_temp  = BASELINE_TEMP_C + np.random.normal(0, 0.5)
        temp_to_rise = ABLATION_TEMP_C - start_temp
        heat_duration = temp_to_rise / (heating_rate * sonication_power / 10)
        heat_duration = max(heat_duration, 5.0)

        heat_t    = np.linspace(0, heat_duration, int(heat_duration * 5))
        heat_temp = start_temp + (temp_to_rise) * (1 - np.exp(-heat_t / (heat_duration * 0.45)))
        heat_temp += np.random.normal(0, 0.3, len(heat_t))   # MR thermometry noise

        # ── Ablation hold phase ────────────────────────────────────────────
        hold_t    = np.linspace(0, ablation_duration, int(ablation_duration * 5))
        hold_temp = ABLATION_TEMP_C + np.random.normal(0, 0.8, len(hold_t))

        # ── Cooling phase ──────────────────────────────────────────────────
        cool_t    = np.linspace(0, cooling_time, int(cooling_time * 5))
        cool_temp = ABLATION_TEMP_C * np.exp(-COOLING_RATE * cool_t / ABLATION_TEMP_C * 10)
        cool_temp = np.clip(cool_temp, BASELINE_TEMP_C, ABLATION_TEMP_C)
        cool_temp += np.random.normal(0, 0.3, len(cool_t))

        # Full pass temperature curve
        pass_temp = np.concatenate([heat_temp, hold_temp, cool_temp])
        pass_time = np.linspace(0, heat_duration + ablation_duration + cooling_time,
                                len(pass_temp))

        # Append to global timeline
        all_time.extend(global_t + pass_time)
        all_temp.extend(pass_temp)
        global_t += heat_duration + ablation_duration + cooling_time

        # ── Thermal dose: CEM43 ────────────────────────────────────────────
        # CEM43 = sum(dt * R^(43-T)) where R=0.5 for T>=43, R=0.25 for T<43
        dt_s = (heat_duration + ablation_duration + cooling_time) / len(pass_temp)
        cem43 = 0.0
        for t_val in pass_temp:
            R = 0.5 if t_val >= 43 else 0.25
            cem43 += dt_s / 60 * (R ** (43 - t_val))
        cumulative_cem43 += cem43

        cumulative_e += pass_energy

        # Peak temp this pass
        peak_temp = float(np.max(pass_temp))

        # Ablation achieved?
        ablation_achieved = peak_temp >= ABLATION_TEMP_C

        passes.append({
            "pass_number":        p,
            "pass_energy_j":      pass_energy,
            "cumulative_energy_j": round(cumulative_e, 1),
            "heat_duration_s":    round(heat_duration, 1),
            "ablation_hold_s":    ablation_duration,
            "cooling_time_s":     cooling_time,
            "total_pass_time_s":  round(heat_duration + ablation_duration + cooling_time, 1),
            "peak_temp_c":        round(peak_temp, 1),
            "ablation_achieved":  ablation_achieved,
            "cem43":              round(cem43, 2),
            "cumulative_cem43":   round(cumulative_cem43, 2),
        })

    return {
        "passes":           passes,
        "n_passes":         n_passes,
        "ablation_duration": ablation_duration,
        "cooling_time":     cooling_time,
        "total_energy_j":   round(cumulative_e, 1),
        "total_cem43":      round(cumulative_cem43, 2),
        "total_time_s":     round(global_t, 1),
        "time_series":      np.array(all_time),
        "temp_series":      np.array(all_temp),
        "base_result":      base_result,
    }


# ── 3. Clinical report ─────────────────────────────────────────────────────────
def interpret_passes(sim: dict) -> str:
    passes  = sim["passes"]
    inputs  = sim["base_result"]["inputs"]
    sex_label = "Female" if inputs["sex"] == 1 else "Male"
    eth_map   = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    eth_label = eth_map.get(int(inputs["ethnicity"]), "Unknown")

    lines = [
        "=" * 65,
        "  MRgFUS PER-PASS ENERGY & THERMAL DOSE REPORT",
        "  ⚠  SIMULATED DATA — NOT FOR CLINICAL USE",
        "=" * 65,
        f"  Patient:  Age {inputs['age']:.0f} | {sex_label} | "
        f"BMI {inputs['bmi']:.1f} | {eth_label}",
        f"  SDR: {inputs['sdr_ratio']:.3f} | "
        f"Skull Vol: {inputs['skull_volume']:.0f} mm³ | "
        f"Tremor: {inputs['baseline_tremor']:.1f} CRST",
        f"  Ablation Hold: {sim['ablation_duration']:.0f} s  |  "
        f"Cooling Window: {sim['cooling_time']:.0f} s",
        "-" * 65,
        f"  {'Pass':<6} {'Energy (J)':<13} {'Cumul. (J)':<13} "
        f"{'Peak °C':<10} {'CEM43':<10} {'Ablation'}",
        "-" * 65,
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
        "-" * 65,
        f"  Total Energy Delivered:   {sim['total_energy_j']:.1f} J",
        f"  Total Thermal Dose:       {sim['total_cem43']:.2f} CEM43",
        f"  Total Procedure Time:     {sim['total_time_s'] / 60:.1f} min",
        "=" * 65,
    ]
    return "\n".join(lines)


# ── 4. Full figure ─────────────────────────────────────────────────────────────
def plot_per_pass(sim: dict, save_path: str = None):

    passes      = sim["passes"]
    time_series = sim["time_series"] / 60     # convert to minutes
    temp_series = sim["temp_series"]
    inputs      = sim["base_result"]["inputs"]
    contribs    = sim["base_result"]["contributions"]

    sex_label = "Female" if inputs["sex"] == 1 else "Male"
    eth_map   = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    eth_label = eth_map.get(int(inputs["ethnicity"]), "Unknown")

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

    shap_order  = sorted(contribs, key=lambda f: abs(contribs[f]))
    shap_labels = [label_map[f] for f in shap_order]
    shap_values = [contribs[f] for f in shap_order]
    shap_colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_values]

    pass_nums    = [p["pass_number"]        for p in passes]
    pass_energy  = [p["pass_energy_j"]      for p in passes]
    pass_cum     = [p["cumulative_energy_j"] for p in passes]
    pass_peak    = [p["peak_temp_c"]        for p in passes]
    pass_cem     = [p["cem43"]              for p in passes]

    # ── Figure layout: 3 rows x 3 cols ────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(
        3, 3,
        hspace=0.52, wspace=0.38,
        height_ratios=[1.1, 1.1, 1.0]
    )

    ax_profile  = fig.add_subplot(gs[:, 0])      # Left column: patient profile
    ax_temp     = fig.add_subplot(gs[0, 1:])     # Top right: temperature curve
    ax_energy   = fig.add_subplot(gs[1, 1])      # Mid center: per-pass energy
    ax_cum      = fig.add_subplot(gs[1, 2])      # Mid right: cumulative energy
    ax_shap     = fig.add_subplot(gs[2, 1])      # Bottom center: SHAP bars
    ax_cem      = fig.add_subplot(gs[2, 2])      # Bottom right: CEM43

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

    profile_rows = [
        ("Age",              f"{inputs['age']:.0f} yrs"),
        ("Sex",              sex_label),
        ("BMI",              f"{inputs['bmi']:.1f} kg/m²"),
        ("Ethnicity",        eth_label),
        ("SDR Ratio",        f"{inputs['sdr_ratio']:.3f}"),
        ("Skull Volume",     f"{inputs['skull_volume']:.0f} mm³"),
        ("Baseline Tremor",  f"{inputs['baseline_tremor']:.1f} CRST"),
        ("Sonication Power", f"{inputs['sonication_power']:.0f} W"),
        ("No. of Passes",    str(sim["n_passes"])),
        ("Ablation Hold",    f"{sim['ablation_duration']:.0f} s"),
        ("Cooling Window",   f"{sim['cooling_time']:.0f} s"),
        ("Total Energy",     f"{sim['total_energy_j']:.1f} J"),
        ("Total CEM43",      f"{sim['total_cem43']:.2f}"),
        ("Procedure Time",   f"{sim['total_time_s']/60:.1f} min"),
    ]

    y_start = 0.88
    row_h   = 0.062

    for i, (field, value) in enumerate(profile_rows):
        y   = y_start - i * row_h
        bg  = "#f0f4f8" if i % 2 == 0 else "white"
        ax_profile.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.032), 1.0, row_h,
            boxstyle="square,pad=0.0",
            linewidth=0, facecolor=bg
        ))
        ax_profile.text(0.04, y + 0.001, field,
                        ha="left", va="center",
                        fontsize=9, color="#555555")
        ax_profile.text(0.96, y + 0.001, value,
                        ha="right", va="center",
                        fontsize=9, fontweight="bold", color="#222222")

    # Total energy risk indicator
    total_e = sim["total_energy_j"]
    if total_e < 2400:
        rc, rl = "#2ecc71", "LOW"
    elif total_e < 3500:
        rc, rl = "#f39c12", "MODERATE"
    else:
        rc, rl = "#e74c3c", "HIGH"

    ax_profile.add_patch(mpatches.FancyBboxPatch(
        (0.1, 0.01), 0.80, 0.085,
        boxstyle="round,pad=0.03",
        linewidth=2, edgecolor=rc, facecolor=rc + "28"
    ))
    ax_profile.text(0.5, 0.072, "Overall Risk",
                    ha="center", va="center", fontsize=8.5, color="#555555")
    ax_profile.text(0.5, 0.032, rl,
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color=rc)

    ax_profile.text(0.5, -0.02,
                    "⚠ Simulated — not for clinical use",
                    ha="center", va="bottom",
                    fontsize=7, color="#e74c3c", style="italic")

    # ══ Temperature Time-Series ════════════════════════════════════════════════
    ax_temp.plot(time_series, temp_series,
                 color="#e74c3c", linewidth=1.4, alpha=0.85, zorder=3)
    ax_temp.axhline(ABLATION_TEMP_C, color="#c0392b", linewidth=1.2,
                    linestyle="--", label=f"Ablation threshold ({ABLATION_TEMP_C}°C)")
    ax_temp.axhline(BASELINE_TEMP_C, color="#3498db", linewidth=1.0,
                    linestyle=":", label=f"Body temp ({BASELINE_TEMP_C}°C)")

    # Shade ablation zones
    ax_temp.fill_between(time_series, ABLATION_TEMP_C, temp_series,
                         where=(temp_series >= ABLATION_TEMP_C),
                         color="#e74c3c", alpha=0.18, label="Ablation zone")

    # Pass boundary markers
    t_cursor = 0.0
    for p in passes:
        t_cursor += p["total_pass_time_s"] / 60
        ax_temp.axvline(t_cursor, color="#888888", linewidth=0.8,
                        linestyle="--", alpha=0.6)
        ax_temp.text(t_cursor - p["total_pass_time_s"] / 120,
                     ax_temp.get_ylim()[1] if ax_temp.get_ylim()[1] != 1 else 75,
                     f"P{p['pass_number']}",
                     ha="center", va="top", fontsize=8, color="#555555")

    ax_temp.set_xlabel("Time (min)", fontsize=10)
    ax_temp.set_ylabel("Tissue Temperature (°C)", fontsize=10)
    ax_temp.set_title("Tissue Temperature Curve — All Passes", fontsize=11, fontweight="bold")
    ax_temp.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)
    ax_temp.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_temp.set_axisbelow(True)

    # ══ Per-Pass Energy Bar Chart ══════════════════════════════════════════════
    bar_colors = ["#2ecc71" if p["ablation_achieved"] else "#e74c3c" for p in passes]
    ax_energy.bar(pass_nums, pass_energy, color=bar_colors,
                  edgecolor="none", width=0.6)

    for i, (pn, pe) in enumerate(zip(pass_nums, pass_energy)):
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

    # ══ Cumulative Energy Line ══════════════════════════════════════════════════
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

    # ══ SHAP Contribution Bar Chart ════════════════════════════════════════════
    ax_shap.barh(shap_labels, shap_values, color=shap_colors,
                 edgecolor="none", height=0.6)
    ax_shap.axvline(0, color="black", linewidth=1.0)

    for label, val in zip(shap_labels, shap_values):
        offset = 0.15 if val >= 0 else -0.15
        ha     = "left" if val >= 0 else "right"
        ax_shap.text(val + offset, shap_labels.index(label),
                     f"{val:+.1f}J",
                     va="center", ha=ha, fontsize=8)

    pos_patch = mpatches.Patch(color="#d73027", label="↑ Increases energy")
    neg_patch = mpatches.Patch(color="#4575b4", label="↓ Decreases energy")
    ax_shap.legend(handles=[pos_patch, neg_patch],
                   fontsize=8, loc="lower right", framealpha=0.9)
    ax_shap.set_xlabel("SHAP Contribution (J)", fontsize=10)
    ax_shap.set_title("Feature Contributions (SHAP)", fontsize=11, fontweight="bold")
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    ax_shap.spines["left"].set_visible(False)
    ax_shap.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax_shap.set_axisbelow(True)

    # ══ CEM43 Thermal Dose per Pass ════════════════════════════════════════════
    ax_cem.bar(pass_nums, pass_cem, color="#e67e22",
               edgecolor="none", width=0.6)

    # CEM43 safety threshold reference (240 min is common brain tissue limit)
    cem_threshold = 240
    ax_cem.axhline(cem_threshold, color="#c0392b", linewidth=1.1,
                   linestyle="--", label=f"Safety limit ({cem_threshold} CEM43)")

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


# ── 5. Main pipeline ───────────────────────────────────────────────────────────
def run_per_pass_tool(
    sdr_ratio, age, bmi, sonication_power,
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
    ablation_duration : float
        Seconds tissue is held at or above ablation temperature per pass.
    cooling_time : float
        Seconds of cooling between passes.
    n_passes : int
        Number of sonication passes planned.
    power_ramp : float
        Fractional power increase per pass (tissue impedance changes).
    targeting_adjust : float
        Fractional energy efficiency gain per pass (improved targeting).
    """
    base = estimate_base_energy(
        sdr_ratio=sdr_ratio, age=age, bmi=bmi,
        sonication_power=sonication_power,
        skull_volume=skull_volume,
        baseline_tremor=baseline_tremor,
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
#  EXAMPLE PATIENT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_per_pass_tool(
        sdr_ratio         = 0.38,
        age               = 72,
        bmi               = 31.2,
        sonication_power  = 62,
        skull_volume      = 138,
        baseline_tremor   = 22,
        sex               = 0,          # Male
        ethnicity         = 0,          # White
        n_passes          = 5,
        ablation_duration = 20.0,       # hold ablation temp for 20 seconds
        cooling_time      = 40.0,       # 40 second cooling window between passes
        power_ramp        = 0.05,       # 5% power increase per pass
        targeting_adjust  = 0.03,       # 3% efficiency gain per pass
        save_plot         = "mrgfus_per_pass_report.png"
    )


