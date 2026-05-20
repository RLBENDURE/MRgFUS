#File 9:

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
#  Thermal Protocol Reference:
#  - Cooling medium: degassed water at 15°C circulated via silicone membrane
#  - Scalp temp maintained <19°C during sonication
#  - Post high-energy (≥40 kJ) sonication cooling: ~20 min
#  - Low-temp protocol (50–54°C) total procedure: 115 ± 34 min
#  - Therapeutic target: 55–60°C at Vim nucleus
#  - Average sonications: 10.0 ± 2.6
#  - Peak temps: 57.0–62.4°C
#  - Thermal dose target: 17 CEM43
#  - SDR Kurtosis cutoff: > -0.32 for favorable candidacy
# ─────────────────────────────────────────────────────────────────────────────

# ── Constants ──────────────────────────────────────────────────────────────────
SHAP_WEIGHTS = {
    "sdr_ratio":        -0.42,
    "sdr_kurtosis":     -0.35,
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
    "sdr_kurtosis":     -0.20,
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
    "sdr_kurtosis":      0.45,
    "age":               9.0,
    "bmi":               5.0,
    "sonication_power":  10.0,
    "skull_volume":      25.0,
    "baseline_tremor":   4.0,
    "sex":               0.48,
    "ethnicity":         1.1,
}

BASE_ENERGY_J         = 10000

# ── Clinically-grounded thermal constants ─────────────────────────────────────
BODY_TEMP_C           = 37.0
SCALP_MAX_TEMP_C      = 19.0    # Scalp must stay below this
COOLING_WATER_TEMP_C  = 15.0    # Degassed circulating water
TARGET_TEMP_LOW_C     = 55.0    # Therapeutic range floor
TARGET_TEMP_HIGH_C    = 60.0    # Therapeutic range ceiling
CONFIRM_TEMP_C        = 45.0    # Low-power targeting confirmation sonication
HIGH_ENERGY_THRESHOLD = 40000   # Joules — triggers 20-min skull cooling
SKULL_COOLING_TIME_S  = 1200    # 20 minutes in seconds (post high-energy)
STANDARD_COOLING_S    = 90      # Standard inter-pass cooling (seconds)
CEM43_TARGET          = 17.0    # Clinical thermal dose target
SDR_KURTOSIS_CUTOFF   = -0.32
HEATING_RATE_BASE     = 0.22


# ── 1. SDR kurtosis interpreter ────────────────────────────────────────────────
def interpret_sdr_kurtosis(kurtosis: float) -> dict:
    favorable = kurtosis > SDR_KURTOSIS_CUTOFF

    if kurtosis > 0.50:
        tier, color          = "EXCELLENT", "#27ae60"
        description          = (
            "Highly peaked SDR distribution.\n"
            "Expect efficient sonication, lower\n"
            "peak temps, and strong long-term\n"
            "tremor improvement (>55%)."
        )
        energy_modifier      = -0.12
        tremor_prognosis     = "55–62% CRST improvement @ 12 months"
    elif kurtosis > SDR_KURTOSIS_CUTOFF:
        tier, color          = "FAVORABLE", "#2ecc71"
        description          = (
            "Above-threshold kurtosis.\n"
            "Suitable candidate even if mean\n"
            "SDR is below 0.40.\n"
            "Expected improvement: 44–55%."
        )
        energy_modifier      = -0.06
        tremor_prognosis     = "44–55% CRST improvement @ 12 months"
    elif kurtosis > -0.60:
        tier, color          = "BORDERLINE", "#f39c12"
        description          = (
            "Below cutoff. Broader SDR\n"
            "distribution with low-SDR tail.\n"
            "Additional CT skull analysis\n"
            "recommended before proceeding."
        )
        energy_modifier      = 0.04
        tremor_prognosis     = "30–44% CRST improvement @ 12 months"
    else:
        tier, color          = "UNFAVORABLE", "#e74c3c"
        description          = (
            "Flat, broad SDR distribution.\n"
            "Significant low-SDR tail.\n"
            "Higher energy demand & elevated\n"
            "thermal risk. MDT review advised."
        )
        energy_modifier      = 0.10
        tremor_prognosis     = "<30% CRST improvement @ 12 months"

    return {
        "kurtosis":         kurtosis,
        "tier":             tier,
        "color":            color,
        "favorable":        favorable,
        "description":      description,
        "energy_modifier":  energy_modifier,
        "tremor_prognosis": tremor_prognosis,
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
        "sonication_power": sonication_power,
        "skull_volume":     skull_volume,
        "baseline_tremor":  baseline_tremor,
        "sex":              float(sex),
        "ethnicity":        float(ethnicity),
    }

    kurtosis_info    = interpret_sdr_kurtosis(sdr_kurtosis)
    contributions    = {}
    total_adjustment = 0.0

    for feature, value in inputs.items():
        z = (value - POP_MEANS[feature]) / POP_STDS[feature]
        contribution_j = SHAP_WEIGHTS[feature] * z * BASE_ENERGY_J * 0.15
        contributions[feature] = round(contribution_j, 2)
        total_adjustment += contribution_j

    kurtosis_adjustment = BASE_ENERGY_J * kurtosis_info["energy_modifier"]
    predicted_energy    = round(BASE_ENERGY_J + total_adjustment + kurtosis_adjustment, 1)

    return {
        "base_energy_j":  predicted_energy,
        "contributions":  contributions,
        "inputs":         inputs,
        "kurtosis_info":  kurtosis_info,
    }


# ── 3. Per-pass thermal simulation ────────────────────────────────────────────
def simulate_passes(
    base_result:        dict,
    n_passes:           int   = 10,     # ~10 sonications per literature
    ablation_duration:  float = 20.0,   # seconds at target temp
    power_ramp:         float = 0.05,
    targeting_adjust:   float = 0.03,
    low_temp_protocol:  bool  = False,  # True = 50-54°C repeated protocol
    seed:               int   = 42,
) -> dict:
    """
    Simulates n_passes of sonication with clinically accurate thermal protocol:

    Pass 1-2:  Low-power confirmation sonications targeting ~45°C
    Pass 3+:   Gradual power escalation toward 55-60°C therapeutic range
    Cooling:   Standard 90s between passes; 20 min after any ≥40 kJ sonication
    Skull:     Cooled continuously by 15°C degassed water (scalp <19°C)
    Protocol:  Low-temp protocol targets 50-54°C with longer total time
    """
    np.random.seed(seed)

    base_energy      = base_result["base_energy_j"]
    sonication_power = base_result["inputs"]["sonication_power"]
    kurtosis_info    = base_result["kurtosis_info"]
    heating_rate     = HEATING_RATE_BASE / (
        1 + 0.1 * base_result["inputs"]["bmi"] / 27
    )

    # SDR ratio modifies heating efficiency
    sdr = base_result["inputs"]["sdr_ratio"]
    sdr_heating_factor = 0.7 + (sdr / 0.45) * 0.6   # lower SDR = less efficient

    # Kurtosis modifies peak temp slightly
    k_temp_mod = -kurtosis_info["energy_modifier"] * 8.0

    # Protocol-dependent target temperature
    if low_temp_protocol:
        target_temp = np.random.uniform(50.0, 54.0)
    else:
        target_temp = np.random.uniform(
            TARGET_TEMP_LOW_C, TARGET_TEMP_HIGH_C
        )

    passes           = []
    cumulative_e     = 0.0
    cumulative_cem43 = 0.0
    all_time         = []
    all_temp         = []
    all_scalp_temp   = []
    global_t         = 0.0
    skull_heat_accum = 0.0     # tracks residual skull heat across passes

    for p in range(1, n_passes + 1):

        # ── Determine pass type ────────────────────────────────────────────
        is_confirmation = p <= 2
        if is_confirmation:
            pass_target_temp = CONFIRM_TEMP_C + np.random.normal(0, 0.5)
            pass_energy      = round(base_energy * 0.25 + np.random.normal(0, 20), 1)
        else:
            # Gradual power escalation from pass 3 onward
            escalation_factor = min(1.0, 0.65 + (p - 3) * 0.07)
            pass_target_temp  = CONFIRM_TEMP_C + (
                target_temp - CONFIRM_TEMP_C
            ) * escalation_factor + np.random.normal(0, 0.8)
            pass_energy = round(
                base_energy
                * escalation_factor
                * (1 + power_ramp * (p - 3))
                * (1 - targeting_adjust * max(0, p - 3))
                + np.random.normal(0, 0.025) * base_energy,
                1
            )
        pass_energy = max(pass_energy, 150.0)

        # ── Cooling time decision ──────────────────────────────────────────
        # 20-min skull cooling required after high-energy sonications
        if cumulative_e >= HIGH_ENERGY_THRESHOLD:
            inter_pass_cooling = SKULL_COOLING_TIME_S
        else:
            inter_pass_cooling = STANDARD_COOLING_S

        # ── Heating phase ──────────────────────────────────────────────────
        start_temp   = BODY_TEMP_C + skull_heat_accum + np.random.normal(0, 0.4)
        temp_to_rise = pass_target_temp - start_temp
        heat_duration = max(
            5.0,
            temp_to_rise / (
                heating_rate * sdr_heating_factor * sonication_power / 10
            )
        )

        heat_t    = np.linspace(0, heat_duration, int(heat_duration * 5))
        heat_temp = start_temp + temp_to_rise * (
            1 - np.exp(-heat_t / (heat_duration * 0.45))
        )
        heat_temp += np.random.normal(0, 0.4, len(heat_t))

        # ── Ablation hold phase ────────────────────────────────────────────
        hold_t    = np.linspace(0, ablation_duration, int(ablation_duration * 5))
        hold_temp = pass_target_temp + np.random.normal(0, 0.8, len(hold_t))

        # ── Cooling phase (inter-pass) ─────────────────────────────────────
        cool_t = np.linspace(
            0, inter_pass_cooling, int(inter_pass_cooling * 2)
        )
        # Brain cools toward body temp; skull retains some residual heat
        # moderated by 15°C circulating water
        brain_cool_tau  = inter_pass_cooling * 0.30
        skull_cool_rate = 1.0 - np.exp(
            -inter_pass_cooling / (SKULL_COOLING_TIME_S * 0.8)
        )
        cool_temp = BODY_TEMP_C + (
            pass_target_temp - BODY_TEMP_C
        ) * np.exp(-cool_t / brain_cool_tau)
        cool_temp = np.clip(cool_temp, BODY_TEMP_C, pass_target_temp)
        cool_temp += np.random.normal(0, 0.3, len(cool_t))

        # Residual skull heat for next pass (diminished by water cooling)
        skull_heat_accum = max(
            0.0,
            skull_heat_accum * (1 - skull_cool_rate) + (
                pass_target_temp - BODY_TEMP_C
            ) * 0.08
        )

        # ── Scalp temperature simulation ───────────────────────────────────
        # Scalp is continuously cooled by 15°C water
        # Scalp temp rises slightly during sonication but stays <19°C
        full_pass_len = len(heat_temp) + len(hold_temp) + len(cool_temp)
        scalp_base    = COOLING_WATER_TEMP_C + np.random.normal(0, 0.3)
        scalp_rise    = np.linspace(0, 2.8, len(heat_temp) + len(hold_temp))
        scalp_cool_   = np.linspace(2.8, 0.0, len(cool_temp))
        scalp_profile = np.concatenate([scalp_base + scalp_rise,
                                        scalp_base + scalp_cool_])
        scalp_profile = np.clip(
            scalp_profile + np.random.normal(0, 0.2, full_pass_len),
            COOLING_WATER_TEMP_C, SCALP_MAX_TEMP_C
        )

        # ── Concatenate pass curve ─────────────────────────────────────────
        pass_temp = np.concatenate([heat_temp, hold_temp, cool_temp])
        pass_time = np.linspace(
            0,
            heat_duration + ablation_duration + inter_pass_cooling,
            len(pass_temp)
        )

        all_time.extend(global_t + pass_time)
        all_temp.extend(pass_temp)
        all_scalp_temp.extend(scalp_profile)
        global_t += heat_duration + ablation_duration + inter_pass_cooling

        # ── CEM43 thermal dose ─────────────────────────────────────────────
        dt_s  = (
            heat_duration + ablation_duration + inter_pass_cooling
        ) / len(pass_temp)
        cem43 = 0.0
        for t_val in pass_temp:
            R = 0.5 if t_val >= 43 else 0.25
            cem43 += dt_s / 60 * (R ** (43 - t_val))

        cumulative_cem43 += cem43
        cumulative_e     += pass_energy

        ablation_achieved = (
            float(np.max(pass_temp)) >= TARGET_TEMP_LOW_C
            and not is_confirmation
        )
        cem43_target_met = cumulative_cem43 >= CEM43_TARGET

        passes.append({
            "pass_number":          p,
            "pass_type":            "Confirmation" if is_confirmation else "Therapeutic",
            "pass_energy_j":        pass_energy,
            "cumulative_energy_j":  round(cumulative_e, 1),
            "target_temp_c":        round(pass_target_temp, 1),
            "peak_temp_c":          round(float(np.max(pass_temp)), 1),
            "scalp_peak_c":         round(float(np.max(scalp_profile)), 1),
            "heat_duration_s":      round(heat_duration, 1),
            "ablation_hold_s":      ablation_duration,
            "inter_pass_cooling_s": inter_pass_cooling,
            "total_pass_time_s":    round(
                heat_duration + ablation_duration + inter_pass_cooling, 1
            ),
            "ablation_achieved":    ablation_achieved,
            "cem43":                round(cem43, 4),
            "cumulative_cem43":     round(cumulative_cem43, 4),
            "cem43_target_met":     cem43_target_met,
            "skull_heat_residual":  round(skull_heat_accum, 2),
        })

    return {
        "passes":             passes,
        "n_passes":           n_passes,
        "ablation_duration":  ablation_duration,
        "low_temp_protocol":  low_temp_protocol,
        "total_energy_j":     round(cumulative_e, 1),
        "total_cem43":        round(cumulative_cem43, 4),
        "total_time_s":       round(global_t, 1),
        "cem43_target_met":   cumulative_cem43 >= CEM43_TARGET,
        "time_series":        np.array(all_time),
        "temp_series":        np.array(all_temp),
        "scalp_series":       np.array(all_scalp_temp),
        "base_result":        base_result,
    }


# ── 4. Clinical report ─────────────────────────────────────────────────────────
def interpret_passes(sim: dict) -> str:
    passes        = sim["passes"]
    inputs        = sim["base_result"]["inputs"]
    kurtosis_info = sim["base_result"]["kurtosis_info"]
    sex_label     = "Female" if inputs["sex"] == 1 else "Male"
    eth_map       = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    eth_label     = eth_map.get(int(inputs["ethnicity"]), "Unknown")
    protocol      = "Low-Temp (50–54°C)" if sim["low_temp_protocol"] else "Standard (55–60°C)"

    lines = [
        "=" * 72,
        "  MRgFUS PER-PASS ENERGY & THERMAL DOSE REPORT",
        "  ⚠  SIMULATED DATA — NOT FOR CLINICAL USE",
        "=" * 72,
        f"  Patient:   Age {inputs['age']:.0f} | {sex_label} | "
        f"BMI {inputs['bmi']:.1f} | {eth_label}",
        f"  SDR: {inputs['sdr_ratio']:.3f} | "
        f"SDR Kurtosis: {inputs['sdr_kurtosis']:.3f} [{kurtosis_info['tier']}] | "
        f"Skull Vol: {inputs['skull_volume']:.0f} mm³",
        f"  Protocol:  {protocol} | "
        f"Cooling Water: {COOLING_WATER_TEMP_C}°C | "
        f"Ablation Hold: {sim['ablation_duration']:.0f} s",
        f"  CEM43 Target: {CEM43_TARGET} min  |  "
        f"Achieved: {'✔ Yes' if sim['cem43_target_met'] else '✘ No'}",
        "-" * 72,
        f"  {'#':<4} {'Type':<14} {'Energy(J)':<11} {'Cumul(J)':<11} "
        f"{'Target°C':<10} {'Peak°C':<9} {'Scalp°C':<9} "
        f"{'CEM43':<8} {'Ablation'}",
        "-" * 72,
    ]

    for p in passes:
        ablated = "✔" if p["ablation_achieved"] else (
            "CONF" if p["pass_type"] == "Confirmation" else "✘"
        )
        lines.append(
            f"  {p['pass_number']:<4} "
            f"{p['pass_type']:<14} "
            f"{p['pass_energy_j']:<11.1f} "
            f"{p['cumulative_energy_j']:<11.1f} "
            f"{p['target_temp_c']:<10.1f} "
            f"{p['peak_temp_c']:<9.1f} "
            f"{p['scalp_peak_c']:<9.1f} "
            f"{p['cem43']:<8.4f} "
            f"{ablated}"
        )

    lines += [
        "-" * 72,
        f"  Total Energy:       {sim['total_energy_j']:.1f} J",
        f"  Total CEM43:        {sim['total_cem43']:.4f} min  "
        f"(target: {CEM43_TARGET} min)",
        f"  Total Procedure:    {sim['total_time_s'] / 60:.1f} min",
        f"  Tremor Prognosis:   {kurtosis_info['tremor_prognosis']}",
        "=" * 72,
    ]
    return "\n".join(lines)


# ── 5. Full figure ─────────────────────────────────────────────────────────────
def plot_per_pass(sim: dict, save_path: str = None):

    passes        = sim["passes"]
    time_min      = sim["time_series"] / 60
    temp_series   = sim["temp_series"]
    scalp_series  = sim["scalp_series"]
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
    pass_cum_cem= [p["cumulative_cem43"]    for p in passes]
    pass_scalp  = [p["scalp_peak_c"]        for p in passes]
    pass_peak   = [p["peak_temp_c"]         for p in passes]
    pass_types  = [p["pass_type"]           for p in passes]
    kt_color    = kurtosis_info["color"]

    # ── Layout: 4 rows x 3 cols ────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(
        4, 3,
        hspace=0.58, wspace=0.38,
        height_ratios=[1.15, 1.0, 1.0, 0.90]
    )

    ax_profile  = fig.add_subplot(gs[:, 0])
    ax_temp     = fig.add_subplot(gs[0, 1:])
    ax_kurtosis = fig.add_subplot(gs[1, 1])
    ax_energy   = fig.add_subplot(gs[1, 2])
    ax_cum      = fig.add_subplot(gs[2, 1])
    ax_cem      = fig.add_subplot(gs[2, 2])
    ax_shap     = fig.add_subplot(gs[3, 1:])

    # ══ Patient Profile ════════════════════════════════════════════════════════
    ax_profile.set_xlim(0, 1)
    ax_profile.set_ylim(0, 1)
    ax_profile.axis("off")

    ax_profile.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.93), 1.0, 0.07,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#2c3e50"
    ))
    ax_profile.text(0.5, 0.965, "PATIENT PROFILE",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

    protocol_label = (
        "Low-Temp (50–54°C)" if sim["low_temp_protocol"] else "Standard (55–60°C)"
    )

    profile_rows = [
        ("Age",              f"{inputs['age']:.0f} yrs"),
        ("Sex",              sex_label),
        ("BMI",              f"{inputs['bmi']:.1f} kg/m²"),
        ("Ethnicity",        eth_label),
        ("SDR Ratio",        f"{inputs['sdr_ratio']:.3f}"),
        ("SDR Kurtosis",     f"{inputs['sdr_kurtosis']:.3f}  [{kurtosis_info['tier']}]"),
        ("Skull Volume",     f"{inputs['skull_volume']:.0f} mm³"),
        ("Baseline Tremor",  f"{inputs['baseline_tremor']:.1f} CRST"),
        ("Sonication Power", f"{inputs['sonication_power']:.0f} W"),
        ("Protocol",         protocol_label),
        ("Cooling Water",    f"{COOLING_WATER_TEMP_C}°C (degassed)"),
        ("Scalp Limit",      f"<{SCALP_MAX_TEMP_C}°C"),
        ("No. of Passes",    str(sim["n_passes"])),
        ("Ablation Hold",    f"{sim['ablation_duration']:.0f} s"),
        ("Total Energy",     f"{sim['total_energy_j']:.1f} J"),
        ("Total CEM43",      f"{sim['total_cem43']:.3f} / {CEM43_TARGET} min"),
        ("CEM43 Met",        "✔ Yes" if sim["cem43_target_met"] else "✘ No"),
        ("Procedure Time",   f"{sim['total_time_s']/60:.1f} min"),
        ("Tremor Prognosis", kurtosis_info["tremor_prognosis"]),
    ]

    y_start = 0.91
    row_h   = 0.047

    for i, (field, value) in enumerate(profile_rows):
        y  = y_start - i * row_h
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        if field == "SDR Kurtosis":
            bg = kt_color + "22"
        if field == "CEM43 Met":
            bg = "#d5f5e3" if sim["cem43_target_met"] else "#fadbd8"

        ax_profile.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.025), 1.0, row_h,
            boxstyle="square,pad=0.0",
            linewidth=0, facecolor=bg
        ))
        ax_profile.text(0.04, y + 0.001, field,
                        ha="left", va="center",
                        fontsize=8.2, color="#555555")
        val_color = (
            kt_color if field == "SDR Kurtosis"
            else ("#27ae60" if (field == "CEM43 Met" and sim["cem43_target_met"])
                  else ("#e74c3c" if field == "CEM43 Met"
                        else "#222222"))
        )
        ax_profile.text(0.96, y + 0.001, value,
                        ha="right", va="center",
                        fontsize=8.2, fontweight="bold", color=val_color)

    ax_profile.text(0.5, 0.005,
                    "⚠ Simulated data — not for clinical use",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#e74c3c", style="italic")

    # ══ Temperature & Scalp Curve ══════════════════════════════════════════════
    ax_temp.plot(time_min, temp_series,
                 color="#e74c3c", linewidth=1.5, alpha=0.85,
                 zorder=3, label="Brain target temp")
    ax_temp.plot(time_min, scalp_series,
                 color="#3498db", linewidth=1.2, alpha=0.75,
                 zorder=3, linestyle="-.", label="Scalp temp (cooled)")

    ax_temp.axhline(TARGET_TEMP_LOW_C, color="#c0392b", linewidth=1.1,
                    linestyle="--",
                    label=f"Therapeutic floor ({TARGET_TEMP_LOW_C}°C)")
    ax_temp.axhline(TARGET_TEMP_HIGH_C, color="#922b21", linewidth=1.0,
                    linestyle=":",
                    label=f"Therapeutic ceiling ({TARGET_TEMP_HIGH_C}°C)")
    ax_temp.axhline(CONFIRM_TEMP_C, color="#f39c12", linewidth=0.9,
                    linestyle="--", alpha=0.7,
                    label=f"Confirmation target ({CONFIRM_TEMP_C}°C)")
    ax_temp.axhline(SCALP_MAX_TEMP_C, color="#2980b9", linewidth=0.9,
                    linestyle=":", alpha=0.8,
                    label=f"Scalp limit ({SCALP_MAX_TEMP_C}°C)")

    # Shade therapeutic zone
    ax_temp.fill_between(
        time_min, TARGET_TEMP_LOW_C, TARGET_TEMP_HIGH_C,
        alpha=0.08, color="#e74c3c", label="Therapeutic zone"
    )
    # Shade ablation zones
    ax_temp.fill_between(
        time_min, TARGET_TEMP_LOW_C, temp_series,
        where=(temp_series >= TARGET_TEMP_LOW_C),
        color="#e74c3c", alpha=0.20, zorder=2
    )

    # Pass boundary markers & type labels
    t_cursor = 0.0
    for p in passes:
        t_cursor += p["total_pass_time_s"] / 60
        lc = "#aaaaaa" if p["pass_type"] == "Confirmation" else "#555555"
        ax_temp.axvline(t_cursor, color=lc, linewidth=0.7,
                        linestyle="--", alpha=0.55)
        lbl = f"C{p['pass_number']}" if p["pass_type"] == "Confirmation" \
              else f"T{p['pass_number']}"
        ax_temp.text(
            t_cursor - p["total_pass_time_s"] / 120,
            ax_temp.get_ylim()[1] if ax_temp.get_ylim()[1] != 1 else 65,
            lbl, ha="center", va="top", fontsize=7.5,
            color="#888888" if p["pass_type"] == "Confirmation" else "#333333"
        )

    ax_temp.set_xlabel("Time (min)", fontsize=10)
    ax_temp.set_ylabel("Temperature (°C)", fontsize=10)
    ax_temp.set_title(
        "Brain Target & Scalp Temperature — All Passes\n"
        "(C = Confirmation pass, T = Therapeutic pass  |  "
        f"Skull cooled by {COOLING_WATER_TEMP_C}°C degassed water)",
        fontsize=11, fontweight="bold"
    )
    ax_temp.legend(fontsize=7.8, loc="upper right",
                   framealpha=0.9, ncol=2)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)
    ax_temp.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_temp.set_axisbelow(True)

    # ══ SDR Kurtosis Panel ═════════════════════════════════════════════════════
    ax_kurtosis.set_xlim(0, 1)
    ax_kurtosis.set_ylim(0, 1)
    ax_kurtosis.axis("off")
    ax_kurtosis.set_title("SDR Kurtosis Assessment", fontsize=11, fontweight="bold")

    x     = np.linspace(0, 1, 300)
    k     = inputs["sdr_kurtosis"]
    sigma = max(0.05, 0.18 - k * 0.04)
    mu    = inputs["sdr_ratio"]
    y_dist = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y_dist /= y_dist.max()

    ax_hist = ax_kurtosis.inset_axes([0.05, 0.44, 0.90, 0.48])
    ax_hist.fill_between(x, y_dist, alpha=0.35, color=kt_color)
    ax_hist.plot(x, y_dist, color=kt_color, linewidth=2.0)
    ax_hist.axvline(mu, color=kt_color, linewidth=1.0,
                    linestyle="--", alpha=0.6, label=f"Mean SDR {mu:.2f}")
    ax_hist.axvline(0.40, color="#888888", linewidth=0.9,
                    linestyle=":", alpha=0.7, label="SDR 0.40 cutoff")
    ax_hist.legend(fontsize=6.5, loc="upper left", framealpha=0.8)
    ax_hist.set_xlim(0.1, 0.9)
    ax_hist.set_xlabel("SDR Value", fontsize=7.5)
    ax_hist.set_ylabel("Density", fontsize=7.5)
    ax_hist.tick_params(labelsize=6.5)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.set_title("SDR Distribution Shape", fontsize=8, pad=3)

    ax_kurtosis.text(0.5, 0.37, f"Kurtosis = {k:.3f}",
                     ha="center", va="center",
                     fontsize=13, fontweight="bold", color=kt_color)
    ax_kurtosis.text(0.5, 0.27, f"Tier:  {kurtosis_info['tier']}",
                     ha="center", va="center",
                     fontsize=10, color=kt_color)
    ax_kurtosis.text(0.5, 0.17, kurtosis_info["tremor_prognosis"],
                     ha="center", va="center",
                     fontsize=8, color="#333333", style="italic")
    ax_kurtosis.text(0.5, 0.07, kurtosis_info["description"],
                     ha="center", va="center",
                     fontsize=7, color="#555555", linespacing=1.4)

    # ══ Per-Pass Energy ════════════════════════════════════════════════════════
    bar_colors = []
    for p in passes:
        if p["pass_type"] == "Confirmation":
            bar_colors.append("#95a5a6")
        elif p["ablation_achieved"]:
            bar_colors.append("#2ecc71")
        else:
            bar_colors.append("#e74c3c")

    ax_energy.bar(pass_nums, pass_energy, color=bar_colors,
                  edgecolor="none", width=0.65)
    for pn, pe in zip(pass_nums, pass_energy):
        ax_energy.text(pn, pe + 8, f"{pe:.0f}J",
                       ha="center", va="bottom", fontsize=7.5)

    conf_patch = mpatches.Patch(color="#95a5a6", label="Confirmation pass")
    ach_patch  = mpatches.Patch(color="#2ecc71", label="Ablation achieved")
    mis_patch  = mpatches.Patch(color="#e74c3c", label="Not reached")
    ax_energy.legend(handles=[conf_patch, ach_patch, mis_patch],
                     fontsize=7.5, loc="upper left", framealpha=0.85)
    ax_energy.set_xlabel("Pass Number", fontsize=10)
    ax_energy.set_ylabel("Energy (J)", fontsize=10)
    ax_energy.set_title("Energy per Pass", fontsize=11, fontweight="bold")
    ax_energy.set_xticks(pass_nums)
    ax_energy.spines["top"].set_visible(False)
    ax_energy.spines["right"].set_visible(False)
    ax_energy.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_energy.set_axisbelow(True)

    # ══ Cumulative Energy + Scalp Peak ════════════════════════════════════════
    ax_cum.plot(pass_nums, pass_cum, color="#8e44ad",
                linewidth=2.2, marker="o", markersize=6, zorder=3,
                label="Cumulative energy (J)")
    ax_cum.fill_between(pass_nums, pass_cum, alpha=0.10, color="#8e44ad")

    ax2_cum = ax_cum.twinx()
    ax2_cum.plot(pass_nums, pass_scalp, color="#3498db",
                 linewidth=1.6, marker="s", markersize=5,
                 linestyle="--", zorder=3, label="Scalp peak temp (°C)")
    ax2_cum.axhline(SCALP_MAX_TEMP_C, color="#2980b9", linewidth=0.9,
                    linestyle=":", alpha=0.8)
    ax2_cum.set_ylabel("Scalp Peak Temp (°C)", fontsize=9, color="#3498db")
    ax2_cum.tick_params(axis="y", labelcolor="#3498db")
    ax2_cum.set_ylim(10, 25)

    for pn, pc in zip(pass_nums, pass_cum):
        ax_cum.text(pn, pc + 30, f"{pc:.0f}J",
                    ha="center", va="bottom", fontsize=7.5, color="#8e44ad")

    lines1, labels1 = ax_cum.get_legend_handles_labels()
    lines2, labels2 = ax2_cum.get_legend_handles_labels()
    ax_cum.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=7.5, loc="upper left", framealpha=0.85)
    ax_cum.set_xlabel("Pass Number", fontsize=10)
    ax_cum.set_ylabel("Cumulative Energy (J)", fontsize=10, color="#8e44ad")
    ax_cum.set_title("Cumulative Energy & Scalp Temperature",
                     fontsize=11, fontweight="bold")
    ax_cum.set_xticks(pass_nums)
    ax_cum.spines["top"].set_visible(False)
    ax_cum.yaxis.grid(True, linestyle="--", alpha=0.30)
    ax_cum.set_axisbelow(True)

    # ══ CEM43 per pass + cumulative ════════════════════════════════════════════
    ax_cem.bar(pass_nums, pass_cem, color="#e67e22",
               edgecolor="none", width=0.65, label="Per-pass CEM43")
    ax_cem.plot(pass_nums, pass_cum_cem, color="#d35400",
                linewidth=2.0, marker="D", markersize=5,
                zorder=4, label="Cumulative CEM43")
    ax_cem.axhline(CEM43_TARGET, color="#c0392b", linewidth=1.2,
                   linestyle="--",
                   label=f"Clinical target ({CEM43_TARGET} CEM43)")

    for pn, pc in zip(pass_nums, pass_cem):
        ax_cem.text(pn, pc + 0.0002, f"{pc:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax_cem.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    ax_cem.set_xlabel("Pass Number", fontsize=10)
    ax_cem.set_ylabel("CEM43 (min)", fontsize=10)
    ax_cem.set_title("Thermal Dose per Pass & Cumulative (CEM43)",
                     fontsize=11, fontweight="bold")
    ax_cem.set_xticks(pass_nums)
    ax_cem.spines["top"].set_visible(False)
    ax_cem.spines["right"].set_visible(False)
    ax_cem.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_cem.set_axisbelow(True)

    # ══ SHAP full-width bottom ════════════════════════════════════════════════
    ax_shap.barh(shap_labels, shap_values, color=shap_colors,
                 edgecolor="none", height=0.6)
    ax_shap.axvline(0, color="black", linewidth=1.0)

    for i, (label, val) in enumerate(zip(shap_labels, shap_values)):
        offset = 0.15 if val >= 0 else -0.15
        ha     = "left" if val >= 0 else "right"
        ax_shap.text(val + offset, i, f"{val:+.1f}J",
                     va="center", ha=ha, fontsize=9)

    # Highlight kurtosis bar
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
    n_passes:           int   = 10,
    ablation_duration:  float = 20.0,
    low_temp_protocol:  bool  = False,
    power_ramp:         float = 0.05,
    targeting_adjust:   float = 0.03,
    save_plot:          str   = None,
):
    """
    Full pipeline: estimate → simulate → report → plot.

    Parameters
    ----------
    sdr_kurtosis : float
        Kurtosis of SDR histogram. Cutoff > -0.32 = favorable candidate.
    n_passes : int
        Number of sonications (~10 ± 2.6 per literature).
    ablation_duration : float
        Seconds tissue is held at target temperature per pass.
    low_temp_protocol : bool
        If True, targets 50–54°C (longer procedure, ~115 ± 34 min).
        If False, targets 55–60°C (standard therapeutic range).
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
        low_temp_protocol = low_temp_protocol,
        power_ramp        = power_ramp,
        targeting_adjust  = targeting_adjust,
    )
    print(interpret_passes(sim))
    plot_per_pass(sim, save_path=save_plot)
    return sim


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE PATIENTS
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n--- PATIENT A: Standard protocol, favorable kurtosis ---")
    run_per_pass_tool(
        sdr_ratio         = 0.38,
        sdr_kurtosis      = 0.12,
        age               = 72,
        bmi               = 31.2,
        sonication_power  = 62,
        skull_volume      = 138,
        baseline_tremor   = 22,
        sex               = 0,
        ethnicity         = 0,
        n_passes          = 10,
        ablation_duration = 20.0,
        low_temp_protocol = False,
        save_plot         = "patient_A_standard.png"
    )

    print("\n--- PATIENT B: Low-temp protocol, unfavorable kurtosis ---")
    run_per_pass_tool(
        sdr_ratio         = 0.41,
        sdr_kurtosis      = -0.85,
        age               = 68,
        bmi               = 27.5,
        sonication_power  = 58,
        skull_volume      = 150,
        baseline_tremor   = 19,
        sex               = 1,
        ethnicity         = 1,
        n_passes          = 10,
        ablation_duration = 20.0,
        low_temp_protocol = True,
        save_plot         = "patient_B_lowtemp.png"
    )


