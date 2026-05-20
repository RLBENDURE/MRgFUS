# ─────────────────────────────────────────────────────────────────────────────
#File 13 - Polished Final Outcomes - Outputs were presented to Dr. Frank Bova, PhD - Medical Physics - Specialist in Stereotactic Neurosurgery for Brain Surgery at the University of Florida

#presentation was conducted on April 14th, 2026 as part of BME 6018 Clinical Correlations semester-long student assignment for developing a clinical implementation methodology or strategy, hosted by Dr. Christine Schmidt - PhD - Chemical Engineering

#The objective of this project was to enhance surgical planning methodologies for noninvasive Thalamotomies that utilize FDA approved Magnetic Resonance guided Focused Ultrasound technology (MRgFUS) in patients experiencing refractory tremor.

#As of the project due date, few studies attempt to critically reassess patient inclusion criteria, which does not exclusively include: nuanced skull density ratio measurements, such as SDR Kurtosis. This remains a significant bottleneck to patients seeking treatment due to radiological safety concerns.

#Only one other study has utilized SHAP modeling for MRgFUS thalamotomy applications, and only assesses post-operative outcomes (DOI: 10.3389/fradi.2025.1683274). We attempt to improve pre-operative energy delivery planning by using patient simulation parameters and utilizing SHAP as a communication aid for the Neurosurgeon in planning their surgical pipeline, reducing potential error, time and cost.

#Dr. Frank Bova commented that this project appears to mimic the trends in radiological surgery regarding utilization of simulation/modeling/AI to improve parameter consolidation for neurological and radiological intervention.

#Simulated patient data was utilized to design our tool. NIH cited weights are utilized in the parameter consolidation tools.

#No external funding was utilized for this project. Frank Bova and Justin Hilliard MD provided general overview of medical scope in 1 hour introductory meeting.

#AI assistance was used sparingly in compiling codespace using Claude-sonnet-4.6 embedded within UF Navigator AI System.

#Signed: Riley L. Bendure: May, 20th, 2026. Master's Biomedical Engineering student at the University of Florida. Gainesville, FL. 


# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE: MRgFUS Per-Pass Energy Sonication Estimator
#          & Clinical Interpretation Tool
#  Institution: University of Florida
#  Purpose:     Surgical decision support
#  Data:        Simulated — not for clinical use
#
#  Lesion Imaging Reference:
#  - T2-weighted:     Standard for acute/subacute lesion visualization
#                     High contrast between lesion and surrounding edema
#  - T1-weighted:     Lesions appear hypointense
#                     Less reliable for chronic lesion boundary
#  - T1/T2 ratio:     Improves chronic lesion contrast
#                     Reduces measurement error vs T2 alone
#                     More reliable for long-term lesion volume tracking
#  - Lesion sizes:    T1: 5.3 ± 1.2 mm | T2: 6.2 ± 1.3 mm
#  - T1/T2 ratio ref: ~0.72 ± 0.09 within lesion core
#                     (ratio <0.65 suggests active edema/inflammation)
#                     (ratio >0.80 suggests chronic consolidated lesion)
#
#  CEM43 Reference (PMC):
#  - 13.6 CEM43: 50% tissue damage probability
#  - 17.0 CEM43: ExAblate lesion boundary predictor
#  - 36.0 CEM43: Robust T1 lesion volume predictor (1-day post-op)
#
#  Thermal Protocol:
#  - Cooling: 15°C degassed water | Scalp <19°C
#  - Post ≥40 kJ: 20 min skull cooling
#  - Therapeutic target: 55–60°C at Vim nucleus
#  - SDR Kurtosis cutoff: > -0.32
#
#  Planning Time (NIH):
#  - Pre-treatment planning:  46.2–52.3 min
#  - Total preparation:       101.9–120.6 min
#  - Total procedure:         3–4 hrs
#
#  Neurological Evaluation:
#  - 15-min tremor assessment buffer between therapeutic passes
# ─────────────────────────────────────────────────────────────────────────────

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

BASE_ENERGY_J         = 15000
BODY_TEMP_C           = 37.0
SCALP_MAX_TEMP_C      = 19.0
COOLING_WATER_TEMP_C  = 15.0
TARGET_TEMP_LOW_C     = 55.0
TARGET_TEMP_HIGH_C    = 60.0
CONFIRM_TEMP_C        = 45.0
HIGH_ENERGY_THRESHOLD = 40000
SKULL_COOLING_TIME_S  = 1200
STANDARD_COOLING_S    = 90
HEATING_RATE_BASE     = 0.22
SDR_KURTOSIS_CUTOFF   = -0.32
NEURO_EVAL_TIME_S     = 900
NEURO_EVAL_TIME_MIN   = 15.0

# ── CEM43 thresholds ───────────────────────────────────────────────────────────
CEM43_DAMAGE_50PCT    = 13.6
CEM43_BOUNDARY_PRED   = 17.0
CEM43_T1_LESION_PRED  = 36.0

# ── Lesion imaging reference values ───────────────────────────────────────────
LESION_T1_MEAN_MM     = 5.3
LESION_T1_STD_MM      = 1.2
LESION_T2_MEAN_MM     = 6.2
LESION_T2_STD_MM      = 1.3
LESION_BORDER_ERR_MM  = 0.37
LESION_BORDER_STD_MM  = 0.57
FOCUS_WIDTH_MM        = 1.0

# ── T1/T2 ratio reference values ──────────────────────────────────────────────
# Ratio within lesion core compared to contralateral normal tissue
# T1/T2 ratio improves chronic lesion contrast and reduces measurement error
T1T2_RATIO_CORE_MEAN  = 0.72    # Mean ratio within confirmed lesion core
T1T2_RATIO_CORE_STD   = 0.09
T1T2_EDEMA_THRESHOLD  = 0.65    # Below = active edema/inflammation likely
T1T2_CHRONIC_THRESHOLD = 0.80   # Above = chronic consolidated lesion
T1T2_NORMAL_TISSUE    = 1.00    # Normal brain parenchyma reference

# Planning time constants (NIH)
PLANNING_TIME_LOW_MIN  = 46.2
PLANNING_TIME_HIGH_MIN = 52.3
TOTAL_PREP_LOW_MIN     = 101.9
TOTAL_PREP_HIGH_MIN    = 120.6
TOTAL_PROC_LOW_HRS     = 3.0
TOTAL_PROC_HIGH_HRS    = 4.0


# ── 1. SDR kurtosis interpreter ────────────────────────────────────────────────
def interpret_sdr_kurtosis(kurtosis: float) -> dict:
    if kurtosis > 0.50:
        tier, color      = "EXCELLENT", "#27ae60"
        description      = (
            "Highly peaked SDR distribution.\n"
            "Efficient sonication, lower peak\n"
            "temps, strong tremor improvement\n"
            "(>55% at 12 months)."
        )
        energy_modifier  = -0.12
        tremor_prognosis = "55–62% CRST improvement @ 12 months"
    elif kurtosis > SDR_KURTOSIS_CUTOFF:
        tier, color      = "FAVORABLE", "#2ecc71"
        description      = (
            "Above-threshold kurtosis.\n"
            "Suitable even if mean SDR <0.40.\n"
            "Expected improvement: 44–55%."
        )
        energy_modifier  = -0.06
        tremor_prognosis = "44–55% CRST improvement @ 12 months"
    elif kurtosis > -0.60:
        tier, color      = "BORDERLINE", "#f39c12"
        description      = (
            "Below cutoff. Broader SDR\n"
            "distribution with low-SDR tail.\n"
            "Additional CT skull analysis\n"
            "recommended."
        )
        energy_modifier  = 0.04
        tremor_prognosis = "30–44% CRST improvement @ 12 months"
    else:
        tier, color      = "UNFAVORABLE", "#e74c3c"
        description      = (
            "Flat, broad SDR distribution.\n"
            "Higher energy demand & elevated\n"
            "thermal risk. MDT review advised."
        )
        energy_modifier  = 0.10
        tremor_prognosis = "<30% CRST improvement @ 12 months"

    return {
        "kurtosis":         kurtosis,
        "tier":             tier,
        "color":            color,
        "favorable":        kurtosis > SDR_KURTOSIS_CUTOFF,
        "description":      description,
        "energy_modifier":  energy_modifier,
        "tremor_prognosis": tremor_prognosis,
    }


# ── 2. T1/T2 ratio interpreter ─────────────────────────────────────────────────
def interpret_t1t2_ratio(
    cumulative_cem43: float,
    peak_temp_c:      float,
    pass_number:      int,
    n_passes:         int,
) -> dict:
    """
    Estimates the T1/T2 signal intensity ratio within the lesion core
    based on thermal dose and pass progression.

    T1/T2 ratio interpretation:
    - Ratio < 0.65:  Active edema / acute inflammation (T2 bright, T1 dark)
    - Ratio 0.65–0.80: Subacute lesion, developing gliosis
    - Ratio > 0.80:  Chronic consolidated lesion, reduced edema
    - Reference core: 0.72 ± 0.09 (established lesion)

    Ratio is estimated per pass as thermal dose accumulates:
    - Early passes: low ratio (edema-like, T2 dominates)
    - Later passes: ratio rises toward chronic reference as dose consolidates
    """
    # Base ratio scales with thermal dose and temperature
    # Low CEM43 → acute edema phase (low ratio)
    # High CEM43 → chronic consolidation (ratio approaches reference)
    if cumulative_cem43 <= 0:
        base_ratio = T1T2_NORMAL_TISSUE
    elif cumulative_cem43 < CEM43_DAMAGE_50PCT:
        # Pre-ablation: minimal signal change, near normal tissue
        frac       = cumulative_cem43 / CEM43_DAMAGE_50PCT
        base_ratio = T1T2_NORMAL_TISSUE - frac * 0.25
    elif cumulative_cem43 < CEM43_BOUNDARY_PRED:
        # Acute edema phase: T2 rises sharply, T1 drops → low ratio
        frac       = (
            (cumulative_cem43 - CEM43_DAMAGE_50PCT)
            / (CEM43_BOUNDARY_PRED - CEM43_DAMAGE_50PCT)
        )
        base_ratio = 0.75 - frac * 0.15   # drops toward edema threshold
    elif cumulative_cem43 < CEM43_T1_LESION_PRED:
        # Subacute: ratio recovering as gliosis develops
        frac       = (
            (cumulative_cem43 - CEM43_BOUNDARY_PRED)
            / (CEM43_T1_LESION_PRED - CEM43_BOUNDARY_PRED)
        )
        base_ratio = (T1T2_EDEMA_THRESHOLD
                      + frac * (T1T2_RATIO_CORE_MEAN - T1T2_EDEMA_THRESHOLD))
    else:
        # Chronic consolidation: approaches and can exceed reference mean
        excess     = cumulative_cem43 - CEM43_T1_LESION_PRED
        base_ratio = T1T2_RATIO_CORE_MEAN + min(0.10, excess * 0.002)

    # Temperature modifier: higher peak temp accelerates ratio drop (more edema)
    temp_mod = max(-0.08, -(peak_temp_c - TARGET_TEMP_LOW_C) * 0.004)
    ratio    = round(
        np.clip(base_ratio + temp_mod + np.random.normal(0, 0.01),
                0.40, T1T2_NORMAL_TISSUE),
        3
    )

    # Classification
    if ratio < T1T2_EDEMA_THRESHOLD:
        phase = "ACUTE EDEMA"
        color = "#e74c3c"
        note  = (
            "T2 signal dominant. Active edema/inflammation.\n"
            "Lesion boundary overestimated on T2 alone.\n"
            "T1/T2 ratio imaging recommended for accuracy."
        )
    elif ratio < T1T2_RATIO_CORE_MEAN:
        phase = "SUBACUTE"
        color = "#f39c12"
        note  = (
            "Developing gliosis. T1 signal recovering.\n"
            "T1/T2 ratio imaging improves boundary precision\n"
            "vs T2 alone at this stage."
        )
    elif ratio < T1T2_CHRONIC_THRESHOLD:
        phase = "ESTABLISHED"
        color = "#2ecc71"
        note  = (
            "Lesion within reference ratio range.\n"
            "T1/T2 ratio imaging confirms boundary.\n"
            "Reduced measurement error vs T2 alone."
        )
    else:
        phase = "CHRONIC"
        color = "#27ae60"
        note  = (
            "Chronic consolidated lesion.\n"
            "T1/T2 ratio imaging most reliable here.\n"
            "Edema resolved — precise boundary delineation."
        )

    # Estimated lesion size correction using T1/T2 ratio
    # T2 alone overestimates when ratio is low (edema inflates apparent size)
    # T1/T2 corrected estimate is more accurate for chronic measurement
    t2_overestimate_factor = max(
        0.0, (T1T2_RATIO_CORE_MEAN - ratio) * 1.2
    )
    t1t2_corrected_mm = round(
        LESION_T2_MEAN_MM * (ratio / T1T2_RATIO_CORE_MEAN)
        - t2_overestimate_factor,
        2
    )
    t1t2_corrected_mm = max(0.5, t1t2_corrected_mm)

    return {
        "ratio":               ratio,
        "phase":               phase,
        "color":               color,
        "note":                note,
        "t1t2_corrected_mm":   t1t2_corrected_mm,
        "edema_likely":        ratio < T1T2_EDEMA_THRESHOLD,
        "chronic_confirmed":   ratio >= T1T2_CHRONIC_THRESHOLD,
        "within_ref_range":    (
            abs(ratio - T1T2_RATIO_CORE_MEAN) <= T1T2_RATIO_CORE_STD
        ),
    }


# ── 3. CEM43 clinical interpreter ─────────────────────────────────────────────
def interpret_cem43(cumulative_cem43: float) -> dict:
    if cumulative_cem43 < CEM43_DAMAGE_50PCT:
        lesion_prob   = cumulative_cem43 / CEM43_DAMAGE_50PCT * 50.0
        est_lesion_mm = cumulative_cem43 / CEM43_DAMAGE_50PCT * 3.0
        lesion_tier   = "SUB-THRESHOLD"
        lesion_color  = "#e74c3c"
        lesion_note   = (
            f"Below 50% damage threshold ({CEM43_DAMAGE_50PCT} CEM43)."
        )
    elif cumulative_cem43 < CEM43_BOUNDARY_PRED:
        frac = (
            (cumulative_cem43 - CEM43_DAMAGE_50PCT)
            / (CEM43_BOUNDARY_PRED - CEM43_DAMAGE_50PCT)
        )
        lesion_prob   = 50.0 + frac * 30.0
        est_lesion_mm = 3.0 + frac * (LESION_T1_MEAN_MM - 3.0)
        lesion_tier   = "DEVELOPING"
        lesion_color  = "#f39c12"
        lesion_note   = "Lesion forming — boundary not yet reliable."
    elif cumulative_cem43 < CEM43_T1_LESION_PRED:
        frac = (
            (cumulative_cem43 - CEM43_BOUNDARY_PRED)
            / (CEM43_T1_LESION_PRED - CEM43_BOUNDARY_PRED)
        )
        lesion_prob   = 80.0 + frac * 15.0
        est_lesion_mm = (
            LESION_T1_MEAN_MM
            + frac * (LESION_T2_MEAN_MM - LESION_T1_MEAN_MM)
        )
        lesion_tier   = "THERAPEUTIC"
        lesion_color  = "#2ecc71"
        lesion_note   = f"Above boundary predictor ({CEM43_BOUNDARY_PRED})."
    else:
        lesion_prob   = 95.0 + min(
            5.0, (cumulative_cem43 - CEM43_T1_LESION_PRED) * 0.1
        )
        est_lesion_mm = LESION_T2_MEAN_MM + (
            cumulative_cem43 - CEM43_T1_LESION_PRED
        ) * 0.05
        lesion_tier   = "CONSOLIDATED"
        lesion_color  = "#27ae60"
        lesion_note   = "Above T1 lesion predictor — volume reliable."

    est_t2_mm = est_lesion_mm * (LESION_T2_MEAN_MM / LESION_T1_MEAN_MM)

    return {
        "cumulative_cem43":  cumulative_cem43,
        "lesion_tier":       lesion_tier,
        "lesion_color":      lesion_color,
        "lesion_prob_pct":   round(min(lesion_prob, 99.9), 1),
        "est_lesion_t1_mm":  round(est_lesion_mm, 2),
        "est_lesion_t2_mm":  round(est_t2_mm, 2),
        "lesion_note":       lesion_note,
        "boundary_pred_met": cumulative_cem43 >= CEM43_BOUNDARY_PRED,
        "t1_pred_met":       cumulative_cem43 >= CEM43_T1_LESION_PRED,
        "damage_50pct_met":  cumulative_cem43 >= CEM43_DAMAGE_50PCT,
    }


# ── 4. Base energy estimator ───────────────────────────────────────────────────
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
    predicted_energy    = round(
        BASE_ENERGY_J + total_adjustment + kurtosis_adjustment, 1
    )
    return {
        "base_energy_j":  predicted_energy,
        "contributions":  contributions,
        "inputs":         inputs,
        "kurtosis_info":  kurtosis_info,
    }


# ── 5. Per-pass thermal simulation ────────────────────────────────────────────
def simulate_passes(
    base_result:        dict,
    n_passes:           int   = 10,
    ablation_duration:  float = 20.0,
    power_ramp:         float = 0.05,
    targeting_adjust:   float = 0.03,
    low_temp_protocol:  bool  = False,
    seed:               int   = 42,
) -> dict:
    np.random.seed(seed)

    base_energy        = base_result["base_energy_j"]
    sonication_power   = base_result["inputs"]["sonication_power"]
    kurtosis_info      = base_result["kurtosis_info"]
    heating_rate       = HEATING_RATE_BASE / (
        1 + 0.1 * base_result["inputs"]["bmi"] / 27
    )
    sdr                = base_result["inputs"]["sdr_ratio"]
    sdr_heating_factor = 0.7 + (sdr / 0.45) * 0.6

    if low_temp_protocol:
        target_temp = np.random.uniform(50.0, 54.0)
    else:
        target_temp = np.random.uniform(TARGET_TEMP_LOW_C, TARGET_TEMP_HIGH_C)

    passes           = []
    cumulative_e     = 0.0
    cumulative_cem43 = 0.0
    all_time         = []
    all_temp         = []
    all_scalp_temp   = []
    all_phase_labels = []
    global_t         = 0.0
    skull_heat_accum = 0.0

    for p in range(1, n_passes + 1):
        is_confirmation = p <= 2

        if is_confirmation:
            pass_target_temp = CONFIRM_TEMP_C + np.random.normal(0, 0.5)
            pass_energy      = round(
                base_energy * 0.25 + np.random.normal(0, 20), 1
            )
        else:
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

        thermal_cooling_s = (
            SKULL_COOLING_TIME_S
            if cumulative_e >= HIGH_ENERGY_THRESHOLD
            else STANDARD_COOLING_S
        )
        neuro_eval_s = 0 if is_confirmation else NEURO_EVAL_TIME_S

        # ── Heating ────────────────────────────────────────────────────────
        start_temp    = BODY_TEMP_C + skull_heat_accum + np.random.normal(0, 0.4)
        temp_to_rise  = pass_target_temp - start_temp
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

        # ── Ablation hold ──────────────────────────────────────────────────
        hold_t    = np.linspace(0, ablation_duration, int(ablation_duration * 5))
        hold_temp = pass_target_temp + np.random.normal(0, 0.8, len(hold_t))

        # ── Thermal cooling ────────────────────────────────────────────────
        cool_t = np.linspace(
            0, thermal_cooling_s, int(thermal_cooling_s * 2)
        )
        skull_cool_rate = 1.0 - np.exp(
            -thermal_cooling_s / (SKULL_COOLING_TIME_S * 0.8)
        )
        cool_temp = BODY_TEMP_C + (
            pass_target_temp - BODY_TEMP_C
        ) * np.exp(-cool_t / (thermal_cooling_s * 0.30))
        cool_temp = np.clip(cool_temp, BODY_TEMP_C, pass_target_temp)
        cool_temp += np.random.normal(0, 0.3, len(cool_t))

        skull_heat_accum = max(
            0.0,
            skull_heat_accum * (1 - skull_cool_rate)
            + (pass_target_temp - BODY_TEMP_C) * 0.08
        )

        # ── Neuro eval ─────────────────────────────────────────────────────
        if neuro_eval_s > 0:
            neuro_t    = np.linspace(0, neuro_eval_s, int(neuro_eval_s * 2))
            neuro_temp = BODY_TEMP_C + np.random.normal(0, 0.25, len(neuro_t))
            neuro_temp = np.clip(neuro_temp, BODY_TEMP_C - 0.5, BODY_TEMP_C + 1.0)
        else:
            neuro_t    = np.array([])
            neuro_temp = np.array([])

        # ── Scalp profile ──────────────────────────────────────────────────
        full_len        = (
            len(heat_temp) + len(hold_temp)
            + len(cool_temp) + len(neuro_temp)
        )
        scalp_base      = COOLING_WATER_TEMP_C + np.random.normal(0, 0.3)
        scalp_hh        = np.linspace(0, 2.8, len(heat_temp) + len(hold_temp))
        scalp_cool_down = np.linspace(2.8, 0.3, len(cool_temp))
        scalp_neuro_arr = (
            np.linspace(0.3, 0.0, len(neuro_temp))
            if len(neuro_temp) > 0 else np.array([])
        )
        scalp_parts = [scalp_base + scalp_hh, scalp_base + scalp_cool_down]
        if len(scalp_neuro_arr) > 0:
            scalp_parts.append(scalp_base + scalp_neuro_arr)
        scalp_profile = np.clip(
            np.concatenate(scalp_parts) + np.random.normal(0, 0.2, full_len),
            COOLING_WATER_TEMP_C, SCALP_MAX_TEMP_C
        )

        # ── Combine phases ─────────────────────────────────────────────────
        parts = [heat_temp, hold_temp, cool_temp]
        if len(neuro_temp) > 0:
            parts.append(neuro_temp)
        pass_temp   = np.concatenate(parts)
        total_time  = (
            heat_duration + ablation_duration
            + thermal_cooling_s + neuro_eval_s
        )
        pass_time   = np.linspace(0, total_time, len(pass_temp))

        all_time.extend(global_t + pass_time)
        all_temp.extend(pass_temp)
        all_scalp_temp.extend(scalp_profile)
        all_phase_labels.extend(
            ["heat"]  * len(heat_temp)
            + ["hold"] * len(hold_temp)
            + ["cool"] * len(cool_temp)
            + (["neuro"] * len(neuro_temp) if len(neuro_temp) > 0 else [])
        )
        global_t += total_time

        # ── CEM43 (sonication phases only) ─────────────────────────────────
        sonic_temp = np.concatenate([heat_temp, hold_temp, cool_temp])
        dt_s = (
            heat_duration + ablation_duration + thermal_cooling_s
        ) / len(sonic_temp)
        cem43 = 0.0
        for t_val in sonic_temp:
            R = 0.5 if t_val >= 43 else 0.25
            cem43 += dt_s / 60 * (R ** (43 - t_val))
        cumulative_cem43 += cem43
        cumulative_e     += pass_energy

        cem43_info  = interpret_cem43(cumulative_cem43)
        peak_temp_c = float(np.max(pass_temp))

        # ── T1/T2 ratio estimate for this pass ─────────────────────────────
        t1t2_info = interpret_t1t2_ratio(
            cumulative_cem43 = cumulative_cem43,
            peak_temp_c      = peak_temp_c,
            pass_number      = p,
            n_passes         = n_passes,
        )

        passes.append({
            "pass_number":          p,
            "pass_type":            "Confirmation" if is_confirmation
                                    else "Therapeutic",
            "pass_energy_j":        pass_energy,
            "cumulative_energy_j":  round(cumulative_e, 1),
            "target_temp_c":        round(pass_target_temp, 1),
            "peak_temp_c":          round(peak_temp_c, 1),
            "scalp_peak_c":         round(float(np.max(scalp_profile)), 1),
            "heat_duration_s":      round(heat_duration, 1),
            "ablation_hold_s":      ablation_duration,
            "thermal_cooling_s":    thermal_cooling_s,
            "neuro_eval_s":         neuro_eval_s,
            "total_pass_time_s":    round(total_time, 1),
            "ablation_achieved":    (
                peak_temp_c >= TARGET_TEMP_LOW_C and not is_confirmation
            ),
            "cem43":                round(cem43, 4),
            "cumulative_cem43":     round(cumulative_cem43, 4),
            "cem43_info":           cem43_info,
            "t1t2_info":            t1t2_info,
            "skull_heat_residual":  round(skull_heat_accum, 2),
        })

    final_cem43_info = interpret_cem43(cumulative_cem43)
    final_t1t2_info  = interpret_t1t2_ratio(
        cumulative_cem43 = cumulative_cem43,
        peak_temp_c      = passes[-1]["peak_temp_c"],
        pass_number      = n_passes,
        n_passes         = n_passes,
    )

    return {
        "passes":             passes,
        "n_passes":           n_passes,
        "ablation_duration":  ablation_duration,
        "low_temp_protocol":  low_temp_protocol,
        "total_energy_j":     round(cumulative_e, 1),
        "total_cem43":        round(cumulative_cem43, 4),
        "total_time_s":       round(global_t, 1),
        "cem43_target_met":   cumulative_cem43 >= CEM43_BOUNDARY_PRED,
        "final_cem43_info":   final_cem43_info,
        "final_t1t2_info":    final_t1t2_info,
        "time_series":        np.array(all_time),
        "temp_series":        np.array(all_temp),
        "scalp_series":       np.array(all_scalp_temp),
        "phase_labels":       np.array(all_phase_labels),
        "base_result":        base_result,
    }


# ── 6. Clinical report ─────────────────────────────────────────────────────────
def interpret_passes(sim: dict) -> str:
    passes        = sim["passes"]
    inputs        = sim["base_result"]["inputs"]
    kurtosis_info = sim["base_result"]["kurtosis_info"]
    cem43_info    = sim["final_cem43_info"]
    t1t2_info     = sim["final_t1t2_info"]
    sex_label     = "Female" if inputs["sex"] == 1 else "Male"
    eth_map       = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian/Other"}
    eth_label     = eth_map.get(int(inputs["ethnicity"]), "Unknown")
    protocol      = (
        "Low-Temp (50–54°C)" if sim["low_temp_protocol"]
        else "Standard (55–60°C)"
    )
    comp_ms       = sim.get("computation_time_ms", 0.0)

    lines = [
        "=" * 82,
        "  MRgFUS PER-PASS ENERGY, THERMAL DOSE & LESION IMAGING REPORT",
        "  ⚠  SIMULATED DATA — NOT FOR CLINICAL USE",
        "=" * 82,
        f"  Patient:  Age {inputs['age']:.0f} | {sex_label} | "
        f"BMI {inputs['bmi']:.1f} | {eth_label}",
        f"  SDR: {inputs['sdr_ratio']:.3f} | "
        f"Kurtosis: {inputs['sdr_kurtosis']:.3f} [{kurtosis_info['tier']}]",
        f"  Protocol: {protocol} | "
        f"Cooling: {COOLING_WATER_TEMP_C}°C | "
        f"Neuro eval: {NEURO_EVAL_TIME_MIN:.0f} min/pass",
        "-" * 82,
        "  CEM43 & LESION SUMMARY",
        f"  CEM43: {sim['total_cem43']:.4f}  |  "
        f"Tier: {cem43_info['lesion_tier']}  |  "
        f"Damage prob: {cem43_info['lesion_prob_pct']}%",
        f"  T1 est: {cem43_info['est_lesion_t1_mm']:.2f} mm  |  "
        f"T2 est: {cem43_info['est_lesion_t2_mm']:.2f} mm  |  "
        f"T1/T2 corrected: {t1t2_info['t1t2_corrected_mm']:.2f} mm",
        f"  T1/T2 ratio: {t1t2_info['ratio']:.3f}  |  "
        f"Phase: {t1t2_info['phase']}  |  "
        f"Ref: {T1T2_RATIO_CORE_MEAN} ± {T1T2_RATIO_CORE_STD}",
        f"  Edema likely: {'Yes ⚠' if t1t2_info['edema_likely'] else 'No ✔'}  |  "
        f"Chronic confirmed: {'Yes ✔' if t1t2_info['chronic_confirmed'] else 'No'}  |  "
        f"Within ref range: {'Yes ✔' if t1t2_info['within_ref_range'] else 'No'}",
        "-" * 82,
        f"  Model runtime: {comp_ms:.2f} ms  |  "
        f"Surgeon planning: {PLANNING_TIME_LOW_MIN}–{PLANNING_TIME_HIGH_MIN} min",
        "-" * 82,
        "  PER-PASS DELIVERY PLAN",
        f"  {'#':<4} {'Type':<14} {'J':<9} {'°C':<8} "
        f"{'CEM43':<9} {'T1/T2':<8} {'Ratio Phase':<14} "
        f"{'T1/T2 Corr(mm)':<16} {'Neuro':<8} {'Notes'}",
        "-" * 82,
    ]

    for p in passes:
        ci         = p["cem43_info"]
        ti         = p["t1t2_info"]
        neuro_flag = (
            f"{p['neuro_eval_s']//60:.0f} min"
            if p["neuro_eval_s"] > 0 else "—"
        )
        notes = (
            "Low power — targeting verification" if p["pass_type"] == "Confirmation"
            else "Escalate if target not reached"  if not p["ablation_achieved"]
            else "Therapeutic — maintain hold"
        )
        lines.append(
            f"  {p['pass_number']:<4} "
            f"{p['pass_type']:<14} "
            f"{p['pass_energy_j']:<9.1f} "
            f"{p['target_temp_c']:<8.1f} "
            f"{p['cumulative_cem43']:<9.3f} "
            f"{ti['ratio']:<8.3f} "
            f"{ti['phase']:<14} "
            f"{ti['t1t2_corrected_mm']:<16.2f} "
            f"{neuro_flag:<8} "
            f"{notes}"
        )

    lines += [
        "-" * 82,
        f"  Total Energy:    {sim['total_energy_j']:.1f} J",
        f"  Total CEM43:     {sim['total_cem43']:.4f}",
        f"  Final T1/T2:     {t1t2_info['ratio']:.3f} ({t1t2_info['phase']})",
        f"  T1/T2 corrected: {t1t2_info['t1t2_corrected_mm']:.2f} mm  "
        f"(vs T2 est: {cem43_info['est_lesion_t2_mm']:.2f} mm)",
        f"  Procedure Time:  {sim['total_time_s'] / 60:.1f} min",
        f"  Tremor Prognosis: {kurtosis_info['tremor_prognosis']}",
        "=" * 82,
    ]
    return "\n".join(lines)


# ── 7. Full figure ─────────────────────────────────────────────────────────────
def plot_per_pass(sim: dict, save_path: str = None):

    passes        = sim["passes"]
    time_min      = sim["time_series"] / 60
    temp_series   = sim["temp_series"]
    scalp_series  = sim["scalp_series"]
    phase_labels  = sim["phase_labels"]
    inputs        = sim["base_result"]["inputs"]
    contribs      = sim["base_result"]["contributions"]
    kurtosis_info = sim["base_result"]["kurtosis_info"]
    final_ci      = sim["final_cem43_info"]
    final_ti      = sim["final_t1t2_info"]
    comp_time_ms  = sim.get("computation_time_ms", 0.0)

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

    pass_nums      = [p["pass_number"]           for p in passes]
    pass_energy    = [p["pass_energy_j"]         for p in passes]
    pass_cum       = [p["cumulative_energy_j"]   for p in passes]
    pass_cem       = [p["cem43"]                 for p in passes]
    pass_cum_cem   = [p["cumulative_cem43"]      for p in passes]
    pass_scalp     = [p["scalp_peak_c"]          for p in passes]
    pass_t1t2      = [p["t1t2_info"]["ratio"]    for p in passes]
    pass_t1t2_corr = [p["t1t2_info"]["t1t2_corrected_mm"] for p in passes]
    pass_lesion_t1 = [p["cem43_info"]["est_lesion_t1_mm"] for p in passes]
    pass_lesion_t2 = [p["cem43_info"]["est_lesion_t2_mm"] for p in passes]
    pass_dmg_prob  = [p["cem43_info"]["lesion_prob_pct"]  for p in passes]

    kt_color     = kurtosis_info["color"]
    lesion_color = final_ci["lesion_color"]
    t1t2_color   = final_ti["color"]

    time_arr  = np.array(time_min)
    temp_arr  = np.array(temp_series)
    scalp_arr = np.array(scalp_series)
    phase_arr = np.array(phase_labels)

    # ── Layout: 6 rows x 3 cols ────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 26))
    gs  = gridspec.GridSpec(
        6, 3,
        hspace=0.60, wspace=0.38,
        height_ratios=[1.15, 1.0, 1.0, 1.0, 0.90, 0.85]
    )

    ax_profile  = fig.add_subplot(gs[:, 0])
    ax_temp     = fig.add_subplot(gs[0, 1:])
    ax_kurtosis = fig.add_subplot(gs[1, 1])
    ax_cem43pan = fig.add_subplot(gs[1, 2])
    ax_energy   = fig.add_subplot(gs[2, 1])
    ax_cum      = fig.add_subplot(gs[2, 2])
    ax_cem      = fig.add_subplot(gs[3, 1])
    ax_t1t2     = fig.add_subplot(gs[3, 2])   # NEW: T1/T2 ratio panel
    ax_lesion   = fig.add_subplot(gs[4, 1:])  # Full width lesion tracker
    ax_shap     = fig.add_subplot(gs[5, 1:])

    # ══ Patient Profile ════════════════════════════════════════════════════════
    ax_profile.set_xlim(0, 1)
    ax_profile.set_ylim(0, 1)
    ax_profile.axis("off")

    ax_profile.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.93), 1.0, 0.07,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#2c3e50"
    ))
    ax_profile.text(0.5, 0.966, "PATIENT PROFILE",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

    protocol_label = (
        "Low-Temp (50–54°C)" if sim["low_temp_protocol"]
        else "Standard (55–60°C)"
    )
    n_therapeutic   = sum(1 for p in passes if p["pass_type"] == "Therapeutic")
    total_neuro_min = n_therapeutic * NEURO_EVAL_TIME_MIN

    profile_rows = [
        ("Age",               f"{inputs['age']:.0f} yrs"),
        ("Sex",               sex_label),
        ("BMI",               f"{inputs['bmi']:.1f} kg/m²"),
        ("Ethnicity",         eth_label),
        ("SDR Ratio",         f"{inputs['sdr_ratio']:.3f}"),
        ("SDR Kurtosis",      f"{inputs['sdr_kurtosis']:.3f}  [{kurtosis_info['tier']}]"),
        ("Skull Volume",      f"{inputs['skull_volume']:.0f} mm³"),
        ("Baseline Tremor",   f"{inputs['baseline_tremor']:.1f} CRST"),
        ("Sonication Power",  f"{inputs['sonication_power']:.0f} W"),
        ("Protocol",          protocol_label),
        ("Cooling Water",     f"{COOLING_WATER_TEMP_C}°C (degassed)"),
        ("Scalp Limit",       f"<{SCALP_MAX_TEMP_C}°C"),
        ("No. of Passes",     str(sim["n_passes"])),
        ("Ablation Hold",     f"{sim['ablation_duration']:.0f} s"),
        ("Neuro Eval",        f"{NEURO_EVAL_TIME_MIN:.0f} min × {n_therapeutic}"),
        ("Total Neuro Time",  f"{total_neuro_min:.0f} min"),
        ("Total Energy",      f"{sim['total_energy_j']:.1f} J"),
        ("Total CEM43",       f"{sim['total_cem43']:.3f}"),
        ("CEM43 Tier",        final_ci["lesion_tier"]),
        ("Damage Prob.",      f"{final_ci['lesion_prob_pct']:.1f}%"),
        ("Est. Lesion T1",    f"{final_ci['est_lesion_t1_mm']:.2f} mm"),
        ("Est. Lesion T2",    f"{final_ci['est_lesion_t2_mm']:.2f} mm"),
        # T1/T2 ratio rows
        ("T1/T2 Ratio",       f"{final_ti['ratio']:.3f}"),
        ("T1/T2 Phase",       final_ti["phase"]),
        ("T1/T2 Corr. Size",  f"{final_ti['t1t2_corrected_mm']:.2f} mm"),
        ("Edema Likely",      "Yes ⚠" if final_ti["edema_likely"] else "No ✔"),
        ("Procedure Time",    f"{sim['total_time_s']/60:.1f} min"),
        ("Tremor Prognosis",  kurtosis_info["tremor_prognosis"]),
    ]

    y_start = 0.91
    row_h   = 0.032

    for i, (field, value) in enumerate(profile_rows):
        y  = y_start - i * row_h
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        if field == "SDR Kurtosis":
            bg = kt_color + "22"
        if field == "CEM43 Tier":
            bg = lesion_color + "22"
        if field in ("T1/T2 Ratio", "T1/T2 Phase",
                     "T1/T2 Corr. Size", "Edema Likely"):
            bg = t1t2_color + "18"
        if field in ("Neuro Eval", "Total Neuro Time"):
            bg = "#eaf4fb"
        if field == "Damage Prob.":
            prob = final_ci["lesion_prob_pct"]
            bg   = (
                "#d5f5e3" if prob >= 80
                else "#fef9e7" if prob >= 50
                else "#fadbd8"
            )

        ax_profile.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.018), 1.0, row_h,
            boxstyle="square,pad=0.0",
            linewidth=0, facecolor=bg
        ))
        ax_profile.text(0.04, y + 0.001, field,
                        ha="left", va="center",
                        fontsize=7.2, color="#555555")
        val_color = (
            kt_color      if field == "SDR Kurtosis"
            else "#2980b9" if field in ("Neuro Eval", "Total Neuro Time")
            else t1t2_color if field in (
                "T1/T2 Ratio", "T1/T2 Phase",
                "T1/T2 Corr. Size", "Edema Likely"
            )
            else lesion_color if field in (
                "CEM43 Tier", "Est. Lesion T1",
                "Est. Lesion T2", "Damage Prob."
            )
            else "#222222"
        )
        ax_profile.text(0.96, y + 0.001, value,
                        ha="right", va="center",
                        fontsize=7.2, fontweight="bold", color=val_color)

    # Timing footer
    timing_y = y_start - len(profile_rows) * row_h - 0.003
    ax_profile.add_patch(mpatches.FancyBboxPatch(
        (0.0, timing_y - 0.092), 1.0, 0.10,
        boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor="#aaaaaa", facecolor="#fdfefe"
    ))
    ax_profile.text(0.5, timing_y + 0.002,
                    "⏱  Planning & Computation Times",
                    ha="center", va="center",
                    fontsize=7.8, fontweight="bold", color="#2c3e50")
    ax_profile.text(0.5, timing_y - 0.016,
                    f"Model runtime:  {comp_time_ms:.2f} ms",
                    ha="center", va="center",
                    fontsize=7.5, color="#16a085", fontweight="bold")
    ax_profile.text(0.5, timing_y - 0.034,
                    f"Surgeon planning:  "
                    f"{PLANNING_TIME_LOW_MIN}–{PLANNING_TIME_HIGH_MIN} min",
                    ha="center", va="center", fontsize=7.2, color="#555555")
    ax_profile.text(0.5, timing_y - 0.052,
                    f"Total prep:  "
                    f"{TOTAL_PREP_LOW_MIN}–{TOTAL_PREP_HIGH_MIN} min",
                    ha="center", va="center", fontsize=7.2, color="#555555")
    ax_profile.text(0.5, timing_y - 0.070,
                    f"Total procedure:  "
                    f"{TOTAL_PROC_LOW_HRS:.0f}–{TOTAL_PROC_HIGH_HRS:.0f} hrs",
                    ha="center", va="center", fontsize=7.2, color="#555555")
    ax_profile.text(0.5, 0.003,
                    "⚠ Simulated data — not for clinical use",
                    ha="center", va="bottom",
                    fontsize=7.0, color="#e74c3c", style="italic")

    # ══ Temperature Curve ═════════════════════════════════════════════════════
    ax_temp.plot(time_arr, temp_arr,
                 color="#e74c3c", linewidth=1.5, alpha=0.85,
                 zorder=3, label="Brain target temp")
    ax_temp.plot(time_arr, scalp_arr,
                 color="#3498db", linewidth=1.2, alpha=0.75,
                 zorder=3, linestyle="-.", label="Scalp temp (cooled)")

    # Neuro eval shading
    neuro_mask = phase_arr == "neuro"
    if neuro_mask.any():
        in_block, block_start = False, None
        for idx in range(len(time_arr)):
            if neuro_mask[idx] and not in_block:
                in_block, block_start = True, time_arr[idx]
            elif not neuro_mask[idx] and in_block:
                in_block = False
                ax_temp.axvspan(block_start, time_arr[idx - 1],
                                alpha=0.12, color="#2980b9", zorder=1)
        if in_block:
            ax_temp.axvspan(block_start, time_arr[-1],
                            alpha=0.12, color="#2980b9", zorder=1)

    ax_temp.axhline(TARGET_TEMP_LOW_C, color="#c0392b", linewidth=1.1,
                    linestyle="--",
                    label=f"Therapeutic floor ({TARGET_TEMP_LOW_C}°C)")
    ax_temp.axhline(TARGET_TEMP_HIGH_C, color="#922b21", linewidth=1.0,
                    linestyle=":",
                    label=f"Therapeutic ceiling ({TARGET_TEMP_HIGH_C}°C)")
    ax_temp.axhline(CONFIRM_TEMP_C, color="#f39c12", linewidth=0.9,
                    linestyle="--", alpha=0.7,
                    label=f"Confirmation ({CONFIRM_TEMP_C}°C)")
    ax_temp.axhline(SCALP_MAX_TEMP_C, color="#2980b9", linewidth=0.9,
                    linestyle=":", alpha=0.8,
                    label=f"Scalp limit ({SCALP_MAX_TEMP_C}°C)")
    ax_temp.fill_between(
        time_arr, TARGET_TEMP_LOW_C, TARGET_TEMP_HIGH_C,
        alpha=0.08, color="#e74c3c", label="Therapeutic zone"
    )
    ax_temp.fill_between(
        time_arr, TARGET_TEMP_LOW_C, temp_arr,
        where=(temp_arr >= TARGET_TEMP_LOW_C),
        color="#e74c3c", alpha=0.20, zorder=2
    )

    t_cursor = 0.0
    for p in passes:
        t_cursor += p["total_pass_time_s"] / 60
        lc  = "#aaaaaa" if p["pass_type"] == "Confirmation" else "#555555"
        lbl = (f"C{p['pass_number']}" if p["pass_type"] == "Confirmation"
               else f"T{p['pass_number']}")
        ax_temp.axvline(t_cursor, color=lc, linewidth=0.7,
                        linestyle="--", alpha=0.55)
        ax_temp.text(
            t_cursor - p["total_pass_time_s"] / 120,
            ax_temp.get_ylim()[1] if ax_temp.get_ylim()[1] != 1 else 65,
            lbl, ha="center", va="top", fontsize=7.5,
            color="#888888" if p["pass_type"] == "Confirmation" else "#333333"
        )

    neuro_patch = mpatches.Patch(
        color="#2980b9", alpha=0.25,
        label=f"Neuro eval ({NEURO_EVAL_TIME_MIN:.0f} min)"
    )
    handles, labels = ax_temp.get_legend_handles_labels()
    ax_temp.legend(handles + [neuro_patch], labels + [neuro_patch.get_label()],
                   fontsize=7.5, loc="upper right", framealpha=0.9, ncol=2)
    ax_temp.set_xlabel("Time (min)", fontsize=10)
    ax_temp.set_ylabel("Temperature (°C)", fontsize=10)
    ax_temp.set_title(
        "Brain Target & Scalp Temperature — All Passes\n"
        f"(C=Confirmation, T=Therapeutic  |  "
        f"🔵 = 15-min neuro eval  |  "
        f"Skull cooled by {COOLING_WATER_TEMP_C}°C degassed water)",
        fontsize=11, fontweight="bold"
    )
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)
    ax_temp.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_temp.set_axisbelow(True)

    # ══ SDR Kurtosis Panel ═════════════════════════════════════════════════════
    ax_kurtosis.set_xlim(0, 1)
    ax_kurtosis.set_ylim(0, 1)
    ax_kurtosis.axis("off")
    ax_kurtosis.set_title("SDR Kurtosis Assessment", fontsize=11, fontweight="bold")
    x      = np.linspace(0, 1, 300)
    k      = inputs["sdr_kurtosis"]
    sigma  = max(0.05, 0.18 - k * 0.04)
    mu     = inputs["sdr_ratio"]
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
                     ha="center", va="center", fontsize=10, color=kt_color)
    ax_kurtosis.text(0.5, 0.17, kurtosis_info["tremor_prognosis"],
                     ha="center", va="center",
                     fontsize=8, color="#333333", style="italic")
    ax_kurtosis.text(0.5, 0.07, kurtosis_info["description"],
                     ha="center", va="center",
                     fontsize=7, color="#555555", linespacing=1.4)

    # ══ CEM43 Thresholds Panel ═════════════════════════════════════════════════
    ax_cem43pan.set_xlim(0, 1)
    ax_cem43pan.set_ylim(0, 1)
    ax_cem43pan.axis("off")
    ax_cem43pan.set_title("CEM43 Clinical Thresholds",
                          fontsize=11, fontweight="bold")
    ax_cem43pan.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.88), 1.0, 0.12,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor="#2c3e50"
    ))
    ax_cem43pan.text(0.5, 0.945, "CEM43 Dose Summary",
                     ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color="white")

    thresholds = [
        (CEM43_DAMAGE_50PCT, "50% Damage Prob.",
         final_ci["damage_50pct_met"], "#e67e22"),
        (CEM43_BOUNDARY_PRED, "Lesion Boundary Pred.",
         final_ci["boundary_pred_met"], "#8e44ad"),
        (CEM43_T1_LESION_PRED, "T1 Volume Pred. (1-day)",
         final_ci["t1_pred_met"], "#2980b9"),
    ]
    thresh_y = [0.76, 0.60, 0.44]
    for (thresh, label, met, tcolor), ty in zip(thresholds, thresh_y):
        status_col = "#27ae60" if met else "#e74c3c"
        fill_w     = min(0.98, sim["total_cem43"] / thresh)
        ax_cem43pan.add_patch(mpatches.FancyBboxPatch(
            (0.02, ty - 0.045), 0.96, 0.115,
            boxstyle="round,pad=0.01",
            linewidth=1, edgecolor="#dddddd", facecolor="#f8f9fa"
        ))
        ax_cem43pan.add_patch(mpatches.FancyBboxPatch(
            (0.02, ty - 0.020), fill_w * 0.96, 0.055,
            boxstyle="square,pad=0.0",
            linewidth=0, facecolor=tcolor + "44"
        ))
        ax_cem43pan.text(0.05, ty + 0.052,
                         f"{label}  ({thresh} CEM43)",
                         ha="left", va="center",
                         fontsize=7.5, color="#333333", fontweight="bold")
        ax_cem43pan.text(0.95, ty + 0.052,
                         "✔" if met else "✘",
                         ha="right", va="center",
                         fontsize=11, color=status_col, fontweight="bold")
        ax_cem43pan.text(0.05, ty - 0.002,
                         f"Current: {sim['total_cem43']:.3f}  /  "
                         f"Target: {thresh}",
                         ha="left", va="center",
                         fontsize=7.2, color="#666666")

    ax_cem43pan.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.09), 0.96, 0.22,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor=lesion_color,
        facecolor=lesion_color + "18"
    ))
    ax_cem43pan.text(0.5, 0.29, "Estimated Lesion Size",
                     ha="center", va="center",
                     fontsize=9, fontweight="bold", color=lesion_color)
    ax_cem43pan.text(0.5, 0.218,
                     f"T1:  {final_ci['est_lesion_t1_mm']:.2f} mm  "
                     f"(ref: {LESION_T1_MEAN_MM} ± {LESION_T1_STD_MM} mm)",
                     ha="center", va="center", fontsize=8, color="#333333")
    ax_cem43pan.text(0.5, 0.158,
                     f"T2:  {final_ci['est_lesion_t2_mm']:.2f} mm  "
                     f"(ref: {LESION_T2_MEAN_MM} ± {LESION_T2_STD_MM} mm)",
                     ha="center", va="center", fontsize=8, color="#333333")
    ax_cem43pan.text(0.5, 0.102,
                     f"Damage prob: {final_ci['lesion_prob_pct']:.1f}%  |  "
                     f"Focus: ~{FOCUS_WIDTH_MM} mm",
                     ha="center", va="center", fontsize=7.5, color="#555555")

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
    conf_patch = mpatches.Patch(color="#95a5a6", label="Confirmation")
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

    # ══ Cumulative Energy + Scalp ══════════════════════════════════════════════
    ax_cum.plot(pass_nums, pass_cum, color="#8e44ad",
                linewidth=2.2, marker="o", markersize=6,
                zorder=3, label="Cumulative energy (J)")
    ax_cum.fill_between(pass_nums, pass_cum, alpha=0.10, color="#8e44ad")
    ax2_cum = ax_cum.twinx()
    ax2_cum.plot(pass_nums, pass_scalp, color="#3498db",
                 linewidth=1.6, marker="s", markersize=5,
                 linestyle="--", zorder=3, label="Scalp peak (°C)")
    ax2_cum.axhline(SCALP_MAX_TEMP_C, color="#2980b9",
                    linewidth=0.9, linestyle=":", alpha=0.8)
    ax2_cum.set_ylabel("Scalp Peak Temp (°C)", fontsize=9, color="#3498db")
    ax2_cum.tick_params(axis="y", labelcolor="#3498db")
    ax2_cum.set_ylim(10, 25)
    for pn, pc in zip(pass_nums, pass_cum):
        ax_cum.text(pn, pc + 30, f"{pc:.0f}J",
                    ha="center", va="bottom", fontsize=7.5, color="#8e44ad")
    l1, lb1 = ax_cum.get_legend_handles_labels()
    l2, lb2 = ax2_cum.get_legend_handles_labels()
    ax_cum.legend(l1 + l2, lb1 + lb2,
                  fontsize=7.5, loc="upper left", framealpha=0.85)
    ax_cum.set_xlabel("Pass Number", fontsize=10)
    ax_cum.set_ylabel("Cumulative Energy (J)", fontsize=10, color="#8e44ad")
    ax_cum.set_title("Cumulative Energy & Scalp Temperature",
                     fontsize=11, fontweight="bold")
    ax_cum.set_xticks(pass_nums)
    ax_cum.spines["top"].set_visible(False)
    ax_cum.yaxis.grid(True, linestyle="--", alpha=0.30)
    ax_cum.set_axisbelow(True)

    # ══ CEM43 per pass ═════════════════════════════════════════════════════════
    ax_cem.bar(pass_nums, pass_cem, color="#e67e22",
               edgecolor="none", width=0.65, label="Per-pass CEM43")
    ax_cem.plot(pass_nums, pass_cum_cem, color="#d35400",
                linewidth=2.0, marker="D", markersize=5,
                zorder=4, label="Cumulative CEM43")
    ax_cem.axhline(CEM43_DAMAGE_50PCT, color="#e67e22", linewidth=1.0,
                   linestyle=":", alpha=0.8,
                   label=f"50% damage ({CEM43_DAMAGE_50PCT})")
    ax_cem.axhline(CEM43_BOUNDARY_PRED, color="#8e44ad", linewidth=1.1,
                   linestyle="--",
                   label=f"Boundary ({CEM43_BOUNDARY_PRED})")
    ax_cem.axhline(CEM43_T1_LESION_PRED, color="#2980b9", linewidth=1.0,
                   linestyle="-.",
                   label=f"T1 vol. ({CEM43_T1_LESION_PRED})")
    for pn, pc in zip(pass_nums, pass_cem):
        ax_cem.text(pn, pc + 0.0002, f"{pc:.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax_cem.legend(fontsize=7.2, loc="upper left", framealpha=0.85)
    ax_cem.set_xlabel("Pass Number", fontsize=10)
    ax_cem.set_ylabel("CEM43 (min)", fontsize=10)
    ax_cem.set_title("Thermal Dose per Pass & Cumulative (CEM43)",
                     fontsize=11, fontweight="bold")
    ax_cem.set_xticks(pass_nums)
    ax_cem.spines["top"].set_visible(False)
    ax_cem.spines["right"].set_visible(False)
    ax_cem.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax_cem.set_axisbelow(True)

    # ══ T1/T2 Ratio Panel ══════════════════════════════════════════════════════
    # T1/T2 ratio trajectory per pass
    t1t2_phase_colors = [p["t1t2_info"]["color"] for p in passes]

    ax_t1t2.plot(pass_nums, pass_t1t2, color="#5d6d7e",
                 linewidth=2.0, marker="o", markersize=6,
                 zorder=4, label="T1/T2 ratio")

    # Color each point by phase
    for pn, rv, pc in zip(pass_nums, pass_t1t2, t1t2_phase_colors):
        ax_t1t2.scatter(pn, rv, color=pc, s=55, zorder=5, edgecolors="white",
                        linewidths=0.8)

    # Reference threshold lines
    ax_t1t2.axhline(T1T2_EDEMA_THRESHOLD, color="#e74c3c", linewidth=1.1,
                    linestyle="--",
                    label=f"Edema threshold ({T1T2_EDEMA_THRESHOLD})")
    ax_t1t2.axhline(T1T2_RATIO_CORE_MEAN, color="#2ecc71", linewidth=1.1,
                    linestyle="-",
                    label=f"Core reference ({T1T2_RATIO_CORE_MEAN})")
    ax_t1t2.axhline(T1T2_CHRONIC_THRESHOLD, color="#27ae60", linewidth=1.0,
                    linestyle="-.",
                    label=f"Chronic threshold ({T1T2_CHRONIC_THRESHOLD})")
    ax_t1t2.axhline(T1T2_NORMAL_TISSUE, color="#888888", linewidth=0.8,
                    linestyle=":", alpha=0.7,
                    label=f"Normal tissue ({T1T2_NORMAL_TISSUE})")

    # Reference range band
    ax_t1t2.axhspan(
        T1T2_RATIO_CORE_MEAN - T1T2_RATIO_CORE_STD,
        T1T2_RATIO_CORE_MEAN + T1T2_RATIO_CORE_STD,
        alpha=0.10, color="#2ecc71", label="Core reference range"
    )

    # T1/T2 corrected lesion size on secondary axis
    ax2_t1t2 = ax_t1t2.twinx()
    ax2_t1t2.plot(pass_nums, pass_t1t2_corr, color="#8e44ad",
                  linewidth=1.6, marker="s", markersize=5,
                  linestyle="--", zorder=3, alpha=0.85,
                  label="T1/T2 corrected size (mm)")
    ax2_t1t2.set_ylabel("T1/T2 Corrected Lesion Size (mm)",
                         fontsize=9, color="#8e44ad")
    ax2_t1t2.tick_params(axis="y", labelcolor="#8e44ad")
    ax2_t1t2.set_ylim(0, 10)

    l1, lb1 = ax_t1t2.get_legend_handles_labels()
    l2, lb2 = ax2_t1t2.get_legend_handles_labels()
    ax_t1t2.legend(l1 + l2, lb1 + lb2,
                   fontsize=7.0, loc="lower right", framealpha=0.88, ncol=2)
    ax_t1t2.set_xlabel("Pass Number", fontsize=10)
    ax_t1t2.set_ylabel("T1/T2 Signal Intensity Ratio", fontsize=10)
    ax_t1t2.set_title(
        "T1/T2 Ratio per Pass — Lesion Phase Classification\n"
        f"(Ref: {T1T2_RATIO_CORE_MEAN} ± {T1T2_RATIO_CORE_STD}  |  "
        f"Edema <{T1T2_EDEMA_THRESHOLD}  |  "
        f"Chronic >{T1T2_CHRONIC_THRESHOLD})",
        fontsize=11, fontweight="bold"
    )
    ax_t1t2.set_xticks(pass_nums)
    ax_t1t2.set_ylim(0.35, 1.10)
    ax_t1t2.spines["top"].set_visible(False)
    ax_t1t2.yaxis.grid(True, linestyle="--", alpha=0.30)
    ax_t1t2.set_axisbelow(True)

    # Phase legend patches
    phase_patches = [
        mpatches.Patch(color="#e74c3c", label="Acute Edema (<0.65)"),
        mpatches.Patch(color="#f39c12", label="Subacute (0.65–0.72)"),
        mpatches.Patch(color="#2ecc71", label="Established (0.72–0.80)"),
        mpatches.Patch(color="#27ae60", label="Chronic (>0.80)"),
    ]
    ax_t1t2.legend(
        handles=phase_patches,
        fontsize=7.0, loc="upper left",
        framealpha=0.88, title="T1/T2 Phase", title_fontsize=7
    )

    # ══ Lesion Size Tracker (full width) ══════════════════════════════════════
    ax_lesion.plot(pass_nums, pass_lesion_t1, color="#c0392b",
                   linewidth=2.0, marker="o", markersize=6,
                   label=f"Est. T1  ref: "
                         f"{LESION_T1_MEAN_MM}±{LESION_T1_STD_MM} mm")
    ax_lesion.plot(pass_nums, pass_lesion_t2, color="#922b21",
                   linewidth=1.6, marker="s", markersize=5,
                   linestyle="--",
                   label=f"Est. T2  ref: "
                         f"{LESION_T2_MEAN_MM}±{LESION_T2_STD_MM} mm")
    ax_lesion.plot(pass_nums, pass_t1t2_corr, color="#8e44ad",
                   linewidth=1.8, marker="^", markersize=6,
                   linestyle="-.",
                   label="T1/T2 ratio corrected size")
    ax_lesion.axhspan(
        LESION_T1_MEAN_MM - LESION_T1_STD_MM,
        LESION_T1_MEAN_MM + LESION_T1_STD_MM,
        alpha=0.08, color="#c0392b", label="T1 reference range"
    )
    ax_lesion.axhspan(
        LESION_T2_MEAN_MM - LESION_T2_STD_MM,
        LESION_T2_MEAN_MM + LESION_T2_STD_MM,
        alpha=0.06, color="#922b21", label="T2 reference range"
    )
    ax_lesion.axhline(FOCUS_WIDTH_MM, color="#555555", linewidth=0.9,
                      linestyle=":", label=f"Focus (~{FOCUS_WIDTH_MM} mm)")
    ax2_lesion = ax_lesion.twinx()
    ax2_lesion.plot(pass_nums, pass_dmg_prob, color="#8e44ad",
                    linewidth=1.5, marker="D", markersize=4,
                    linestyle=":", zorder=3, alpha=0.7,
                    label="Damage prob. (%)")
    ax2_lesion.axhline(50, color="#8e44ad", linewidth=0.7,
                       linestyle=":", alpha=0.6)
    ax2_lesion.set_ylabel("Damage Probability (%)", fontsize=9, color="#8e44ad")
    ax2_lesion.tick_params(axis="y", labelcolor="#8e44ad")
    ax2_lesion.set_ylim(0, 105)
    l1, lb1 = ax_lesion.get_legend_handles_labels()
    l2, lb2 = ax2_lesion.get_legend_handles_labels()
    ax_lesion.legend(l1 + l2, lb1 + lb2,
                     fontsize=7.5, loc="upper left", framealpha=0.85, ncol=2)
    ax_lesion.set_xlabel("Pass Number", fontsize=10)
    ax_lesion.set_ylabel("Estimated Lesion Size (mm)", fontsize=10)
    ax_lesion.set_title(
        "Estimated Lesion Size — T1 / T2 / T1:T2 Ratio Corrected per Pass\n"
        f"(T1/T2 corrected size reduces measurement error vs T2 alone  |  "
        f"Border accuracy: {LESION_BORDER_ERR_MM} ± {LESION_BORDER_STD_MM} mm)",
        fontsize=11, fontweight="bold"
    )
    ax_lesion.set_xticks(pass_nums)
    ax_lesion.spines["top"].set_visible(False)
    ax_lesion.yaxis.grid(True, linestyle="--", alpha=0.30)
    ax_lesion.set_axisbelow(True)

    # ══ SHAP ══════════════════════════════════════════════════════════════════
    ax_shap.barh(shap_labels, shap_values, color=shap_colors,
                 edgecolor="none", height=0.6)
    ax_shap.axvline(0, color="black", linewidth=1.0)
    for i, (label, val) in enumerate(zip(shap_labels, shap_values)):
        offset = 0.15 if val >= 0 else -0.15
        ha     = "left" if val >= 0 else "right"
        ax_shap.text(val + offset, i, f"{val:+.1f}J",
                     va="center", ha=ha, fontsize=9)
    for i, label in enumerate(shap_labels):
        if label == "SDR Kurtosis":
            ax_shap.get_yticklabels()[i].set_color(kt_color)
            ax_shap.get_yticklabels()[i].set_fontweight("bold")
    pos_patch = mpatches.Patch(color="#d73027", label="↑ Increases energy output")
    neg_patch = mpatches.Patch(color="#4575b4", label="↓ Decreases energy output")
    ax_shap.legend(handles=[pos_patch, neg_patch],
                   fontsize=9, loc="lower right", framealpha=0.9)
    ax_shap.set_xlabel("SHAP Contribution to Predicted Energy Output (J)",
                       fontsize=10)
    ax_shap.set_title("Feature Contributions (SHAP) — All Features",
                      fontsize=11, fontweight="bold")
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    ax_shap.spines["left"].set_visible(False)
    ax_shap.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax_shap.set_axisbelow(True)

    # ══ Timing footnote ════════════════════════════════════════════════════════
    fig.text(
        0.5, -0.010,
        f"Model computation:  {comp_time_ms:.2f} ms    |    "
        f"Surgeon planning (NIH):  {PLANNING_TIME_LOW_MIN}–"
        f"{PLANNING_TIME_HIGH_MIN} min    |    "
        f"Total prep:  {TOTAL_PREP_LOW_MIN}–{TOTAL_PREP_HIGH_MIN} min    |    "
        f"Full procedure:  {TOTAL_PROC_LOW_HRS:.0f}–"
        f"{TOTAL_PROC_HIGH_HRS:.0f} hrs    |    "
        f"Neuro eval:  {NEURO_EVAL_TIME_MIN:.0f} min × "
        f"{n_therapeutic} passes = {total_neuro_min:.0f} min",
        ha="center", va="center",
        fontsize=8.5, color="#2c3e50", fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#eaf4fb",
            edgecolor="#aed6f1",
            linewidth=1.3
        )
    )

    fig.suptitle(
        "MRgFUS Per-Pass Energy & Thermal Ablation Tool  |  University of Florida\n"
        "⚠  Simulated Data — Not for Clinical Use",
        fontsize=13, fontweight="bold", y=1.01, color="#2c3e50"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ── 8. Main pipeline ───────────────────────────────────────────────────────────
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
    t_start = time.perf_counter()
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
    t_end = time.perf_counter()
    sim["computation_time_ms"] = round((t_end - t_start) * 1000, 3)

    print(f"\n  ⏱  Model computation time: {sim['computation_time_ms']:.3f} ms\n")
    print(interpret_passes(sim))
    plot_per_pass(sim, save_path=save_plot)
    return sim


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
        save_plot         = "patient_A_t1t2.png"
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
        save_plot         = "patient_B_t1t2.png"
    )




