"""
================================================================================
 Supercavitation — Transom Planing Force (Tail-Slap Impact)
================================================================================
Kuyruk ucunun (transom) kavite duvarı ile temasından kaynaklanan kuvvetler.

Model: Dzielski-Kurdila 2003 (IEEE J. Vibration & Control)
"Impact of Cavitation Collapse on Supercavitating Vehicles"

Planing sadece TRANSOM'da (kuyruğun en arkasında) uygulanır — bu, literaturde
sık yapılan hata (araç boyu boyunca dağıtmak) ile tutarsızdır.

İşaret Konvansiyonu:
  - Dikey kuvvet: YUKARI pozitif (legacy); gövde z-aşağıya dönüşüm dynamics/model.py'de
  - α_p: local planing angle (cavity axis slope at transom)
  - F_L: lift perpendicular to cavity surface (upward)
  - My: pitching moment (positive = nose-up)
================================================================================
"""

import numpy as np
from .constants import (
    RHO, V_MIN, G, CD_BASE_EXPOSED, CD_BASE_FULLY_WET,
    TRANSOM_EXIT_START, TRANSOM_EXIT_WIDTH,
)
from .smooth import smoothstep


def planing_force_dzielski_kurdila(V: float, R_v: float, R_c: float,
                                  alpha_p: float, h_immersion: float) -> float:
    """
    Compute transom planing (tail-slap) force using Dzielski-Kurdila 2003 model.

    F_pz = ρ·V²·R_v² · [1 − (Δ/(h+Δ))²] · [(1+h/R_v)/(1+2h/R_v)] · α_p

    where:
      R_v = vehicle radius at transom
      R_c = cavity radius at transom
      Δ = R_c − R_v  (gap between cavity and vehicle)
      h = immersion depth (transom below cavity axis at tail)
      α_p = local planing angle (dh/dx at transom)

    Fiziksel anlam:
      - Transom kavitenin duvarına "çarpar" (impact)
      - Etki: normal kuvvet, asimetrisi α_p ile modüle edilir
      - Işık-dış dalış (h>0): kavite araç etrafında — planing aktif
      - Dahil (h<0): transom kavite içinde — planing = 0

    Formülün unsurları:
      1. ρV² — dinamik basınç etkisi
      2. R_v² — geometrik ölçek (yarıçap karesel)
      3. [1 − (Δ/(h+Δ))²] — gap factor (Δ=0 → 1; Δ=h → 0.75)
      4. [(1+h/R_v)/(1+2h/R_v)] — cavity shape factor
      5. α_p — angle of attack (impact angle)

    Args:
        V: Velocity [m/s]
        R_v: Vehicle radius at transom [m]
        R_c: Cavity radius at transom [m]
        alpha_p: Local planing angle (cavity axis slope) [rad]
                 = dh_cavity/dx at transom
                 Positive = cavity tilted down (transom above axis)
        h_immersion: Immersion depth (delta in literature) [m]
                    = R_v + |h_c| − R_c
                    Positive = transom protruding from cavity

    Returns:
        F_pz: Planing lift force (upward, positive in aeronautical sense) [N]
              Actually returned as force magnitude (magnitude always ≥ 0)
              Sign interpretation: positive = upward (reduces weight)
              In body-fixed z (down): F_pz > 0 means negative F_z contribution

    Sign Convention:
      - F_pz > 0: upward planing force (physical lift)
      - Applied at transom (x = L_vehicle)
      - Contributes negative M_y if x_fin > x_cg (tail lifts → nose-down)

    References:
      - Dzielski, J. E., & Kurdila, A. J. (2003). "A Low-Order Model for
        Supercavitating Vehicles." IEEE J. Ocean. Eng., 28(3), 460-471.
      - Formulation adapted for Logvinovich cavity axis (asymptotic decay)
    """

    V_safe = max(V, V_MIN)
    R_v_safe = max(R_v, 1e-6)
    R_c_safe = max(R_c, 1e-6)
    h_safe = max(h_immersion, -1e-6)  # Allow slightly negative (cavity just covers)

    # Gap between cavity and vehicle
    gap = max(R_c_safe - R_v_safe, 0.0)

    # Only apply planing if h > 0 (transom partially exposed)
    if h_safe <= 0.0:
        return 0.0

    # Factor 1: gap reduction [1 − (Δ/(h+Δ))²]
    denominator_gap = h_safe + gap
    if denominator_gap < 1e-9:
        f1 = 0.0
    else:
        gap_ratio = gap / denominator_gap
        f1 = 1.0 - gap_ratio * gap_ratio

    # Factor 2: cavity shape [(1+h/R_v) / (1+2h/R_v)]
    f2_num = 1.0 + h_safe / R_v_safe
    f2_den = 1.0 + 2.0 * h_safe / R_v_safe
    if abs(f2_den) > 1e-9:
        f2 = f2_num / f2_den
    else:
        f2 = 0.0

    # Magnitude of planing angle
    alpha_p_mag = abs(alpha_p)

    # Main formula
    q = 0.5 * RHO * V_safe * V_safe
    F_pz = q * R_v_safe * R_v_safe * f1 * f2 * alpha_p_mag

    return max(F_pz, 0.0)


def planing_angle_from_cavity_slope(x_transom: float, V: float,
                                    k_dev: float = 0.4,
                                    x_open: float = None,
                                    alpha_eff: float = 0.0) -> float:
    """
    Compute local planing angle at transom (cavity axis slope).

    α_p = dh_cavity/dx = d/dx[g·x²/(2V²) + k_dev·α_eff·x_open·(1−exp(−x/x_open))]

    Türev:
      dh_grav/dx = g·x/V²
      dh_angle/dx = k_dev·α_eff·exp(−x/x_open)

    Fiziksel anlam:
      - Yerçekimi teriminin türevi: lineer artış (x ile)
      - Açı teriminin türevi: exponential decay (uzaklıkla sönümleme)
      - Transom'da (x büyük): α_p ≈ g·x/V² (yerçekimi baskın)

    Args:
        x_transom: Transom position (kavite koordinatında) [m]
        V: Velocity [m/s]
        k_dev: Deviation damping factor (0-1)
        x_open: Characteristic opening length [m]
               If None, returns 0 (no angle contribution)
        alpha_eff: Effective cavitator angle [rad]

    Returns:
        alpha_p: Planing angle (cavity slope at transom) [rad]

    Sign Convention:
      - Positive α_eff + gravity → positive α_p (cavity curves down)
      - Negative α_eff → can reduce α_p (nose-down deflection)
    """

    V_safe = max(V, V_MIN)
    x_safe = max(x_transom, 0.0)

    # Gravity contribution (always positive, increasing with x)
    dh_grav_dx = G * x_safe / (V_safe * V_safe)

    # Angle contribution (decaying exponential)
    if x_open is None or x_open < 1e-6:
        dh_angle_dx = 0.0
    else:
        dh_angle_dx = k_dev * alpha_eff * np.exp(-x_safe / x_open)

    alpha_p = dh_grav_dx + dh_angle_dx

    return alpha_p


def planing_moment_contribution(F_pz: float, x_transom: float, x_cg: float) -> float:
    """
    Compute pitching moment from planing force at transom.

    M_y = (x_cg - x_transom) · F_pz

    Moment arm sign convention (right-hand rule):
      - Transom at x = L_vehicle (aft end)
      - C.G. typically at L_vehicle/2 (middle)
      - x_cg < x_transom (c.g. forward of transom)
      - Upward F_pz (planing) at transom (aft) → negative M_y (nose-down)

    Args:
        F_pz: Planing lift force [N] (positive = upward)
        x_transom: Transom location [m]
        x_cg: Center of gravity location [m]

    Returns:
        M_y: Pitching moment [N·m]
             Negative M_y = nose-down (tail lifts)

    Sign Convention:
      - Right-hand rule: fingers curl x → z, thumb along +y
      - Consistent with vehicle pitch control (elevator-like behavior)
    """

    moment_arm = x_cg - x_transom  # Typically negative (c.g. forward)
    M_y = moment_arm * F_pz

    return M_y


def planing_immersion_depth(R_v: float, h_cavity_axis: float, R_c: float) -> float:
    """
    Compute immersion depth of transom (how much it protrudes from cavity).

    h_imm = R_v + |h_cavity_axis| − R_c

    where:
      R_v = vehicle radius at transom
      h_cavity_axis = vertical offset of cavity axis (from vehicle centerline)
                      Positive = cavity axis below vehicle centerline
      R_c = cavity radius at transom

    Fiziksel anlam:
      - h_imm > 0: transom partially outside cavity (planing region)
      - h_imm = 0: transom just touches cavity wall
      - h_imm < 0: transom fully inside cavity (no planing)
      - h_imm > 2·R_v: transom fully exposed to water (classical hydrodynamics)

    Args:
        R_v: Vehicle radius [m]
        h_cavity_axis: Cavity axis vertical displacement [m]
                       (absolute value used for immersion)
        R_c: Cavity radius [m]

    Returns:
        h_imm: Immersion depth [m]

    Sign Convention:
      - h_imm ≥ 0 (depth is positive quantity)
      - h_imm = 0: critical case (cavity just touches transom)
    """

    R_v_safe = max(R_v, 1e-6)
    R_c_safe = max(R_c, 1e-6)

    h_imm = R_v_safe + abs(h_cavity_axis) - R_c_safe

    return max(h_imm, 0.0)  # Force non-negative (immersion depth is positive quantity)


# ============================================================================
# TRANSOM KUVVETLERİ — legacy ile birebir (satır 1080-1165)
# ============================================================================

def transom_forces_legacy(V: float, R_v: float, rc_tail: float, h_tail: float,
                          alpha_p: float, S_base: float,
                          planing_enabled: bool = True) -> dict:
    """
    Transom planing kuvveti ve taban basınç sürüklemesi (legacy simulate() birebir).

      δ = R_v + |h_tail| − rc_tail
      δ ≤ 0 veya rc_tail ≤ 1e-4      → temas yok: F = 0
      0 < δ < 2R_v  (planing)         → F_up = ρV²R_v²·f1·f2·α_p   (Dzielski-Kurdila,
                                         ρV² — ½ YOK), F_press = F_up·α_p
      δ ≥ 2R_v      (tamamen dışarıda) → F_up = 0, F_press = ½ρV²·S_base·CD_BASE_EXPOSED
      f1 = 1 − (Δ/(δ+Δ))², Δ = max(rc_tail − R_v, 0);  f2 = (1+δ/R_v)/(1+2δ/R_v)

    Not: planing_force_dzielski_kurdila() ½ρV² ve |α_p| kullanır; legacy ρV² ve
    işaretli α_p kullanır. Bu fonksiyon legacy'yi izler.

    Returns dict (legacy konvansiyonu):
        F_planing  dikey kuvvet, YUKARI pozitif [N]
        F_press    taban/planing basınç sürüklemesi, geri yönlü büyüklük [N]
        delta      dalma derinliği [m]
        regime     "disabled" | "covered" | "planing" | "exposed"
    """
    delta = R_v + abs(h_tail) - rc_tail
    out = {"F_planing": 0.0, "F_press": 0.0, "delta": delta, "regime": "covered"}
    if not planing_enabled:
        out["regime"] = "disabled"
        return out
    if delta <= 0.0 or R_v <= 1e-6 or rc_tail <= 1e-4:
        return out
    if delta >= 2.0 * R_v:
        out["F_press"] = 0.5 * RHO * V * V * S_base * CD_BASE_EXPOSED
        out["regime"] = "exposed"
        return out
    gap = max(rc_tail - R_v, 0.0)
    f1 = 1.0 - (gap / max(delta + gap, 1e-9)) ** 2
    f2 = (1.0 + delta / R_v) / (1.0 + 2.0 * delta / R_v)
    F_up = RHO * V * V * R_v * R_v * f1 * f2 * alpha_p
    out.update(F_planing=F_up, F_press=F_up * alpha_p, regime="planing")
    return out


def transom_forces_smooth(V: float, R_v: float, rc_tail: float, h_tail: float,
                          alpha_p: float, S_base: float,
                          planing_enabled: bool = True) -> dict:
    """
    Transom kuvvetleri — rejim geçişleri sürekli (legacy_exact olmayan varsayılan model).

    Legacy'deki sıçramalar ve giderilmesi:
      (1) Kavite ucu transomu geçtiği an (rc_tail: 0 → Rn) planing 0 → ~660 N.
          → Kaplama ağırlığı c = smoothstep(rc_tail/R_v): kavite transomda gövdeden
            dar iken kuyruk büyük ölçüde ıslaktır, Dzielski-Kurdila "duvara çarpma"
            modeli geçerli değildir; c → 1 olunca tam devreye girer.
      (2) Kavite yok ↔ var: legacy taban sürüklemesi 0.15·q·S ↔ 0 anahtarlıyor.
          → Taban sürüklemesi (1 − c)·CD_BASE_FULLY_WET·q·S (kavite transomu örtmedikçe).
      (3) δ = 2R_v eşiği (planing → tamamen dışarıda) anahtarlaması.
          → w = smoothstep((δ/R_v − 1.5)/1.0) ile planing ↔ CD_BASE_EXPOSED harmanlaması.
      δ → 0⁺ girişi zaten süreklidir (Δ > 0 iken f1 → 0).

      F_DK    = ρV²R_v²·f1·f2·α_p   (δ > 0; aksi 0)
      F_up    = c·(1 − w)·F_DK                                     [planing kapalıysa 0]
      F_press = (1 − c)·CD_FW·q·S + c·[(1 − w)·F_DK·α_p + w·CD_EXP·q·S]
    Taban sürüklemesi planing_enabled'dan bağımsızdır (planing değil, taban basıncı).

    Returns: F_planing (yukarı +), F_press (geri +), delta, cover_tail (c), regime
    """
    q = 0.5 * RHO * V * V
    c = smoothstep(rc_tail / R_v) if R_v > 1e-6 else 0.0
    delta = R_v + abs(h_tail) - rc_tail

    F_dk = 0.0
    if delta > 0.0 and R_v > 1e-6 and planing_enabled:
        gap = max(rc_tail - R_v, 0.0)
        f1 = 1.0 - (gap / max(delta + gap, 1e-9)) ** 2
        f2 = (1.0 + delta / R_v) / (1.0 + 2.0 * delta / R_v)
        F_dk = RHO * V * V * R_v * R_v * f1 * f2 * alpha_p
    w = smoothstep((delta / R_v - TRANSOM_EXIT_START) / TRANSOM_EXIT_WIDTH) if R_v > 1e-6 else 0.0

    F_up = c * (1.0 - w) * F_dk
    F_press = ((1.0 - c) * CD_BASE_FULLY_WET * q * S_base
               + c * ((1.0 - w) * F_dk * alpha_p + w * CD_BASE_EXPOSED * q * S_base))

    if c < 0.5:
        regime = "tail_wet"
    elif delta <= 0.0:
        regime = "covered"
    elif w > 0.5:
        regime = "exposed"
    else:
        regime = "planing"
    return {"F_planing": F_up, "F_press": F_press, "delta": delta,
            "cover_tail": c, "regime": regime}


# ============================================================================
# COMPREHENSIVE PLANING FORCE (wrapper)
# ============================================================================

def compute_planing_force(V: float, L_transom: float, R_v: float, R_c: float,
                         h_cavity_axis: float, alpha_eff: float, x_cg: float,
                         k_dev: float = 0.4, x_open: float = None,
                         planing_enabled: bool = True) -> dict:
    """
    Comprehensive transom planing force computation.

    Integrates immersion, angle, and moment calculations.

    Args:
        V: Velocity [m/s]
        L_transom: Transom axial position [m]
        R_v: Vehicle radius at transom [m]
        R_c: Cavity radius at transom [m]
        h_cavity_axis: Cavity axis vertical offset (absolute value used) [m]
        alpha_eff: Effective cavitator angle [rad]
        x_cg: Center of gravity location [m]
        k_dev: Deviation damping factor (0-1)
        x_open: Characteristic opening length [m] (if None, no angle effect)
        planing_enabled: Toggle planing on/off (user control)

    Returns:
        dict with keys:
          - "F_pz": Planing lift force (upward) [N]
          - "M_y": Planing pitching moment [N·m]
          - "h_imm": Immersion depth [m]
          - "alpha_p": Planing angle [rad]
          - "status": "active", "inactive_no_immersion", "inactive_disabled", "inactive_cavity_covers"

    Sign Convention:
      - F_pz > 0: upward (reduces weight)
      - M_y typically < 0 for planing aft (nose-down moment)
    """

    if not planing_enabled:
        return {
            "F_pz": 0.0, "M_y": 0.0,
            "h_imm": 0.0, "alpha_p": 0.0,
            "status": "inactive_disabled",
        }

    if V < V_MIN:
        return {
            "F_pz": 0.0, "M_y": 0.0,
            "h_imm": 0.0, "alpha_p": 0.0,
            "status": "inactive_no_speed",
        }

    # Immersion depth
    h_imm = planing_immersion_depth(R_v, h_cavity_axis, R_c)

    if h_imm <= 0.0:
        return {
            "F_pz": 0.0, "M_y": 0.0,
            "h_imm": 0.0, "alpha_p": 0.0,
            "status": "inactive_cavity_covers",
        }

    # If immersion depth >= 2·R_v, transom fully exposed (classical regime)
    # In this case, planing transitions to classical form drag
    # But we compute via Dzielski-Kurdila formula (should approach 0 as h → large)

    # Planing angle
    alpha_p = planing_angle_from_cavity_slope(L_transom, V, k_dev, x_open, alpha_eff)

    # Planing force
    F_pz = planing_force_dzielski_kurdila(V, R_v, R_c, alpha_p, h_imm)

    if F_pz < 1e-6:
        return {
            "F_pz": 0.0, "M_y": 0.0,
            "h_imm": h_imm, "alpha_p": alpha_p,
            "status": "inactive_zero_force",
        }

    # Moment
    M_y = planing_moment_contribution(F_pz, L_transom, x_cg)

    return {
        "F_pz": F_pz,
        "M_y": M_y,
        "h_imm": h_imm,
        "alpha_p": alpha_p,
        "status": "active",
    }
