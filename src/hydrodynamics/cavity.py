"""
================================================================================
 Supercavitation — Cavity Geometry and Ventilation Models
================================================================================
Kavite boyutları (Lc, Dc) ve efektif kavitasyon numarası hesapları.

Altı model (Garabedian, Savchenko, Semenenko, May, Vasin-Serebryakov,
Logvinovich) sağlanır — dispatcher pattern ile seçilir.

Ventilasyon: Hybrid σ (doğal + yapay) ve basınç (pc) dinamiği.
================================================================================
"""

import numpy as np
from .constants import (
    SIGMA_MIN, V_MIN, CAVITY_MIN,
    CX0_DISK, K_G_CAVITY_DEFAULT,
    RHO, P_VAP, TAU_PC, A_V_DEFAULT, K_LEAK, CAVITY_TAU, SIGMA_ONSET_WIDTH
)
from .smooth import smoothstep


def cavity_geometry(sigma: float, Dn: float, model: str = "savchenko",
                    Cx0: float = CX0_DISK, k_g_val: float = K_G_CAVITY_DEFAULT,
                    K_Dc_val: float = 1.0, K_Lc_val: float = 1.0) -> tuple:
    """
    Compute cavity length (Lc) and maximum diameter (Dc) using selected model.

    İşaret Konvansiyonu (Sign Convention):
      - Lc, Dc: pozitif, büyüklükler [m]
      - σ: efektif kavitasyon numarası (pozitif)
      - Cx: drag katsayısı (pozitif)

    Altı model arasında seçim (6 models):
      1. garabedian: Klasik asimptotik — küçük σ'da uzun kavite
      2. savchenko: Deneysel kalibrasyonu — mermi atışları referansı
      3. semenenko: Kısa, kalın — yüksek σ doygunluğu
      4. may: Orta yol — Birkhoff-Plesset tipi
      5. vasin_serebr: Slender body 0.7 düzeltmesi — ince uzun
      6. logvinovich: Klasik 1969 — kalın çap eğilimi

    Args:
        sigma: Efektif cavitation number (σ = 2(p∞−pc)/(ρV²)) [dimensionless]
        Dn: Cavitator diameter [m]
        model: Model seçimi — yukarıdaki anahtar kelimelerden
        Cx0: Cavitator drag coeff (disk: 0.82, cone için sin²(β/2)·factor)
        k_g_val: Reichardt-Garabedian constant (0.78-1.0 range)
        K_Dc_val: Diameter scaling factor (>1 = thicker, <1 = thinner)
        K_Lc_val: Length scaling factor (<1 = shorter)

    Returns:
        (Lc, Dc, Cx): Cavity length [m], diameter [m], drag coefficient
        Cx = Cx0·(1+σ) per Reichardt

    Raises:
        None — fonksiyon singularities karşısında güvenlidir (guards)
    """

    # Singularity guards
    s = max(sigma, SIGMA_MIN)
    s_safe = max(s, 1e-6)  # Extra safe for logarithm
    Cx = Cx0 * (1.0 + s)

    # Logarithm argument safety
    ln_inv_s = max(np.log(1.0 / s_safe), 0.1)

    # === Çap formülü (Diameter) ===
    # Standart Reichardt-Garabedian: Dc/Dn = √(Cx / (σ·k_g))
    # Bazı modeller özel Dc formülü kullanır.

    if model == "logvinovich":
        # Logvinovich düzeltme — kalın kavite eğilimi (1969)
        Dc = np.sqrt(Cx / s_safe) * Dn * (1.0 + 0.05 / (s_safe**0.3))
    elif model == "may":
        # May 1975 — (1+σ) faktörü ekstra
        Dc = np.sqrt(Cx * (1.0 + s) / s_safe) * Dn
    else:
        # Reichardt-Garabedian standard (Garabedian, Savchenko, Semenenko, Vasin)
        Dc = np.sqrt(Cx / (s_safe * k_g_val)) * Dn

    # === Uzunluk formülü (Length) ===
    # Her model kendi Lc/Dc oranını tanımlıyor

    if model == "garabedian":
        # Klasik asimptotik (1956, teorik):
        # Lc/Dc = (1/σ)·√(ln(1/σ))
        ratio_lc_dc = (1.0 / s_safe) * np.sqrt(ln_inv_s)
        Lc = Dc * ratio_lc_dc

    elif model == "savchenko":
        # Savchenko 2001 deneysel (mermi atışları):
        # Lc/Dc = (2/σ)·√(1 / [ln(1/σ)·(1+σ)])
        denominator = ln_inv_s * (1.0 + s)
        ratio_lc_dc = (2.0 / s_safe) * np.sqrt(1.0 / max(denominator, 1e-6))
        Lc = Dc * ratio_lc_dc

    elif model == "semenenko":
        # Semenenko (doygunluk eğilimi, kısa kavite):
        # Lc/Dn = max(1.92/σ − 3.0, 0)
        ratio_lc_dn = max(1.92 / s_safe - 3.0, 0.0)
        Lc = ratio_lc_dn * Dn

    elif model == "may":
        # May 1975 modifiye:
        # Lc/Dc = √(Cx)/σ · (1+0.5σ)^(−1)
        numerator = np.sqrt(max(Cx, 0.01))
        denominator = s_safe * (1.0 + 0.5 * s)
        ratio_lc_dc = numerator / max(denominator, 1e-6)
        Lc = Dc * ratio_lc_dc

    elif model == "vasin_serebr":
        # Vasin-Serebryakov slender body (0.7 düzeltme, asimptotik):
        # Lc/Dc = (1/σ)·√(ln(1/σ))·(1−0.3σ)·0.7
        ratio_lc_dc = ((1.0 / s_safe) * np.sqrt(ln_inv_s)
                       * (1.0 - 0.3 * s) * 0.7)
        Lc = Dc * ratio_lc_dc

    elif model == "logvinovich":
        # Logvinovich klasik (1969, (1−σ) düzeltme):
        # Lc/Dc = (1/σ)·√(ln(1/σ))·(1−σ)·0.85
        ratio_lc_dc = ((1.0 / s_safe) * np.sqrt(ln_inv_s)
                       * (1.0 - s) * 0.85)
        Lc = Dc * ratio_lc_dc

    else:
        # Bilinmeyen model → Savchenko'ya dön (güvenli varsayılan)
        denominator = ln_inv_s * (1.0 + s)
        ratio_lc_dc = (2.0 / s_safe) * np.sqrt(1.0 / max(denominator, 1e-6))
        Lc = Dc * ratio_lc_dc

    # === Kalibrasyon çarpanları (Calibration factors) ===
    # CFD veya deneysel uyum için ince ayar
    Dc = max(Dc * K_Dc_val, CAVITY_MIN)
    Lc = max(Lc * K_Lc_val, CAVITY_MIN)

    return Lc, Dc, Cx


def compute_ventilation_sigma(Cq_in: float, A_v: float = A_V_DEFAULT) -> float:
    """
    Compute ventilation-induced cavitation number (Spurk-Semenenko model).

    σ_vent = A_v / Cq_in

    where A_v is ventilation calibration constant (literature: 0.005-0.30).

    Fiziksel anlam:
      - Cq = Q / (V·Dn²) — boyutsuz havalandırma katsayısı
      - Daha büyük Cq (daha çok gaz) → daha küçük σ_vent (daha geniş kavite)
      - A_v deneysel kalibrasyon (Semenenko 2001, Epshtein 1970)

    Args:
        Cq_in: Ventilation coefficient = Q_in / (V·Dn²) [dimensionless]
        A_v: Ventilation calibration factor (default 0.035)

    Returns:
        σ_vent: Ventilation-limited σ [dimensionless]

    Sign Convention:
      - σ_vent > 0 (always)
      - Cq_in > 0 (positive volume flow) → σ_vent inverse ile orantılı
    """

    Cq_safe = max(Cq_in, 1e-9)  # Avoid division by zero
    sigma_vent = A_v / Cq_safe

    return max(sigma_vent, SIGMA_MIN)


def update_cavity_pressure(pc_prev: float, pc_target: float, dt: float,
                          tau_pc: float = TAU_PC) -> float:
    """
    Update cavity pressure using first-order relaxation dynamics.

    Kullanıcı-belirlenmiş hedef basınç (pc_target) zamanla pc'ye yaklaşır.
    Zaman sabiti: τ_pc ≈ 0.3s (havalandırma basınç tepkisi).

    Model: dpc/dt = (pc_target − pc) / τ_pc
    Çözüm (first-order): pc(t+dt) = pc + (pc_target − pc)·(dt/τ)

    Args:
        pc_prev: Previous cavity pressure [Pa]
        pc_target: Target pressure (from σ_eff) [Pa]
        dt: Time step [s]
        tau_pc: Time constant [s]

    Returns:
        pc_new: Updated cavity pressure [Pa]

    Sign Convention:
      - pc ≥ P_VAP (vapor pressure)
      - pc ≤ p_inf (ambient)
    """

    tau_safe = max(tau_pc, 1e-3)  # Avoid dt/tau → ∞
    alpha = dt / tau_safe  # Relaxation factor [0, 1)

    pc_new = pc_prev + (pc_target - pc_prev) * alpha
    pc_new = max(pc_new, P_VAP)  # Can't go below vapor pressure

    return pc_new


def compute_hybrid_sigma(sigma_vapor: float, sigma_vent: float) -> float:
    """
    Compute effective cavitation number as minimum of vapor and ventilation limits.

    σ_eff = min(σ_vapor, σ_vent)

    Fiziksel anlam:
      - σ_vapor: Doğal (buharlaşma) sınırı — pc = p_vapor durumu
      - σ_vent: Havalandırma sınırı — pc > p_vapor ama gaz enjeksiyonu limitli
      - Hangisi daha kısıtlayıcıysa (daha küçük σ), o geçerli olur
      - Daha küçük σ → daha geniş kavite (kontrol)

    Args:
        sigma_vapor: Natural cavitation number (from vapor pressure)
        sigma_vent: Ventilation-limited cavitation number

    Returns:
        sigma_hybrid: Effective σ

    Sign Convention:
      - Tüm input'lar pozitif
      - Output = min(input'lar) → pozitif
    """

    sigma_v_safe = max(sigma_vapor, SIGMA_MIN)
    sigma_vent_safe = max(sigma_vent, SIGMA_MIN)

    return min(sigma_v_safe, sigma_vent_safe)


def compute_pc_target(p_inf: float, sigma_hybrid: float, V: float) -> float:
    """
    Back-compute cavity pressure from effective cavitation number.

    σ = 2(p_inf − pc) / (ρV²)
    → pc = p_inf − σ·ρV²/2

    Args:
        p_inf: Ambient pressure [Pa]
        sigma_hybrid: Effective cavitation number
        V: Velocity [m/s]

    Returns:
        pc_target: Target cavity pressure [Pa]

    Sign Convention:
      - pc ≥ P_VAP (minimum)
      - pc < p_inf (maximum)
    """

    V_safe = max(V, V_MIN)
    q = 0.5 * RHO * V_safe * V_safe

    pc_target = p_inf - sigma_hybrid * q
    pc_target = max(pc_target, P_VAP)
    pc_target = min(pc_target, p_inf - 1.0)  # Must be below ambient

    return pc_target


# ============================================================================
# KAVİTE DURUM TÜREVİ (Lc, Dc, pc) — dynamics/model.py sözleşmesi
# ============================================================================

def cavity_state_derivative(Lc: float, Dc: float, pc: float, V: float, p_inf: float,
                            Cq_in: float, Dn: float, model: str = "savchenko",
                            k_g: float = K_G_CAVITY_DEFAULT, K_Dc: float = 1.0,
                            K_Lc: float = 1.0, A_v: float = A_V_DEFAULT,
                            tau_cav: float = CAVITY_TAU, tau_pc: float = TAU_PC,
                            smooth: bool = False, gas_p_ref: float = None) -> dict:
    """
    Kavite durumlarının zaman türevi (legacy simulate() satır 547-651'in ODE formu).

      σ_vapor   = 2(p∞ − p_v)/(ρV²)
      σ_vent    = A_v / Cq            (Cq ≤ 1e-9 → ∞, havalandırma yok)
      pc_target = p∞ − ½ρV²·min(σ_vapor, σ_vent)   ∈ [p_v, p∞ − 1]
      dpc/dt    = (pc_target − pc)/τ_pc
      σ         = 2(p∞ − pc)/(ρV²)
      (Lc_ss, Dc_ss) = cavity_geometry(σ)  (σ < 1; aksi 0 — kavite yok)
                        Şekil için her zaman disk Cx0=0.82 (legacy satır 630).
      dLc/dt = (Lc_ss − Lc)/τ ,  dDc/dt = (Dc_ss − Dc)/τ

    Durumlar girdi değişince sıfırlanmaz; sadece hedefe doğru gecikmeyle ilerler.

    gas_p_ref [Pa]: None → legacy (Cq kavite basıncındaki hacim debisinden). Verilirse Cq_in,
    p_ref basıncında ölçülmüş hacim debisinden hesaplanmıştır; gaz kavitede (izotermal)
    pc'ye genleşir: Cq_eff = Cq_in·p_ref/pc. pc'nin kendisi Cq'ya bağlı olduğundan denge
    (hedef) basıncında kapalı form çözülür:
        pc_t = p∞ / (1 + ½ρV²·A_v/(Cq_in·p_ref)),   σ_vent = A_v·pc_t/(Cq_in·p_ref)
    (gecikmeli pc durumu kullanılmaz → cebirsel döngü yok; başlangıçtaki pc = p_v'de
    sonsuz genleşme olmaz).

    smooth=True: legacy σ = 1'de (Lc_ss, Dc_ss) ≈ (1.3 m, 0.29 m) → 0 sıçrıyor. Hedef
    boyutlar smoothstep((1 − σ)/SIGMA_ONSET_WIDTH) ile σ → 1'de sürekli 0'a indirilir
    (σ ≤ 1 − w'de legacy ile aynı).

    Returns dict: dLc, dDc, dpc [SI/s], sigma (≤3'e kırpılmış, legacy çıktısı),
        sigma_raw, sigma_vapor, sigma_vent, pc_target, Lc_ss, Dc_ss
    """
    V_safe = max(V, V_MIN)
    rho_v2 = RHO * V_safe * V_safe

    sigma_vapor = 2.0 * (p_inf - P_VAP) / rho_v2
    if Cq_in <= 1e-9:
        sigma_vent = np.inf
    elif gas_p_ref is None:
        sigma_vent = A_v / max(Cq_in, 1e-6)
    else:
        Cq_ref = max(Cq_in, 1e-6) * gas_p_ref
        pc_vent = p_inf / (1.0 + 0.5 * rho_v2 * A_v / Cq_ref)
        sigma_vent = A_v * pc_vent / Cq_ref
    sigma_hybrid = min(sigma_vapor, sigma_vent)

    pc_target = p_inf - 0.5 * rho_v2 * sigma_hybrid
    pc_target = min(max(pc_target, P_VAP), p_inf - 1.0)

    pc_eff = max(pc, P_VAP)
    dpc = (pc_target - pc) / max(tau_pc, 1e-6)

    s = 2.0 * (p_inf - pc_eff) / rho_v2
    if s < 1.0:
        Lc_ss, Dc_ss, _ = cavity_geometry(s, Dn, model=model, Cx0=CX0_DISK,
                                          k_g_val=k_g, K_Dc_val=K_Dc, K_Lc_val=K_Lc)
        if smooth:
            w = smoothstep((1.0 - s) / SIGMA_ONSET_WIDTH)
            Lc_ss, Dc_ss = w * Lc_ss, w * Dc_ss
    else:
        Lc_ss, Dc_ss = 0.0, 0.0

    tau_c = max(tau_cav, 1e-6)
    return {
        "dLc": (Lc_ss - Lc) / tau_c,
        "dDc": (Dc_ss - Dc) / tau_c,
        "dpc": dpc,
        "sigma": min(s, 3.0),
        "sigma_raw": s,
        "sigma_vapor": sigma_vapor,
        "sigma_vent": sigma_vent,
        "pc_target": pc_target,
        "Lc_ss": Lc_ss,
        "Dc_ss": Dc_ss,
    }


# ============================================================================
# HELPER: Cavity axis offset due to gravity and angle
# ============================================================================

def cavity_axis_offset(x: float, alpha_eff: float, V: float,
                       k_dev: float = 0.4, x_open: float = None) -> float:
    """
    Compute cavity axis vertical displacement (Logvinovich-Serebryakov asymptotic).

    δh = gravity + angle contributions
    h(x) = g·x²/(2V²) + k_dev·α_eff·x_open·(1 − exp(−x/x_open))

    Fiziksel anlam:
      - Yerçekimi: parabolik sag (yukarı doğru), her zaman pozitif
      - Açı sapması: asimptotik - x=0'da sıfır, x→∞'de α_eff·x_open'a saturate
      - k_dev: sönümleme faktörü (0-1), havalandırma stabilitesi
      - x_open: karakteristik uzunluk (akış koşullarına bağlı)

    Args:
        x: Axial position along cavity [m]
        alpha_eff: Effective cavitator angle = α_aoa + δ_cav [rad]
        V: Velocity [m/s]
        k_dev: Deviation damping factor (0-1)
        x_open: Characteristic opening length [m]
                If None, computed as 10·Rn/√(Cx) (requires external Cx)

    Returns:
        h: Cavity axis vertical displacement [m] (positive = cavity moves down)

    Sign Convention:
      - h > 0: Kavite aşağı kaymış (alt yüzey ıslak)
      - h < 0: Kavite yukarı kaymış (üst yüzey ıslak) — nadir, sadece negatif α
    """

    V_safe = max(V, V_MIN)
    G = 9.81  # Assume global import available elsewhere

    # Gravity term (always positive, parabolic)
    from .constants import G as G_const
    h_gravity = G_const * x * x / (2.0 * V_safe * V_safe)

    # Angle term (asymptotic, saturates at x_open)
    if x_open is None or x_open < 1e-6:
        h_angle = 0.0  # No angle effect if x_open not provided
    else:
        x_safe = max(x, 0.0)
        h_angle = k_dev * alpha_eff * x_open * (1.0 - np.exp(-x_safe / x_open))

    return h_gravity + h_angle
