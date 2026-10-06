"""
================================================================================
 Supercavitation — Fin (Control Surface) Forces
================================================================================
Dört kontrol kanadının (4 independent fins) kuvvet ve moment hesapları.

NACA 16-009 profili (low-drag, symmetric, supercavitation standard).
Sadece kavite zarfı DIŞINDA kalan kısım ıslak (kuvvet üretir).

İşaret Konvansiyonu (Sign Convention):
  - Lift: orthogonal to fin surface, positive = upward (aeronautical convention)
  - Drag: opposing flow direction
  - Moment: right-hand rule about y-axis (pitch), positive = nose-up
  - Azimuth: 90°=top, 0°=right, 270°=bottom, 180°=left (looking aft)
================================================================================
"""

import numpy as np
from .constants import (
    RHO, V_MIN, CAVITY_MIN,
    FIN_CL_ALPHA_2D, FIN_CD0, FIN_OSWALD_E, FIN_STALL_ANGLE
)


def fin_3d_lift_slope(fin_chord: float, fin_span: float) -> float:
    """
    Compute 3D lift slope for finite wing (Prandtl lifting-line theory).

    CL_α(3D) = CL_α(2D) / [1 + (CL_α(2D) / (π·AR·e))]

    For small AR corrections: CL_α(3D) ≈ (2π·AR) / (AR + 2)  (standard approximation)

    Fiziksel anlam:
      - Wing aspect ratio: AR = span / chord
      - Winglet/tip loss: e = Oswald efficiency (0.85 typical)
      - Finite wing → daha düşük lift slope (induced drag etkisi)

    Args:
        fin_chord: Fin chord length [m]
        fin_span: Fin span (root to tip) [m]

    Returns:
        CL_α: 3D lift slope [1/rad]

    Sign Convention:
      - CL_α > 0 always (lift increases with positive angle)
      - Units: 1/rad (convert angle from degrees to radians)
    """

    if fin_chord < 1e-6:
        return 0.0

    fin_AR = fin_span / fin_chord

    if fin_AR < 0.1:
        # Very low AR (flat plate-like) — use 2D value
        return FIN_CL_ALPHA_2D

    # Prandtl finite wing (standard form)
    # CL_α(3D) = (2π·AR) / (AR + 2)
    CL_3d = (2.0 * np.pi * fin_AR) / (fin_AR + 2.0)

    return max(CL_3d, 0.0)


def fin_drag_coefficient(alpha_local: float, CL: float,
                        fin_chord: float, fin_span: float) -> float:
    """
    Compute total fin drag coefficient (profile + induced).

    CD = CD0 + k_induced · CL²

    where k_induced = 1/(π·AR·e)  — induced drag factor.

    Fiziksel anlam:
      - CD0 ≈ 0.0085 (NACA 16-009 profile drag, low-speed)
      - k_induced CL² term → increasing with lift (vortex drag)
      - Stall effects: CD increases sharply beyond ~14°, but model remains linear

    Args:
        alpha_local: Local fin angle of attack [rad] (not used directly, for reference)
        CL: Local lift coefficient (from CL_α · α)
        fin_chord: Fin chord [m]
        fin_span: Fin span [m]

    Returns:
        CD: Drag coefficient (dimensionless)

    Sign Convention:
      - CD > 0 always (drag opposes flow)
      - High α (stall) → high CD, model does not include post-stall rise
    """

    if fin_chord < 1e-6 or fin_span < 1e-6:
        return FIN_CD0

    fin_AR = fin_span / fin_chord
    if fin_AR > 1e-3:
        k_induced = 1.0 / (np.pi * fin_AR * FIN_OSWALD_E)
    else:
        k_induced = 0.0

    CD = FIN_CD0 + k_induced * CL * CL

    return max(CD, 0.0)


def fin_immersion_ratio(R_cavity: float, R_vehicle: float, fin_span: float) -> float:
    """
    Compute effective wetted span of fin (outside cavity envelope).

    Fin root: R_vehicle (vehicle surface)
    Fin tip: R_vehicle + fin_span (maximum extent)

    Cavity envelope: R_cavity (radial distance)

    Effective wetted span:
      - If R_cavity < R_vehicle: entire fin wetted (fully outside cavity)
      - If R_cavity ≥ R_vehicle + fin_span: no wet portion (fully inside cavity)
      - Otherwise: wet span = tip distance - cavity distance

    Fiziksel anlam:
      - Gaz içindeki kanat kuvvet üretmez (ρ_gas << ρ_water)
      - Sadece suyun içindeki kısım etkilidir
      - Partial immersion reduces effective span and force

    Args:
        R_cavity: Cavity envelope radius at fin location [m]
        R_vehicle: Vehicle radius at fin location (fin root) [m]
        fin_span: Fin span (root to tip extent) [m]

    Returns:
        wet_span: Effective wetted fin span [m]
                  Range: [0, fin_span]

    Sign Convention:
      - All radii > 0
      - wet_span ≥ 0
      - wet_span ≤ fin_span
    """

    R_cavity_safe = max(R_cavity, 0.0)
    R_vehicle_safe = max(R_vehicle, 0.0)
    fin_span_safe = max(fin_span, 0.0)

    r_tip = R_vehicle_safe + fin_span_safe

    if R_cavity_safe < R_vehicle_safe:
        # Cavity inside vehicle surface — entire fin wetted
        wet_span = fin_span_safe
    elif R_cavity_safe >= r_tip:
        # Fin fully inside cavity — no wetted portion
        wet_span = 0.0
    else:
        # Partial immersion
        wet_span = r_tip - R_cavity_safe

    return max(min(wet_span, fin_span_safe), 0.0)


def fin_aoa_effective(alpha_aoa: float, delta_fin: float, azimuth: float) -> float:
    """
    Compute effective angle of attack for fin (world-frame to fin-frame conversion).

    Fin pitch angle (world-z) projected onto fin coordinate system:
      α_eff = (α_aoa + δ_fin) · sin(azimuth)

    Fiziksel anlam:
      - Üst kanat (azim=90°, sin=+1): tam dünya pitch açısı görür
      - Alt kanat (azim=270°, sin=-1): ters pitch (taban kanat)
      - Yan kanatlar (azim=0° or 180°, sin=0): pitch'i görmez

    Havalandırma kaçağı (yaw) modeli: not implemented in this module
      (would require β_dot, yaw rate, additional azimuth calculation)

    Args:
        alpha_aoa: Angle of attack (world, pitch) [rad]
        delta_fin: Fin deflection angle [rad]
        azimuth: Fin position azimuth [rad]
                 0° = starboard (right), 90° = top, 270° = bottom, 180° = port (left)

    Returns:
        alpha_eff: Effective fin AoA [rad]

    Sign Convention:
      - Positive α (nose-up) + top fin → positive α_eff (lift up)
      - Bottom fin sees negative α_eff (sin(270°)=-1)
      - Can be negative (angle reversal, stall regions)
    """

    alpha_total = alpha_aoa + delta_fin
    sin_azim = np.sin(azimuth)

    alpha_eff = alpha_total * sin_azim

    return alpha_eff


def fin_wetted_segment(R_cavity: float, R_root: float, fin_span: float,
                       az_rad: float, h_y: float = 0.0, h_z: float = 0.0) -> tuple:
    """
    Kanadın ıslak açıklığı ve kuvvet yarıçapı — kavite ekseni GÖVDE EKSENİNDEN SAPMIŞKEN.

    `fin_immersion_ratio` kaviteyi gövdeyle eş merkezli varsayar; o durumda dört kanat
    aynı ıslaklığı görür ve roll katkıları birebir sönümlenir (K ≡ 0). Kavite kesiti
    gerçekte akış doğrultusunu izler: α_eff ve β ile gövde ekseninden (h_y, h_z) kadar
    kayar (bkz. cavity.cavity_axis_offset_components) → her kanat farklı ıslaklık görür
    → β'dan ve α'dan doğal roll momenti doğar.

    Geometri (gövde çerçevesi y-z kesiti, y sancak, z aşağı):
      Kanat ışını:  P(s) = s·e,  e = (cos az, −sin az),  s ∈ [R_root, R_root + span]
      Kavite kesiti: |P − C| = R_cavity,  C = (h_y, h_z)
      |s·e − C|² = R_c²  →  s² − 2s(e·C) + |C|² − R_c² = 0
      d = e·C,  disc = R_c² − (|C|² − d²) = R_c² − (C'nin ışına dik uzaklığı)²
      disc ≤ 0 → ışın kaviteye hiç girmez (kanat tamamen ıslak)
      disc > 0 → kuru aralık s ∈ [d − √disc, d + √disc]

    Islak küme [R_root, R_tip] fark [s1, s2] olduğundan **iki parçalı** olabilir: kavite
    kanadın orta bandını örtüp kökü ve ucu suda bırakabilir (çok sapmış kavite).
    Kuvvet için toplam ıslak uzunluk ve (sabit veter varsayımıyla) alan merkezi döner.

    h_y = h_z = 0'da sonuç `fin_immersion_ratio` + legacy r_mid ile **birebir** aynıdır
    (eş merkezli dal ayrıca kısa yoldan hesaplanır → bit düzeyinde aynı).

    Süreklilik: wet_span her yerde süreklidir, ancak kanat ışını kavite çemberine TEĞET
    geçerken kuru aralık 2·√disc ile açıldığı için o noktada **dikey teğetli**dir (sonsuz
    eğim; çember kirişinin gerçek geometrisi — kavite kapanma profilindeki aynı desen,
    bkz. body.cavity_radius_logvinovich). Teğetlik ancak sapma kanat açıklığı ölçeğine
    (|h| ≳ 0.4 m) çıkınca oluşur; çalışma aralığında (2°'de h ≈ 0.02 m) fonksiyon
    Lipschitz'tir (|Δwet| ≤ |Δh|). Bkz. test_fin_offset.py bölüm A8.

    Args:
        R_cavity: Kanat istasyonundaki kavite yarıçapı [m]
        R_root: Kanat kökü yarıçapı = o istasyondaki gövde yarıçapı [m]
        fin_span: Kanat açıklığı (kök → uç) [m]
        az_rad: Kanat azimutu [rad] (0 = sancak, π/2 = üst; gövde: y sancak, z aşağı)
        h_y, h_z: Kavite ekseninin gövde ekseninden sapması [m] (+y sancak, +z aşağı)

    Returns:
        (wet_span, r_force): Islak açıklık [m] ∈ [0, fin_span] ve kuvvetin etkidiği
                             yarıçap [m] (ıslak kısmın alan merkezi; ıslak değilse
                             kök + açıklık/2 — kuvvet sıfır olduğu için önemsiz)

    İşaret: wet_span ≥ 0, r_force > 0.
    """
    R_c = max(float(R_cavity), 0.0)
    R_root = max(float(R_root), 0.0)
    span = max(float(fin_span), 0.0)
    r_tip = R_root + span
    if span <= 0.0:
        return 0.0, r_tip

    # Eş merkezli kavite: legacy yolla birebir aynı kalsın (bit düzeyinde)
    if h_y == 0.0 and h_z == 0.0:
        wet = fin_immersion_ratio(max(R_c, R_root), R_root, span)
        return wet, r_tip - 0.5 * wet

    d = h_y * np.cos(az_rad) - h_z * np.sin(az_rad)
    C2 = h_y * h_y + h_z * h_z
    disc = R_c * R_c - max(C2 - d * d, 0.0)
    if disc <= 0.0:
        # Kanat ışını kavite kesitini hiç kesmiyor → tamamen ıslak
        return span, r_tip - 0.5 * span

    root_disc = np.sqrt(disc)
    s1, s2 = d - root_disc, d + root_disc          # kuru aralık

    # [R_root, r_tip] fark [s1, s2] → en çok iki parça (iç: kök tarafı, dış: uç tarafı)
    wet = 0.0
    moment = 0.0
    for lo, hi in ((R_root, min(r_tip, s1)), (max(R_root, s2), r_tip)):
        if hi > lo:
            wet += hi - lo
            moment += 0.5 * (lo + hi) * (hi - lo)
    if wet <= 0.0:
        return 0.0, r_tip - 0.5 * span
    return min(wet, span), moment / wet


def cavity_radius_ellipse(x_cav: float, Lc: float, Dc: float) -> float:
    """
    Kanat konumundaki kavite yarıçapı — elipsoid yaklaşımı (legacy satır 971-977).

      ξ = 2x/Lc − 1 ∈ [−1, 1],   R_c = (Dc/2)·√(1 − ξ²)
    Kavite yok (Lc, Dc ≤ 1e-6) veya x ≥ Lc ise 0.
    """
    if Lc <= 1e-6 or Dc <= 1e-6 or x_cav >= Lc:
        return 0.0
    xi = max(-1.0, min(1.0, 2.0 * x_cav / Lc - 1.0))
    return 0.5 * Dc * np.sqrt(max(0.0, 1.0 - xi * xi))


def fin_lift_and_drag(V: float, fin_chord: float, wet_span: float,
                      CL_alpha: float, alpha_eff: float,
                      span_for_AR: float = None) -> tuple:
    """
    Compute lift and drag forces on fin at given angle.

    L = q · S_wet · CL  (orthogonal to flow)
    D = q · S_wet · CD  (parallel to flow)

    where:
      q = ½ρV²
      S_wet = fin_chord · wet_span  (wetted planform area)
      CL = CL_α · α (linear approximation, stall not modeled)
      CD = CD0 + k_ind · CL²  (parabolic induced drag)

    Args:
        V: Velocity [m/s]
        fin_chord: Fin chord [m]
        wet_span: Wetted (immersed) fin span [m]
        CL_alpha: 3D lift slope [1/rad]
        alpha_eff: Effective angle of attack [rad]
        span_for_AR: İndüklenmiş drag AR'si için span [m]. None → wet_span.
                     Legacy tam span kullanır (fin_k_induced sabit).

    Returns:
        (F_L, F_D): Lift [N], Drag [N]
                    Lift > 0 = upward; Drag > 0 = retarding

    Sign Convention:
      - F_L > 0: upward (positive in aeronautical convention)
      - F_D > 0: opposing flow (always positive)
      - Stall not modeled — linear through ±14°
    """

    V_safe = max(V, V_MIN)
    S_wet = fin_chord * wet_span

    if S_wet < 1e-9:
        return 0.0, 0.0

    # Dynamic pressure
    q = 0.5 * RHO * V_safe * V_safe

    # Stall protection (±14° for NACA 16-009)
    alpha_clipped = np.clip(alpha_eff, -FIN_STALL_ANGLE, FIN_STALL_ANGLE)

    # Lift coefficient
    CL = CL_alpha * alpha_clipped

    # Drag coefficient (induced + profile)
    CD = fin_drag_coefficient(alpha_clipped, CL, fin_chord,
                              wet_span if span_for_AR is None else span_for_AR)

    # Forces
    F_L = q * S_wet * CL
    F_D = q * S_wet * CD

    return F_L, F_D


def fin_vertical_component(F_L: float, azimuth: float) -> float:
    """
    Convert fin lift to vertical (z-direction) component.

    Lift acts perpendicular to fin surface. For fins at azimuth θ:
      F_z = F_L · sin(θ)

    Fiziksel anlam:
      - Üst kanat (θ=90°): F_z = F_L (full lift in z-direction)
      - Alt kanat (θ=270°): F_z = -F_L (opposite sign)
      - Yan kanatlar (θ=0° or 180°): F_z = 0 (no vertical component)

    Args:
        F_L: Fin lift force [N] (perpendicular to fin surface)
        azimuth: Fin azimuth [rad]

    Returns:
        F_L_z: Vertical component [N]

    Sign Convention:
      - Body-fixed z: positive = downward
      - Lift upward (+F_L) → negative contribution to F_z (reduces weight)
      - Top fin (sin=+1) with +F_L → F_L_z = +F_L (but interpretation: -1×F_L_z in body-z)

      Actually for consistency:
      - Return F_L · sin(azim) where positive F_L (up) and top fin (sin=+1)
      - Caller applies sign convention (F_z = -F_L_z or similar)
    """

    sin_azim = np.sin(azimuth)
    F_L_z = F_L * sin_azim

    return F_L_z


def fin_moment_contribution(F_L_z: float, x_fin: float, x_cg: float) -> float:
    """
    Compute pitching moment contribution from fin lift.

    M_y = (x_cg - x_fin) · F_L_z

    Moment arm (right-hand rule about y-axis):
      - x_fin < x_cg: fin forward of c.g. → upward force (+F_L_z) → positive M_y (nose-up)
      - x_fin > x_cg: fin aft of c.g. → upward force (+F_L_z) → negative M_y (nose-down)

    Args:
        F_L_z: Vertical component of fin lift [N]
        x_fin: Fin location along fuselage [m]
        x_cg: Center of gravity location [m]

    Returns:
        M_y: Pitching moment contribution [N·m]
             Positive = nose-up, negative = nose-down

    Sign Convention:
      - Right-hand rule: fingers curl from +x to +z, thumb points along y
      - Consistent with vehicle pitch dynamics
    """

    moment_arm = x_cg - x_fin
    M_y = moment_arm * F_L_z

    return M_y


# ============================================================================
# COMPREHENSIVE FIN FORCES (wrapper)
# ============================================================================

def compute_fin_forces(V: float, fin_chord: float, fin_span: float,
                      R_cavity: float, R_vehicle: float,
                      alpha_aoa: float, delta_fin: float, azimuth: float,
                      x_fin: float, x_cg: float) -> dict:
    """
    Comprehensive fin force and moment computation.

    Integrates immersion, lift/drag, and moment calculations.

    Args:
        V: Velocity [m/s]
        fin_chord: Fin chord [m]
        fin_span: Fin span [m]
        R_cavity: Cavity radius at fin location [m]
        R_vehicle: Vehicle radius at fin location [m]
        alpha_aoa: Vehicle angle of attack [rad]
        delta_fin: Fin deflection [rad]
        azimuth: Fin azimuth [rad] (0=right, π/2=top, 3π/2=bottom, π=left)
        x_fin: Fin axial position [m]
        x_cg: Center of gravity position [m]

    Returns:
        dict with keys:
          - "F_L": Lift perpendicular to fin [N]
          - "F_D": Drag along flow [N]
          - "F_L_z": Vertical (pitch) component [N]
          - "M_y": Pitching moment [N·m]
          - "wet_span": Wetted span [m]
          - "immersion": Ratio of wet_span / fin_span
          - "alpha_eff": Effective AoA [rad]

    Sign Convention:
      - F_L, F_D, F_L_z > 0: magnitude convention (see detailed functions)
      - M_y > 0: nose-up pitch
    """

    # 3D lift slope
    CL_α = fin_3d_lift_slope(fin_chord, fin_span)

    if CL_α < 1e-6 or V < V_MIN:
        # No lift or stalled — return zeros
        return {
            "F_L": 0.0, "F_D": 0.0, "F_L_z": 0.0, "M_y": 0.0,
            "wet_span": 0.0, "immersion": 0.0, "alpha_eff": 0.0,
        }

    # Immersion
    wet_span = fin_immersion_ratio(R_cavity, R_vehicle, fin_span)
    immersion = wet_span / max(fin_span, 1e-6) if fin_span > 1e-6 else 0.0

    if wet_span < 1e-6:
        # No wetted area
        return {
            "F_L": 0.0, "F_D": 0.0, "F_L_z": 0.0, "M_y": 0.0,
            "wet_span": 0.0, "immersion": 0.0, "alpha_eff": 0.0,
        }

    # Effective AoA
    alpha_eff = fin_aoa_effective(alpha_aoa, delta_fin, azimuth)

    # Lift and drag
    F_L, F_D = fin_lift_and_drag(V, fin_chord, wet_span, CL_α, alpha_eff)

    # Vertical component
    F_L_z = fin_vertical_component(F_L, azimuth)

    # Moment
    M_y = fin_moment_contribution(F_L_z, x_fin, x_cg)

    return {
        "F_L": F_L,
        "F_D": F_D,
        "F_L_z": F_L_z,
        "M_y": M_y,
        "wet_span": wet_span,
        "immersion": immersion,
        "alpha_eff": alpha_eff,
    }
