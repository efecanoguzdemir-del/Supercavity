"""
================================================================================
 Supercavitation Hydrodynamics — SI Constants and Physical Parameters
================================================================================
Tüm sabitler SI birimlerinde (m, kg, s, N, Pa, rad).
Kavitatöre ait tüm açılar radyan; kullanıcı arabirimi dereceyle çalışır.

Referanslar:
  - Logvinovich (1969): Hydrodynamics of Cavitating Flows
  - Reichardt (1946): The Law of Flow in Rough Pipes
  - Savchenko (2001): Supercavitating Flows Around Bodies of Revolution
  - Garabedian (1956): Cavitation Theory
================================================================================
"""

import numpy as np

# ============================================================================
# TEMEL FİZİKSEL SABİTLER (seawater, 15°C)
# ============================================================================

RHO = 1025.0        # Seawater density [kg/m³]
G = 9.81            # Gravitational acceleration [m/s²]
P_ATM = 101325.0    # Atmospheric pressure (sea level) [Pa]
P_VAP = 2340.0      # Water vapor pressure [Pa]

# Dinamik viskozite (15°C seawater)
MU_WATER = 1.19e-3  # Dynamic viscosity [Pa·s]
NU_WATER = MU_WATER / RHO   # Kinematik viskozite [m²/s] ≈ 1.161e-6 (ITTC-57 Re için)

# ============================================================================
# TEKILLIK (SINGULARITY) KORUMA GÜARDLARİ
# ============================================================================
# Sıfıra yaklaşan değişkenler karşısında sayısal istikrar için minimum eşikler

SIGMA_MIN = 1e-4    # Minimum cavitation number (σ → 0 koruması)
V_MIN = 0.1         # Minimum velocity [m/s] (division by zero koruması)
CAVITY_MIN = 1e-6   # Minimum cavity length/diameter [m]

# ============================================================================
# KAVİTATÖR SABİTLERİ
# ============================================================================

# Reichardt-Garabedian drag coefficient sabiti
CX0_DISK = 0.82                    # Flat disk cavitator (β=180°)
K_G_CAVITY_DEFAULT = 0.78          # Cavity diameter calibration (Vasin/CFD)

# Konik kavitatör viskoz düzeltme (May 1975, CFD)
K_V_CONE_DEFAULT = 0.35            # Viscous correction for conical tips

# Kavitatör lift katsayısı — hibrit model
K_SLENDER_DEFAULT = 1.0            # Slender body cross-flow lift (Munk: 1.0-2.0)

# ============================================================================
# KANAT (FIN) SABİTLERİ — NACA 16-009
# ============================================================================

# NACA 16-009 profili (low-drag symmetric, supercavitation standard)
FIN_CL_ALPHA_2D = 2.0 * np.pi      # 2D lift slope [1/rad]
FIN_CD0 = 0.0085                   # Minimum drag coefficient
FIN_OSWALD_E = 0.85                # Oswald efficiency factor
FIN_STALL_ANGLE = np.radians(14.0) # Stall angle [rad]

# ============================================================================
# HAVALANDIRMA (VENTILATION) SABİTLERİ
# ============================================================================

A_V_DEFAULT = 0.035                # Ventilation calibration factor (Semenenko)
                                   # Range: 0.005-0.30
TAU_PC = 0.3                       # Cavity pressure time constant [s]
K_LEAK = 0.15                      # Epshtein twin-vortex leakage (0.1-0.3)

# ============================================================================
# GÖVDE (BODY) SABİTLERİ
# ============================================================================

CL_ALPHA_BODY_DEFAULT = 1.5        # Munk slender body lift slope factor
CDC_BODY_DEFAULT = 0.4             # Body crossflow drag coefficient (low-α)
CF_SKIN = 0.003                    # Wetted-body skin friction coefficient (legacy Cf, sabit)
# ITTC-57 sürtünme modeli (friction_model="ittc"; legacy_exact=False varsayılanı):
#   Cf = (1+k)·0.075/(log10(Re) − 2)²,  Re = V·L/ν
# Sabit 0.003 legacy'den gelir ve Re bağımsızdır: V=20 m/s, L=4 m'de (Re=6.9e7) ITTC
# 0.00220 verir → legacy %36 yüksek; V=40'ta (Re=1.4e8) 0.00199 → %51 yüksek. Tam ıslak
# başlangıç koşulunda CFD ile karşılaştırmada bu fark X kuvvetinin ~%6'sıydı.
FORM_FACTOR_DEFAULT = 1.1          # (1+k): gövde form etkisi (Hoerner L/D=13 için ~1.03)
                                   # + pürüzlülük payı (ITTC ΔCf ≈ 3e-4) ≈ 1.1
RE_MIN_FRICTION = 1.0e5            # ITTC formülünün alt sınırı (V→0 koruması)
CDC_CROSSFLOW_SECTION = 1.2        # Hoerner crossflow Cdc used in per-section body integral
N_BODY_SECTIONS = 40               # Number of axial body sections (legacy n_x)
CAVITY_TAU = 0.15                  # Cavity Lc/Dc first-order lag time constant [s] (legacy tau)
CD_BASE_EXPOSED = 0.20             # Transom tamamen kavite dışında: taban basınç sürüklemesi (legacy)
CD_BASE_FULLY_WET = 0.15           # Kavite hiç yok (tam ıslak): taban basınç sürüklemesi (legacy)

# ============================================================================
# KAVİTE YOKKEN (TAM ISLAK) KAVİTATÖR/BURUN DİRENCİ
# ============================================================================
# Cx = Cx0·(1+σ) (Reichardt) kavitatörün ARKASINDA kavite varken geçerlidir: taban
# basıncı pc'dir ve σ ≲ 0.3-0.5 aralığında doğrulanmıştır. Kavite yokken (σ → 1,
# havalandırma başlamadan) burun ayrılmamış ıslak akış içindedir ve direnci klasik
# koni/disk basınç direncidir — Reichardt ekstrapolasyonundan belirgin biçimde düşük.
# Hoerner, Fluid-Dynamic Drag (Bölüm 3, koni direnci; taban alanına göre, Re ~ 1e6+):
CONE_CD_WET_APEX_DEG = (0.0, 20.0, 30.0, 40.0, 60.0, 90.0, 120.0, 180.0)
CONE_CD_WET_TABLE = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.17)
CD_DISK_WET = 1.17                 # Tam ıslak düz disk (β=180°) — tablonun uç değeri

# Kavitenin "kurulmuş" sayıldığı ölçüt (burun direnci harmanlaması, smooth_transitions):
# Klasik süperkavite direnç yasasında Cx0·(1+σ) kavitatör direncine gövdenin basınç
# direncinin tamamı yüklenir; bu ancak gövde kavitenin İÇİNDEYKEN, yani kavite gövdeden
# geniş olduğunda (Dc > D) geçerlidir. Dc < D iken kavite ön gövde üzerinde kapanan bir
# "kısmi kavite"dir, akış gövdeye yeniden yapışır ve burun pratikte ıslak akıştadır →
# tam ıslak koni/disk basınç direnci (cavitator.wetted_cavitator_Cx) kullanılır.
# Harmanlama Dc/D ∈ [1, 1+w] aralığında yapılır.
#
# Ölçüt σ DEĞİL kavite DURUMUDUR. σ tabanlı eşik (eski hali) iki yerde kırılıyordu:
#   * σ hız arttıkça kavite yokken de düşer — V = 25 m/s'de t = 0'da σ = 0.62 → düzeltme
#     hiç devreye girmiyordu (başlangıç direnci %20 yüksek).
#   * Havalandırma başlar başlamaz pc yükselip σ ms'ler içinde düşüyor → düzeltme
#     V = 20 m/s'de ~30 ms'de kapanıyor, sürükleme %23 geri sıçrıyordu.
# Referans: Dc = D (= 1.5·Dn, Case 1) yaklaşık σ ≈ 0.39'a, Dc = 1.5·D ise σ ≈ 0.24'e
# denk gelir — Reichardt'ın doğrulandığı σ ≲ 0.3-0.5 bandı.
CAVITY_ESTAB_WIDTH = 0.5

# Sürekli geçiş genişlikleri (smooth_transitions=True; legacy_exact modunda kullanılmaz)
SIGMA_ONSET_WIDTH = 0.2            # Lc_ss, Dc_ss: σ ∈ [1−w, 1] aralığında 0'a iner
ARC_FREE_BLEND = 0.2               # ıslak yay: rc ∈ [R_v, (1+w)·R_v] bitişik→serbest harmanlama
TRANSOM_EXIT_START = 1.5           # planing→tamamen dışarıda: δ/R_v ∈ [başlangıç, başlangıç+genişlik]
TRANSOM_EXIT_WIDTH = 1.0

# ============================================================================
# PLANING (TAIL-SLAP) SABİTLERİ — Dzielski-Kurdila 2003
# ============================================================================

# Kullanıcı bu değerleri tuning yapabilir; algoritmada hardcoded olan değerler
# yok — tüm hesaplar parametresel.

# ============================================================================
# GÖVDE-KAVİTE ETKİLEŞİMİ
# ============================================================================

BODY_VOLUME_CORRECTION = True      # Superposition: R_zarf² = R_logv² + R_v²
K_DEV_DEFAULT = 0.4                # Cavity axis deviation (Logvinovich asymptotic)
                                   # Controls asymmetric wetting (gravity + angle)

# ============================================================================
# KALIBRASYON ÇARPANLARİ
# ============================================================================

K_DC_DEFAULT = 1.0                 # Cavity diameter scaling factor
K_LC_DEFAULT = 1.0                 # Cavity length scaling factor

# ============================================================================
# LEGACY CASE 1 — REGRESSION TEST REFERENCE
# ============================================================================
# Supercavitation GUI v10 nominal scenario
# Test amacı: ±%5 tolerans içinde F_drag, Fz, My eşleşmesi

LEGACY_CASE_1 = {
    # Input parameters
    "V": 40.0,                      # [m/s]
    "alpha_aoa_deg": -1.0,          # [°] — Aircraft convention: nose-up = +
    "delta_cav_deg": 2.0,           # [°] — Cavitator pitch
    "depth": 10.0,                  # [m]
    "diam_cav": 0.050,              # [m] cavitator diameter
    "mass": 100.0,                  # [kg]
    "thrust": 0.0,                  # [N] (not used for forces test)
    "veh_len": 2.0,                 # [m]
    "veh_diam": 0.15,               # [m]
    "x_cg": 1.0,                    # [m] center of gravity (aft from nose)

    # Hydrodynamic parameters (nominal)
    "geom_model": "savchenko",      # Cavity geometry model
    "k_g_cavity": 0.78,             # Reichardt-Garabedian constant
    "K_Dc_factor": 1.0,             # Diameter scaling
    "K_Lc_factor": 1.0,             # Length scaling
    "cavitator_type": "disk",       # Not cone
    "cone_drag_visc": 0.35,         # Irrelevant (disk)
    "CL_alpha_body": 1.5,           # Slender body
    "Cdc_body": 0.4,                # Crossflow drag
    "K_slender": 1.0,               # Slender body lift
    "K_cone_lift": 0.0,             # No cone lift gain (disk)
    "cone_lift_gain": 0.0,          # Irrelevant

    # Ventilation (natural cavitation only)
    "vent_mode": "Q",
    "gas_flow": 0.0,                # [L/min] — natural only
    "A_v": 0.035,

    # Fins disabled, planing disabled
    "fins_enabled": False,
    "planing_enable": False,

    # Other defaults
    "steady_v_mode": True,          # V constant (steady-state)
    "L_taper": 0.3,                 # [m] nose taper
    "k_dev": 0.4,

    # Simulation time (for one snapshot)
    "t_max": 0.01,
    "dt": 0.001,

    # Expected output (±%5 tolerance)
    "expected_F_drag_N": 5764.0,    # ±276 N (5%)
    "expected_Fz_N": -2109.0,       # ±105 N (5%)
    "expected_My_Nm": 3344.0,       # ±167 N·m (5%)
}

# Tolerance: ±5% on forces, moments
LEGACY_CASE_1_TOL = 0.05  # 5%

# ============================================================================
# KOORDINAT KONVANSIYONLARI (sign conventions)
# ============================================================================
"""
Vehicle body-fixed frame (aeronautical convention):
  x: Forward (nose direction) [m]
  y: Starboard (right) [m]
  z: Down (gravity direction) [m]

Forces:
  Fx: Forward drag (positive = retarding)
  Fz: Vertical (positive = downward in body frame, but lift up = negative)
  My: Pitching moment (positive = nose-up rotation, sağ-el kuralı x-ekseni etraf)

Angles (all in radians internally; UI uses degrees):
  α (alpha_aoa): angle of attack, positive = nose-up relative to flow
  δ (delta_cav): cavitator pitch, positive = nose-up deflection
  β (cone_apex): cone half-angle

Sign rules for moments:
  - Force at position (x_force, z_force) applied to c.g. at (x_cg, z_cg)
  - Moment arm: M = (x_cg - x_force) · F_z  (right-hand rule about y-axis)
  - Positive M_y = nose-up pitch
"""

# ============================================================================
# DERIVED QUANTITIES (computed, not user input)
# ============================================================================

def compute_sigma_vapor(p_inf: float, V: float) -> float:
    """
    Compute natural (vapor) cavitation number.

    σ = 2(p∞ − pv) / (ρV²)

    Args:
        p_inf: Ambient pressure [Pa]
        V: Velocity [m/s]

    Returns:
        σ: Cavitation number (dimensionless, positive)
    """
    V_safe = max(V, V_MIN)
    return (2.0 * (p_inf - P_VAP)) / (RHO * V_safe**2)


def compute_ambient_pressure(depth: float) -> float:
    """
    Compute ambient pressure at depth.

    p_inf = P_atm + ρg·depth

    Args:
        depth: Depth below surface [m]

    Returns:
        p_inf: Ambient pressure [Pa]
    """
    return P_ATM + RHO * G * depth
