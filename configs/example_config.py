"""
Simülasyon konfigürasyonu: parametreler, kontrol modu, PID kazançları, referanslar.

Açık çevrim (open_loop): zaman-tabanlı schedule'lar tüm girdiler için
Kapalı çevrim (closed_loop): 3-DOF AttitudeAutopilot + δc/Cq açık çevrim
"""

# =============================================================================
# GENEL SİMÜLASYON PARAMETRELERİ
# =============================================================================

SIM_PARAMS = {
    # Zaman
    "t_max": 10.0,          # simülasyon süresi [s]
    "dt": 0.001,            # zaman adımı [s] — 1 ms otopilot rate

    # Ortam
    "depth": 10.0,          # derinlik [m]

    # Araç fiziksel parametreleri
    "mass": 100.0,          # toplam kütle [kg]
    "veh_len": 2.5,         # gövde boyu [m]
    "veh_diam": 0.15,       # gövde çapı [m]

    # Kavitatör
    "diam_cav": 0.05,       # kavitatör çapı [m]
    "cavitator_type": "disk",  # "disk" veya "cone"
    "cone_apex_deg": 90.0,  # koni tepe açısı [deg] (sadece cone için)

    # Başlangıç koşulları
    "V_init": 40.0,         # başlangıç hızı [m/s]
    "alpha_aoa_deg": -1.0,  # başlangıç hücum açısı [deg]
    "delta_cav_deg": 2.0,   # başlangıç kavitatör açısı [deg]

    # Kanatlar (fins)
    "fins_enabled": True,
    "fin_chord": 0.040,     # kanat kord boyu [m]
    "fin_span": 0.060,      # kanat uzanımı [m]
    "fin_x_pos": 2.125,     # kanat x-pozisyonu [m] (L_veh * 0.85)

    # Kavite
    "geom_model": "savchenko",  # "garabedian", "savchenko", "semenenko", etc.
    "k_g_cavity": 0.78,     # Reichardt-Garabedian sabiti
    "K_Dc_factor": 1.0,     # kavite çapı kalibrasyon çarpanı
    "K_Lc_factor": 1.0,     # kavite boyu kalibrasyon çarpanı

    # Havalandırma
    "vent_mode": "Q",       # "Q" (L/min) veya "Cq" (boyutsuz)
    "A_v": 0.035,           # ventilasyon sabiti (Semenenko)
}

# =============================================================================
# KONTROL MODU SEÇİMİ
# =============================================================================

CONTROL_MODE = "closed_loop"  # "open_loop" veya "closed_loop"

# =============================================================================
# KAPALΙ ÇEVRIM (closed_loop) — 3-DOF AttitudeAutopilot
# =============================================================================

if CONTROL_MODE == "closed_loop":
    # 3-DOF PID kazançları
    AUTOPILOT = {
        "pid_yaw": {
            "Kp": 0.5,          # yaw proportional
            "Ki": 0.1,          # yaw integral
            "Kd": 0.2,          # yaw derivative
            "output_max": 20.0, # rudder saturation [deg]
        },
        "pid_pitch": {
            "Kp": 1.0,          # pitch proportional
            "Ki": 0.05,         # pitch integral
            "Kd": 0.3,          # pitch derivative
            "output_max": 15.0, # elevator saturation [deg]
        },
        "pid_roll": {
            "Kp": 0.3,          # roll proportional
            "Ki": 0.02,         # roll integral
            "Kd": 0.15,         # roll derivative
            "output_max": 15.0, # aileron saturation [deg]
        },
    }

    # Açık çevrim girdiler (δc, Cq)
    CONTROL = {
        "δc_schedule": [
            (0.0, 2.0),
            (10.0, 2.0),
        ],
        "Cq_schedule": [
            (0.0, 0.02),
            (10.0, 0.02),
        ],
        # Attitude referansları
        "attitude_refs": {
            "ψ_desired": 0.0,    # hedef yaw [rad]
            "θ_desired": 0.0,    # hedef pitch [rad] (deniz seviyesi)
            "φ_desired": 0.0,    # hedef roll [rad]
            "δc_fixed": 0.0,     # opsiyonel fallback (kullanılmıyor)
        },
    }

# =============================================================================
# AÇIK ÇEVRIM (open_loop) — Zaman-tabanlı schedule'lar
# =============================================================================

else:  # CONTROL_MODE == "open_loop"
    CONTROL = {
        "δc_schedule": [
            (0.0, 2.0),
            (10.0, 2.0),
        ],
        "δe_schedule": [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        "δr_schedule": [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        "δa_schedule": [
            (0.0, 0.0),
            (10.0, 0.0),
        ],
        "Cq_schedule": [
            (0.0, 0.02),
            (10.0, 0.02),
        ],
    }

# =============================================================================
# ÇIKTI AYARLARI
# =============================================================================

OUTPUT_DIR = "outputs"
RUN_ID = "sim_default"  # Farklı koşumlar için değiştirilir

# =============================================================================
# DOĞRULAMA / KALIBRASYONA AIT PARAMETRELERİ
# =============================================================================

# Legacy Case 1 referansi (supercavitation_gui_LIVE_v10.py'den)
LEGACY_CASE_1 = {
    "V": 40.0,
    "alpha_aoa_deg": -1.0,
    "delta_cav_deg": 2.0,
    "F_drag_target": 5764.0,    # [N]
    "Fz_target": -2109.0,       # [N] (negatif = aşağı)
    "My_target": 3344.0,        # [N·m]
    "tolerance_pct": 5.0,       # ±%5 tolerans
}
