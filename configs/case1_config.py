"""
Legacy Case 1 aracı ile senaryolar.

Çalıştır:
  python src/simulation/run_simulation.py --config configs/case1_config.py

Her senaryo bağımsız bir koşumdur (kendi t=0 başlangıcı). Tek koşumun içinde durum
hiçbir zaman sıfırlanmaz; kontrol girdisi değişince kavite mevcut boyutundan
yeni hedefe doğru gecikmeyle ilerler.

Kontrolcü tipleri: "schedule" (açık çevrim) ve "autopilot" (1 ms pitch+yaw duruş tutma).
"""

# Araç ve hidrodinamik parametreleri — legacy GUI anahtar adları, Case 1 değerleri
VEHICLE = dict(
    # geometri / kütle
    mass=350.0, veh_len=4.0, veh_diam=0.30, L_taper=0.60, x_cg=2.4,
    # Ixx=..., Iyy=..., Izz=...   # verilmezse homojen silindir
    c_roll_damp=0.0,
    # kavitatör
    diam_cav=0.20, cavitator_type="cone", cone_apex_deg=40.0, cone_drag_visc=0.18,
    K_slender=0.5, cone_lift_gain=0.50, K_cone_lift=0.0,
    # kavite / havalandırma
    geom_model="savchenko", k_g_cavity=0.78, K_Dc_factor=1.0, K_Lc_factor=1.0,
    vent_mode="Q", A_v=0.020, k_dev=0.4, body_volume_correction=True,
    # gövde
    CL_alpha_body=2.5, Cdc_body=0.4,
    # kanatlar (artı düzen: sağ/sol elevator, üst/alt rudder)
    fins_enabled=True, fin_chord=0.10, fin_span=0.25, fin_x_pos=3.80,
    fins_use_steady_cavity=False,
    # planing
    planing_enable=True,
    # False: rejim geçişleri sürekli (varsayılan); True: legacy anahtarlamaları birebir
    legacy_exact=False,
)

SIMULATION = dict(t_max=0.5, dt=0.001, dt_control=0.001, log_every=1)

# Case 1 açık çevrim komutları (legacy: fin1=fin3=−0.80° ≡ δe=−0.80°)
CASE1_CONTROL = dict(type="schedule", delta_e_deg=-0.80, delta_r_deg=0.0,
                     delta_c_deg=2.0, thrust=6000.0, gas_flow=15000.0)

# Otopilot kazançları (legacy_exact=False fiziğiyle): 5 çalışma noktasında (kavite yok →
# kavite 9 m, kanat ıslak oranı 1.0 → 0.23, kontrol etkinliği ~6× düşük; son iki noktada
# araç AÇIK ÇEVRİMDE STATİK KARARSIZ, özdeğer +1.8/+1.6) doğrusallaştırılmış modelde en kötü
# durum adım yanıtı + min sönüm oranı (ζ≥0.6 hedef) + rate döngüsü bant genişliği (≤80 rad/s)
# kısıtlarıyla optimize edildi; doğrusal olmayan modelde doğrulandı (test_autopilot.py).
# En kötü ζ ≈ 0.54. Birimler: Kp [rad/rad], Ki [1/s], Kd [s].
AXIS_GAINS = dict(Kp=2.9, Ki=10.5, Kd=0.19, b=0.5, limit_deg=15.0, i_limit_deg=10.0,
                  rate_limit_deg_s=None)
AUTOPILOT = dict(type="autopilot", pitch=AXIS_GAINS, yaw=AXIS_GAINS, V_ref=40.0,
                 theta_ref_deg=0.0, psi_ref_deg=0.0,
                 delta_c_deg=2.0, thrust=6000.0, gas_flow=15000.0)

# Legacy'de pitch/heave/lateral hareket yok: bunları kilitleyince legacy ile kıyaslanabilir
LOCK_ALL_BUT_SURGE = ["v", "w", "p", "q", "r", "phi", "theta", "psi", "Y", "Z"]

SCENARIOS = [
    dict(
        name="case1_kilitli",
        description="Legacy Case 1: sadece ileri hareket serbest, legacy_exact (legacy ile kıyaslanabilir)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=CASE1_CONTROL,
        vehicle=dict(legacy_exact=True),
        locked_states=LOCK_ALL_BUT_SURGE,
    ),
    dict(
        name="case1_serbest_6dof",
        description="Case 1 başlangıcı, tüm serbestlikler açık, açık çevrim (trim dengeli değil)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=CASE1_CONTROL,
    ),
    dict(
        name="case1_gaz_adimi",
        description="Kilitli Case 1; t=0.25 s'de gaz debisi 15000→5000 L/min (kavite sürekliliği)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=dict(CASE1_CONTROL, gas_flow=[(0.0, 15000.0), (0.25, 5000.0)]),
        locked_states=LOCK_ALL_BUT_SURGE,
    ),
    dict(
        name="otopilot_tutma",
        description="Kapalı çevrim: Case 1 başlangıcı, θ=ψ=0 tutma (gaz/itki açık çevrim)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=AUTOPILOT,
        simulation=dict(t_max=1.0),
    ),
    dict(
        name="otopilot_adimlar",
        description="Kapalı çevrim: θ_ref 0→2° (t=0.5 s), ψ_ref 0→2° (t=1.0 s), θ_ref 2→0° (t=1.5 s)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=dict(AUTOPILOT, theta_ref_deg=[(0.0, 0.0), (0.5, 2.0), (1.5, 0.0)],
                     psi_ref_deg=[(0.0, 0.0), (1.0, 2.0)]),
        simulation=dict(t_max=3.0),
    ),
    dict(
        name="dusuk_hiz_baslangic",
        description="Near-inception: V0=20 m/s, kavite yok, serbest 6-DOF",
        initial=dict(V=20.0, alpha_deg=0.0, depth=10.0),
        control=dict(CASE1_CONTROL, delta_e_deg=0.0, delta_c_deg=0.0),
        simulation=dict(t_max=0.3),
    ),
]
