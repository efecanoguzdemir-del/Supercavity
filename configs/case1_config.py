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
    # Gaz debisi Q'nun ölçüldüğü derinlik (kullanıcı, 2026-09-16): 5 m hidrostatik basınçta
    # hacim debisi; kavitede pc'ye genleşir. None → legacy (Q kavite basıncında).
    gas_flow_ref_depth=5.0,
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

# Çalışma noktası (2026-09-16, kullanıcı): itki 8000 N, V0 = 20 m/s, gaz debisi = gövdeyi tamamen
# örtecek minimum debi (kullanıcı tahmini 250-400 L/s). Q, 5 m derinlikteki hacim debisi
# (VEHICLE.gas_flow_ref_depth). Tarama (otopilot + derinlik tutma, 15 s) — tam örtülme için
# en düşük debi: 5 m 55-85, 10 m 85-100, 20 m 200-215 L/s (kavite basıncı ~107 kPa < 150 kPa
# referans → 10 m'de gaz genleşir). 10 m çalışma derinliği için 125 L/s (~%25 pay) seçildi;
# kanat ıslak oranı ~0.47. 20 m ve daha derin için ≥ 250 L/s gerekir.
# Önceki (legacy Q tanımı) tarama: ≥215 L/s, 250 seçilmişti.
THRUST_N = 8000.0
GAS_FLOW_LPS = 125.0              # [L/s], 5 m'de; model girdisi L/min
GAS_FLOW = GAS_FLOW_LPS * 60.0
V0 = 20.0

# Çalışma noktası açık çevrim komutları
OP_CONTROL = dict(type="schedule", delta_e_deg=0.0, delta_r_deg=0.0,
                  delta_c_deg=2.0, thrust=THRUST_N, gas_flow=GAS_FLOW)

# Otopilot kazançları (2026-09-16, çalışma noktası 8000 N / V0=20 / 250 L/s): src/control/tuning.py
# — otopilot_derinlik koşumundan 12 çalışma noktası (kavite yok → tam örtülü, kanat ıslak oranı
# 1.0 → 0.19; t≈1.2-1.5 s'de araç açık çevrimde statik kararsız, +2.6/+3.3 rad/s), donmuş
# doğrusallaştırma, kanat etkinliği çizelgeli k_s ve k_s×{0.7, 1, 1.4} üzerinde en kötü durum
# (ITAE + aşım + ts5 + ζ≥0.6 + |s|≤80 rad/s + kanatta 1° giriş bozucusu reddi + integratör
# payı); b ≥ 0.3. Birimler: Kp [rad/rad], Ki [1/s], Kd [s].
# 125 L/s (Q @5 m) + trim ileri besleme + sensörlerle yeniden doğrulandı: doğrusal en kötü
# ζ=0.77; daha yüksek kazançlı aday (Kp 7.3/Ki 53) derinlik adımında δe'yi 14.3°'ye (sınır 15°)
# çıkardığı için seçilmedi.
AXIS_GAINS = dict(Kp=4.445, Ki=27.69, Kd=0.2501, b=0.30, limit_deg=15.0, i_limit_deg=40.0,
                  rate_limit_deg_s=None)
# Kazanç çizelgesi k_s = clip((V_ref/V)²·eta_ref/eta, k_s_min, k_s_max); eta kanat ıslak
# açıklık oranı, model tabanlı kavite tahmincisinden (estimator=dict() → etkin).
# Derinlik tutma: Z hatası → θ_ref (sadece senaryoda depth_ref_m verilirse etkin).
# Kazançlar: tuning.py (Z_ref adımı + 0.5 m/s² dikey bozucu, en kötü durum) Kp=4.1/Ki=0.71/
# Kd=0.70 verdi; doğrusal olmayan modelde kavite geçişindeki kaldırma kaybı (~−900 N, ~2.6 m/s²)
# 0.68 m batma bıraktı → doğrusal olmayan ızgara taraması + doğrusal sönüm kontrolüyle (en kötü
# ζ ≥ 0.5, k_s×{0.7,1,1.4}) sertleştirildi: geçişte |ΔZ| 0.36 m, 1 m adım aşım %2.6, t95 1.16 s.
DEPTH_GAINS = dict(Kp_deg_m=8.0, Ki_deg_m_s=1.0, Kd_deg_s_m=2.5, theta_limit_deg=5.0,
                   i_limit_deg=4.0)
# Sensörler: kavite basıncı (pc) + kuyruk gaz sensörü tahminciyi düzeltir. Trim ileri beslemesi:
# tahmin edilen kavitede moment dengesi (10 ms'de bir). Referans ön filtresi: adım anındaki
# kanat sıçramasını ve anlık doyumu önler.
AUTOPILOT = dict(type="autopilot", pitch=AXIS_GAINS, yaw=AXIS_GAINS, V_ref=40.0,
                 eta_ref=0.5, k_s_min=0.25, k_s_max=8.0, estimator=dict(eta_min=0.1),
                 sensors=dict(use_pc=True, use_tail=True, pc_noise_pa=0.0),
                 trim_ff=dict(gain=1.0, every=10), ref_tau_s=0.05,
                 depth_hold=DEPTH_GAINS, depth_ref_m=None,
                 theta_ref_deg=0.0, psi_ref_deg=0.0,
                 delta_c_deg=2.0, thrust=THRUST_N, gas_flow=GAS_FLOW)

# Legacy'de pitch/heave/lateral hareket yok: bunları kilitleyince legacy ile kıyaslanabilir
LOCK_ALL_BUT_SURGE = ["v", "w", "p", "q", "r", "phi", "theta", "psi", "Y", "Z"]

SCENARIOS = [
    dict(
        name="case1_kilitli",
        description="Legacy Case 1: sadece ileri hareket serbest, legacy_exact (legacy ile kıyaslanabilir)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=CASE1_CONTROL,
        vehicle=dict(legacy_exact=True, gas_flow_ref_depth=None),
        locked_states=LOCK_ALL_BUT_SURGE,
    ),
    dict(
        name="case1_serbest_6dof",
        description="Case 1 başlangıcı, tüm serbestlikler açık, açık çevrim (trim dengeli değil)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=CASE1_CONTROL,
        vehicle=dict(gas_flow_ref_depth=None),
    ),
    dict(
        name="case1_gaz_adimi",
        description="Kilitli Case 1; t=0.25 s'de gaz debisi 15000→5000 L/min (kavite sürekliliği)",
        initial=dict(V=40.0, alpha_deg=-1.0, depth=10.0),
        control=dict(CASE1_CONTROL, gas_flow=[(0.0, 15000.0), (0.25, 5000.0)]),
        vehicle=dict(gas_flow_ref_depth=None),
        locked_states=LOCK_ALL_BUT_SURGE,
    ),
    dict(
        name="otopilot_tutma",
        description="Kapalı çevrim: V0=20 m/s, 8000 N, 125 L/s; θ=ψ=0 tutma, derinlik döngüsü kapalı",
        initial=dict(V=V0, alpha_deg=0.0, depth=10.0),
        control=AUTOPILOT,
        simulation=dict(t_max=5.0),
    ),
    dict(
        name="otopilot_adimlar",
        description="Kapalı çevrim, V0=20 m/s: θ_ref 0→2° (t=1.0 s, rejim geçişi sırasında), "
                    "ψ_ref 0→2° (t=3.0 s), θ_ref 2→0° (t=4.0 s)",
        initial=dict(V=V0, alpha_deg=0.0, depth=10.0),
        control=dict(AUTOPILOT, theta_ref_deg=[(0.0, 0.0), (1.0, 2.0), (4.0, 0.0)],
                     psi_ref_deg=[(0.0, 0.0), (3.0, 2.0)]),
        simulation=dict(t_max=5.5),
    ),
    dict(
        name="otopilot_derinlik",
        description="Kapalı çevrim, V0=20 m/s: derinlik tutma 10 m, t=5 s'de Z_ref 10→11 m, "
                    "t=8 s'de ψ_ref 0→2°",
        initial=dict(V=V0, alpha_deg=0.0, depth=10.0),
        control=dict(AUTOPILOT, depth_ref_m=[(0.0, 10.0), (5.0, 11.0)],
                     psi_ref_deg=[(0.0, 0.0), (8.0, 2.0)]),
        simulation=dict(t_max=12.0),
    ),
    dict(
        name="dusuk_hiz_baslangic",
        description="Çalışma noktası açık çevrim: V0=20 m/s, kavite yok, 8000 N, 125 L/s, serbest 6-DOF",
        initial=dict(V=V0, alpha_deg=0.0, depth=10.0),
        control=OP_CONTROL,
        # Açık çevrimde duruş ıraksar: 125 L/s'de burun sürekli düşer (θ 3.5 s'de ~−14°) ve
        # kavite gövdeden ayrılır (örtülme 0.45). (250 L/s legacy Q'da: burun kalkıp takla.)
        simulation=dict(t_max=3.5),
    ),
]
