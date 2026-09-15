"""
================================================================================
 SÜPERKAVİTASYON GEÇİŞ DİNAMİĞİ - İNTERAKTİF SİMÜLASYON
================================================================================
 Akademik araştırma için zamana bağlı süperkavitasyon geçiş modeli.

 Temel fiziksel denklemler:
   σ(t)    = 2·(p∞ - pv) / (ρ·V(t)²)            Kavitasyon numarası
   Cx(σ)   = Cx0·(1 + σ),   Cx0 ≈ 0.82          Reichardt sürükleme katsayısı
   Dc/Dn   = √(Cx / (kσ))                        Garabedian kavite çapı
   Lc/Dn   = (1/σ)·√(Cx·ln(1/σ))                 Garabedian kavite boyu
   dLc/dt  = (Lc_ss - Lc) / τ                    Logvinovich olgunlaşma gecikmesi
   m·dV/dt = T - ½ρV²·(S·Cx + Sw·Cf·(1-κ) + ...) Araç dinamiği

 Kavite zarf profili (Logvinovich ince cisim):
   R(x) = Rn + (Rmax - Rn)·f(ξ),  ξ = x/Lc
   Yüksek σ: küt, simetriye yakın   ·   Düşük σ: uzun, ince, sivri kuyruklu

 Referanslar:
   Logvinovich (1969), Garabedian (1956), Reichardt (1946),
   Semenenko (2001), Savchenko (1996)

 Kullanım:
   python supercavitation_gui.py

 Gereksinimler:
   pip install matplotlib numpy
================================================================================
"""

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Polygon

# ============================================================================
# FİZİKSEL SABİTLER
# ============================================================================
RHO   = 1025.0
G     = 9.81
P_ATM = 101325.0
P_VAP = 2340.0

COLORS = {
    "bg": "#0a1320", "panel": "#0d1826", "border": "#1f2d42",
    "text": "#e8eef7", "mute": "#6a7f9a", "accent": "#4ba3c7",
    "accent2": "#d4a548", "cavity": "#7ec4e0", "drag": "#e07b3a",
    "skin": "#95c17a", "press": "#b58acc", "vehicle": "#2a3442",
    "vehicle2": "#5a6a7a", "water1": "#0a2540", "water2": "#061a30",
}


# ============================================================================
# FİZİK ÇÖZÜCÜSÜ
# ============================================================================
def cavity_geometry(sigma, Dn, model="savchenko", Cx0=0.82, k_g_val=0.85,
                     K_Dc_val=1.0, K_Lc_val=1.0):
    """Kavite uzunluğu ve maksimum çapı için literatürdeki farklı modellerin
    karşılaştırılmasına olanak sağlayan dispatcher.

    Cx0: kavitatör drag katsayısı (disk için 0.82, konik için sin²(β) faktörü)

    Modellerin pratik özeti (σ=0.02, Dn=0.05m için):
      ╔═════════════════╦═══════╦═════════╦══════════╦══════════════════╗
      ║ Model           ║ Lc/Dc ║ Lc (m)  ║ Dc (mm)  ║ Karakteristik    ║
      ╠═════════════════╬═══════╬═════════╬══════════╬══════════════════╣
      ║ garabedian      ║  99   ║  32.6   ║   330    ║ Klasik teori,    ║
      ║                 ║       ║         ║          ║ küçük σ'da uzun  ║
      ║ savchenko       ║  50   ║  16.5   ║   330    ║ Deneysel kalibre ║
      ║                 ║       ║         ║          ║ (mermi atışları) ║
      ║ semenenko       ║  14   ║   4.7   ║   330    ║ Doygunluk eğil., ║
      ║                 ║       ║         ║          ║ kalın iç kavite  ║
      ║ may             ║  45   ║  15.0   ║   327    ║ Birkhoff-Plesset ║
      ║                 ║       ║         ║          ║ tipi orta yol    ║
      ║ vasin_serebr    ║  69   ║  22.7   ║   330    ║ Slender body 0.7 ║
      ║                 ║       ║         ║          ║ asimptotik       ║
      ║ logvinovich     ║  44   ║  16.6   ║   376    ║ Kalın çap eğil., ║
      ║                 ║       ║         ║          ║ daha büyük Dc    ║
      ╚═════════════════╩═══════╩═════════╩══════════╩══════════════════╝

    Parametreler:
        sigma : kavitasyon numarası (efektif, σ_eff)
        Dn    : kavitatör çapı [m]
        model : seçilen formül (yukarıdaki anahtar kelimelerden)
        Cx0   : kavitatör drag katsayısı (disk: 0.82, konik için sin²(β))

    Döndürür: (Lc, Dc, Cx) — kavite uzunluğu, çapı, sürükleme katsayısı
    """
    s = max(sigma, 1e-4)
    Cx = Cx0 * (1.0 + s)
    ln_t = max(np.log(1.0 / s), 0.1)

    # Çap formülü — modellere göre değişir
    if model == "logvinovich":
        # Logvinovich düzeltmeli — kalın kavite eğilimi
        Dc = np.sqrt(Cx / s) * Dn * (1.0 + 0.05 / (s ** 0.3))
    elif model == "may":
        # May 1975 — (1+σ) faktörü ek
        Dc = np.sqrt(Cx * (1.0 + s) / s) * Dn
    else:
        # Reichardt-Garabedian standart formül:
        #   Dc/Dn = √(Cx / (σ·k_g))
        # k_g (kavite çap kalibrasyon sabiti):
        #   Reichardt 1946: 0.96  (klasik)
        #   Garabedian 1956: 0.97
        #   May 1975:        0.85
        #   Logvinovich:     0.82  (daha büyük Dc)
        #   Newman 1977:     1.00  (daha küçük Dc)
        # Varsayılan 0.85 (May/Logvinovich uyumlu, mod. CFD-friendly)
        # k_g küçükse → Dc büyük; CFD'de daha büyük Dc gözükürse k_g azalt.
        Dc = np.sqrt(Cx / (s * k_g_val)) * Dn

    # Uzunluk formülü
    if model == "garabedian":
        # Klasik asimptotik: Lc/Dc = (1/σ)·√(ln(1/σ))
        Lc = Dc * (1.0 / s) * np.sqrt(ln_t)
    elif model == "savchenko":
        # Savchenko 2001 deneysel: Lc/Dc = (2/σ)·√(1/[ln(1/σ)(1+σ)])
        Lc = Dc * (2.0 / s) * np.sqrt(1.0 / (ln_t * (1.0 + s)))
    elif model == "semenenko":
        # Semenenko (Logvinovich-tipi yaklaşım) — kalın, kısa kavite
        # Lc/Dn = 1.92/σ - 3.0  (deney verisinden, küçük σ'da doygunluk eğilimi)
        Lc_Dn = max(1.92 / s - 3.0, 0.0)
        Lc = Lc_Dn * Dn
    elif model == "may":
        # May 1975 modifiye: Lc/Dc = √(Cx)/σ · (1+0.5σ)^(-1)
        Lc = Dc * np.sqrt(Cx) / s / (1.0 + 0.5 * s)
    elif model == "vasin_serebr":
        # Vasin-Serebryakov slender body asimptotik (0.7 düzeltme)
        Lc = Dc * (1.0 / s) * np.sqrt(ln_t) * (1.0 - 0.3 * s) * 0.7
    elif model == "logvinovich":
        # Logvinovich klasik: Lc/Dc = (1/σ)·√(ln(1/σ))·(1−σ) düzeltmesi
        Lc = Dc * (1.0 / s) * np.sqrt(ln_t) * (1.0 - s) * 0.85
    else:
        # Bilinmeyen model — Savchenko'ya dön
        Lc = Dc * (2.0 / s) * np.sqrt(1.0 / (ln_t * (1.0 + s)))

    # Boyut kalibrasyon çarpanları (CFD/deney uyumu için)
    # K_Dc: çap büyüt/küçült (1.0=değişmez, >1=daha kalın, <1=daha ince)
    # K_Lc: boy büyüt/küçült (1.0=değişmez, <1=daha kısa)
    # Çap/boy oranını ayarlama için: K_Dc=1.2, K_Lc=0.8 → kalın kısa kavite
    Dc = Dc * K_Dc_val
    Lc = Lc * K_Lc_val
    return max(Lc, 0.0), max(Dc, 0.0), Cx


def simulate(params, t_max=None, dt=None):
    """Hibrit süperkavitasyon simülasyonu — doğal + yapay (havalandırmalı).

    t_max ve dt önceden parametre olarak geçirilebilir veya params içinden
    alınabilir (UI öncelikli). dt çok küçük olursa hesap uzun sürer; çok
    büyük olursa pc dinamiği (τ_pc=0.3s) doğru izlenmez.

    Parametre:
        gas_flow [L/min] — kavite içine enjekte edilen hava debisi
            0 ise: saf doğal buhar kavitasyonu (pc = p_vapor)
            >0 ise: havalandırmalı, pc > p_vapor olabilir

    Hibrit fizik:
        Cq = Q / (V · Dn²)                     — havalandırma katsayısı
        Fr = V / √(g · Dn)                     — Froude sayısı
        σ_eff = 2(p∞ − pc) / (ρV²)             — efektif kavitasyon numarası
        Epshtein denge (twin-vortex rejimi):
            Q_in = Q_out = k_leak · σ² · V · Dn² / Fr²
        Bu denklemden σ_target çözülür; pc_target buradan geri hesaplanır.
        pc zaman gecikmesi τ_pc ile pc_target'a yaklaşır.
    """
    depth     = params["depth"]
    Dn        = params["diam_cav"]
    m         = params["mass"]
    T_init    = params["thrust"]     # başlangıç itki (sabit, schedule yoksa)
    L_veh     = params["veh_len"]
    D_veh     = params["veh_diam"]

    # =========================================================================
    # ZAMAN ÇİZELGELERİ (TIME SCHEDULES)
    # =========================================================================
    # Bazı kontrol parametreleri zamana göre adım-adım değişebilir (step).
    # Bu, harici kod tarafından canlı güncelleme veya parametrik analiz için
    # kullanılır. Schedule formatı:
    #
    #   params["alpha_aoa_schedule"] = [(t0, val0), (t1, val1), (t2, val2), ...]
    #
    # Anlam: t < t0: ilk değer (val0). t0 <= t < t1: val0. t1 <= t < t2: val1.
    # Yani schedule[k] noktası, [t_k, t_{k+1}) aralığında etkili.
    #
    # Eğer schedule verilmezse, klasik sabit parametre kullanılır.
    # Schedule değerleri **DERECE** (açılar için), **N** (thrust), **L/min** (Q).
    #
    # Desteklenen parametreler:
    #   alpha_aoa_schedule  → α [°]
    #   delta_cav_schedule  → δ_c [°]
    #   thrust_schedule     → T [N]
    #   gas_flow_schedule   → Q [L/min] veya Cq (vent_mode'a göre)
    #   fin_delta_1_schedule → δ_üst [°]
    #   fin_delta_2_schedule → δ_sağ [°]
    #   fin_delta_3_schedule → δ_alt [°]
    #   fin_delta_4_schedule → δ_sol [°]
    def step_value(t_now, schedule, default):
        """Schedule listesinden t_now anındaki değeri ZOH (zero-order hold) döndürür.
        schedule: [(t0, v0), (t1, v1), ...] — sıralı zaman noktaları
        Boş veya None ise default döndürür.
        """
        if not schedule:
            return default
        # En son geçilen noktanın değerini bul
        current = schedule[0][1]   # t < t0 ise ilk değer
        for (tk, vk) in schedule:
            if t_now >= tk:
                current = vk
            else:
                break
        return current

    alpha_sched  = params.get("alpha_aoa_schedule", None)
    delta_sched  = params.get("delta_cav_schedule", None)
    thrust_sched = params.get("thrust_schedule",    None)
    gas_sched    = params.get("gas_flow_schedule",  None)
    # 4 kanat açısı için ayrı schedule (her biri opsiyonel)
    fin1_sched   = params.get("fin_delta_1_schedule", None)
    fin2_sched   = params.get("fin_delta_2_schedule", None)
    fin3_sched   = params.get("fin_delta_3_schedule", None)
    fin4_sched   = params.get("fin_delta_4_schedule", None)
    # =========================================================================
    # Konik geçiş bölgesi uzunluğu (kavitatörden araç tam çapına kadar)
    # Gerçek araçlarda kavitatör çapı (Dn) ile araç çapı (D_veh) arasında
    # konik bir burun bölgesi vardır. Bu bölgenin profili lineer:
    #   x ∈ [0, L_taper]: R(x) = Dn/2 + (D_veh/2 − Dn/2) · (x/L_taper)
    #   x > L_taper:      R(x) = D_veh/2  (tam silindirik)
    # Varsayılan: araç boyunun %15'i (Şkval benzeri)
    L_taper = float(params.get("L_taper", 0.15 * L_veh))
    L_taper = max(0.0, min(L_taper, L_veh))
    gas_flow_const = params.get("gas_flow", 0.0)   # L/min (Q-mode) veya boyutsuz Cq (Cq-mode) — başlangıç
    geom_model = params.get("geom_model", "savchenko")  # geometri formülü
    vent_mode  = params.get("vent_mode", "Q")  # "Q" sabit veya "Cq" sabit
    # Gövde hacmi telafisi: R_c² = R_logv² + R_body² (Pythagorean)
    body_vol_corr = bool(params.get("body_volume_correction", True))
    # Moment referans noktası (araç burnundan ölçülen mesafe, m)
    # Varsayılan: araç ortası (L_veh/2). Tipik kütle merkezi konumu.
    x_cg_ref  = float(params.get("x_cg", L_veh / 2.0))
    # Sınır içinde tut
    x_cg_ref  = max(0.0, min(x_cg_ref, L_veh))

    # Kütle merkezi konumu — ağırlığın uygulandığı nokta (moment için)
    # Varsayılan: aracın aşağısı %50 (orta noktası). Genelde araçlarda
    # CG, geometrik merkezden biraz farklı (motor, yakıt, vs.)
    x_cg_mass = float(params.get("x_cg_mass", L_veh / 2.0))
    x_cg_mass = max(0.0, min(x_cg_mass, L_veh))

    # Başlangıç hızı — UI parametresi (varsayılan 3 m/s)
    V_init = float(params.get("V_init", 3.0))
    V_init = max(0.1, V_init)   # sıfır veya negatif olamaz

    # ---- HÜCUM AÇISI ve KAVİTATÖR AÇISI (kontrol girdileri) ----
    #
    # alpha_aoa: Araç gövde ekseni ile akış yönü arasındaki açı [derece]
    #   Pozitif = burun yukarı (gövde ekseni akış vektörünün üzerinde)
    #   Etkileri:
    #     - Asimetrik ıslaklık: gövde alt yüzü daha çok dalar (α<0 için)
    #     - Tail-slap planing kuvvetinde α_p teriminin değişmesi
    #     - Pitching moment: kuvvet kollarının α ile döndürülmesi
    #
    # delta_cav: Kavitatör eğim açısı [derece]
    #   Pozitif = kavitatör yüzü öne yukarı eğik
    #     → akış kavitatör altına çarpar → reaksiyon AŞAĞI yönde
    #     → araç burnu yukarı kalkar (pitching moment pozitif)
    #   Bu, süperkavitasyon araçlarının ana kontrol mekanizmasıdır.
    #   Etkileri:
    #     - Kavitatörde dikey kuvvet F_cav_z = -q·S_n·C_L·cos(δ)
    #       C_L = (π/2)·sin(2δ)/(1+σ) ≈ π·δ/(1+σ) küçük açılar
    #     - Kavite ekseni başlangıç eğimi değişir (h_c'ye lineer katkı)
    #     - Pitching moment: F_cav_z burunda etkir, kol = -x_cg
    # Hücum açısı (α) ve kavitatör açısı (δ_c) — derece olarak girilir
    # α: araç ekseni ile akış arasındaki açı (gövde nose-up = pozitif)
    # δ_c: kavitatör diskin gövde eksenine göre eğimi (kontrol kanadı gibi)
    # Toplam efektif kavitatör akış açısı: α_eff = α + δ_c
    alpha_aoa_deg = float(params.get("alpha_aoa_deg",
                          params.get("alpha_aoa",
                          params.get("alpha_AoA", 0.0))))
    delta_cav_deg = float(params.get("delta_cav_deg",
                          params.get("delta_cav", 0.0)))
    alpha_aoa_deg = max(-15.0, min(alpha_aoa_deg, 15.0))
    delta_cav_deg = max(-30.0, min(delta_cav_deg, 30.0))
    # Başlangıç değerleri (schedule yoksa kullanılır)
    # NOT: alpha_aoa, delta_cav, alpha_eff değerleri döngünün İÇİNDE schedule'dan
    # alınan değerlere göre her adımda güncellenir. Buradaki değerler sadece
    # schedule None ise referans olarak kullanılır.
    alpha_aoa = np.radians(alpha_aoa_deg)
    delta_cav = np.radians(delta_cav_deg)
    alpha_eff = alpha_aoa + delta_cav    # toplam efektif açı (akışa göre, başlangıç)

    # Kavite ekseni sapma modeli (Logvinovich-Serebryakov asimptotik):
    #
    # Naif lineer model: y_c(x) = α · x — kavite sonsuza kadar eğilir, gerçekçi DEĞİL
    #
    # Gerçekçi model: kavite kavitatörün hemen arkasında α açısıyla eğilir,
    # ama belli bir karakteristik uzunlukta (~10·Rn) yatay yönde "stabilize" olur.
    #   y_c(x) = α · x_open · (1 - exp(-x/x_open))
    # Bu, başlangıçta dy/dx ≈ α (eğilme yönü), büyük x'te y → α·x_open (asimptot)
    # Yani toplam yer değiştirme x_open mertebesine sınırlı kalır.
    #
    # k_dev: ek sönümleme katsayısı (deneysel, tipik 0.3-0.5)
    k_dev = float(params.get("k_dev", 0.4))
    k_dev = max(0.0, min(k_dev, 1.0))

    # Gövde lift slope (kanat formülü, slender body)
    # Klasik silindir: C_L_alpha ≈ 1.0–2.0 (aspect ratio'ya bağlı)
    # Slender body teorisi: dCL/dα = 2 (Munk 1924)
    # Pratik silindirik gövde için 1.0–1.5 aralığı tipik
    CL_alpha_body = float(params.get("CL_alpha_body", 1.5))
    CL_alpha_body = max(0.0, min(CL_alpha_body, 6.28))   # ≤ 2π üst sınır

    # Konik kavitatör yan yüzey lift kalibrasyonu (CFD ile uydurma için)
    # 0.0 = klasik hibrit formül (Birkhoff-May, β'ya bağlı)
    # 0.5 = orta etki
    # 1.0 = tam yan yüzey alanı çarpanı (Vasin-Logvinovich)
    # > 1.0 = CFD'ye göre ek artırım
    K_cone_lift = float(params.get("K_cone_lift", 0.0))
    K_cone_lift = max(0.0, min(K_cone_lift, 3.0))

    # Simülasyon süresi ve adımı — UI'dan veya argümandan
    # Çok uzun süre + çok küçük adım → bellek/zaman yükü; otomatik dengeleme:
    # N = t_max/dt > 50000 olursa dt'yi otomatik büyüt (uyarı yok, sessiz limit)
    if t_max is None:
        t_max = float(params.get("t_max", 5.0))
    if dt is None:
        dt = float(params.get("dt", 0.01))
    t_max = max(0.001, min(t_max, 600.0))         # 1ms — 10 dakika (canlı mod chunks için düşürüldü)
    dt    = max(1e-6, min(dt, 0.1))               # 1 µs — 100 ms (min düşürüldü ki dt convergence testi yapılabilsin)
    if t_max / dt > 60000:
        dt = t_max / 60000.0                       # otomatik koruma

    p_inf = P_ATM + RHO * G * depth
    Sn    = np.pi * (Dn / 2) ** 2
    Sb    = np.pi * (D_veh / 2) ** 2
    Sw    = np.pi * D_veh * L_veh

    # ---- KAVİTATÖR TİPİ (disk veya konik) ----
    # Disk: tepe açısı β = 180° (düz disk, klasik)
    # Konik: tepe açısı β < 180°
    #   β=90°  → dik açılı koni (yaygın, orta drag)
    #   β=60°  → keskin koni (düşük drag)
    #   β=45°  → çok keskin (minimum drag, dar kavite)
    # Cx0 ampirik formülü (May 1975, Epshtein 1971):
    #   C_x0(β) = 0.82 · sin²(β/2)
    # Bu, β=180° için 0.82 (disk), β=60° için 0.205 (sivri koni)
    cav_type = params.get("cavitator_type",
                params.get("cav_type", "disk")).lower()  # "disk" veya "cone"
    cone_apex_deg = float(params.get("cone_apex_deg",
                          params.get("cone_half_angle", 90.0) * 2.0))
    cone_apex_deg = max(20.0, min(cone_apex_deg, 180.0))
    cone_apex_rad = np.radians(cone_apex_deg)

    if cav_type == "cone":
        # KONİK KAVİTATÖR Cx FORMÜLÜ — Logvinovich-Serebryakov (viskoz düzeltmeli)
        #
        # Saf lineer teori: Cx_cone = Cx_disk · sin²(β/2)
        #   Bu formül β azaldıkça (sivri konik) drag'i AŞIRI AZ tahmin eder.
        #   Gerçek (CFD/deneysel) drag, viskoz, kavite duvarı sürtünmesi ve
        #   ayrılma noktası etkileriyle teorikten 1.4-2× daha yüksektir.
        #
        # Viskoz düzeltmeli formül:
        #   Cx_cone = Cx_disk · [sin²(β/2) + K_v · cos²(β/2)]
        #
        # K_v (viskoz katkı katsayısı):
        #   = 0     : saf teorik (Logvinovich 1969, May 1975 deneyselden %40-50 az)
        #   = 0.15  : May 1975 deneysel verilere uyumlu
        #   = 0.35  : CFD-uyumlu varsayılan (high-Re viskoz katkıyı içerir)
        #   = 0.50+ : yüksek viskoz katkı (kalibrasyon için)
        K_v_cone = float(params.get("cone_drag_visc", 0.35))
        K_v_cone = max(0.0, min(K_v_cone, 0.50))
        beta_half = cone_apex_rad / 2.0
        sin2_b = np.sin(beta_half) ** 2
        cos2_b = np.cos(beta_half) ** 2
        Cx0 = 0.82 * (sin2_b + K_v_cone * cos2_b)
    else:
        Cx0 = 0.82      # disk (β=180° için sin²(90°)=1)

    Cx0_base = Cx0   # cavity_geometry için backup (alias)

    # =========================================================================
    # KANATLAR (FIN'LER) — 4 bağımsız kontrol yüzeyi (× yerleşim)
    # =========================================================================
    # Süperkavitasyon araçlarında kontrol kanatları gövdeden radyal uzanır.
    # Kavite zarfı içinde kalan kısımlar KURU, dışarı çıkanlar ISLAK.
    # Sadece ıslak kısımlar kuvvet/moment üretir.
    #
    # NACA 16-009 profili (süperkavitasyon için tipik düşük-drag simetrik):
    #   CL_α(2D) = 2π ≈ 6.28 /rad,  CD0 ≈ 0.0085,  stall ≈ 14°
    #   3D düzeltme: CL_α = 2π·AR/(AR+2)  (Prandtl finite-wing)
    fins_enabled = bool(params.get("fins_enabled", True))
    # YENİ: transom slap (F_planing) toggle — default OFF
    planing_enable = bool(params.get("planing_enable", False))
    # YENİ: V sabit modu — thrust drag'e eşitlenir, V değişmez (steady-state)
    steady_v_mode = bool(params.get("steady_v_mode", True))
    fin_chord    = float(params.get("fin_chord",   0.040))   # [m] kord boyu
    fin_span     = float(params.get("fin_span",    0.060))   # [m] gövdeden radyal uzanım
    fin_x_pos    = float(params.get("fin_x_pos",   L_veh * 0.85))  # [m] araç burnundan
    fin_x_pos    = max(0.0, min(fin_x_pos, L_veh))
    # 4 kanadın deflection açıları (her biri bağımsız, ±20°)
    fin_delta_1 = max(-20.0, min(float(params.get("fin_delta_1", 0.0)), 20.0))  # üst
    fin_delta_2 = max(-20.0, min(float(params.get("fin_delta_2", 0.0)), 20.0))  # sağ
    fin_delta_3 = max(-20.0, min(float(params.get("fin_delta_3", 0.0)), 20.0))  # alt
    fin_delta_4 = max(-20.0, min(float(params.get("fin_delta_4", 0.0)), 20.0))  # sol
    # (azimut_deg, delta_deg) — azimut: 90°=üst, 0°=sağ, 270°=alt, 180°=sol
    # Pitch için DİKEY bileşen sin(azimut)·F_L:
    #   üst (sin=+1): yukarı; alt (sin=-1): aşağı; yan (sin=0): sıfır
    fins_list = [
        (90.0,  fin_delta_1, "1-üst"),
        (0.0,   fin_delta_2, "2-sağ"),
        (270.0, fin_delta_3, "3-alt"),
        (180.0, fin_delta_4, "4-sol"),
    ]
    # 3D kanat lift slope (Prandtl finite wing)
    if fin_chord > 1e-6:
        fin_AR = fin_span / fin_chord
        fin_CL_alpha = (2.0 * np.pi * fin_AR) / (fin_AR + 2.0)
    else:
        fin_AR = 0.0
        fin_CL_alpha = 0.0
    fin_CD0 = 0.0085   # NACA 16-009 minimum drag
    fin_oswald_e = 0.85   # Oswald efficiency (NACA 16 serisi)
    # Induced drag coefficient: k = 1/(π·AR·e)
    fin_k_induced = (1.0 / (np.pi * fin_AR * fin_oswald_e)) if fin_AR > 1e-6 else 0.0
    # =========================================================================

    tau, Cf = 0.15, 0.003
    # k_g (Reichardt-Garabedian sabiti) — UI'dan kalibre edilebilir
    # 0.78 (Vasin/CFD-uyumlu), 0.85 (May), 0.96 (klasik Reichardt), 1.00 (Newman)
    k_g = float(params.get("k_g_cavity", 0.78))
    k_g = max(0.50, min(k_g, 1.20))
    # Bağımsız çap/boy kalibrasyon çarpanları (çap/boy oranı ayarı için)
    K_Dc = float(params.get("K_Dc_factor", 1.0))
    K_Dc = max(0.5, min(K_Dc, 2.0))
    K_Lc = float(params.get("K_Lc_factor", 1.0))
    K_Lc = max(0.3, min(K_Lc, 2.0))

    # Havalandırma kontrolü:
    # - "Q" mod: kullanıcı sabit Q [L/min] verir; araç hızlandıkça Cq=Q/(V·Dn²)
    #   düşer ve havalandırma "sulanır" → kavite zamanla küçülür
    # - "Cq" mod: kullanıcı sabit Cq verir; sistem her V için gerekli Q'yu
    #   üretir; bu durumda kavite kararlı kalır
    # Q-mode pratikte gerçekçidir (mekanik gaz pompası sabit debili).
    # Cq-mode literatür/kontrol senaryolarında kullanılır.
    if vent_mode == "Cq":
        # Cq doğrudan kullanıcıdan (sabit başlangıç)
        Cq_const_init = gas_flow_const   # parametreyi Cq olarak yorumla (boyutsuz)
        Q_in_static_init = 0.0     # zamanla hesaplanacak
    else:
        # Q-mode: L/min → m³/s
        Q_in_static_init = gas_flow_const / 60000.0
        Cq_const_init = None

    # Epshtein twin-vortex kaçış modeli sabiti (literatür: 0.1—0.3)
    k_leak = 0.15
    tau_pc = 0.3    # pc → pc_target relaksasyon sabiti [s]

    N = int(t_max / dt)
    t       = np.zeros(N); V      = np.zeros(N); x       = np.zeros(N)
    sigma   = np.zeros(N); Lc     = np.zeros(N); Dc      = np.zeros(N)
    Cx      = np.zeros(N); Fd     = np.zeros(N)
    F_cav   = np.zeros(N); F_skin = np.zeros(N); F_press = np.zeros(N)
    cover   = np.zeros(N); a      = np.zeros(N)
    pc_arr  = np.zeros(N); Cq_arr = np.zeros(N); Fr_arr  = np.zeros(N)
    beta_arr = np.zeros(N); Qout_arr = np.zeros(N)
    hg_arr   = np.zeros(N)   # yerçekimi kavite eksen kaldırması [m]
    Fz_arr   = np.zeros(N)   # NET dikey kuvvet (planing + buoy − ağırlık) [N]
    F_planing_arr = np.zeros(N)  # tail-slap planing kuvveti [N]
    F_cavz_arr    = np.zeros(N)  # kavitatör δ_c dikey kuvveti [N]
    F_bodylift_arr = np.zeros(N) # gövde α-lifti (kanat formülü, ıslak orantılı) [N]
    F_bodydrag_arr = np.zeros(N) # gövde α²-induced drag [N]
    F_body_lift_arr = np.zeros(N)  # gövde α-lift (slender body) [N]
    F_body_drag_arr = np.zeros(N)  # gövde α-drag (induced) [N]
    F_buoy_arr    = np.zeros(N)  # Arşimet kaldırma kuvveti [N]
    F_grav_arr    = np.zeros(N)  # yerçekimi (-m·g) [N]
    V_sub_arr     = np.zeros(N)  # batık hacim [m³]
    My_arr        = np.zeros(N)  # pitching momenti — ağırlık dahil [N·m]
    My_no_grav_arr = np.zeros(N) # pitching momenti — ağırlık hariç [N·m]
    # Kanat (fin) izleme dizileri
    F_fin_total_arr = np.zeros(N)  # toplam dikey kanat lift [N]
    F_fin_drag_arr  = np.zeros(N)  # toplam kanat drag [N]
    M_fin_arr       = np.zeros(N)  # toplam kanat momenti [N·m]
    F_fin_each_arr  = [np.zeros(N) for _ in range(4)]  # her kanat ayrı
    # Schedule izleme dizileri — zamana göre değişen kontrol girdileri
    alpha_aoa_arr = np.zeros(N)  # her t'de uygulanan α [deg]
    delta_cav_arr = np.zeros(N)  # her t'de uygulanan δ_c [deg]
    thrust_arr    = np.zeros(N)  # her t'de uygulanan T [N]
    gas_flow_arr  = np.zeros(N)  # her t'de uygulanan Q (veya Cq) [L/min veya boyutsuz]
    fin_delta_1_arr = np.zeros(N)  # üst kanat δ [deg]
    fin_delta_2_arr = np.zeros(N)  # sağ kanat δ [deg]
    fin_delta_3_arr = np.zeros(N)  # alt kanat δ [deg]
    fin_delta_4_arr = np.zeros(N)  # sol kanat δ [deg]

    V[0] = V_init
    # Canlı mod için state initialization: kavite olgunlaşma state'ini de al
    Lc_r = float(params.get("initial_Lc", 0.0))
    Dc_r = float(params.get("initial_Dc", 0.0))
    pc = float(params.get("initial_pc", P_VAP))

    for i in range(N):
        t[i] = i * dt
        vi = V[i] if i == 0 else V[i-1]
        t_now = t[i]

        # ---- Zaman çizelgesi değerlendirmesi (step / ZOH) ----
        # Her parametre için, varsa schedule değeri o anki t'ye göre uygulanır.
        # Schedule yoksa başlangıç sabiti kullanılır.
        alpha_aoa_deg_t = step_value(t_now, alpha_sched, alpha_aoa_deg)
        delta_cav_deg_t = step_value(t_now, delta_sched, delta_cav_deg)
        T_t             = step_value(t_now, thrust_sched, T_init)
        gas_flow_t      = step_value(t_now, gas_sched, gas_flow_const)
        # Kanat açıları schedule (her biri opsiyonel)
        fin_delta_1_t = step_value(t_now, fin1_sched, fin_delta_1)
        fin_delta_2_t = step_value(t_now, fin2_sched, fin_delta_2)
        fin_delta_3_t = step_value(t_now, fin3_sched, fin_delta_3)
        fin_delta_4_t = step_value(t_now, fin4_sched, fin_delta_4)
        # Sınır kontrolü (±20° kanat aralığı)
        fin_delta_1_t = max(-20.0, min(fin_delta_1_t, 20.0))
        fin_delta_2_t = max(-20.0, min(fin_delta_2_t, 20.0))
        fin_delta_3_t = max(-20.0, min(fin_delta_3_t, 20.0))
        fin_delta_4_t = max(-20.0, min(fin_delta_4_t, 20.0))

        # Sınır kontrolü
        alpha_aoa_deg_t = max(-15.0, min(alpha_aoa_deg_t, 15.0))
        delta_cav_deg_t = max(-30.0, min(delta_cav_deg_t, 30.0))
        # Açıları radyana çevir (her adımda günceltir)
        alpha_aoa = np.radians(alpha_aoa_deg_t)
        delta_cav = np.radians(delta_cav_deg_t)
        alpha_eff = alpha_aoa + delta_cav

        # Bu adımdaki dinamik thrust
        T = T_t

        # Schedule izleme dizilerine kaydet
        alpha_aoa_arr[i] = alpha_aoa_deg_t
        delta_cav_arr[i] = delta_cav_deg_t
        thrust_arr[i]    = T_t
        gas_flow_arr[i]  = gas_flow_t
        fin_delta_1_arr[i] = fin_delta_1_t
        fin_delta_2_arr[i] = fin_delta_2_t
        fin_delta_3_arr[i] = fin_delta_3_t
        fin_delta_4_arr[i] = fin_delta_4_t

        # Doğal buhar σ (referans, karşılaştırma için)
        sigma_v = (2.0 * (p_inf - P_VAP)) / (RHO * vi * vi)

        # Froude sayısı
        Fr = vi / np.sqrt(G * Dn)
        Fr_arr[i] = Fr

        # ---- Kavite içi basınç pc (hibrit: doğal + yapay) ----
        # Monotonik havalandırma-σ ilişkisi (Spurk, Semenenko tipi):
        #     Cq = A_v / σ         →  σ_vent = A_v / Cq_in
        # Daha çok gaz enjekte edilirse σ azalır (kavite büyür, pc artar).
        # A_v literatürden (Semenenko 2001, Epshtein 1970) için ≈ 0.02-0.10.
        # Yüksek değerler (0.10-0.15) Logvinovich rejiminde kullanılır.
        # Kullanıcı CFD veya deney verisi ile kalibrasyon yapabilir.
        # Hibrit σ = min(σ_vapor, σ_vent) — hangisi KÜÇÜKSE kavite o seviyede.
        A_v = float(params.get("A_v", 0.035))
        A_v = max(0.005, min(A_v, 0.30))

        # Bu adımdaki Cq ve Q_in — moda göre + schedule
        # gas_flow_t değeri yukarıda schedule'dan veya başlangıç sabitinden okundu
        if vent_mode == "Cq":
            # Cq doğrudan kullanıcıdan
            Cq_in = gas_flow_t
            Q_in_now = Cq_in * vi * (Dn ** 2) if vi > 0.5 else 0.0
        else:
            # Q-mode: L/min → m³/s
            Q_in_now = gas_flow_t / 60000.0
            if Q_in_now > 1e-10 and vi > 0.5 and Dn > 1e-4:
                Cq_in = Q_in_now / (vi * Dn ** 2)
            else:
                Cq_in = 0.0

        if Cq_in > 1e-9:
            sigma_vent = A_v / max(Cq_in, 1e-6)
        else:
            sigma_vent = np.inf   # havalandırma yok → doğala düş

        # Hibrit efektif σ — ikisinin MİNİMUMU (hangi rejim baskınsa o)
        sigma_hybrid = min(sigma_v, sigma_vent)

        # σ'dan pc_target geri çöz
        pc_target = p_inf - 0.5 * RHO * vi * vi * sigma_hybrid
        pc_target = max(pc_target, P_VAP)
        pc_target = min(pc_target, p_inf - 1.0)

        # Zaman gecikmeli yaklaşma (pc dinamiği)
        pc += (pc_target - pc) * (dt / tau_pc)
        pc = max(pc, P_VAP)
        pc_arr[i] = pc

        # Efektif σ (havalandırma dahil) — pc'den geri hesapla
        s = (2.0 * (p_inf - pc)) / (RHO * vi * vi)
        sigma[i] = min(s, 3.0)

        # Kayıp debisi (görselleştirme için; dengeli durumda Q_in'e eşit)
        s_safe = max(s, 1e-4)
        Q_out_est = A_v / s_safe * vi * (Dn ** 2)  # Cq_out · V · Dn²
        Qout_arr[i] = Q_out_est

        # Cq (havalandırma katsayısı — bu adımda hesaplanan)
        Cq_arr[i] = Cq_in

        # Paryshev kararlılık parametresi β = σ_vapor / σ_eff
        # Havalandırma pc'yi yükseltirse σ düşer → β büyür
        # β > 2.645 kararsızlık bölgesi (Paryshev)
        if s > 1e-4:
            beta_arr[i] = sigma_v / s
        else:
            beta_arr[i] = 1.0

        # ---- Kavite boyutları — seçilen modele göre ----
        # Birden fazla literatür modeli arasında geçiş için cavity_geometry()
        # dispatcher'ı kullanılır. Karşılaştırma yapmak için params["geom_model"]
        # KAVİTE ŞEKLİ vs DRAG için farklı Cx kullanılır:
        # • DRAG kuvveti: Cx0 (konik için viskoz düzeltmeli, sin² formülü)
        # • KAVİTE ŞEKLİ (Dc, Lc): Cx_disk = 0.82 sabit kullanılır
        #
        # NEDEN AYRI? Birkhoff-Plesset 1957 ve Vasin-Paryshev 2002'ye göre
        # konik kavitatörde kavite ayrılma çizgisi konik tabandan (Dn çapı)
        # olur ve sonrası DİSK-EŞDEĞER Logvinovich elipsoidi şeklini alır.
        # Yani konik kavitatörün KAVİTE ŞEKLİ disk'le aynıdır, sadece DRAG
        # kuvveti viskoz nedeniyle farklıdır (Cx küçük). Eğer Reichardt
        # formülünde konik Cx kullanılırsa, Dc gereksiz küçük çıkar.
        Cx_for_shape = 0.82   # her zaman disk değeri (kavite şekli için)
        Lc_ss, Dc_ss, Cx_i = 0.0, 0.0, Cx0
        if s < 1.0:
            Lc_ss, Dc_ss, _ = cavity_geometry(s, Dn, model=geom_model,
                                              Cx0=Cx_for_shape, k_g_val=k_g,
                                              K_Dc_val=K_Dc, K_Lc_val=K_Lc)
            # Cx_i drag için, konik viskoz formülünden gelir
            Cx_i = Cx0 * (1.0 + s)

        # ---- KAVİTATÖR AÇI ETKİSİ (Reichardt-Garabedian) ----
        # Eğik kavitatör drag'ı azaltır ve dikey kuvvet (lift) üretir.
        # Küçük açı için disk kavitatör:
        #   C_x_eff = C_x · cos²(α_eff)              (drag azalır)
        #   C_L_cav = Cx0 · sin(α_eff) · cos(α_eff)  (lift ortaya çıkar)
        # α_eff = α (hücum) + δ_c (kavitatör eğimi).
        cos_a = np.cos((alpha_aoa + delta_cav))
        sin_a = np.sin((alpha_aoa + delta_cav))
        Cx_i = Cx_i * cos_a * cos_a
        C_L_cav = Cx0 * sin_a * cos_a   # küçük açı: ≈ Cx0·α_eff

        Lc_r += (Lc_ss - Lc_r) * (dt / tau)
        Dc_r += (Dc_ss - Dc_r) * (dt / tau)
        Lc[i] = Lc_r; Dc[i] = Dc_r; Cx[i] = Cx_i

        # ---- Yerçekimi + açı kaynaklı kavite eksen kayması ----
        # Savchenko-Semenenko parabolik + lineer açı katkısı:
        #   h(x) = g·x²/(2V²) + α_eff·x
        # Pozitif α_eff (kavitatör/burun yukarı) → kavite ekseni yukarı kayar
        # → araç altı suya temas eder → planing kuvveti artar (kontrol).
        if vi > 0.1:
            h_g_tail = G * L_veh * L_veh / (2.0 * vi * vi) + (alpha_aoa + delta_cav) * L_veh
            h_g_cav  = (G * Lc_r * Lc_r / (2.0 * vi * vi) + (alpha_aoa + delta_cav) * Lc_r
                        if Lc_r > 1e-4 else 0.0)
        else:
            h_g_tail = 0.0
            h_g_cav  = 0.0
        # hg_arr: araç sonundaki eğilme (asimetrik ıslaklık ile ilgili olan)
        hg_arr[i] = h_g_tail

        # ---- ASİMETRİK ISLAKLIK ve PLANING KUVVETLERİ ----
        # Yerçekimi kavite eksenini yukarı eğer: h_c(x) = g·x²/(2V²).
        # Araç ekseni düz gidiyorsa, araç kesit merkezi kavite kesit
        # merkezinden Δz(x) = h_c(x) kadar AŞAĞIDA kalır. Eğer
        # δ(x) = R_v + h_c(x) − R_c(x) > 0 ise aracın ALT yüzeyinin
        # bir kısmı kaviteden çıkıp suyla temas eder (kısmi ıslanma).
        #
        # Islak yay açısı (dairesel kesit için): θ = 2·arccos(1 − δ/R_v)
        # Islak çevre:   s_wet(x) = R_v · θ_wet(x)
        # Islak uzunluk: L_wet = ∫ 1_{δ>0} dx
        #
        # Kuvvetler (Logvinovich-Paryshev / Wagner planing):
        #   dF_skin = q · C_f · s_wet · dx           (ıslak yay boyunca sürtünme)
        #   dF_lift = q · C_p · R_v · δ(x) · dx      (aşağıdan yukarı kaldırma)
        #   dM_y    = (x − x_cg) · dF_lift           (pitching momenti)
        # Yerel dalma açısı: α_p(x) = dh_c/dx = g·x/V²
        # İleri (drag) yönlü basınç bileşeni: dF_press_drag = dF_lift · sin(α_p)

        q = 0.5 * RHO * vi * vi
        R_v_max = D_veh / 2.0       # tam silindirik gövde yarıçapı
        R_n = Dn / 2.0              # kavitatör yarıçapı (burunda)
        x_cg = x_cg_ref      # kullanıcı parametresi (params["x_cg"])

        # Kavitatör genişliği — kavite koordinat sıfırı kavitenin AYRILMA
        # noktasıdır (Garabedian/Logvinovich konvansiyonu).
        # Disk: kavitatörün ön yüzü x=0, arka yüzü x=cav_w → araç x=cav_w'den başlar
        # Konik: geniş taban (kavite ayrılma) x=0 → araç DOĞRUDAN x=0'dan başlar
        cav_w = max(Dn * 0.25, 0.01)
        if cav_type == "cone":
            x_body_start = 0.0     # araç burnu doğrudan kavite ayrılma noktasında
        else:
            x_body_start = cav_w   # araç kavitatörün arkasında başlar

        # Yardımcı: araç gövdesinin yerel yarıçapı (konik burun + silindir)
        def vehicle_radius(xs_local_):
            """Araç burnundan xs_local_ kadar uzakta gövde yarıçapı.
            Konik geçiş: x ∈ [0, L_taper]: lineer Rn → R_v_max
                        x > L_taper:      sabit R_v_max
            """
            if L_taper <= 1e-6:
                return R_v_max
            if xs_local_ < L_taper:
                return R_n + (R_v_max - R_n) * (xs_local_ / L_taper)
            return R_v_max

        F_skin_tot = 0.0
        wet_len = 0.0        # toplam ıslak uzunluk (en az bir nokta ıslak)
        wet_area = 0.0       # gerçek ıslak yüzey alanı [m²]
        # Kaldırma kuvveti (Arşimet) toplamları
        F_buoy_tot = 0.0     # toplam yukarı yönlü kaldırma [N]
        M_buoy_tot = 0.0     # kaldırmadan kaynaklı moment (x_cg etrafında) [N·m]
        V_submerged_tot = 0.0 # toplam batık hacim [m³]
        # Gövde α-bağımlı lift/drag (slender body)
        F_body_lift_tot = 0.0  # gövde lift kuvveti [N]
        F_body_drag_tot = 0.0  # gövde indükletilmiş drag [N]
        M_body_lift_tot = 0.0  # gövde liftinin momenti [N·m]
        # Toplam yan yüzey alanı — konik burun (kesik koni) + silindir kısmı
        if L_taper > 1e-6:
            # Kesik koni yan yüzeyi: π · (R_n + R_v_max) · √(L_taper² + (R_v_max−R_n)²)
            slant = np.sqrt(L_taper ** 2 + (R_v_max - R_n) ** 2)
            A_taper = np.pi * (R_n + R_v_max) * slant
            A_cyl   = 2.0 * np.pi * R_v_max * (L_veh - L_taper)
            total_area = A_taper + A_cyl
        else:
            total_area = 2.0 * np.pi * R_v_max * L_veh

        # Kesit sayısı — araç boyu boyunca integrasyon (sürtünme için)
        n_x = 40
        dx_seg = L_veh / n_x

        if Lc_r > 1e-4 and Dc_r > 1e-4:
            for xs_local in np.linspace(0.5 * dx_seg, L_veh - 0.5 * dx_seg, n_x):
                # xs_local: araç burnundan ölçülen yerel konum
                # xs_cav: kavite koordinatında konum (kavitatör ön kenarı=0)
                xs_cav = xs_local + x_body_start

                # Bu kesitteki ARAÇ yarıçapı (konik burun dahil)
                R_v = vehicle_radius(xs_local)

                # Bu kesitteki kavite eksen yüksekliği:
                #   - Yerçekimi: h_g(x) = g·x²/(2V²)
                #   - Açı sapması (Logvinovich-Serebryakov asimptotik):
                #       h_α(x) = k_dev · α_eff · x_open · (1 − exp(−x/x_open))
                #     Naif lineer (α_eff·x) kavite sonsuza kadar eğilir, gerçekçi DEĞİL.
                #     Asimptotik model: kavite belli bir karakteristik uzunlukta
                #     (~10·Rn) yatay yöne doğru "stabilize" olur, h_α → α_eff·x_open
                #   k_dev: ek sönümleme katsayısı (deneysel, tipik 0.3-0.5)
                # x_open burada hesaplanmaz; aşağıda tanımlanan değer ile çakışır.
                # Önce x_open'ı hesaplayalım (Cx_i'ye bağlı, σ-bağımsız ölçek)
                x_open_loc = max(10.0 * R_n / max(np.sqrt(Cx_i), 0.5), R_n * 2.0)
                if vi > 0.5:
                    h_grav = G * xs_cav * xs_cav / (2.0 * vi * vi)
                    h_angle = (k_dev * alpha_eff * x_open_loc
                               * (1.0 - np.exp(-xs_cav / x_open_loc)))
                    hc_x = h_grav + h_angle
                else:
                    hc_x = 0.0
                # Bu kesitteki kavite yarıçapı — Logvinovich asimptotik formül:
                #   R²(x) = Rn² + (Rmax² − Rn²) · S(x) · D(x)
                # S, x_open mesafesinde σ-bağımsız açılma; D, Lc/2'den sonra
                # parabolik kapanma. σ küçüldükçe (Lc uzadıkça) aynı fiziksel
                # x noktasında Rc gerçekten büyür — Logvinovich bağımsızlık ilkesi.
                if xs_cav < Lc_r:
                    Rmax = Dc_r / 2.0
                    x_open = max(10.0 * R_n / max(np.sqrt(Cx_i), 0.5), R_n * 2.0)
                    x_mid  = Lc_r * 0.5
                    S = 1.0 - np.exp(-xs_cav / x_open)
                    if xs_cav < x_mid:
                        D = 1.0
                    else:
                        D = max(0.0, 1.0 - ((xs_cav - x_mid) / max(Lc_r - x_mid, 1e-6)) ** 2)
                    rc2 = R_n**2 + (Rmax**2 - R_n**2) * S * D
                    rc_logv = np.sqrt(max(rc2, 0.0))

                    # GÖVDE-KAVİTE ETKİLEŞİMİ
                    # İki rejim tanımlanır:
                    #
                    # (a) FREE-STANDING (R_logv ≥ R_v): kavite gövdeyi rahat sarıyor.
                    #     Superposition uygulanır: R_zarf² = R_logv² + R_v²
                    #     (gaz hacmi korunumu, Logvinovich 1969)
                    #
                    # (b) ATTACHED (R_logv < R_v): doğal kavite gövdeden KÜÇÜK.
                    #     Kavite gövdeye YAPIŞIK kalır. Gerçekte gövdenin
                    #     sadece **kavitatör arkasındaki dar bir gaz tabakası**
                    #     kadar kuru kalır. Geri kalan gövde yüzeyi ıslak.
                    #     Bu durumu işaretlemek için rc_x'i R_logv olarak alıyoruz
                    #     (gerçek gaz tabakasının dış sınırı). delta hesabı:
                    #       delta = R_v + |h_c| - R_logv → büyük → ıslak çıkar
                    #     Bu da fiziksel olarak doğru.
                    if rc_logv >= R_v:
                        # Free-standing — superposition geçerli (gaz hacmi korunumu)
                        rc_x = (np.sqrt(rc_logv ** 2 + R_v ** 2)
                                if body_vol_corr else rc_logv)
                    else:
                        # Attached cavity — sadece kavitatörden gelen gaz tabakası
                        # rc_x = R_logv (gaz tabakası R_logv yarıçapında)
                        # delta = R_v + |h_c| - R_logv (büyük → çok ıslak)
                        rc_x = rc_logv
                else:
                    rc_x = 0.0

                # Alt/üst dalma — AKIŞ-BAĞLI KOORDİNAT SİSTEMİ:
                # Görselde gövde yatay sabit, kavite α_eff açısıyla eğilir.
                # Kavite ekseni dikey kayması: h_c = g·x²/(2V²) + α_eff·x
                # h_c > 0: kavite yukarı kaymış → ALT yüzey suya batık
                # h_c < 0: kavite aşağı kaymış → ÜST yüzey havada (suyla teması)
                # Mutlak değer kullanılır çünkü iki yönde de ıslaklık yaratır.
                # NOT: Yerçekimi her zaman pozitif h_c üretir, sadece negatif α_eff
                # ile h_c negatif olabilir (ve sadece yerçekiminin baskın olmadığı
                # noktalarda).
                delta = R_v + abs(hc_x) - rc_x

                # ISLAKLIK — gövde-kavite rejimine göre AYRI MANTIK:
                #
                # (A) FREE-STANDING (rc_logv ≥ R_v): kavite gövdeden büyük.
                #     Burada normal "alt-üst ıslaklık" mantığı geçerli:
                #     yerçekimi/açı kavite ekseni kaydırır → alt veya üst ıslak.
                #     delta = R_v + |h_c| - rc_zarf (zarf = sqrt(R²+R_v²))
                #
                # (B) ATTACHED (rc_logv < R_v): kavite gövdeden KÜÇÜK.
                #     Kavite gövde etrafında dar bir tabakadır. Bu kesitte
                #     gövdenin yüzeyi:
                #       - rc_logv > 0 (kavitatöre yakın, gaz tabakası var):
                #           gövdenin ÇEVRESEL %∼= rc_logv/R_v kadarı gaz içinde
                #           Yani gövde kısmen ıslak (alttan halka şeklinde)
                #       - rc_logv ≈ 0 (kavite kapanmış, x > Lc_attached):
                #           tam ıslak
                #     Asimetri (yerçekimi/α) bu rejimde önemsiz çünkü kavite
                #     zaten zorla gövdeye yapışık, aşağı-yukarı kayamaz.
                if rc_x > 1e-4 and R_v > 1e-6:
                    if rc_logv >= R_v:
                        # ---- FREE-STANDING REJIM ----
                        if delta >= 2.0 * R_v:
                            theta_wet = 2.0 * np.pi
                            s_wet = 2.0 * np.pi * R_v
                            delta_eff = 2.0 * R_v
                        elif delta > 0:
                            arg = max(min(1.0 - delta / R_v, 1.0), -1.0)
                            theta_wet = 2.0 * np.arccos(arg)
                            s_wet = R_v * theta_wet
                            delta_eff = delta
                        else:
                            # Tamamen kuru — kavite gövdeyi sarıyor
                            theta_wet = 0.0
                            s_wet = 0.0
                            delta_eff = 0.0
                    else:
                        # ---- ATTACHED CAVITY REJIM ----
                        # Kavite gövdenin ETRAFINDA ince çevresel tabaka.
                        # rc_logv: gaz tabakasının dış yarıçapı
                        # R_v: gövde yarıçapı
                        # Eğer rc_logv > 0.1·R_v: kabul edilebilir gaz tabakası
                        #    → çevresel olarak %~ rc_logv/R_v kadarı kuru
                        # Eğer rc_logv → 0: kavite kapanmış, tam ıslak
                        if rc_logv < 0.05 * R_v:
                            # Pratik olarak kavite yok — tam ıslak
                            theta_wet = 2.0 * np.pi
                            s_wet = 2.0 * np.pi * R_v
                            delta_eff = 2.0 * R_v
                        else:
                            # Kısmi ıslak — gaz tabakasının uzantısı
                            # Gaz tabakası gövde etrafında halka, gövdenin
                            # alt kısmı suyla temasta. theta_wet ~ 2π·(1 - rc_logv/R_v)
                            # Ancak çevresel olduğundan tam tersi mantık:
                            # gövdenin (R_v - rc_logv)/R_v oranı kuru hava
                            # ile temasta, geri kalanı ıslak
                            wet_fraction = max(0.0, 1.0 - rc_logv / R_v)
                            theta_wet = 2.0 * np.pi * wet_fraction
                            s_wet = R_v * theta_wet
                            delta_eff = R_v - rc_logv
                else:
                    # Kavite henüz yok (Lc gecikmesi) — tam ıslak
                    theta_wet = 2.0 * np.pi
                    s_wet = 2.0 * np.pi * R_v
                    delta_eff = 2.0 * R_v

                if s_wet > 0.0:
                    wet_len += dx_seg
                # Gerçek ıslak yüzey alanı: ıslak yay × dx
                wet_area += s_wet * dx_seg

                # Sürtünme (ıslak yay üzerinde) — bu kısım fizik doğru
                dF_skin = q * Cf * s_wet * dx_seg
                F_skin_tot += dF_skin

                # ---- KALDIRMA KUVVETİ (BUOYANCY) — Arşimet prensibi ----
                # Aracın suyla temas eden bölümlerinde, batık hacim suyun
                # ağırlığı kadar yukarı yönlü kaldırma kuvveti üretir.
                # Bir kesitin batık alanı, ıslak yay oranıyla orantılıdır:
                #   A_batık = (θ_wet / 2π) · π · R_v² = θ_wet · R_v² / 2
                # dV_batık = A_batık · dx
                # dF_b = ρ·g·dV_batık (yukarı yönlü)
                if theta_wet > 0:
                    A_submerged = 0.5 * theta_wet * R_v * R_v   # batık kesit alanı [m²]
                    dV_submerged = A_submerged * dx_seg
                    # Kaldırma katkısı (yukarı = pozitif)
                    dF_buoy = RHO * G * dV_submerged
                    F_buoy_tot += dF_buoy
                    # Hacim merkezinde uygulanır (kesit merkezi → x_local)
                    # Sağ-el kuralı: M_y = (x_cg - x_force) · F_z
                    M_buoy_tot += (x_cg - xs_local) * dF_buoy
                    # Toplam batık hacim (görselleştirme/metrik için)
                    V_submerged_tot += dV_submerged

                # ---- GÖVDE α-LİFT ve α-DRAG (Munk slender body teorisi) ----
                # Hücum açısı (α_aoa) altındaki ıslak silindir gövdenin
                # kanat-benzeri lift ve indükletilmiş drag üretmesi.
                # Munk (1924) slender body limit: silindir gövde için
                #   F_L = ρV²·A_taban·sin(2α)/2 ≈ ρV²·πR²·α  (küçük α)
                # PLUS Hoerner crossflow viskoz katkı (büyük α'da):
                #   F_N_crossflow = ρV²·sin²α·(2R·dx)·C_dc
                # Burada A_taban = πR² (silindir kesit alanı)
                # Crossflow plan alan: 2R·dx (yandan görünüş)
                # f_wet ile ıslak orantı uygulanır.
                if theta_wet > 0 and abs(alpha_aoa) > 1e-6:
                    f_wet = theta_wet / (2.0 * np.pi)
                    Cdc = 1.2  # Hoerner crossflow drag katsayısı
                    sa = np.sin(alpha_aoa); ca = np.cos(alpha_aoa)
                    s2a = np.sin(2.0 * alpha_aoa)
                    sgn = np.sign(alpha_aoa)
                    A_section = np.pi * R_v * R_v
                    w_plan = 2.0 * R_v
                    # Munk lineer + crossflow viskoz (yönlü)
                    dFL_per_dx = f_wet * (
                        RHO * vi * vi * A_section * s2a * 0.5      # Munk
                      + RHO * vi * vi * w_plan * Cdc * sa * sa * ca * sgn  # crossflow yönlü
                    )
                    dF_body_lift = dFL_per_dx * dx_seg
                    dF_body_drag = abs(dF_body_lift) * abs(np.tan(alpha_aoa))
                    F_body_lift_tot += dF_body_lift
                    F_body_drag_tot += dF_body_drag
                    M_body_lift_tot += (x_cg - xs_local) * dF_body_lift

                # NOT: Planing kuvveti (F_z) ve drag bileşeni (F_press) artık
                # araç boyu boyunca integrasyon ile değil, sadece TRANSOM'da
                # (kuyruk uçunda) Dzielski-Kurdila formülüyle hesaplanır.
                # Aşağıdaki tek-noktada integral artık yapılmıyor — döngü
                # sadece sürtünme ve ıslak alan için.

        # ---- KANATLAR (FIN'LER) — 4 bağımsız kontrol yüzeyi ----
        # Süperkavitasyon araçlarında stabilizer/kontrol kanatları gövdeden
        # radyal uzanır. Sadece kavite zarfı DIŞINDA kalan kısımlar ıslak,
        # kuvvet üretir. NACA 16-009 profili kullanılır.
        # Bu adım için schedule değerleri (zamana göre değişir):
        fins_list_t = [
            (90.0,  fin_delta_1_t, "1-üst"),
            (0.0,   fin_delta_2_t, "2-sağ"),
            (270.0, fin_delta_3_t, "3-alt"),
            (180.0, fin_delta_4_t, "4-sol"),
        ]
        F_fin_lift_z_total = 0.0    # toplam dikey kanat lift'i [N]
        F_fin_drag_total   = 0.0    # toplam kanat drag'ı [N]
        M_fin_total        = 0.0    # toplam kanat momenti (pitch) [N·m]
        F_fin_individual   = [0.0, 0.0, 0.0, 0.0]   # her kanadın dikey kuvveti
        if fins_enabled and fin_chord > 1e-6 and fin_span > 1e-6:
            # Kanadın x konumundaki kavite yarıçapı — Logvinovich asimptotik
            x_fin_local = fin_x_pos    # araç burnundan
            # x_fin'in kavite x-koordinatındaki konumu (kavitatör arkasından)
            x_fin_cav = x_fin_local + 0.013   # x_body_start offset
            # Kavite yarıçapı x_fin'de — basit Logvinovich kavite şekli
            Lc_now = Lc_ss
            Dc_now = Dc_ss
            if Lc_now > 1e-6 and Dc_now > 1e-6 and x_fin_cav < Lc_now:
                # Logvinovich elipsoid yaklaşımı: R(x) = (Dc/2)·√(1 − (1 − 2x/Lc)²)
                xi = 2.0 * x_fin_cav / Lc_now - 1.0  # -1: burun, 0: orta, +1: kuyruk
                xi = max(-1.0, min(1.0, xi))
                R_c_at_fin = (Dc_now / 2.0) * np.sqrt(max(0.0, 1.0 - xi * xi))
            else:
                R_c_at_fin = 0.0    # kavite yok veya kanat kavitesi dışı

            # Gövde yarıçapı x_fin'de
            if L_taper > 1e-6 and x_fin_local < L_taper:
                R_v_at_fin = R_n + (R_v_max - R_n) * (x_fin_local / L_taper)
            else:
                R_v_at_fin = R_v_max
            # Effective kavite yarıçapı (gövde superposition)
            R_eff_at_fin = max(R_c_at_fin, R_v_at_fin)

            # Her kanadı işle
            x_cg_local = x_cg   # araç burnundan x_cg
            for k_fin, (azim_deg, delta_fin_deg, label) in enumerate(fins_list_t):
                # Kanat root: R_v_at_fin (gövde yüzeyinde)
                # Kanat tip:  R_v_at_fin + fin_span
                r_root = R_v_at_fin
                r_tip  = R_v_at_fin + fin_span
                # Islak span: kavite zarfının (R_eff) dışındaki kısım
                # Eğer R_eff < r_root: tüm kanat ıslak
                # Eğer R_eff > r_tip:  tüm kanat kuru (kavite içinde)
                # Aksi: r_tip - R_eff kadar ıslak
                if R_eff_at_fin < r_root:
                    wet_span_fin = fin_span    # tam ıslak
                elif R_eff_at_fin >= r_tip:
                    wet_span_fin = 0.0         # tam kuru
                else:
                    wet_span_fin = r_tip - R_eff_at_fin   # kısmen ıslak
                if wet_span_fin <= 1e-6 or vi <= 0.5:
                    continue

                # Islak alan
                S_wet_fin = fin_chord * wet_span_fin
                # Etkin açı: α_aoa (araç) + δ_fin (kanat sapması)
                # Dikkat: yatay kanatlarda (sağ/sol) α_aoa pitch'i değiştirmez
                # ama δ_fin kendisi etki eder. Pitch için sadece üst+alt önemli.
                # Genel formül: α_eff_fin = α_aoa·sin(azimut)·[-1 if 270 deg-effect] + δ_fin
                # Pratikte: üst kanat AoA'yı tam görür, alt kanat ters görür.
                # Üst kanat (azim=90): α_eff = α_aoa + δ_fin
                # Alt kanat (azim=270): α_eff = -α_aoa + δ_fin
                sin_azim = np.sin(np.radians(azim_deg))
                cos_azim = np.cos(np.radians(azim_deg))
                # δ ve α_aoa DÜNYA-z koordinatında tanımlı (uçak elevator
                # konvansiyonu). Her kanadın görme açısı azimut'a göre projekte:
                #   - Üst kanat (azim=90°, sin=+1): tam dünya açısı görür
                #   - Alt kanat (azim=270°, sin=-1): ters görür (kanada-yerel)
                #   - Yan kanatlar (sin=0): hiç görmez (sadece yaw için aktif)
                # alpha_world = α_aoa + δ_fin (dünya-z'de hedef açı)
                # alpha_yerel (kanada-göreli) = alpha_world · sin(azim)
                alpha_world = alpha_aoa + np.radians(delta_fin_deg)
                alpha_local_fin = alpha_world * sin_azim
                # Stall koruması (NACA 16-009: ~14°)
                alpha_local_fin = max(np.radians(-14.0),
                                      min(alpha_local_fin, np.radians(14.0)))

                # Lift ve drag (kanada-yerel CL, CD hesabı)
                CL_fin = fin_CL_alpha * alpha_local_fin
                CD_fin = fin_CD0 + fin_k_induced * CL_fin * CL_fin
                F_L_fin = q * S_wet_fin * CL_fin     # kanada-dik lift
                F_D_fin = q * S_wet_fin * CD_fin     # akışa zıt drag
                # Lift'in DÜNYA-z bileşeni: F_L_yerel · sin(azim)
                # Her iki yatay kanat (üst+alt) aynı δ → AYNI yönde dünya-z lift
                # (sin² etkisiyle iki kanat birikir, antimetrik durumda iptal eder)
                F_L_z = F_L_fin * sin_azim
                # Bu kanadın araç pitch'ine etkisi
                F_fin_individual[k_fin] = F_L_z
                F_fin_lift_z_total += F_L_z
                F_fin_drag_total   += F_D_fin
                # Moment kolu: M_y = (x_cg - x_fin) · F_L_z (sağ-el kuralı)
                # Pozitif M_y = burun yukarı çevirici moment
                # Kanat CG'nin ARKASINDA (x_fin > x_cg) + yukarı lift (+F_z)
                # → kuyrukta yukarı kuvvet → BURUN AŞAĞI → NEGATIF M_y ✓
                # Kavitatör (x=0, x_cg>0) + yukarı lift → BURUN YUKARI → POZITIF M_y
                # Bu konvansiyon tüm moment formülleriyle tutarlıdır.
                M_fin_total += (x_cg_local - x_fin_local) * F_L_z

        # ---- TRANSOM'DA PLANING KUVVETİ (Dzielski-Kurdila 2003) ----
        # Süperkavitasyon literatüründe planing kuvveti, aracın kuyruk
        # ucunun (transom) kavite duvarıyla teması sonucu oluşur. Araç
        # boyu boyunca dağıtılmış değildir; bu çok yaygın bir hatadır.
        #
        # Formül (Dzielski-Kurdila 2003, IEEE J. Vibration & Control):
        #   F_pz = ρV²R_v² · [1 − (Δ/(h+Δ))²] · [(1+h/R_v)/(1+2h/R_v)] · α
        #
        # burada:
        #   R_v = araç yarıçapı (transom)
        #   R_c = transom'daki kavite yarıçapı
        #   Δ   = R_c − R_v   (kavite-araç boşluğu, pozitif: araç kavite içinde)
        #   h   = dalma derinliği — araç kaviteden ÇIKMIŞSA pozitif
        #   α   = yerel dalma açısı (kavite ekseni eğimi tail'de = g·L/V²)
        #
        # Eğer R_c > R_v (araç kavite içinde) ve dalma yoksa F_pz = 0.
        # Eğer R_c < R_v veya yerçekimi yeterince büyük dalma yaratırsa
        # planing kuvveti devreye girer.
        F_lift_tot = 0.0
        F_press_tot = 0.0
        M_y_tot = 0.0
        F_cav_z = 0.0    # kavitatör eğim kuvvetinin dikey bileşeni (lift)
        M_cav_z = 0.0    # kavitatör momenti (x=0'da etki)

        # NOT: Kavitatör F_cav_z hesabı, F_cav drag hesabı ile birlikte
        # aşağıda "Sürükleme bileşenleri" bloğunda yapılır (F_n vektörünün
        # cos/sin bileşenleri). Burada sadece sıfır olarak başlatıldı.

        if Lc_r > 1e-4 and Dc_r > 1e-4:
            # Transom konumu — kavite koordinatında
            x_tail_cav = L_veh + x_body_start
            # Transom'daki kavite yarıçapı (asimptotik formül)
            if x_tail_cav < Lc_r:
                Rn_t = Dn / 2.0
                Rmax_t = Dc_r / 2.0
                x_open_t = max(10.0 * Rn_t / max(np.sqrt(Cx_i), 0.5), Rn_t * 2.0)
                x_mid_t = Lc_r * 0.5
                S_t = 1.0 - np.exp(-x_tail_cav / x_open_t)
                if x_tail_cav < x_mid_t:
                    D_t = 1.0
                else:
                    D_t = max(0.0,
                              1.0 - ((x_tail_cav - x_mid_t) / max(Lc_r - x_mid_t, 1e-6)) ** 2)
                rc_tail = np.sqrt(max(Rn_t**2 + (Rmax_t**2 - Rn_t**2) * S_t * D_t, 0.0))
            else:
                rc_tail = 0.0   # transom kaviteden öteye geçti

            # Transomda kavite ekseni dikey kayması (yerçekimi + asimptotik açı):
            #   h_c(x) = g·x²/(2V²) + k_dev · α_eff · x_open · (1 − exp(−x/x_open))
            # Asimptotik form: kavite uzun mesafelerde α·x_open ile sınırlı kalır.
            x_open_tail = max(10.0 * R_n / max(np.sqrt(Cx_i), 0.5), R_n * 2.0)
            if vi > 0.5:
                h_grav_tail = G * x_tail_cav * x_tail_cav / (2.0 * vi * vi)
                h_angle_tail = (k_dev * alpha_eff * x_open_tail
                                * (1.0 - np.exp(-x_tail_cav / x_open_tail)))
                hg_tail = h_grav_tail + h_angle_tail
            else:
                hg_tail = 0.0
            R_v_tail = R_v_max
            # AKIŞ-BAĞLI KOORDİNAT — gövde sabit yatay, kavite α_eff açısıyla eğilir.
            # Transom (kuyruk) noktasındaki kavite-araç dalması:
            #   δ = R_v + |h_c(transom)| − R_c(transom)
            # |h_c| çünkü kavite hem yukarı (alt ıslak) hem aşağı (üst ıslak) kayabilir.
            delta_tail = R_v_tail + abs(hg_tail) - rc_tail

            # planing_enable=False ise F_planing hep 0 (kullanıcı zorlaması)
            if not planing_enable:
                delta_tail = -1.0   # F_planing bloğuna girmeyecek

            # F_planing = TAIL-SLAP IMPACTI (Dzielski-Kurdila)
            # Bu sadece şu özel durumda anlamlıdır:
            #   - Kavite mevcut (Lc > 0)
            #   - Kavite kuyruğa kadar uzanmamış (transom kısmen kavite duvarında)
            #   - Aracın transom'u kavite duvarına ÇARPMIŞ ama tam suya batık DEĞİL
            # Yani delta küçükten büyüğe geçişte planing ile form drag arasında
            # bir geçiş bölgesidir.
            #
            # KRİTİK: Eğer kavite hiç gelişmemiş (Lc<<L_veh) veya geniş ölçüde
            # tam ıslak ise, F_planing UYGULANMAZ — bu durumda gövde lifti
            # (F_body_lift) baskındır, planing özel olayı yoktur.
            # Pratik şart: delta < 2R_v VE rc_tail > 1e-4 (kavite kuyruğa eriş.)
            #
            # Tail kavite içinde (delta ≤ 0) → F_planing = 0 (kavite kuyruğu sarıyor)
            # Tail kavite duvarında (0 < delta < 2R_v) → F_planing aktif (Dzielski)
            # Tail tamamen dışarıda (delta ≥ 2R_v) → F_planing = 0, F_press klasik
            if delta_tail > 0 and R_v_tail > 1e-6 and rc_tail > 1e-4:
                if delta_tail >= 2.0 * R_v_tail:
                    # Araç tail'i tamamen dışarıda — KLASİK rejim
                    F_press_tot = q * Sb * 0.20
                    F_lift_tot = 0.0
                    M_y_tot = 0.0
                else:
                    h_imm = delta_tail
                    gap = max(rc_tail - R_v_tail, 0.0)

                    f1 = 1.0 - (gap / max(h_imm + gap, 1e-9)) ** 2
                    f2 = (1.0 + h_imm / R_v_tail) / (1.0 + 2.0 * h_imm / R_v_tail)

                    # Yerel planing açısı (akış-bağlı koordinat):
                    # Kavite ekseninin yerel eğimi (asimptotik formülün türevi):
                    #   dh_c/dx = g·x/V² + k_dev·α_eff·exp(−x/x_open)
                    # Açı katkısı x ile sönümlenir (mesafeyle birlikte uzaklaşır).
                    if vi > 0.5:
                        dh_grav = G * x_tail_cav / (vi * vi)
                        dh_angle = (k_dev * alpha_eff
                                    * np.exp(-x_tail_cav / x_open_tail))
                        alpha_p = dh_grav + dh_angle
                    else:
                        alpha_p = (k_dev * alpha_eff
                                   * np.exp(-x_tail_cav / x_open_tail))
                    F_lift_tot = RHO * vi * vi * R_v_tail * R_v_tail * f1 * f2 * alpha_p

                    F_press_tot = F_lift_tot * alpha_p
                    M_y_tot = (x_cg - L_veh) * F_lift_tot

        # Tam ıslak (kavite hiç oluşmadıysa) klasik basınç sürüklemesi
        if Lc_r < 1e-4 or Dc_r < 1e-4:
            F_press_tot = q * Sb * 0.15
            # Kavite henüz oluşmadı — araç tam ıslak
            wet_area = total_area
            wet_len = L_veh
            # Tam ıslak: tüm gövde hacmi batık → tam Arşimet kaldırması
            # Gövde hacmi (kesik koni burun + silindir):
            if L_taper > 1e-6:
                V_taper_full = (np.pi * L_taper / 3.0) * (R_n**2 + R_n*R_v_max + R_v_max**2)
                V_cyl_full = np.pi * R_v_max**2 * (L_veh - L_taper)
                V_body_total = V_taper_full + V_cyl_full
            else:
                V_body_total = np.pi * R_v_max**2 * L_veh
            V_submerged_tot = V_body_total
            F_buoy_tot = RHO * G * V_body_total
            # Tam ıslakta simetri var → moment yok (yerçekimi)
            M_buoy_tot = 0.0
            # ---- TAM IS̆LAK GÖVDE α-LİFT ve α-DRAG (Munk slender body) ----
            # MUNK + ALLEN-PERKINS BİRLEŞİK MODEL
            # F_L = ρV²·A_avg·sin(2α)/2     ← lineer Munk (silindir kesit alanı)
            #     + ρV²·S_plan·Cdc·η·sin²α·cosα  ← crossflow (viskoz, yüksek-α)
            #
            # Cdc literatür: silindir crossflow drag = 1.2, ama bu YÜKSEK α içindir.
            # Düşük α (<10°) için "viskoz crossflow verim faktörü" η ≈ 0.3-0.5
            # ile çarpılır (Hoerner). Etkin Cdc·η ≈ 0.4-0.6.
            #
            # Yeni varsayılan: Cdc_eff = 0.4 (kalibre edilmiş, küçük α'da makul)
            # Kullanıcı UI'dan Cdc_body_factor değiştirebilir (kalibrasyon).
            if abs(alpha_aoa) > 1e-6:
                Cdc_eff = float(params.get("Cdc_body", 0.4))   # kalibrasyon
                Cdc_eff = max(0.0, min(Cdc_eff, 2.0))
                sa = np.sin(alpha_aoa); ca = np.cos(alpha_aoa)
                s2a = np.sin(2.0 * alpha_aoa)
                sgn = np.sign(alpha_aoa)
                A_avg = V_body_total / L_veh
                S_plan_total = ((R_n + R_v_max) * L_taper
                                + 2.0 * R_v_max * (L_veh - L_taper))
                # CL_alpha_body kullanıcının "Munk gücünü" ayarlamasını sağlar
                # default 1.5 = orta seviye (CL_α=2 Munk tam, 1.0 zayıf gövde)
                # CL_alpha_body 1.0 = klasik Munk, 1.5-2.0 = takviyeli, <1 = pratik düşük
                munk_factor = max(0.0, min(CL_alpha_body, 6.28)) / 2.0
                F_body_lift_tot = (
                    munk_factor * RHO * vi * vi * A_avg * s2a       # Munk lineer (×CL_α/2)
                  + RHO * vi * vi * S_plan_total * Cdc_eff * sa * sa * ca * sgn  # crossflow
                )
                # Yalnızca ıslak orana göre ölçekle (kavite içindeki gövde lift üretmez)
                # wet_frac henüz cover[i] hesaplanmadığı için doğrudan wet_area/total_area
                if total_area > 1e-9:
                    wet_frac_now = max(0.0, min(wet_area / total_area, 1.0))
                else:
                    wet_frac_now = 1.0
                F_body_lift_tot *= wet_frac_now
                F_body_drag_tot = abs(F_body_lift_tot) * abs(np.tan(alpha_aoa))
                # Lift etki noktası: konik bölge merkezi (Munk'a göre lift sivri burunda)
                # Pratik ortalama: konik orta noktası ~L_taper/2 + silindir merkezi karışımı
                if L_taper > 1e-6:
                    x_lift_center = L_taper * 0.5  # Munk'a göre lift kesit değişen yerde
                else:
                    x_lift_center = L_veh * 0.5
                M_body_lift_tot = (x_cg - x_lift_center) * F_body_lift_tot
            # else: F_body_lift_tot = 0 zaten başlangıçtan

        # KAPSAMA = 1 − (ıslak alan / toplam yüzey alanı)
        # Bu metrik, hem uzunluk-bazlı (eskisi) hem de asimetri-bazlı (yeni)
        # bilgiyi birleştirir. Yerçekimi nedeniyle alt yüzey açıkta kalsa
        # bile bu küçük cov verir (gerçekçi).
        if total_area > 1e-9:
            cov = 1.0 - (wet_area / total_area)
        else:
            cov = 0.0
        cover[i] = max(0.0, min(cov, 1.0))

        # ---- Sürükleme bileşenleri ----
        # Kavitatör KUVVETLERİ — May 1975 (Birkhoff), Dzielski-Kurdila 2003
        #
        # Eğimli kavitatörde iki kuvvet bileşeni:
        #   F_drag (yatay) = q · Sn · Cx · cos(α_eff)
        #   F_lift (dikey) = q · Sn · C_L,cav · α_eff
        #
        # Disk kavitatör için (May 1975, Birkhoff slender body):
        #   C_L,α = π / (1 + σ)   (literatür, π yaklaşımı)
        # KAVİTATÖR LİFT FORMÜLÜ — Hibrit (disk + slender body)
        # ============================================================
        # DİSK kavitatör (β=180°): flat plate cavitator teorisi
        #   CL_α = π/(1+σ)  (Logvinovich-May 1975)
        #
        # KONİK kavitatör (β<180°): slender body teorisi baskın olur
        # çünkü konik aslında bir ogive/slender gövdedir.
        #   CL_α = 2·cos²(β/2)  (Munk 1924 slender body limit)
        #
        # Pürüzsüz geçiş için hibrit formül:
        #   CL_α = (π/(1+σ))·sin²(β/2) + 2·cos²(β/2)
        #
        # Sınır kontrolleri:
        #   β=180° (disk):  π/(1+σ)·1 + 2·0    = π/(1+σ) ≈ 3.0 ✓
        #   β=90° :         π/(1+σ)·0.5 + 2·0.5 ≈ 2.5
        #   β=60° :         π/(1+σ)·0.25 + 2·0.75 ≈ 2.25
        #   β=40° :         π/(1+σ)·0.117 + 2·0.88 ≈ 2.12
        #   β=0°  (ideal sivri): 0 + 2 = 2.0 ✓ (slender body)
        #
        # NEDEN: Sivri konik kavitatör 'kanat gibi' davranır, slender body
        # teorisi geçerli. Disk için (β=180°) ise düz-plak teorisi (Logvinovich).
        # Önceki formül sadece disk fiziğine dayanıyordu, konik için 5-8×
        # eksik lift veriyordu.
        sigma_now = max(sigma[i], 0.001)
        # Konik tepe açısı (rad) — disk için β=180° (sin²(90°)=1)
        beta_half = cone_apex_rad / 2.0 if cav_type == "cone" else (np.pi / 2.0)
        # Hibrit lift katsayısı: disk (π/(1+σ)) ↔ slender body (CL_α=K_slender) interpolasyonu
        # K_slender: slender body cross-flow lift terim katsayısı
        #   • 2.0 = Munk teorik üst sınır (tam-batık silindir)
        #   • 1.0 = Hoerner cross-flow (gerçek silindir, varsayılan)
        #   • 0.5 = aşırı düşük (viskoz kayıp baskın)
        #   • 0.0 = drag-bağlantılı (sadece disk-mantığı)
        # Kullanıcı CFD/deneye göre kalibre edebilir (UI'dan).
        K_slender = float(params.get("K_slender", 1.0))
        K_slender = max(0.0, min(K_slender, 2.0))
        CL_alpha_base = ((np.pi / (1.0 + sigma_now)) * np.sin(beta_half) ** 2
                        + K_slender * np.cos(beta_half) ** 2)
        # KONİK YAN YÜZEY KATKISI (Logvinovich-Vasin geometric extension):
        # Sivri konik kavitatörde akış uzun yan yüzey boyunca geçer ve
        # daha büyük momentum değişimi yaratır → yan yüzey alanına orantılı
        # lift kazanımı. Yumuşak geometric düzeltme:
        #   K_lateral = 1/√(sin(β/2))
        # → Disk (β=180°): K=1.00 (değişmez)
        # → β=90°: K=1.19,  β=60°: K=1.41,  β=45°: K=1.62,  β=30°: K=1.97
        # CFD verisi ile kalibrasyon: kullanıcı UI'dan cone_lift_gain ile
        # bu etkiyi 0-2 katı arasında ölçekleyebilir (0 = etkisiz, 1 = nominal)
        # VARSAYILAN: 0 (kapalı) — çünkü modeli abartmayalım, kullanıcı CFD ile
        # uyumsuzluk görürse açabilir.
        if cav_type == "cone":
            K_lateral_max = 1.0 / np.sqrt(max(np.sin(beta_half), 0.05))
            cone_lift_gain = float(params.get("cone_lift_gain", 0.0))
            cone_lift_gain = max(0.0, min(cone_lift_gain, 2.0))
            K_lateral = 1.0 + (K_lateral_max - 1.0) * cone_lift_gain
        else:
            K_lateral = 1.0
        CL_alpha_cav = CL_alpha_base * K_lateral

        # YAN YÜZEY ALANI KATKISI (konik kavitatör için, K_cone_lift ile kalibre)
        # Sivri konik kavitatörün uzun yan yüzeyi akışı kademeli yönlendirir;
        # bu da lift coefficient'ini disk'e göre arttırabilir.
        # Yan yüzey alanı: S_lateral = πR²/sin(β/2)  (Birkhoff-Vasin)
        # Etkin yüzey çarpanı: 1/sin(β/2) → sivri olunca büyür
        #
        # K_cone_lift kullanıcı kalibrasyonu (default 0.0 = etkisiz, mevcut formül):
        #   0.0 → klasik hibrit formül (Birkhoff-May)
        #   0.5 → orta etki (yarı yan yüzey)
        #   1.0 → tam yan yüzey alanı kullanılır (Vasin-Logvinovich)
        #   1.5-2.0 → CFD'ye göre sivri konik için ek artırım
        # Kullanıcı CFD ile karşılaştırarak değeri ayarlamalı.
        if cav_type == "cone":
            lateral_ratio = 1.0 / max(np.sin(beta_half), 0.1)
            lateral_factor = 1.0 + K_cone_lift * (lateral_ratio - 1.0)
        else:
            lateral_factor = 1.0
        CL_alpha_cav *= lateral_factor

        F_n_total  = q * Sn * Cx_i
        F_cav[i]   = F_n_total * np.cos(alpha_eff)         # yatay drag
        # KAVİTATÖR LIFT — May/Birkhoff slender body
        F_cav_z    = q * Sn * CL_alpha_cav * alpha_eff      # dikey lift (literatür)
        F_skin[i]  = F_skin_tot
        F_press[i] = F_press_tot

        # ---- GÖVDE α-LIFT/DRAG (MUNK SLENDER BODY TEORİSİ, 1924) ----
        # ÖNEMLİ: Süperkavitasyon araçları için planform-bazlı lift YANLIŞTIR.
        # Slender body teorisi (Munk 1924, Newman 1977) der ki:
        #
        #   F_L_body = 2 · q · A_max · sin(α)
        #
        # Bu formül silindirik kısmın lift üretmediğini, sadece KESİT ALANI
        # DEĞİŞEN bölgelerin (konik burun) lift ürettiğini gösterir.
        # A_max: aracın maksimum kesit alanı (silindir bölgesinde)
        #
        # Yani: gövde lifti uzunluğa BAĞIMSIZDIR, sadece çapına bağlıdır.
        # Bu, planform-bazlı (S=D·L) formülünden ~5-10x DAHA KÜÇÜK lift verir.
        #
        # Süperkavitasyon adaptasyonu:
        # - Kavite içindeki gövde lift üretmez (gaz çok seyrek, ρ_gaz/ρ_su ≈ 0.001)
        # - Sadece ıslak kısım etkilidir → wet_frac ile çarpılır
        # - CL_alpha_body parametresi, formülde 2'yi temsil eder (CL_α=2)
        # - Kullanıcı CFD/deneye göre kalibre edebilir (0-2π aralık)
        A_max_body = np.pi * (D_veh / 2.0) ** 2          # maksimum kesit alanı
        wet_frac = max(0.0, min(1.0 - cover[i], 1.0))    # ıslak oran
        # F_body_lift = CL_α · q · A_max · α · wet_frac
        # CL_alpha_body varsayılan 2.0 (Munk slender body teorisi)
        F_body_lift_tot = (CL_alpha_body * q * A_max_body
                           * alpha_aoa * wet_frac)
        # Induced drag: F_D = F_L · α (küçük açı yaklaşımı)
        F_body_drag_tot = abs(F_body_lift_tot * alpha_aoa)

        # Gövde lifti planform merkezinde etkir (yaklaşık L_veh/2)
        x_body_center = L_veh / 2.0
        M_body_lift_tot = (x_cg - x_body_center) * F_body_lift_tot

        # Toplam yatay sürüklemeye α-induced drag eklenir
        Fd[i] = F_cav[i] + F_skin[i] + F_press[i] + F_body_drag_tot

        # Kavitatör momenti — kavitatör araç burnundadır (x = 0).
        # x_cg referansından kol: -x_cg (kavitatör ön taraftadır)
        # Çapraz çarpım: M = (x_cav - x_cg) · F_z = -x_cg · F_cav_z
        # Pozitif kavitatör lift (+F_cav_z) → kuvvet yukarı, kavitatör önde
        # → BURUN YUKARI çevirici moment
        # Konvansiyon: burun yukarı = pozitif M_y
        # → M_cav_z = +x_cg · F_cav_z (kol pozitif olarak alınır)
        # Çünkü |kol| = x_cg, yön: kuvvet yukarı + araç burnu önde → +pitch
        M_cav_z = x_cg * F_cav_z

        # ---- Asimetri kaynaklı dikey kuvvet ve moment ----
        # F_z TOPLAM dikey kuvvet bileşenleri:
        #   1. Planing (Dzielski-Kurdila) — tail-slap durumunda, transom'a etkir
        #   2. Kavitatör lift (eğimli kavitatör) — burunda etki eder
        #   3. Gövde α-lift (slender body) — ıslak gövde merkezinde
        #   4. Kaldırma (Arşimet) — kesit-bazlı, hacim merkezinde uygulanır
        #   5. Yerçekimi (-m·g) — kütle merkezi (x_cg_mass)'ta uygulanır
        # Net: F_z_net = F_planing + F_cav_z + F_body_lift + F_fin + F_buoy − m·g
        F_grav = -m * G   # aşağı yönlü (negatif)
        F_z_net = (F_lift_tot + F_cav_z + F_body_lift_tot + F_fin_lift_z_total
                   + F_buoy_tot + F_grav)
        Fz_arr[i] = F_z_net
        F_planing_arr[i]   = F_lift_tot
        F_cavz_arr[i]      = F_cav_z
        F_body_lift_arr[i] = F_body_lift_tot
        F_body_drag_arr[i] = F_body_drag_tot
        F_buoy_arr[i]      = F_buoy_tot
        F_grav_arr[i]      = F_grav
        V_sub_arr[i]       = V_submerged_tot
        # Kanat kuvvet izleme
        F_fin_total_arr[i] = F_fin_lift_z_total
        F_fin_drag_arr[i]  = F_fin_drag_total
        M_fin_arr[i]       = M_fin_total
        for k_fin in range(4):
            F_fin_each_arr[k_fin][i] = F_fin_individual[k_fin]

        # MOMENT — beş bileşen (planing, kavitatör, gövde, kaldırma, ağırlık)
        # Sağ-el kuralı (M_y = (x_cg - x_force)·F_z) — tüm momentler tutarlı:
        #   x_force < x_cg: kuvvet ÖNDE, +F_z (yukarı) → +M_y (burun yukarı)
        #   x_force > x_cg: kuvvet ARKADA, +F_z (yukarı) → -M_y (burun aşağı)
        M_grav = (x_cg - x_cg_mass) * F_grav

        My_no_grav_arr[i] = (M_y_tot + M_cav_z + M_body_lift_tot
                             + M_buoy_tot + M_fin_total)               # ağırlık hariç
        My_arr[i]         = (M_y_tot + M_cav_z + M_body_lift_tot
                             + M_buoy_tot + M_fin_total + M_grav)      # ağırlık dahil

        # ---- Araç dinamiği ----
        # steady_v_mode: thrust drag'e eşitlenir → net kuvvet 0, V sabit kalır
        # Bu, "V=V_init'te steady-state analiz" için (kullanıcının kalibrasyon senaryosu)
        if steady_v_mode:
            acc = 0.0
        else:
            # Kanat drag'ı toplam direnç kuvvetine eklenir (geriye doğru)
            acc = (T - Fd[i] - F_fin_drag_total) / m
        a[i] = acc
        if i < N - 1:
            V[i+1] = max(vi + acc * dt, 0.1)
            x[i+1] = x[i] + V[i+1] * dt

    return {"t": t, "V": V, "x": x, "sigma": sigma,
            "Lc": Lc, "Dc": Dc, "Cx": Cx, "Fd": Fd,
            "F_cav": F_cav, "F_skin": F_skin, "F_press": F_press,
            "cover": cover, "a": a,
            "pc": pc_arr, "Cq": Cq_arr, "Fr": Fr_arr,
            "beta": beta_arr, "Q_out": Qout_arr, "hg": hg_arr,
            "Fz": Fz_arr, "My": My_arr,
            "F_planing": F_planing_arr, "F_buoy": F_buoy_arr,
            "F_cavz": F_cavz_arr,
            "F_body_lift": F_body_lift_arr,
            "F_body_drag": F_body_drag_arr,
            "F_grav": F_grav_arr, "V_sub": V_sub_arr,
            "My_no_grav": My_no_grav_arr,
            "alpha_aoa": alpha_aoa, "delta_cav": delta_cav,
            "alpha_eff": alpha_eff,
            # Schedule izleme — zamana göre değişen kontrol girdileri
            "alpha_aoa_t": alpha_aoa_arr,
            "delta_cav_t": delta_cav_arr,
            "thrust_t":    thrust_arr,
            "gas_flow_t":  gas_flow_arr,
            "fin_delta_1_t": fin_delta_1_arr,
            "fin_delta_2_t": fin_delta_2_arr,
            "fin_delta_3_t": fin_delta_3_arr,
            "fin_delta_4_t": fin_delta_4_arr,
            # Kanat (fin) verileri
            "F_fin_total": F_fin_total_arr,
            "F_fin_drag":  F_fin_drag_arr,
            "M_fin":       M_fin_arr,
            "F_fin_1":     F_fin_each_arr[0],
            "F_fin_2":     F_fin_each_arr[1],
            "F_fin_3":     F_fin_each_arr[2],
            "F_fin_4":     F_fin_each_arr[3]}


def regime_of(sigma):
    if sigma > 1.0:   return ("Kavitasyonsuz",             "#7a8a99")
    if sigma > 0.3:   return ("Kısmi Kavitasyon",          "#d4a548")
    if sigma > 0.1:   return ("Geçiş Rejimi",              "#e07b3a")
    if sigma > 0.05:  return ("Süperkavitasyon",           "#4ba3c7")
    return            ("Gelişmiş Süperkavitasyon",          "#7ec4e0")


def cavity_profile(Lc, Dc, Dn, sigma, n_points=80,
                    D_veh=None, L_veh=None, L_taper=None, x_body_start=None,
                    body_volume_correction=True):
    """Logvinovich asimptotik kavite zarf profili — GÖVDE KELEPÇELİ.

    Bağımsızlık ilkesi: kavite kesiti, kavitatörden geçen su ile ÖLÇEKlenir,
    Lc'den bağımsız bir karakteristik uzunlukta (~10·Rn) açılır. Daha sonra
    Garabedian kapanma yaklaşımıyla Lc/2 civarında zirveye, Lc'de sıfıra düşer.

    R²(x) = Rn² + (Rmax² − Rn²) · S(x) · D(x)
        S(x) = 1 − exp(−x/x_open)         açılma (σ-bağımsız ölçek)
        D(x) = 1                          (x < x_mid)
               1 − ((x−x_mid)/(Lc−x_mid))²   (x > x_mid)

    GÖVDE KELEPÇESİ (body clamping):
        Eğer gövde geometrisi verilirse (D_veh, L_veh, L_taper), o zaman:
            R(x) = max(R_logvinovich(x), R_body(x))
        Yani kavite zarfı gövde yüzeyinin DAHA İÇİNE geçemez. Eğer doğal
        Logvinovich kavitesi gövdeyi saramayacak kadar küçükse, kavite zarfı
        gövde yüzeyiyle çakışır ("attached cavity" / "body-supported cavity").
        Bu, fiziksel olarak doğru süperkavitasyon başlangıç davranışıdır:
        kavite oluşurken önce gövdeye yapışık başlar, σ azaldıkça gövdeden
        ayrılarak free-standing süperkavite haline gelir.
    """
    if Lc <= 1e-4 or Dc <= 1e-4:
        return np.array([]), np.array([])
    Rn   = Dn / 2.0
    Rmax = Dc / 2.0
    Cx   = 0.82 * (1.0 + sigma)
    # Karakteristik açılma uzunluğu — Logvinovich asimptotik
    x_open = max(10.0 * Rn / max(np.sqrt(Cx), 0.5), Rn * 2.0)
    x_mid  = Lc * 0.5     # Garabedian: max çap Lc/2'de

    x = np.linspace(0.0, Lc, n_points)
    # Açılma faktörü
    S = 1.0 - np.exp(-x / x_open)
    # Kapanma faktörü (Lc/2'den sonra parabolik düşüş)
    D = np.where(x < x_mid, 1.0,
                 np.maximum(0.0, 1.0 - ((x - x_mid) / max(Lc - x_mid, 1e-6)) ** 2))
    R2 = Rn ** 2 + (Rmax ** 2 - Rn ** 2) * S * D
    R  = np.sqrt(np.maximum(R2, 0.0))
    R[0]  = Rn
    R[-1] = max(Rn * 0.12, 1e-4)

    # GÖVDE-KAVİTE ETKİLEŞİMİ (cavity-body interaction, NOT blockage correction)
    #
    # ÖNEMLİ AYRIM:
    #   - "Blockage correction" (Wu-Whitney-Brennen 1971) → su tüneli DUVARI etkisi
    #     (kavite dışındaki sınırlı akış alanı kaviteyi sıkıştırır)
    #   - "Cavity-body interaction" (Logvinovich 1969) → kavite İÇİNDEKİ gövde etkisi
    #     (gövdenin yer kapladığı hacim kaviteyi dışarı iter)
    #
    # Bizim modellediğimiz İKİNCİ olgudur. Logvinovich bağımsızlık prensibi'ne göre,
    # her kavite kesiti bağımsız genişler — gövde bir noktada bulunuyorsa, o kesit
    # kavitatörden gelen serbest kavite YANINDA gövdenin kesit alanını da içerir.
    #
    # Pratik formülasyon (Logvinovich 1969, Serebryakov 2009):
    #   Kavite gaz kesit alanı + gövde kesit alanı = toplam zarf kesit alanı
    #   π·R_c² + π·R_body² = π·R_zarf²
    #   →  R_zarf² = R_c² + R_body²
    #
    # Bu, gaz hacminin korunmasını otomatik sağlar ve aynı zamanda
    # R_zarf >= R_body koşulunu trivial olarak gerçekleştirir.
    if (D_veh is not None and L_veh is not None and
        L_taper is not None and x_body_start is not None):
        Rv_max = D_veh / 2.0
        # YENİ MANTIK — "Küt silindirik kuyruk" görüntüsü için:
        #
        # ARAÇ İÇİNDE (0 ≤ xs_body ≤ L_veh): TAM clamping, R_zarf = max(R_logv, R_v_x)
        #   → kavite gövdeye yapışık (silindir gövde paralel)
        #   → araç boyunca düzgün silindir çizgi
        #
        # TRANSOM'DAN SONRA (xs_body > L_veh, wake bölgesi):
        #   D_veh kadar mesafede kavite kademeli olarak natural Logvinovich'e geçer
        #   → deep transom wake collapse modeli
        #
        # NOT: Yumuşatma SADECE transom'dan SONRA — araç içinde keskin köşe yok
        # çünkü clamping doğal olarak sürekli (max fonksiyonu C⁰).
        smooth_start = L_veh                  # transom'da başlar
        smooth_post  = D_veh * 1.0            # wake yumuşatma uzunluğu
        smooth_end   = L_veh + smooth_post
        for k in range(len(x)):
            xs_cav = x[k]
            xs_body = xs_cav - x_body_start

            # KAVITATÖR ÖNÜ veya WAKE SONRASI: doğal Logvinovich
            if xs_body < 0 or xs_body > smooth_end:
                continue

            # Bu kesitte gövde yarıçapı
            if 0.0 <= xs_body <= L_veh:
                # ARAÇ İÇİ
                if L_taper > 1e-6 and xs_body < L_taper:
                    R_body = Rn + (Rv_max - Rn) * (xs_body / L_taper)
                else:
                    R_body = Rv_max
            else:
                # transom dışında, wake bölgesi — referans olarak transom çapı
                R_body = Rv_max

            R_logv = R[k]

            # CLAMPING — kavite gövdeye yapışık veya serbest
            if body_volume_correction:
                # Pythagorean superposition (yumuşak büyütme)
                R_clamped = np.sqrt(R_logv ** 2 + R_body ** 2)
            else:
                # Basit clamping (gövde yüzeyinde tam yapışık)
                R_clamped = max(R_logv, R_body)

            if xs_body <= L_veh:
                # ARAÇ İÇİNDE: tam clamping (silindir görünüm)
                R[k] = R_clamped
            else:
                # WAKE BÖLGESİ — transom'dan sonra smooth collapse
                t_norm = (xs_body - L_veh) / smooth_post
                t_norm = max(0.0, min(t_norm, 1.0))
                # Quintic smoothstep — C² sürekli
                smooth = t_norm**3 * (t_norm * (t_norm * 6.0 - 15.0) + 10.0)
                # transom değerinde: R_clamped (transom çapı + olası R_logv)
                # Blend: transom değeri → doğal Logvinovich
                R[k] = (1.0 - smooth) * R_clamped + smooth * R_logv

    return x, R


# ============================================================================
# ANA UYGULAMA
# ============================================================================
class SupercavitationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Süperkavitasyon Geçiş Simülasyonu — CANLI MOD [v8 · slider ↔ entry senkron]")
        self.root.configure(bg=COLORS["bg"])
        self.root.geometry("1500x1050")
        self.root.minsize(1200, 700)   # çok küçültülünce UI kırılmasın

        self.params = {
            # ===== CASE 1 SİSTEM PARAMETRELERİ =====
            "depth":    tk.DoubleVar(value=10.0),
            "diam_cav": tk.DoubleVar(value=0.20),    # kavitatör çapı Dn [m]
            "mass":     tk.DoubleVar(value=350.0),
            "thrust":   tk.DoubleVar(value=6000.0),   # V=40 kararlı tutmak için
            "veh_len":  tk.DoubleVar(value=4.0),
            "veh_diam": tk.DoubleVar(value=0.30),
            "L_taper":  tk.DoubleVar(value=0.60),    # konik burun uzunluğu [m] (≈0.15·L_veh)
            "gas_flow": tk.DoubleVar(value=15000.0), # havalandırma debisi [L/min]
            "t_max":    tk.DoubleVar(value=0.5),      # steady-state ulaşmak için
            "dt":       tk.DoubleVar(value=0.001),    # ölçekle hızlı çözüm
            "x_cg":     tk.DoubleVar(value=2.4),     # moment referans noktası [m araç burnundan]
            "x_mg":     tk.DoubleVar(value=2.4),     # kütle merkezi
            "x_cg_mass": tk.DoubleVar(value=2.4),    # kütle merkezi (alias)
            "V_init":   tk.DoubleVar(value=40.0),    # başlangıç hızı [m/s]
            # ===== CASE 1 KONTROL =====
            "alpha_aoa": tk.DoubleVar(value=-1.0),   # CASE 1: α = -1°
            "delta_cav": tk.DoubleVar(value=2.0),    # δ_c = 2°
            # ===== KALİBRASYON PARAMETRELERİ =====
            "CL_alpha_body": tk.DoubleVar(value=2.5),   # gövde lift slope
            "Cdc_body": tk.DoubleVar(value=0.4),
            "k_dev": tk.DoubleVar(value=0.4),
            "A_v": tk.DoubleVar(value=0.020),           # havalandırma kalibrasyon
            "K_slender": tk.DoubleVar(value=0.5),       # slender body
            "cone_lift_gain": tk.DoubleVar(value=0.50), # konik lift kazanım
            "K_cone_lift": tk.DoubleVar(value=0.0),     # konik yan-lift kapalı
            "cone_apex_deg": tk.DoubleVar(value=40.0),  # konik tepe açısı
            "cone_drag_visc": tk.DoubleVar(value=0.18), # konik drag K_v (V-sabit steady için)
            "k_g_cavity": tk.DoubleVar(value=0.78),     # Reichardt k_g
            "K_Dc_factor": tk.DoubleVar(value=1.0),
            "K_Lc_factor": tk.DoubleVar(value=1.0),
            # ===== KANATLAR (CASE 1 trim açıları) =====
            "fin_chord":    tk.DoubleVar(value=0.10),    # kord [m]
            "fin_span":     tk.DoubleVar(value=0.25),    # span [m]
            "fin_x_pos":    tk.DoubleVar(value=3.80),    # x konumu [m]
            "fin_delta_1":  tk.DoubleVar(value=-0.80),   # üst kanat trim (asimetri proxy)
            "fin_delta_2":  tk.DoubleVar(value=0.0),     # sağ
            "fin_delta_3":  tk.DoubleVar(value=-0.80),   # alt kanat trim
            "fin_delta_4":  tk.DoubleVar(value=0.0),     # sol
        }
        # Geometri modeli — dropdown ile seçilir
        self.geom_model_var = tk.StringVar(value="savchenko")
        # Havalandırma modu — Q sabit ya da Cq sabit
        self.vent_mode_var = tk.StringVar(value="Q")
        # Kavitatör tipi: "disk" (klasik) veya "cone" (konik)
        # Disk: Cx0=0.82 (sabit), tüm fizik formülleri yaygın bilinir
        # Cone: Cx0=0.82·sin²(β/2), β = tepe açısı (vertex angle)
        #   Konik kavitatörler düşük drag, küçük kavite üretir.
        self.cav_type_var = tk.StringVar(value="cone")
        # Gövde-kavite hacim etkileşimi açık/kapalı
        # True: superposition formülü R²=R_logv²+R_body² (önerilen, fiziksel)
        # False: sadece clamping max(R_logv, R_body) (daha basit, karşılaştırma için)
        self.body_volume_var = tk.BooleanVar(value=True)
        # Kanatlar aktif/pasif toggle (varsayılan kapalı, kullanıcı açar)
        self.fins_enabled_var = tk.BooleanVar(value=True)
        # YENİ: analiz modu toggle'ları
        self.steady_v_var = tk.BooleanVar(value=False)       # V-sabit default AÇIK
        self.planing_enable_var = tk.BooleanVar(value=True) # transom slap default KAPALI

        self.data = None
        self.idx = 0
        # Schedule sözlüğü — None ise sabit parametre kullanılır
        self.schedules = {
            "alpha_aoa_schedule":  None,    # [(t, deg), ...]
            "delta_cav_schedule":  None,
            "thrust_schedule":     None,
            "gas_flow_schedule":   None,
            "fin_delta_1_schedule": None,   # üst kanat δ [(t, deg), ...]
            "fin_delta_2_schedule": None,   # sağ kanat δ
            "fin_delta_3_schedule": None,   # alt kanat δ
            "fin_delta_4_schedule": None,   # sol kanat δ
        }
        self.playing = False   # başlangıçta DURAKLATILMIŞ
        self._internal_slider_update = False    # re-entrance koruması

        # ===== CANLI MOD STATE =====
        self._live_running = False
        self._live_V = None       # anlık hız (state)
        self._live_t = 0.0        # birikimli zaman
        self._live_data = None    # birikimli sonuç arrays
        self._live_chunk_ms = 50  # chunk boyutu (ms wall-clock)
        self._live_chunk_dur = 0.02  # chunk simülasyon süresi (s)
        self._live_dt = 0.001     # her chunk içindeki dt

        self._build_ui()
        self._run_sim()
        self._draw_plots_static()
        self._update_frame()
        self._animate()

    # ================================================================= UI
    def _build_ui(self):
        header = tk.Frame(self.root, bg=COLORS["bg"])
        header.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(header, text="SÜPERKAVİTASYON GEÇİŞ DİNAMİĞİ",
                 fg=COLORS["text"], bg=COLORS["bg"],
                 font=("Georgia", 18)).pack(side="left")
        tk.Label(header,
                 text="  Logvinovich · Garabedian · Reichardt formülasyonu",
                 fg=COLORS["accent"], bg=COLORS["bg"],
                 font=("Consolas", 9, "italic")).pack(side="left", padx=(12, 0))
        tk.Label(header,
                 text="ρ=1025 kg/m³  ·  p_v=2.34 kPa  ·  dt=10 ms",
                 fg=COLORS["mute"], bg=COLORS["bg"],
                 font=("Consolas", 8)).pack(side="right")

        tk.Frame(self.root, bg=COLORS["border"], height=1).pack(fill="x", padx=16)

        main = tk.Frame(self.root, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        left = tk.Frame(main, bg=COLORS["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # --- Sağ panel: KAYDIRILABİLİR ---
        # Pencere yüksekliği yeterli olmayabilir (7 input + metrikler +
        # kontroller). Bu yüzden sağ paneli Canvas + Scrollbar ile sarıp
        # dikey kaydırma ekliyoruz. Fare tekerleği de bağlanır.
        right_container = tk.Frame(main, bg=COLORS["bg"], width=380)
        right_container.pack(side="right", fill="y")
        right_container.pack_propagate(False)

        # Scrollbar
        vscroll = tk.Scrollbar(right_container, orient="vertical",
                               bg=COLORS["panel"],
                               troughcolor=COLORS["bg"],
                               activebackground=COLORS["accent"],
                               highlightthickness=0, bd=0, width=12)
        vscroll.pack(side="right", fill="y")

        # Canvas — asıl kaydırma kabı
        right_canvas = tk.Canvas(right_container, bg=COLORS["bg"],
                                  highlightthickness=0, bd=0,
                                  yscrollcommand=vscroll.set)
        right_canvas.pack(side="left", fill="both", expand=True)
        vscroll.config(command=right_canvas.yview)

        # İçerik frame'i — kaydırılan kısım burası
        right = tk.Frame(right_canvas, bg=COLORS["bg"])
        right_window = right_canvas.create_window((0, 0), window=right,
                                                   anchor="nw")

        # İçerik genişliği Canvas genişliğine uysun
        def _on_canvas_configure(event):
            right_canvas.itemconfig(right_window, width=event.width)
        right_canvas.bind("<Configure>", _on_canvas_configure)

        # Scrollregion, içerik değiştikçe güncellensin
        def _on_frame_configure(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        right.bind("<Configure>", _on_frame_configure)

        # Fare tekerleği — Windows/macOS/Linux farklı event'ler gönderir
        def _on_mousewheel(event):
            # Windows & macOS: event.delta 120'nin katları
            # Linux: Button-4 (yukarı) / Button-5 (aşağı)
            if event.num == 4:
                right_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                right_canvas.yview_scroll(1, "units")
            else:
                right_canvas.yview_scroll(int(-event.delta / 120), "units")

        # Tüm widget hiyerarşisine bind et (imleç sağ panel üstündeyken)
        def _bind_wheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)    # Win/Mac
            widget.bind("<Button-4>", _on_mousewheel)       # Linux yukarı
            widget.bind("<Button-5>", _on_mousewheel)       # Linux aşağı
        _bind_wheel(right_canvas)
        _bind_wheel(right)
        # Tüm alt widget'lara da uygulansın (build_controls çağrıldıktan sonra)
        self._right_canvas = right_canvas
        self._bind_wheel_fn = _bind_wheel

        self._build_plots(left)
        self._build_controls(right)

        # _build_controls çağrıldıktan sonra tüm child widget'lara tekerlek bind'i
        def _bind_recursive(widget):
            _bind_wheel(widget)
            for child in widget.winfo_children():
                _bind_recursive(child)
        _bind_recursive(right)

    def _build_plots(self, parent):
        # 5 satır × 2 sütun:
        #   Satır 1 (span 2): araç kesit
        #   Satır 2: V/σ           | Lc/Dc
        #   Satır 3: Sürükleme     | Kaplama+ivme
        #   Satır 4 (span 2): Havalandırma (pc, Cq, β)
        #   Satır 5 (span 2): Asimetri kuvvetleri (F_z, M_y)
        self.fig = Figure(figsize=(11, 12), facecolor=COLORS["bg"])
        gs = self.fig.add_gridspec(5, 2, height_ratios=[1.4, 1, 1, 1.1, 1.0],
                                    hspace=0.65, wspace=0.28,
                                    left=0.07, right=0.94,
                                    top=0.97, bottom=0.04)

        self.ax_vehicle = self.fig.add_subplot(gs[0, :])
        self.ax_vehicle.set_facecolor(COLORS["water2"])

        self.ax_vsigma = self.fig.add_subplot(gs[1, 0])
        self.ax_cavity = self.fig.add_subplot(gs[1, 1])
        self.ax_drag   = self.fig.add_subplot(gs[2, 0])
        self.ax_cover  = self.fig.add_subplot(gs[2, 1])
        self.ax_vent   = self.fig.add_subplot(gs[3, :])
        self.ax_force  = self.fig.add_subplot(gs[4, :])      # F_z + M_y

        # Twin axes bir kere oluştur
        self.ax_vsigma2 = self.ax_vsigma.twinx()
        self.ax_cover2  = self.ax_cover.twinx()
        self.ax_vent2   = self.ax_vent.twinx()       # Cq için 2. y-ekseni
        self.ax_force2  = self.ax_force.twinx()      # M_y için 2. y-ekseni

        for ax in [self.ax_vsigma, self.ax_cavity, self.ax_drag, self.ax_cover,
                   self.ax_vent, self.ax_force,
                   self.ax_vsigma2, self.ax_cover2, self.ax_vent2, self.ax_force2]:
            ax.set_facecolor(COLORS["panel"])

        # subplots_adjust artık gridspec'in kendi parametreleriyle halledildi

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_controls(self, parent):
        # --- Oynatma kontrolü ---
        ctrl = tk.Frame(parent, bg=COLORS["panel"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        ctrl.pack(fill="x", pady=(0, 10))

        tk.Label(ctrl, text="  OYNATMA KONTROLÜ",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 4))

        btn_row = tk.Frame(ctrl, bg=COLORS["panel"])
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        self.btn_play = tk.Button(btn_row, text="▶ BAŞLAT",
            command=self._toggle_play,
            bg=COLORS["accent"], fg=COLORS["bg"],
            font=("Consolas", 9, "bold"), relief="flat",
            activebackground=COLORS["accent2"])
        self.btn_play.pack(side="left", padx=(0, 4), fill="x", expand=True)

        tk.Button(btn_row, text="↺ SIFIRLA",
            command=self._reset,
            bg=COLORS["border"], fg=COLORS["text"],
            font=("Consolas", 9), relief="flat",
            activebackground=COLORS["accent"]).pack(side="left", fill="x", expand=True)

        # YENİ: CANLI MOD butonu - simülasyon adım adım ilerler,
        # UI'daki parametre değişiklikleri anlık etkin olur
        live_row = tk.Frame(ctrl, bg=COLORS["panel"])
        live_row.pack(fill="x", padx=10, pady=(0, 6))
        self.btn_live = tk.Button(live_row, text="▶ CANLI BAŞLAT",
            command=self._live_toggle,
            bg="#40a068", fg="white",
            font=("Consolas", 9, "bold"), relief="flat",
            activebackground="#50c078")
        self.btn_live.pack(fill="x", expand=True)
        tk.Label(ctrl, text="  Canlı modda parametre değişimleri anlık etkin",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", padx=10)

        # ===== CANLI KONTROL SLIDER'LARI =====
        # Bu slider'lar hem batch hem canlı modda simülasyonu anlık etkiler.
        # self.params dict'inden aynı DoubleVar'ları kullanarak entry ile eşleşir.
        live_ctrl = tk.LabelFrame(ctrl, text=" CANLI KONTROL ",
                                   fg=COLORS["accent"], bg=COLORS["panel"],
                                   font=("Consolas", 8, "bold"),
                                   bd=1, relief="solid")
        live_ctrl.pack(fill="x", padx=10, pady=(4, 8))

        def make_live_slider(parent, label_text, key, from_val, to_val, resolution, unit):
            """Slider + entry çift yönlü senkron oluştur.

            - Slider hareket → var.set → entry StringVar güncelle → label güncelle
            - Entry commit → var.set (mevcut _commit_entry) → trace ile slider ve label güncelle
            """
            var = self.params[key]
            row = tk.Frame(parent, bg=COLORS["panel"])
            row.pack(fill="x", padx=5, pady=2)
            tk.Label(row, text=label_text, fg=COLORS["text"],
                     bg=COLORS["panel"], font=("Consolas", 8),
                     width=8, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text=f"{var.get():+.2f}{unit}",
                               fg=COLORS["accent"], bg=COLORS["panel"],
                               font=("Consolas", 8, "bold"), width=8)
            val_lbl.pack(side="right")

            # Slider — variable=var ile DoubleVar'a doğrudan bağla
            # command=_on_slide → entry StringVar ve label güncelle + sim yeniden koş
            def _on_slide(v):
                try:
                    new_val = float(v)
                    # Entry StringVar güncelle (varsa)
                    if key in self.param_entries:
                        _, sv = self.param_entries[key]
                        fmt = self.param_fmts.get(key, "{:+.2f}")
                        sv.set(fmt.format(new_val))
                    # Label güncelle
                    val_lbl.config(text=f"{new_val:+.2f}{unit}")
                    # Canlı modda değil ama batch modda çalışıyorsa sim yenile
                    if not self._live_running:
                        self._run_sim()
                        self._draw_plots_static()
                        self._update_frame()
                except Exception:
                    pass

            sc = tk.Scale(row, from_=from_val, to=to_val, resolution=resolution,
                          orient="horizontal", showvalue=False,
                          bg=COLORS["panel"], troughcolor=COLORS["border"],
                          highlightthickness=0, activebackground=COLORS["accent"],
                          sliderrelief="flat", length=100,
                          variable=var,       # ← ÇİFT YÖNLÜ BAĞLAM
                          command=_on_slide)
            sc.pack(side="left", fill="x", expand=True, padx=4)
            # Explicit initial değer — variable bağlantısı bazen yuvarlar/reset eder
            try:
                initial = float(var.get())
                sc.set(initial)   # slider pozisyonunu kesin ayarla
                var.set(initial)  # DoubleVar'ı geri set et (slider clamp etmişse)
            except Exception:
                pass

            # DoubleVar'a trace: entry commit veya başka yerden güncellenirse
            # slider ve label otomatik uyum sağlar
            def _sync_from_var(*_args):
                try:
                    new_val = float(var.get())
                    val_lbl.config(text=f"{new_val:+.2f}{unit}")
                except Exception:
                    pass
            var.trace_add("write", _sync_from_var)

            # Başlangıç değeriyle senkronize et
            _sync_from_var()
            return sc

        make_live_slider(live_ctrl, "α (AoA)", "alpha_aoa",
                         -15.0, 15.0, 0.1, "°")
        make_live_slider(live_ctrl, "δ_c cav", "delta_cav",
                         -30.0, 30.0, 0.1, "°")
        make_live_slider(live_ctrl, "δ_üst", "fin_delta_1",
                         -20.0, 20.0, 0.1, "°")
        make_live_slider(live_ctrl, "δ_alt", "fin_delta_3",
                         -20.0, 20.0, 0.1, "°")
        make_live_slider(live_ctrl, "İtki", "thrust",
                         0.0, 15000.0, 100.0, "N")

        self.time_var = tk.DoubleVar(value=0.0)
        self.time_slider = tk.Scale(ctrl, from_=0, to=499,
            orient="horizontal", variable=self.time_var,
            command=self._on_time_scrub, showvalue=False,
            bg=COLORS["panel"], fg=COLORS["text"],
            troughcolor=COLORS["border"], highlightthickness=0,
            activebackground=COLORS["accent"], sliderrelief="flat")
        self.time_slider.pack(fill="x", padx=10, pady=(0, 8))

        # --- Rejim göstergesi ---
        self.regime_frame = tk.Frame(parent, bg=COLORS["panel"],
                                      highlightbackground=COLORS["border"],
                                      highlightthickness=1)
        self.regime_frame.pack(fill="x", pady=(0, 10))
        tk.Label(self.regime_frame, text="  ANLIK AKIŞ REJİMİ",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 8)).pack(anchor="w", pady=(8, 2))
        self.regime_label = tk.Label(self.regime_frame, text="—",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Georgia", 15))
        self.regime_label.pack(anchor="w", padx=10, pady=(0, 4))

        # Geometrik durum: Lc/L_veh ve ıslak alan oranı
        self.geom_label = tk.Label(self.regime_frame, text="—",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 8))
        self.geom_label.pack(anchor="w", padx=10, pady=(0, 8))

        # --- Metrikler ---
        metrics_frame = tk.Frame(parent, bg=COLORS["panel"],
                                  highlightbackground=COLORS["border"],
                                  highlightthickness=1)
        metrics_frame.pack(fill="x", pady=(0, 10))
        tk.Label(metrics_frame, text="  ANLIK DEĞERLER",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 4))

        self.metric_labels = {}
        metrics = [
            ("t",     "Zaman",          "s",     COLORS["text"]),
            ("V",     "Hız",            "m/s",   COLORS["accent"]),
            ("sigma", "Kav. Num. σ",    "",      COLORS["accent2"]),
            ("Lc",    "Kavite Boyu",    "m",     COLORS["cavity"]),
            ("Dc",    "Kavite Çapı",    "m",     COLORS["cavity"]),
            ("Fd",    "Sürük. Kuv.",    "N",     COLORS["drag"]),
            ("Cx",    "Cx (kavitatör)", "",      COLORS["press"]),
            ("a",     "İvme",           "m/s²",  COLORS["skin"]),
            ("x",     "Yol",            "m",     COLORS["text"]),
            ("pc",    "Kavite içi pc",  "kPa",   "#b8e0f2"),
            ("Cq",    "Cq (havaland.)", "",      "#95c17a"),
            ("Fr",    "Fr (Froude)",    "",      "#d4a548"),
            ("beta",  "β Paryshev",     "",      "#e07b3a"),
            ("hg",    "h_g araç sonu",  "mm",    "#ffcc66"),
            ("cavity_clear", "Kav.Dışı Payı",   "mm",    "#ff6b4a"),   # YENİ: pozitif → dışarıda
            ("Fz",    "F_z (net)",       "N",     "#c19ae0"),
            ("F_planing", "F_transom_slap", "N", "#c19ae0"),
            ("F_buoy", "F_kaldırma",     "N",     "#95c17a"),
            ("F_cavz", "F_kav_lift",  "N",     "#ffcc66"),
            ("F_body_lift", "F_gövde_L",  "N",     "#a8d4ff"),
            ("F_body_drag", "F_gövde_D",  "N",     "#a8d4ff"),
            ("F_grav", "F_ağırlık",      "N",     "#dd8888"),
            ("V_sub", "V_batık",         "L",     "#95c17a"),
            ("My",    "M_y (top.)",      "N·m",   "#c19ae0"),
            ("My_no_grav","M_y (w-hariç)","N·m",  "#ffcc66"),
            # ===== KANAT METRİKLERİ =====
            ("F_fin_total", "F_fin_lift",  "N",   "#d4a548"),
            ("F_fin_drag",  "F_fin_drag",  "N",   "#d4a548"),
            ("M_fin",       "M_fin",       "N·m", "#d4a548"),
        ]
        mg = tk.Frame(metrics_frame, bg=COLORS["panel"])
        mg.pack(fill="x", padx=10, pady=(0, 8))
        for key, label, unit, color in metrics:
            row = tk.Frame(mg, bg=COLORS["panel"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, fg=COLORS["mute"], bg=COLORS["panel"],
                     font=("Consolas", 8), width=15, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text="—", fg=color, bg=COLORS["panel"],
                               font=("Consolas", 10, "bold"), anchor="e")
            val_lbl.pack(side="right")
            tk.Label(row, text=unit, fg=COLORS["mute"], bg=COLORS["panel"],
                     font=("Consolas", 7), width=5, anchor="w").pack(side="right", padx=(2, 0))
            self.metric_labels[key] = val_lbl

        # ========== GEOMETRİ MODELİ SEÇİCİSİ ==========
        # Lc/Dc hesabında kullanılan literatür modelini seç
        gm_frame = tk.Frame(parent, bg=COLORS["panel"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        gm_frame.pack(fill="x", pady=(0, 10))
        tk.Label(gm_frame, text="  GEOMETRİ MODELİ",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(gm_frame,
                 text="  Lc, Dc hesabı için literatür formülü",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))

        # Model seçenekleri ve tanımları
        self.geom_models = [
            ("savchenko",    "Savchenko 2001 (deneysel kalibre)"),
            ("garabedian",   "Garabedian 1956 (klasik asimptotik)"),
            ("semenenko",    "Semenenko (kalın doygunluk)"),
            ("may",          "May 1975 (Birkhoff-Plesset)"),
            ("vasin_serebr", "Vasin-Serebryakov (slender body)"),
            ("logvinovich",  "Logvinovich (kalın çap)"),
        ]

        # Radio button şeklinde göster (dropdown yerine — daha akademik)
        for key, label in self.geom_models:
            rb = tk.Radiobutton(gm_frame, text=label, value=key,
                                variable=self.geom_model_var,
                                fg=COLORS["text"], bg=COLORS["panel"],
                                selectcolor=COLORS["bg"],
                                activebackground=COLORS["panel"],
                                activeforeground=COLORS["accent"],
                                font=("Consolas", 8),
                                anchor="w", relief="flat",
                                command=self._on_geom_model_change)
            rb.pack(fill="x", padx=12, pady=1)

        # Alt boşluk
        tk.Frame(gm_frame, bg=COLORS["panel"], height=6).pack()

        # ========== HAVALANDIRMA MODU SEÇİCİSİ ==========
        # Q sabit (gerçekçi: pompa sabit debili) vs Cq sabit (kontrollü)
        vm_frame = tk.Frame(parent, bg=COLORS["panel"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        vm_frame.pack(fill="x", pady=(0, 10))
        tk.Label(vm_frame, text="  HAVALANDIRMA MODU",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(vm_frame,
                 text="  Q sabit: gerçek pompa  |  Cq sabit: kontrol",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))

        for key, label in [("Q",  "Q sabit [L/min]  (V↑ ile Cq↓, kavite küçülür)"),
                           ("Cq", "Cq sabit [boyutsuz]  (Q hızla orantılı artar)")]:
            rb = tk.Radiobutton(vm_frame, text=label, value=key,
                                variable=self.vent_mode_var,
                                fg=COLORS["text"], bg=COLORS["panel"],
                                selectcolor=COLORS["bg"],
                                activebackground=COLORS["panel"],
                                activeforeground=COLORS["accent"],
                                font=("Consolas", 8),
                                anchor="w", relief="flat",
                                command=self._on_geom_model_change)
            rb.pack(fill="x", padx=12, pady=1)
        tk.Frame(vm_frame, bg=COLORS["panel"], height=6).pack()

        # ========== KAVİTATÖR TİPİ (DISK / CONE) ==========
        ct_frame = tk.Frame(parent, bg=COLORS["panel"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        ct_frame.pack(fill="x", pady=(0, 10))
        tk.Label(ct_frame, text="  KAVİTATÖR TİPİ",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(ct_frame,
                 text="  Disk: Cx0=0.82  |  Cone: Cx0=0.82·sin²(β/2)",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))

        for key, label in [
                ("disk", "Disk kavitatör (klasik, yüksek drag, geniş kavite)"),
                ("cone", "Konik kavitatör (düşük drag, dar kavite — β ile ayarlanır)")]:
            rb = tk.Radiobutton(ct_frame, text=label, value=key,
                                variable=self.cav_type_var,
                                fg=COLORS["text"], bg=COLORS["panel"],
                                selectcolor=COLORS["bg"],
                                activebackground=COLORS["panel"],
                                activeforeground=COLORS["accent"],
                                font=("Consolas", 8),
                                anchor="w", relief="flat",
                                command=self._on_geom_model_change)
            rb.pack(fill="x", padx=12, pady=1)
        tk.Frame(ct_frame, bg=COLORS["panel"], height=6).pack()

        # ========== ZAMAN ÇİZELGESİ (TIME SCHEDULES) ==========
        sch_frame = tk.Frame(parent, bg=COLORS["panel"],
                              highlightbackground=COLORS["border"], highlightthickness=1)
        sch_frame.pack(fill="x", pady=(0, 10))
        tk.Label(sch_frame, text="  ZAMAN ÇİZELGESİ (STEP)",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(sch_frame,
                 text="  α, δ_c, T, Q zamanla DEĞİŞEBİLİR  ·  JSON yükle/temizle",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))

        # Schedule durumu (kaç adım var)
        self.sch_status_label = tk.Label(sch_frame,
            text="  [hiç schedule yok — sabit parametre]",
            fg=COLORS["mute"], bg=COLORS["panel"],
            font=("Consolas", 7))
        self.sch_status_label.pack(anchor="w", padx=10, pady=(0, 4))

        btn_row = tk.Frame(sch_frame, bg=COLORS["panel"])
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        tk.Button(btn_row, text="📂 JSON YÜKLE",
            command=self._load_schedules,
            bg=COLORS["accent"], fg=COLORS["bg"],
            font=("Consolas", 8, "bold"), relief="flat",
            activebackground=COLORS["accent2"]
        ).pack(side="left", padx=(0, 4), fill="x", expand=True)

        tk.Button(btn_row, text="✕ TEMİZLE",
            command=self._clear_schedules,
            bg=COLORS["border"], fg=COLORS["text"],
            font=("Consolas", 8), relief="flat"
        ).pack(side="left", fill="x", expand=True)

        tk.Button(btn_row, text="💾 ÖRNEK JSON",
            command=self._save_example_schedule,
            bg=COLORS["border"], fg=COLORS["text"],
            font=("Consolas", 8), relief="flat"
        ).pack(side="left", padx=(4, 0), fill="x", expand=True)

        # ========== GÖVDE-KAVİTE ETKİLEŞİMİ ==========
        bv_frame = tk.Frame(parent, bg=COLORS["panel"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        bv_frame.pack(fill="x", pady=(0, 10))
        tk.Label(bv_frame, text="  GÖVDE-KAVİTE ETKİLEŞİMİ",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(bv_frame,
                 text="  Logvinovich superposition: R²=R_c²+R_body²",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))

        cb = tk.Checkbutton(bv_frame,
                             text="Gövde hacmi ile kavite genişlemesi (önerilen)",
                             variable=self.body_volume_var,
                             fg=COLORS["text"], bg=COLORS["panel"],
                             selectcolor=COLORS["bg"],
                             activebackground=COLORS["panel"],
                             activeforeground=COLORS["accent"],
                             font=("Consolas", 8),
                             anchor="w", relief="flat",
                             command=self._on_geom_model_change)
        cb.pack(fill="x", padx=12, pady=1)
        tk.Label(bv_frame,
                 text="  Kapalıysa: sadece R≥R_body clamping (basit)",
                 fg="#5a708a", bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))
        tk.Frame(bv_frame, bg=COLORS["panel"], height=6).pack()

        # ========== KANATLAR (FIN'LER) ==========
        fn_frame = tk.Frame(parent, bg=COLORS["panel"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        fn_frame.pack(fill="x", pady=(0, 10))
        tk.Label(fn_frame, text="  KANATLAR (4 BAĞIMSIZ FIN)",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(fn_frame,
                 text="  NACA 16-009  ·  Azim: 90°üst, 0°sağ, 270°alt, 180°sol",
                 fg=COLORS["mute"], bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))

        cb_fin = tk.Checkbutton(fn_frame,
                             text="Kanatları AKTİF et (kuvvet ve momente etki eder)",
                             variable=self.fins_enabled_var,
                             fg=COLORS["text"], bg=COLORS["panel"],
                             selectcolor=COLORS["bg"],
                             activebackground=COLORS["panel"],
                             activeforeground=COLORS["accent"],
                             font=("Consolas", 8),
                             anchor="w", relief="flat",
                             command=self._on_geom_model_change)
        cb_fin.pack(fill="x", padx=12, pady=1)

        # ANALİZ MODU: V-sabit ve planing toggle
        cb_steady = tk.Checkbutton(fn_frame,
                             text="V-SABİT modu (thrust=drag, V değişmez, steady-state)",
                             variable=self.steady_v_var,
                             fg=COLORS["accent"], bg=COLORS["panel"],
                             selectcolor=COLORS["bg"],
                             activebackground=COLORS["panel"],
                             activeforeground=COLORS["accent"],
                             font=("Consolas", 8, "bold"),
                             anchor="w", relief="flat",
                             command=self._on_geom_model_change)
        cb_steady.pack(fill="x", padx=12, pady=(6, 1))

        cb_planing = tk.Checkbutton(fn_frame,
                             text="Transom slap (F_planing) AKTİF (kapatmak için tıkla)",
                             variable=self.planing_enable_var,
                             fg=COLORS["text"], bg=COLORS["panel"],
                             selectcolor=COLORS["bg"],
                             activebackground=COLORS["panel"],
                             activeforeground=COLORS["accent"],
                             font=("Consolas", 8),
                             anchor="w", relief="flat",
                             command=self._on_geom_model_change)
        cb_planing.pack(fill="x", padx=12, pady=1)
        tk.Label(fn_frame,
                 text="  Sadece kavite DIŞINDA kalan kısım kuvvet üretir",
                 fg="#5a708a", bg=COLORS["panel"],
                 font=("Consolas", 7, "italic")).pack(anchor="w", pady=(0, 4))
        tk.Frame(fn_frame, bg=COLORS["panel"], height=6).pack()

        # ========== PARAMETRE GİRİŞ KUTULARI (Entry) ==========
        pf = tk.Frame(parent, bg=COLORS["panel"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        # fill='x' (expand=True DEĞİL) — scrollable canvas içinde doğru çalışır
        pf.pack(fill="x", pady=(0, 10))
        tk.Label(pf, text="  PARAMETRELER  (Enter ile uygula)",
                 fg=COLORS["accent"], bg=COLORS["panel"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(8, 4))

        # Parametre tanımları: (anahtar, etiket, varsayılan, birim, biçim)
        # gas_flow için 0 geçerli (havalandırma kapalı demek)
        # _section_marker: True ise parametre öncesinde bölüm başlığı göster
        inputs = [
            # === KONTROL AÇILARI (CASE 1) ===
            ("alpha_aoa","Hücum Açısı α",  -1.0,    "°",      "{:+.2f}", True),
            ("delta_cav","Kavitatör δ_c",  2.0,     "°",      "{:+.2f}", True),
            ("CL_alpha_body","Gövde C_L_α", 2.5,    "1/rad",  "{:.2f}",  True),
            ("Cdc_body", "Crossflow Cdc·η", 0.4,   "",       "{:.2f}",  True),
            ("k_dev",    "Kav. Sapma k", 0.4,    "",       "{:.2f}",  True),
            ("A_v",      "Havalandırma A_v", 0.020, "",     "{:.4f}",  True),
            ("K_slender","Slender K", 0.5,         "",       "{:.2f}",  True),
            ("cone_lift_gain","Konik Lift K", 0.50, "",       "{:.2f}",  True),
            ("K_cone_lift", "Konik Yan-Lift K", 0.0, "",   "{:.2f}",  True),
            ("cone_apex_deg","Konik Tepe β", 40.0,"°",      "{:.1f}",  False),
            ("cone_drag_visc","Konik Drag K_v", 0.18, "",  "{:.3f}",  True),
            ("k_g_cavity",   "Reichardt k_g", 0.78, "",   "{:.3f}",  True),
            ("K_Dc_factor",  "Çap Çarpanı K_Dc", 1.0, "",  "{:.3f}",  True),
            ("K_Lc_factor",  "Boy Çarpanı K_Lc", 1.0, "",  "{:.3f}",  True),
            # ===== KANATLAR (NACA 16-009, CASE 1 trim açıları) =====
            ("fin_chord",   "Kanat Kord",    0.10,   "m",      "{:.3f}", False),
            ("fin_span",    "Kanat Span",    0.25,   "m",      "{:.3f}", False),
            ("fin_x_pos",   "Kanat x Konum", 3.80,   "m",      "{:.2f}", False),
            ("fin_delta_1", "δ_üst (azim 90°)",  -0.80, "°",   "{:.2f}", True),
            ("fin_delta_2", "δ_sağ (azim 0°)",    0.0,  "°",   "{:.2f}", True),
            ("fin_delta_3", "δ_alt (azim 270°)", -0.80, "°",   "{:.2f}", True),
            ("fin_delta_4", "δ_sol (azim 180°)",  0.0,  "°",   "{:.2f}", True),
            # === GENEL PARAMETRELER (CASE 1) ===
            ("depth",    "Derinlik",        10.0,    "m",      "{:.2f}",  False),
            ("diam_cav", "Kavitatör Çapı",  0.20,    "m",      "{:.4f}",  False),
            ("mass",     "Araç Kütlesi",    350.0,   "kg",     "{:.2f}",  False),
            ("thrust",   "İtki Kuvveti",    6000.0,  "N",      "{:.1f}",  False),
            ("veh_len",  "Araç Uzunluğu",   4.0,     "m",      "{:.3f}",  False),
            ("veh_diam", "Araç Çapı",       0.30,    "m",      "{:.4f}",  False),
            ("L_taper",  "Konik Burun L",   0.60,    "m",      "{:.3f}",  True),
            ("gas_flow", "Havalandırma Q", 15000.0, "L/min",  "{:.1f}",  True),
            ("V_init",   "Başlangıç Hızı", 40.0,    "m/s",    "{:.2f}",  False),
            ("x_cg",     "Moment Noktası", 2.4,     "m",      "{:.3f}",  True),
            ("x_mg",     "Kütle Merkezi",  2.4,     "m",      "{:.3f}",  True),
            ("t_max",    "Analiz Süresi",   0.5,     "s",      "{:.3f}",  False),
            ("dt",       "Zaman Adımı",     0.001,   "s",      "{:.4f}",  False),
        ]
        self.param_entries = {}
        self.param_fmts = {}
        self.param_allow_zero = {}

        for key, label, _default, unit, fmt, allow_zero in inputs:
            self.param_fmts[key] = fmt
            self.param_allow_zero[key] = allow_zero

            row = tk.Frame(pf, bg=COLORS["panel"])
            row.pack(fill="x", padx=10, pady=5)

            top = tk.Frame(row, bg=COLORS["panel"])
            top.pack(fill="x")
            tk.Label(top, text=label, fg=COLORS["mute"], bg=COLORS["panel"],
                     font=("Consolas", 8)).pack(side="left")
            # Hint metni — açı parametreleri için özel (negatif değer kabul)
            if key == "alpha_aoa":
                hint = "(±15°, signed)"
            elif key == "delta_cav":
                hint = "(±30°, signed)"
            elif key in {"fin_delta_1", "fin_delta_2", "fin_delta_3", "fin_delta_4"}:
                hint = "(±20°, signed)"
            else:
                hint = f"(≥0, {unit})" if allow_zero else f"(pozitif, {unit})"
            tk.Label(top, text=hint,
                     fg="#4a5d75", bg=COLORS["panel"],
                     font=("Consolas", 7, "italic")).pack(side="right")

            entry_row = tk.Frame(row, bg=COLORS["panel"])
            entry_row.pack(fill="x", pady=(2, 0))

            sv = tk.StringVar(value=fmt.format(self.params[key].get()))
            entry = tk.Entry(entry_row, textvariable=sv,
                             bg=COLORS["bg"], fg=COLORS["text"],
                             insertbackground=COLORS["accent"],
                             font=("Consolas", 10, "bold"),
                             relief="flat", bd=0,
                             highlightthickness=1,
                             highlightbackground=COLORS["border"],
                             highlightcolor=COLORS["accent"])
            entry.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
            tk.Label(entry_row, text=unit, fg=COLORS["mute"],
                     bg=COLORS["panel"], font=("Consolas", 9),
                     width=4, anchor="w").pack(side="left")

            entry.bind("<Return>",
                       lambda e, k=key, s=sv: self._on_entry_commit(k, s))
            entry.bind("<FocusOut>",
                       lambda e, k=key, s=sv: self._on_entry_commit(k, s))

            self.param_entries[key] = (entry, sv)

        apply_btn = tk.Button(pf, text="↻ UYGULA ve SIMÜLASYONU YENİLE",
            command=self._apply_all_entries,
            bg=COLORS["accent"], fg=COLORS["bg"],
            font=("Consolas", 9, "bold"), relief="flat",
            activebackground=COLORS["accent2"])
        apply_btn.pack(fill="x", padx=10, pady=(8, 10))

        tk.Label(parent,
                 text="Logvinovich 1969 · Garabedian 1956 · Reichardt 1946\nSemenenko 2001 · Savchenko 1996",
                 fg=COLORS["mute"], bg=COLORS["bg"],
                 font=("Consolas", 7), justify="left").pack(anchor="w", pady=(10, 0))

    # ===================================================== schedule mekanizması
    def _load_schedules(self):
        """JSON dosyasından zaman çizelgesi yükle.
        Beklenen format:
            {
              "alpha_aoa_schedule":  [[0.0, 0], [1.0, 5], [3.0, -3]],
              "delta_cav_schedule":  [[0.0, 0], [2.0, 10]],
              "thrust_schedule":     [[0.0, 8000], [4.0, 12000]],
              "gas_flow_schedule":   [[0.0, 0], [1.5, 500]]
            }
        Tüm anahtarlar opsiyonel; eksik olanlar sabit parametreyi kullanır.
        """
        try:
            from tkinter import filedialog, messagebox
            import json
        except Exception:
            return
        fname = filedialog.askopenfilename(
            title="Schedule JSON dosyası seç",
            filetypes=[("JSON dosyaları", "*.json"), ("Tüm dosyalar", "*.*")])
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            try:
                messagebox.showerror("Schedule yükleme hatası", str(e))
            except Exception:
                print(f"[schedule HATA] {e}")
            return
        # Geçerli anahtarları tanı
        valid_keys = ["alpha_aoa_schedule", "delta_cav_schedule",
                      "thrust_schedule",    "gas_flow_schedule",
                      "fin_delta_1_schedule", "fin_delta_2_schedule",
                      "fin_delta_3_schedule", "fin_delta_4_schedule"]
        loaded = []
        for k in valid_keys:
            if k in data and isinstance(data[k], list) and len(data[k]) > 0:
                # [[t, val], [t, val], ...] formatı bekleniyor
                try:
                    sched = [(float(p[0]), float(p[1])) for p in data[k]]
                    sched.sort(key=lambda x: x[0])  # zamana göre sırala
                    self.schedules[k] = sched
                    loaded.append(f"{k.replace('_schedule','')}({len(sched)})")
                except Exception:
                    self.schedules[k] = None
            else:
                # Bu schedule yok — None bırak (sabit parametre kullanılır)
                pass
        # Durum etiketini güncelle
        if loaded:
            txt = "  → " + " · ".join(loaded)
        else:
            txt = "  [JSON'da geçerli schedule yok]"
        self.sch_status_label.config(text=txt, fg=COLORS["accent"])
        # Yeniden çalıştır
        self._full_reset_after_param_change()

    def _clear_schedules(self):
        """Tüm schedule'ları temizle, sabit parametre moduna dön."""
        for k in self.schedules:
            self.schedules[k] = None
        self.sch_status_label.config(
            text="  [hiç schedule yok — sabit parametre]",
            fg=COLORS["mute"])
        self._full_reset_after_param_change()

    def _save_example_schedule(self):
        """Örnek schedule JSON dosyası kaydet — kullanıcıya format şablonu."""
        try:
            from tkinter import filedialog, messagebox
            import json
        except Exception:
            return
        fname = filedialog.asksaveasfilename(
            title="Örnek JSON dosyası kaydet",
            defaultextension=".json",
            initialfile="schedule_ornek.json",
            filetypes=[("JSON dosyaları", "*.json")])
        if not fname:
            return
        example = {
            "_aciklama": "Format: [[t_saniye, deger], ...]. t < t0 ise ilk deger. ZOH (zero-order hold).",
            "_birimler": {
                "alpha_aoa_schedule":  "derece (±15°)",
                "delta_cav_schedule":  "derece (±30°)",
                "thrust_schedule":     "Newton",
                "gas_flow_schedule":   "L/min (Q-mode) veya boyutsuz (Cq-mode)",
                "fin_delta_1_schedule": "derece, üst kanat (±20°)",
                "fin_delta_2_schedule": "derece, sağ kanat (±20°)",
                "fin_delta_3_schedule": "derece, alt kanat (±20°)",
                "fin_delta_4_schedule": "derece, sol kanat (±20°)"
            },
            "alpha_aoa_schedule":    [[0.0, 0.0], [1.0, 3.0], [3.0, -2.0], [5.0, 0.0]],
            "delta_cav_schedule":    [[0.0, 0.0], [2.0, 10.0], [4.0, -5.0]],
            "thrust_schedule":       [[0.0, 8000.0], [3.0, 15000.0]],
            "gas_flow_schedule":     [[0.0, 0.0], [1.0, 500.0], [4.0, 0.0]],
            "fin_delta_1_schedule":  [[0.0, 0.0], [1.5, 5.0], [3.5, -3.0]],
            "fin_delta_2_schedule":  [[0.0, 0.0]],
            "fin_delta_3_schedule":  [[0.0, 0.0], [1.5, 5.0], [3.5, -3.0]],
            "fin_delta_4_schedule":  [[0.0, 0.0]]
        }
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(example, f, indent=2, ensure_ascii=False)
            try:
                messagebox.showinfo("Örnek JSON kaydedildi",
                    f"Şablon dosyası kaydedildi:\n{fname}\n\n"
                    "Düzenleyip 'JSON YÜKLE' ile yükleyebilirsin.")
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Kaydetme hatası", str(e))
            except Exception:
                print(f"[schedule kaydetme HATA] {e}")

    # ========================================================== simülasyon
    def _collect_params(self):
        """UI'dan güncel params dict oluşturur (canlı ve batch mod ortak kullanır)."""
        p = {k: v.get() for k, v in self.params.items()}
        p["geom_model"] = self.geom_model_var.get()
        p["vent_mode"] = self.vent_mode_var.get()
        p["cavitator_type"] = self.cav_type_var.get()
        p["cav_type"] = self.cav_type_var.get()
        p["body_volume_correction"] = self.body_volume_var.get()
        p["fins_enabled"] = self.fins_enabled_var.get()
        p["steady_v_mode"] = self.steady_v_var.get()
        p["planing_enable"] = self.planing_enable_var.get()
        if "x_mg" in p:
            p["x_cg_mass"] = p["x_mg"]
        return p

    def _run_sim(self):
        p = self._collect_params()
        # Schedule'ları aktar — None ise simulate sabiti kullanır
        for k, v in self.schedules.items():
            p[k] = v
        self.data = simulate(p)
        self.time_slider.configure(to=len(self.data["t"]) - 1)

    # ================================================================= CANLI MOD
    def _live_toggle(self):
        """Canlı simülasyonu başlat/durdur."""
        if self._live_running:
            self._live_stop()
        else:
            self._live_start()

    def _live_start(self):
        """Canlı simülasyonu başlat - chunk chunk ilerler, UI güncel params okur."""
        self._live_running = True
        # State init: UI'daki V_init'ten başla
        p = self._collect_params()
        self._live_V = float(p.get("V_init", 40.0))
        self._live_Lc = 0.0    # kavite boyu state (olgunlaşma gecikmesi için)
        self._live_Dc = 0.0    # kavite çapı state
        self._live_pc = None   # kavite basıncı state (P_VAP olarak simulate init)
        self._live_t = 0.0
        # Boş data dict - chunk'lar buraya biriktirilir
        self._live_data = None
        # Buton görünümünü değiştir
        if hasattr(self, "btn_live"):
            self.btn_live.config(text="⏸ CANLI DURDUR", bg="#c85040")
        # İlk tick'i planla
        self.root.after(10, self._live_tick)

    def _live_stop(self):
        """Canlı simülasyonu durdur (birikmiş veri kalır, grafikte gözükür)."""
        self._live_running = False
        if hasattr(self, "btn_live"):
            self.btn_live.config(text="▶ CANLI BAŞLAT", bg="#40a068")

    def _live_tick(self):
        """Bir chunk simüle et, arrays'e ekle, grafikleri güncelle, sonrakini planla."""
        if not self._live_running:
            return

        # 1. UI'dan güncel params al (kullanıcı arada değiştirmiş olabilir!)
        p = self._collect_params()
        # State'i override et: V + kavite state önceki chunk'ın son değerleri
        p["V_init"] = self._live_V
        p["initial_Lc"] = self._live_Lc
        p["initial_Dc"] = self._live_Dc
        if self._live_pc is not None:
            p["initial_pc"] = self._live_pc
        # Chunk süresi (kısa - kullanıcı müdahalesine hızlı yanıt için)
        p["t_max"] = self._live_chunk_dur
        p["dt"] = self._live_dt
        # Schedule'lar çalışmasın (canlı modda anlık kontrol)
        for k in self.schedules:
            p[k] = None

        # 2. Chunk simüle et
        try:
            chunk = simulate(p)
        except Exception as e:
            print(f"[canlı sim hata] {e}")
            self._live_stop()
            return

        # 3. State güncelle (V + kavite boyu/çapı + basıncı)
        self._live_V = float(chunk["V"][-1])
        self._live_Lc = float(chunk["Lc"][-1])
        self._live_Dc = float(chunk["Dc"][-1])
        if "pc" in chunk:
            self._live_pc = float(chunk["pc"][-1])

        # 4. Verileri birikimli arrays'e ekle (zaman offset ile)
        t_offset = self._live_t
        self._live_t += self._live_chunk_dur

        if self._live_data is None:
            # İlk chunk - direkt kopyala
            self._live_data = {}
            for k, v in chunk.items():
                try:
                    arr = np.asarray(v)
                    if arr.ndim > 0 and len(arr) > 0:
                        if k == "t":
                            self._live_data[k] = arr + t_offset
                        else:
                            self._live_data[k] = arr.copy()
                    else:
                        self._live_data[k] = v
                except Exception:
                    self._live_data[k] = v
        else:
            # Sonraki chunk'lar - concatenate
            for k, v in chunk.items():
                if k not in self._live_data:
                    continue
                try:
                    new_arr = np.asarray(v)
                    if new_arr.ndim > 0 and len(new_arr) > 0:
                        if k == "t":
                            new_arr = new_arr + t_offset
                        self._live_data[k] = np.concatenate([self._live_data[k], new_arr])
                except Exception:
                    pass

        # 5. Grafik güncelle - data pointer'ı canlı data'ya çevir
        self.data = self._live_data
        # Slider max'ı güncelle
        self.time_slider.configure(to=len(self.data["t"]) - 1)
        # Anlık idx'i sona getir (kullanıcı canlı takip eder)
        self.idx = len(self.data["t"]) - 1
        # Slider da sona
        self._internal_slider_update = True
        self.time_slider.set(self.idx)
        self._internal_slider_update = False

        # 6. Grafik yenile (statik + anlık cursor)
        try:
            self._draw_plots_static()
            self._update_frame()
        except Exception as e:
            print(f"[canlı grafik hata] {e}")

        # 7. Sonraki tick'i planla
        if self._live_running:
            self.root.after(self._live_chunk_ms, self._live_tick)

    def _on_geom_model_change(self):
        """Geometri modeli değiştirildiğinde simülasyonu yeniden çalıştır
        ve aynı zaman noktasında çizimleri güncelle."""
        old_idx = self.idx
        self._run_sim()
        # Yeni veri uzunluğuna sığ
        self.idx = min(old_idx, len(self.data["t"]) - 1)
        self._draw_plots_static()
        self._update_frame()

    def _on_entry_commit(self, key, str_var):
        """Bir girişe Enter/FocusOut: doğrula, uygula, sim yeniden, çiz.
        Üç tip parametre var:
          - allow_zero=False: sıkı pozitif (mass, depth, V_init vb.)
          - allow_zero=True ve sayısal: ≥0 (gas_flow, x_cg vb.)
          - SIGNED (alpha_aoa, delta_cav): negatif değer ALINIR
        """
        fmt = self.param_fmts[key]
        raw = str_var.get().strip().replace(",", ".")

        try:
            val = float(raw)
        except ValueError:
            # Geçersiz sayı → eski değere dön, kırmızı flash
            str_var.set(fmt.format(self.params[key].get()))
            self._flash_entry(key, ok=False)
            return

        # İşaret kuralları
        SIGNED_KEYS = {"alpha_aoa", "delta_cav",
                       "fin_delta_1", "fin_delta_2",
                       "fin_delta_3", "fin_delta_4"}   # negatif kabul edilenler
        if key in SIGNED_KEYS:
            # Sınırlı aralık (kavitatör/hücum açısı pratik aralıkları)
            if key == "delta_cav":
                limit = 30.0
            elif key == "alpha_aoa":
                limit = 15.0
            else:
                limit = 20.0    # kanat deflection ±20°
            if abs(val) > limit:
                str_var.set(fmt.format(self.params[key].get()))
                self._flash_entry(key, ok=False)
                return
            # Negatif değer kabul edilir, koşul yok
        else:
            allow_zero = self.param_allow_zero.get(key, False)
            if allow_zero:
                if val < 0:
                    str_var.set(fmt.format(self.params[key].get()))
                    self._flash_entry(key, ok=False)
                    return
            else:
                if val <= 0:
                    str_var.set(fmt.format(self.params[key].get()))
                    self._flash_entry(key, ok=False)
                    return

        old = self.params[key].get()
        str_var.set(fmt.format(val))

        if abs(val - old) < 1e-12:
            self._flash_entry(key, ok=True)
            return

        self.params[key].set(val)
        # DoubleVar.set çağrısı trace mekanizmasını tetikler → canlı slider
        # ve label otomatik olarak senkronize olur (make_live_slider içindeki
        # var.trace_add("write", _sync_from_var) hookunu üzerinden).
        self._flash_entry(key, ok=True)
        self._full_reset_after_param_change()

    def _apply_all_entries(self):
        """UYGULA butonu: tüm Entry'leri okur, doğrular, tek sim çalıştırır.
        Geçersiz değerler eski haline döner. gas_flow için 0 kabul edilir.
        alpha_aoa, delta_cav için negatif değer kabul edilir."""
        SIGNED_KEYS = {"alpha_aoa", "delta_cav",
                       "fin_delta_1", "fin_delta_2",
                       "fin_delta_3", "fin_delta_4"}
        for key, (entry, sv) in self.param_entries.items():
            fmt = self.param_fmts[key]
            raw = sv.get().strip().replace(",", ".")
            try:
                val = float(raw)
            except ValueError:
                sv.set(fmt.format(self.params[key].get()))
                continue
            if key in SIGNED_KEYS:
                if key == "delta_cav":
                    limit = 30.0
                elif key == "alpha_aoa":
                    limit = 15.0
                else:
                    limit = 20.0
                if abs(val) > limit:
                    sv.set(fmt.format(self.params[key].get()))
                    continue
            else:
                allow_zero = self.param_allow_zero.get(key, False)
                if allow_zero:
                    if val < 0:
                        sv.set(fmt.format(self.params[key].get()))
                        continue
                else:
                    if val <= 0:
                        sv.set(fmt.format(self.params[key].get()))
                        continue
            sv.set(fmt.format(val))
            self.params[key].set(val)
        self._full_reset_after_param_change()

    def _flash_entry(self, key, ok=True):
        """Kısa süreli yeşil (ok) / turuncu (hata) kenar."""
        entry, _ = self.param_entries[key]
        color = COLORS["skin"] if ok else COLORS["drag"]
        entry.config(highlightbackground=color, highlightcolor=color)
        self.root.after(400, lambda: entry.config(
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"]))

    def _full_reset_after_param_change(self):
        """Parametre değişince: sim yeniden, animasyon t=0'a, tüm grafikler
        baştan çizilir, animasyon DURAKLI bekler."""
        self._run_sim()
        self.idx = 0
        self.time_var.set(0)
        self.playing = False
        self.btn_play.config(text="▶ BAŞLAT")
        self._draw_plots_static()
        self._update_frame()

    def _toggle_play(self):
        if not self.playing and self.data is not None \
                and self.idx >= len(self.data["t"]) - 1:
            self.idx = 0
            self.time_var.set(0)
        self.playing = not self.playing
        self.btn_play.config(text="❚❚ DURDUR" if self.playing else "▶ BAŞLAT")

    def _reset(self):
        """Animasyonu başa sar ve DURAKLI bekle."""
        self.idx = 0
        self.time_var.set(0)
        self.playing = False
        self.btn_play.config(text="▶ BAŞLAT")
        self._update_frame()

    def _on_time_scrub(self, val):
        # Animasyon time_var.set yaparsa bu callback de tetiklenir.
        # Re-entrance engellemek için bayrak kontrolü.
        if getattr(self, "_internal_slider_update", False):
            return
        new_idx = int(float(val))
        if new_idx == self.idx:
            return
        # Kullanıcı slider'ı çekti — animasyonu durdur
        self.idx = new_idx
        if self.playing:
            self.playing = False
            self.btn_play.config(text="▶ BAŞLAT")
        self._update_frame()

    # ========================================================== çizimler
    def _style_axis(self, ax, title):
        ax.set_title(title, color=COLORS["text"], fontsize=9, loc="left",
                     family="monospace", pad=6)
        ax.tick_params(colors=COLORS["mute"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.4, alpha=0.5)

    def _draw_plots_static(self):
        """Tüm grafikleri BAŞTAN çizer — twin axes dahil clear()."""
        d = self.data

        # V ve σ (çift eksen)
        ax  = self.ax_vsigma
        ax2 = self.ax_vsigma2
        ax.clear(); ax2.clear()
        ax.set_facecolor(COLORS["panel"])
        l1, = ax.plot(d["t"], d["V"], color=COLORS["accent"], lw=1.6,
                      label="V [m/s]")
        l2, = ax2.plot(d["t"], d["sigma"], color=COLORS["accent2"], lw=1.6,
                       label="σ")
        l3 = ax2.axhline(0.1, color=COLORS["drag"], lw=0.8, ls="--",
                          alpha=0.8, label="σ = 0.1 (süperkav. eşiği)")
        ax.set_xlabel("t [s]", color=COLORS["mute"], fontsize=8)
        ax.set_ylabel("V [m/s]", color=COLORS["accent"], fontsize=8)
        ax2.set_ylabel("σ", color=COLORS["accent2"], fontsize=8)
        ax2.tick_params(colors=COLORS["mute"], labelsize=7)
        for sp in ax2.spines.values(): sp.set_color(COLORS["border"])
        ax.legend(handles=[l1, l2, l3], loc="center right",
                  fontsize=7, facecolor=COLORS["panel"],
                  edgecolor=COLORS["border"], labelcolor=COLORS["text"],
                  framealpha=0.85)
        self._style_axis(ax, "HIZ V(t) ve KAVİTASYON NUMARASI σ(t)")
        self.vline_vsigma = ax.axvline(0, color=COLORS["drag"], lw=0.8, ls=":")

        # Kavite boyutları
        ax = self.ax_cavity
        ax.clear(); ax.set_facecolor(COLORS["panel"])
        ax.plot(d["t"], d["Lc"], color=COLORS["cavity"], lw=1.6,
                label="Lc — kavite boyu [m]")
        ax.plot(d["t"], d["Dc"] * 10, color="#a8dce8", lw=1.6,
                label="Dc × 10 — kavite çapı [m]")
        ax.fill_between(d["t"], 0, d["Lc"], color=COLORS["cavity"], alpha=0.1)
        ax.set_xlabel("t [s]", color=COLORS["mute"], fontsize=8)
        ax.set_ylabel("m", color=COLORS["mute"], fontsize=8)
        ax.legend(loc="upper left", fontsize=7, facecolor=COLORS["panel"],
                  edgecolor=COLORS["border"], labelcolor=COLORS["text"],
                  framealpha=0.85)
        self._style_axis(ax, "KAVİTE BÜYÜMESİ: Lc(t), Dc(t)")
        self.vline_cavity = ax.axvline(0, color=COLORS["drag"], lw=0.8, ls=":")

        # Sürükleme bileşenleri
        ax = self.ax_drag
        ax.clear(); ax.set_facecolor(COLORS["panel"])
        ax.plot(d["t"], d["F_cav"],   color=COLORS["drag"],  lw=1.6,
                label="F_kavitatör")
        ax.plot(d["t"], d["F_skin"],  color=COLORS["skin"],  lw=1.6,
                label="F_sürtünme")
        ax.plot(d["t"], d["F_press"], color=COLORS["press"], lw=1.6,
                label="F_basınç")
        ax.plot(d["t"], d["Fd"],      color=COLORS["text"],  lw=1.0,
                ls="--", label="F_toplam")
        ax.set_xlabel("t [s]", color=COLORS["mute"], fontsize=8)
        ax.set_ylabel("F [N]", color=COLORS["mute"], fontsize=8)
        ax.legend(loc="upper right", fontsize=7, facecolor=COLORS["panel"],
                  edgecolor=COLORS["border"], labelcolor=COLORS["text"],
                  framealpha=0.85, ncol=2)
        self._style_axis(ax, "SÜRÜKLEME KUVVETİ BİLEŞENLERİ")
        self.vline_drag = ax.axvline(0, color=COLORS["drag"], lw=0.8, ls=":")

        # Kaplama + Kavite içi basınç pc + Paryshev β (çift eksen)
        # Havalandırma etkilerinin görsel izleme paneli
        ax  = self.ax_cover
        ax2 = self.ax_cover2
        ax.clear(); ax2.clear()
        ax.set_facecolor(COLORS["panel"])

        # Sol eksen: Kaplama (%)
        l1, = ax.plot(d["t"], d["cover"] * 100, color=COLORS["accent"], lw=1.8,
                      label="Kaplama [%]")
        ax.fill_between(d["t"], 0, d["cover"] * 100,
                        color=COLORS["accent"], alpha=0.15)

        # Sağ eksen: ivme
        l2, = ax2.plot(d["t"], d["a"], color=COLORS["drag"], lw=1.4,
                       label="İvme a [m/s²]")
        l3 = ax2.axhline(0, color=COLORS["mute"], lw=0.5, alpha=0.6, label="a = 0")

        ax.set_xlabel("t [s]", color=COLORS["mute"], fontsize=8)
        ax.set_ylabel("Kaplama [%]", color=COLORS["accent"], fontsize=8)
        ax2.set_ylabel("a [m/s²]", color=COLORS["drag"], fontsize=8)
        ax2.tick_params(colors=COLORS["mute"], labelsize=7)
        for sp in ax2.spines.values(): sp.set_color(COLORS["border"])
        ax.set_ylim(0, 105)
        ax.legend(handles=[l1, l2, l3], loc="center right",
                  fontsize=7, facecolor=COLORS["panel"],
                  edgecolor=COLORS["border"], labelcolor=COLORS["text"],
                  framealpha=0.85)
        self._style_axis(ax, "KAVİTE KAPLAMASI ve ARAÇ İVMESİ")
        self.vline_cover = ax.axvline(0, color=COLORS["drag"], lw=0.8, ls=":")

        # --- Havalandırma grafiği (hibrit model) ---
        ax  = self.ax_vent
        ax2 = self.ax_vent2
        ax.clear(); ax2.clear()
        ax.set_facecolor(COLORS["panel"])

        # Sol eksen: pc [kPa] ve β × 10 (aynı ölçeğe sığar)
        pc_kPa = d["pc"] / 1000.0
        l1, = ax.plot(d["t"], pc_kPa, color="#b8e0f2", lw=1.8,
                      label="pc — kavite içi basınç [kPa]")
        ax.fill_between(d["t"], d["pc"].min()/1000, pc_kPa,
                        color="#b8e0f2", alpha=0.08)
        l2, = ax.plot(d["t"], d["beta"] * 10, color="#e07b3a", lw=1.4,
                      label="β × 10 (Paryshev kararlılık)")
        l3 = ax.axhline(26.45, color=COLORS["drag"], lw=0.7, ls="--",
                        alpha=0.7, label="β = 2.645 kararsızlık eşiği")

        # Sağ eksen: Cq (havalandırma katsayısı) ve Fr (Froude)
        # Cq genelde 0–1 arası, Fr onlar-yüzler → Fr'yi ölçekleyelim
        l4, = ax2.plot(d["t"], d["Cq"], color="#95c17a", lw=1.6,
                       label="Cq (havalandırma katsayısı)")
        # Fr'yi 50'ye bölerek aynı ölçeğe getir
        l5, = ax2.plot(d["t"], d["Fr"] / 50, color="#d4a548", lw=1.2, ls="-.",
                       label="Fr / 50 (Froude sayısı)")

        ax.set_xlabel("t [s]", color=COLORS["mute"], fontsize=8)
        ax.set_ylabel("pc [kPa]  ·  β × 10",
                      color="#b8e0f2", fontsize=8)
        ax2.set_ylabel("Cq  ·  Fr/50", color="#95c17a", fontsize=8)
        ax2.tick_params(colors=COLORS["mute"], labelsize=7)
        for sp in ax2.spines.values(): sp.set_color(COLORS["border"])

        ax.legend(handles=[l1, l2, l3, l4, l5], loc="upper right",
                  fontsize=7, facecolor=COLORS["panel"],
                  edgecolor=COLORS["border"], labelcolor=COLORS["text"],
                  framealpha=0.85, ncol=2)
        self._style_axis(ax,
            "HAVALANDIRMA DİNAMİĞİ · pc, β (Paryshev), Cq, Fr")
        self.vline_vent = ax.axvline(0, color=COLORS["drag"], lw=0.8, ls=":")

        # ---- DİKEY KUVVET (F_z) ve PİTCH MOMENT (M_y) zaman serisi ----
        # Grafik: F_z bileşenleri (sol) + M_y iki versiyon (sağ)
        ax  = self.ax_force
        ax2 = self.ax_force2
        ax.clear(); ax2.clear()
        ax.set_facecolor(COLORS["panel"])

        # Sol eksen: F_z bileşenleri [kN]
        F_planing_kN = d["F_planing"] / 1000.0
        F_buoy_kN    = d["F_buoy"]    / 1000.0
        F_cavz_kN    = d["F_cavz"]    / 1000.0
        F_body_kN    = d["F_body_lift"] / 1000.0
        F_grav_kN    = d["F_grav"]    / 1000.0
        Fz_total_kN  = d["Fz"]        / 1000.0

        # Bileşenler — ince çizgiler
        l_p, = ax.plot(d["t"], F_planing_kN, color="#c19ae0", lw=1.1,
                       label="F_transom_slap", alpha=0.9)
        l_b, = ax.plot(d["t"], F_buoy_kN, color="#5cd47b", lw=1.1,
                       label="F_kaldırma", alpha=0.9)
        l_c, = ax.plot(d["t"], F_cavz_kN, color="#ffcc66", lw=1.1,
                       label="F_kav_lift", alpha=0.9)
        l_bl, = ax.plot(d["t"], F_body_kN, color="#a8d4ff", lw=1.1,
                        label="F_gövde_lift", alpha=0.9)
        l_g, = ax.plot(d["t"], F_grav_kN, color="#dd8888", lw=1.0,
                       label="F_ağırlık", alpha=0.7, ls=":")
        # Toplam — kalın
        l_tot, = ax.plot(d["t"], Fz_total_kN, color="#ffffff", lw=2.0,
                         label="F_z TOPLAM", alpha=0.95)
        ax.axhline(0, color=COLORS["mute"], lw=0.5, alpha=0.5)

        # Sağ eksen: M_y iki versiyon [kN·m]
        My_full_kNm  = d["My"]        / 1000.0   # ağırlık dahil
        My_no_g_kNm  = d["My_no_grav"]/ 1000.0   # ağırlık hariç

        l_my,  = ax2.plot(d["t"], My_full_kNm, color="#e07b3a", lw=1.6,
                          label="M_y (toplam)")
        l_my2, = ax2.plot(d["t"], My_no_g_kNm, color="#ffcc66", lw=1.3,
                          ls="-.", label="M_y (ağırlık hariç)")
        ax2.axhline(0, color=COLORS["mute"], lw=0.5, alpha=0.5)

        ax.set_xlabel("t [s]", color=COLORS["mute"], fontsize=8)
        ax.set_ylabel("F_z bileşenleri [kN]", color="#c19ae0", fontsize=8)
        ax2.set_ylabel("M_y [kN·m]", color="#e07b3a", fontsize=8)
        ax2.tick_params(colors=COLORS["mute"], labelsize=7)
        for sp in ax2.spines.values(): sp.set_color(COLORS["border"])

        # İki ayrı legend
        leg1 = ax.legend(handles=[l_p, l_b, l_c, l_bl, l_g, l_tot],
                         loc="upper left", fontsize=6.5,
                         facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                         labelcolor=COLORS["text"], framealpha=0.85,
                         title="F_z bileşenleri", title_fontsize=7, ncol=2)
        leg1.get_title().set_color("#c19ae0")
        leg2 = ax2.legend(handles=[l_my, l_my2],
                          loc="upper right", fontsize=6.5,
                          facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                          labelcolor=COLORS["text"], framealpha=0.85,
                          title="M_y", title_fontsize=7)
        leg2.get_title().set_color("#e07b3a")

        x_cg_disp = self.params["x_cg"].get()
        try:
            x_mg_disp = self.params["x_mg"].get()
        except KeyError:
            x_mg_disp = x_cg_disp
        try:
            alpha_disp = self.params["alpha_aoa_deg"].get()
            delta_disp = self.params["delta_cav_deg"].get()
            angle_label = f"  ·  α={alpha_disp:+.1f}° δ_c={delta_disp:+.1f}°"
        except KeyError:
            angle_label = ""

        self._style_axis(ax,
            f"DİKEY KUVVET F_z ve PİTCH MOMENT M_y · "
            f"x_cg={x_cg_disp:.3f}m  x_mg={x_mg_disp:.3f}m{angle_label}")
        self.vline_force = ax.axvline(0, color=COLORS["drag"], lw=0.8, ls=":")

        self.canvas.draw_idle()

    def _draw_vehicle(self):
        """Araç + kavite yanal kesiti. Eksen limitleri Dn, D_veh, Dc_max'e
        duyarlı. Kavite zarfı σ'ya bağlı Logvinovich profili."""
        ax = self.ax_vehicle
        ax.clear()
        ax.set_facecolor(COLORS["water2"])

        p = {k: v.get() for k, v in self.params.items()}
        d = self.data
        i = min(self.idx, len(d["t"]) - 1)

        L_veh = p["veh_len"]
        D_veh = p["veh_diam"]
        Dn    = p["diam_cav"]
        L_taper_v = max(0.0, min(p.get("L_taper", 0.15 * L_veh), L_veh))
        Lc_i  = d["Lc"][i]
        Dc_i  = d["Dc"][i]

        Dc_max = float(np.max(d["Dc"])) if d["Dc"].size else 0.0
        Lc_max = float(np.max(d["Lc"])) if d["Lc"].size else 0.0
        hg_max = float(np.max(d["hg"])) if d["hg"].size else 0.0

        # x-ekseni stratejisi: araç + asimetrik bölge görünür kalsın.
        # Sigma küçükken Lc çok büyük olabilir (32m+) ama aracın çevresindeki
        # kavite şeklini görmek istiyoruz. x_span'ı araç boyunun 2 katı ile
        # sınırlandır, kavite ucu çok ötedeyse görüntü dışı uyarısı eklenecek.
        # Hala Dc'nin maksimuma ulaştığı bölgeyi göster (Lc*0.5 yakını).
        # Kavitatör tipini erkenden oku (cav_w hesabı için lazım)
        try:
            cav_type_view = self.cav_type_var.get()
        except (KeyError, AttributeError):
            cav_type_view = "disk"

        # Kavitatör genişliği — disk veya konik için farklı
        # Disk: dikdörtgen kavitatör, genişlik cav_w; ön yüz x=0, arka yüz x=cav_w
        #       Araç burnu kavitatörün arkasından başlar → x_body0 = cav_w
        # Konik: sivri uç önde (-L_cone), geniş taban x=0; kavite x=0'dan ayrılır
        #       Araç burnu DOĞRUDAN x=0'dan başlar → x_body0 = 0
        cav_w = max(Dn * 0.25, 0.01)   # disk için kalınlık
        if cav_type_view == "cone":
            x_body0 = 0.0    # konik kavitatörün geniş tabanı x=0 → araç hemen burada başlar
        else:
            x_body0 = cav_w  # disk kavitatörden sonra araç başlar
        x_view_min = L_veh * 1.6 + cav_w
        x_view_max = L_veh * 3.0 + cav_w

        # Kontrol açılarını UI'dan oku — fonksiyon boyunca tek kaynak.
        # alpha_aoa: gövde ekseninin akışla yaptığı açı (gövde rotasyonu)
        # delta_cav: kavitatör eğimi
        # alpha_eff: kavitatöre çarpan akışın etkin açısı (kuvvet hesabı için)
        # (k_dev * alpha_eff): k_dev ile çarpılmış (kavite ekseni geometrisi için)
        try:
            alpha_aoa_view = np.radians(self.params["alpha_aoa"].get())
            delta_cav_view = np.radians(self.params["delta_cav"].get())
        except KeyError:
            alpha_aoa_view = 0.0
            delta_cav_view = 0.0
        alpha_eff_view = alpha_aoa_view + delta_cav_view
        # Görselde k_dev uygulanmış kavite ekseni
        try:
            k_dev_view = self.params["k_dev"].get()
        except KeyError:
            k_dev_view = 0.4
        k_dev_view = max(0.0, min(k_dev_view, 1.0))
        # Saturating profile için x_open hesabı (fizik tarafıyla aynı formül)
        R_n_view = Dn / 2.0
        x_open_view = max(10.0 * R_n_view / np.sqrt(0.82), R_n_view * 2.0)

        def cavity_axis_y_view(x_arr, V_loc):
            """Görselde kavite ekseninin y kayması — fizikle aynı formül."""
            x_arr = np.asarray(x_arr) if hasattr(x_arr, '__len__') else x_arr
            if V_loc <= 0.5:
                grav_part = 0.0
            else:
                grav_part = G * x_arr * x_arr / (2.0 * V_loc * V_loc)
            angle_part = (k_dev_view * alpha_eff_view * x_open_view
                          * (1.0 - np.exp(-x_arr / x_open_view)))
            return grav_part + angle_part
        # Kavite çok kısa ise tüm kaviteyi göster
        x_span = max(min(Lc_max * 1.15, x_view_max), x_view_min)
        max_diameter = max(Dn, D_veh, Dc_max, 0.02)
        y_half = max_diameter * 3.5
        # Görüntülenen aralıktaki maksimum yerçekimi kaldırma
        hg_visible = G * x_span * x_span / (2.0 * max(d["V"][i], 0.5) ** 2) if i >= 0 else hg_max
        hg_visible = min(hg_visible, hg_max)
        y_top    = y_half + max(hg_visible, 0.0)
        y_bottom = -y_half

        # x ekseni: sol kenara biraz boşluk (kavitatör ön kenarı x=0)
        x_left = -x_span * 0.05
        ax.set_xlim(x_left, x_span)
        ax.set_ylim(y_bottom, y_top)
        ax.set_aspect("auto")
        ax.axis("off")

        # ÖLÇEK REFERANS ÇUBUĞU — kullanıcı gerçek boyutları anlasın
        # Görselde x ekseni (boy) ve y ekseni (çap) farklı ölçekte!
        # 100 mm referans hem x hem y yönünde gösterilir.
        scale_len = 0.1  # 100 mm referans
        ref_x = x_left + x_span * 0.04
        ref_y = y_bottom + (y_top - y_bottom) * 0.10
        # Yatay 100mm
        ax.plot([ref_x, ref_x + scale_len], [ref_y, ref_y],
                color="#ffcc66", lw=2.0, zorder=15,
                solid_capstyle="butt")
        ax.text(ref_x + scale_len/2, ref_y - (y_top-y_bottom)*0.025,
                "100mm (x)", color="#ffcc66", fontsize=6,
                family="monospace", ha="center", va="top", zorder=15)
        # Dikey 100mm — y ekseninde farklı ölçek olduğunu vurgular
        ax.plot([ref_x, ref_x], [ref_y, ref_y + scale_len],
                color="#ffcc66", lw=2.0, zorder=15,
                solid_capstyle="butt")
        ax.text(ref_x + x_span * 0.005, ref_y + scale_len/2,
                "100mm (y)", color="#ffcc66", fontsize=6,
                family="monospace", ha="left", va="center", zorder=15)

        # Kavite görüntü dışına taşıyorsa kullanıcıyı bilgilendir
        if Lc_i > x_span * 1.05:
            ax.text(x_span * 0.98, y_top * 0.95,
                    f"⮕ Kavite uzunluğu {Lc_i:.1f} m (görüntü dışına uzanıyor)",
                    color="#ffcc66", fontsize=8, ha="right", va="top",
                    family="monospace", zorder=20,
                    bbox=dict(facecolor=COLORS["panel"], edgecolor="#ffcc66",
                              alpha=0.85, boxstyle="round,pad=0.3"))

        # Akış çizgileri (tüm dikey kapsam boyunca)
        for y in np.linspace(y_bottom*0.92, y_top*0.92, 9):
            ax.plot([x_left, x_span], [y, y], color="#1d3d63",
                    lw=0.4, ls=(0, (1, 8)), alpha=0.35)

        # ---- KAVİTE ZARFI (Logvinovich profili, σ-duyarlı şekil) ----
        # GEOMETRIK KISITLAMA: Doğal kavite, araç gövdesinden daha küçük ise
        # gerçekte kavite gövde duvarına temas eder ve gövdeyi sararak şekillenir.
        # Bu, araç tasarımı için kritik: doğal kavite çapı R_cavity_natural(x)
        # araç çapı R_vehicle(x)'dan küçükse, fizik açısından üç durum mümkündür:
        #   1) Tam ıslak rejim: kavite gövdeyi saramaz, kısmi kavitasyon
        #   2) Geçiş: havalandırma ile kavite içi basınç yükselir, kavite zorla
        #      gövdeyi sarar (artificial supercavity)
        #   3) Süperkavitasyon: doğal kavite zaten gövdeden büyük
        #
        # Görsel olarak: zarfı her zaman max(R_natural, R_vehicle)'e genişletiyoruz
        # ki kavite gövdenin İÇİNDE görünmesin. Bu ayrıca havalandırmanın
        # geometrik etkisini doğru gösterir.

        if Lc_i > 0.05 and Dc_i > 0.01:
            sigma_i_plot = d["sigma"][i]
            # Kavite profili: cavity_profile() zaten gövde superposition'ı
            # uyguluyor (R² = R_logv² + R_body²). Bu yüzden burada ekstra
            # clamping veya R_min uygulamasına gerek yok.
            x_prof, R_prof = cavity_profile(
                Lc_i, Dc_i, Dn, sigma_i_plot,
                D_veh=D_veh, L_veh=L_veh,
                L_taper=L_taper_v, x_body_start=x_body0,
                body_volume_correction=self.body_volume_var.get()
            )

            if len(x_prof) > 0:
                # Kavite ekseninin dikey kayması — saturating profile (fizikle aynı)
                V_i = d["V"][i]
                y_lift = cavity_axis_y_view(x_prof, V_i)

                # Doğal Logvinovich kavitesini de hesapla (referans için)
                # — bu, gövde superposition uygulanmamış halidir
                if self.body_volume_var.get():
                    x_natural, R_natural = cavity_profile(
                        Lc_i, Dc_i, Dn, sigma_i_plot,
                        body_volume_correction=False
                    )
                    # Aynı x ızgarasında olduğundan doğrudan karşılaştırılabilir
                    # forced_expansion: kavitenin gövde tarafından genişletildiği
                    # anlamına gelir (body-attached cavity rejimi)
                    forced_expansion = bool(np.any(R_prof > R_natural * 1.01))
                else:
                    R_natural = R_prof  # toggle kapalıysa zaten doğal
                    forced_expansion = False

                # Üst ve alt profilleri h(x) kadar yukarı kaydır
                upper_y = R_prof + y_lift
                lower_y = -R_prof + y_lift

                poly_x = np.concatenate([x_prof, x_prof[::-1]])
                poly_y = np.concatenate([upper_y, lower_y[::-1]])
                envelope = list(zip(poly_x, poly_y))

                # Renk: gövde-genişletilmiş kavite turuncu, doğal süperkav. mavi
                if forced_expansion:
                    fill_color = "#d4a548"
                    edge_color = "#ffcc66"
                    fill_alpha = 0.18
                else:
                    fill_color = COLORS["cavity"]
                    edge_color = "#b8e0f2"
                    fill_alpha = 0.22

                # Dolgu
                ax.add_patch(Polygon(envelope,
                    facecolor=fill_color, alpha=fill_alpha,
                    edgecolor="none", zorder=2, closed=True))
                # Kontur
                ax.add_patch(Polygon(envelope,
                    facecolor="none", edgecolor=edge_color,
                    lw=2.2, zorder=4, closed=True))

                # İç parlaklık (eğimli merkez boyunca) — sadece doğal modda
                if not forced_expansion:
                    inner_x_base = x_prof * 0.95 + Lc_i * 0.025
                    inner_lift = G * inner_x_base ** 2 / (2.0 * max(V_i, 0.5) ** 2)
                    inner_R = R_prof * 0.78
                    inner_upper = inner_R + inner_lift
                    inner_lower = -inner_R + inner_lift
                    inner_x = np.concatenate([inner_x_base, inner_x_base[::-1]])
                    inner_y = np.concatenate([inner_upper, inner_lower[::-1]])
                    ax.add_patch(Polygon(list(zip(inner_x, inner_y)),
                        facecolor="#e8f4ff", alpha=0.08,
                        edgecolor="none", zorder=3, closed=True))

                # "Gövde-genişletilmiş" durumda doğal kaviteyi gri ile göster
                if forced_expansion:
                    natural_upper = R_natural + y_lift
                    natural_lower = -R_natural + y_lift
                    ax.plot(x_prof, natural_upper, color="#7a8a9a", lw=0.8,
                            ls="--", alpha=0.7, zorder=3,
                            label="doğal kavite (gövde olmadan)")
                    ax.plot(x_prof, natural_lower, color="#7a8a9a", lw=0.8,
                            ls="--", alpha=0.7, zorder=3)
                    ax.text(x_body0 + L_veh*0.5, y_top * 0.78,
                            "✦ Gövde-genişletilmiş kavite (R² = R_doğal² + R_gövde²)",
                            color="#d4a548", fontsize=8, ha="center",
                            family="monospace", zorder=20,
                            bbox=dict(facecolor=COLORS["panel"],
                                      edgecolor="#d4a548",
                                      alpha=0.9, boxstyle="round,pad=0.3"))

                # Kavite merkez çizgisi (eksen eğrisi) — yerçekimi etkisini
                # net gösterir. Sadece kayda değer eğilmede çiz (>5% Dc).
                hg_end = y_lift[-1]
                if hg_end > 0.05 * Dc_i:
                    ax.plot(x_prof, y_lift, color="#ffcc66",
                            lw=1.0, ls=":", alpha=0.75, zorder=6)
                    ax.text(x_prof[-1], y_lift[-1],
                            f"  h_g = {hg_end*1000:.1f} mm",
                            color="#ffcc66", fontsize=7, family="monospace",
                            va="center", zorder=9)

                # Max çap konumu (eğim dahil)
                i_max = int(np.argmax(R_prof))
                x_Dmax = x_prof[i_max]
                R_Dmax = R_prof[i_max] + y_lift[i_max]

                # ZARF etiketi
                ax.annotate("KAVİTE ZARFI",
                    xy=(x_Dmax + Lc_i * 0.08, R_Dmax * 0.85),
                    xytext=(x_Dmax + Lc_i * 0.20, R_Dmax + y_half*0.15),
                    color="#b8e0f2", fontsize=8, family="monospace",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#b8e0f2", lw=0.8),
                    zorder=8)

                # Lc boyut oku
                arrow_y = -y_half * 0.55
                ax.annotate("", xy=(Lc_i, arrow_y), xytext=(0, arrow_y),
                            arrowprops=dict(arrowstyle="<->", color="#b8e0f2",
                                            lw=1.2), zorder=9)
                tick = y_half * 0.04
                ax.plot([0, 0], [arrow_y - tick, arrow_y + tick],
                        color="#b8e0f2", lw=1.2, zorder=9)
                ax.plot([Lc_i, Lc_i], [arrow_y - tick, arrow_y + tick],
                        color="#b8e0f2", lw=1.2, zorder=9)
                ax.text(Lc_i / 2, arrow_y - tick*2.5, f"Lc = {Lc_i:.2f} m",
                        color="#b8e0f2", fontsize=8, family="monospace",
                        ha="center", fontweight="bold", zorder=9)

                # Dc boyut oku — gövde-genişletilmişse zarftan, değilse Dc
                # parametresinden ölçülmüş gibi göster
                Dc_label = R_Dmax * 2 if forced_expansion else Dc_i
                ax.annotate("", xy=(x_Dmax, R_Dmax), xytext=(x_Dmax, -R_Dmax),
                            arrowprops=dict(arrowstyle="<->", color="#b8e0f2",
                                            lw=1.2), zorder=9)
                ax.text(x_Dmax + x_span * 0.015, 0,
                        f"Dc = {Dc_label:.3f} m", color="#b8e0f2",
                        fontsize=8, family="monospace",
                        ha="left", va="center", fontweight="bold", zorder=9)

        # ---- Araç gövdesi (gerçek D_veh çapı) ----
        # KOORDINAT KONVANSIYONU:
        #   x = 0      →  kavitatör ÖN yüzü (zarfın ayrılma noktası)
        #   x = cav_w  →  kavitatör arka yüzü = araç burnu
        #   x ∈ [cav_w, cav_w + L_veh]  →  araç gövdesi
        # Bu, Garabedian/Logvinovich literatür konvansiyonuyla uyumlu.
        veh_h = D_veh
        R_v_draw = D_veh / 2.0
        n_seg = 60
        seg_dx = L_veh / n_seg
        V_now = max(d["V"][i], 0.5)

        # cav_w ve x_body0 zaten yukarıda tanımlandı (kavite çizimi ile tutarlı)

        # Ana gövde silüeti — konik burun + silindirik gövde
        # Konik kısım: x ∈ [x_body0, x_body0 + L_taper]
        #   yarıçap: Dn/2 → D_veh/2 lineer
        # Silindirik kısım: x ∈ [x_body0 + L_taper, x_body0 + L_veh]
        Rn_v = Dn / 2.0
        Rmax_v = D_veh / 2.0
        if L_taper_v > 1e-6:
            # Polygon noktaları (saat yönünde, üst kenardan başla)
            body_pts = [
                (x_body0, Rn_v),                                    # burun üst
                (x_body0 + L_taper_v, Rmax_v),                      # konik son üst
                (x_body0 + L_veh, Rmax_v),                          # transom üst
                (x_body0 + L_veh, -Rmax_v),                         # transom alt
                (x_body0 + L_taper_v, -Rmax_v),                     # konik son alt
                (x_body0, -Rn_v),                                   # burun alt
            ]
        else:
            # Tam silindir
            body_pts = [
                (x_body0, Rmax_v),
                (x_body0 + L_veh, Rmax_v),
                (x_body0 + L_veh, -Rmax_v),
                (x_body0, -Rmax_v),
            ]
        # NOT: Akış-bağlı koordinat sistemi kullanılır:
        # → Gövde her zaman YATAY çizilir (akış vektörüne hizalı)
        # → Hücum açısı α: kavitenin gövdeye göre eğikliği olarak görünür
        # → Bu, "akıştan bakan" referans çerçevesidir, fiziksel olarak ekvivalent
        # NOT 2: Kuyruk DÜZ (boru gibi). Süperkavitasyon araçlarında transom
        # düz kesilir — ek görsel vurgu yok.
        ax.add_patch(Polygon(body_pts,
            facecolor=COLORS["vehicle"], edgecolor=COLORS["vehicle2"],
            lw=1.2, zorder=5, alpha=0.85, closed=True))

        # ========== KANATLAR (FIN'LER) — yan görünüş ==========
        # Yatay görüntüde: üst kanat (azim 90°) ve alt kanat (azim 270°) görünür
        # Yan kanatlar (sağ 0°, sol 180°) perspektifte sıfır kalınlık → görünmez
        # Her kanat: chord boyu × span yüksekliği, NACA 16-009 simetrik
        try:
            fins_on = self.fins_enabled_var.get()
        except Exception:
            fins_on = False
        if fins_on:
            try:
                fc = float(self.params["fin_chord"].get())
                fs = float(self.params["fin_span"].get())
                fx = float(self.params["fin_x_pos"].get())
                fd1 = float(self.params["fin_delta_1"].get())  # üst
                fd3 = float(self.params["fin_delta_3"].get())  # alt
            except (KeyError, ValueError):
                fc, fs, fx, fd1, fd3 = 0.04, 0.06, L_veh*0.85, 0.0, 0.0

            fin_x_abs = x_body0 + fx   # araç koordinatında

            # Gövde yarıçapı x_fin'de
            xs_body_loc = fx - 0.0  # araç başlangıcı = 0
            if L_taper_v > 1e-6 and xs_body_loc < L_taper_v:
                R_body_at_fin = Rn_v + (Rmax_v - Rn_v) * (xs_body_loc / L_taper_v)
            else:
                R_body_at_fin = Rmax_v

            # Kavite yarıçapı x_fin'de (sadece görsel için)
            R_cav_at_fin = 0.0
            if d is not None and d.get("Lc") is not None and self.idx < len(d["Lc"]):
                Lc_now = d["Lc"][self.idx]
                Dc_now = d["Dc"][self.idx]
                if Lc_now > 1e-6 and Dc_now > 1e-6 and fx < Lc_now:
                    xi = 2.0 * fx / Lc_now - 1.0
                    xi = max(-1.0, min(1.0, xi))
                    R_cav_at_fin = (Dc_now / 2.0) * np.sqrt(max(0.0, 1.0 - xi*xi))

            R_eff = max(R_cav_at_fin, R_body_at_fin)

            # ÜST KANAT (azimut 90°, yukarı yönde uzanır)
            r_root_up = R_body_at_fin
            r_tip_up  = R_body_at_fin + fs
            # Kanat polygon (chord × span dikdörtgen, deflection ile rotasyon)
            for sign, delta_deg in [(+1.0, fd1), (-1.0, fd3)]:
                cs_f = np.cos(np.radians(delta_deg))
                sn_f = np.sin(np.radians(delta_deg))
                # 4 köşe (root-leading, tip-leading, tip-trailing, root-trailing)
                # Kanat chord/2 önde, chord/2 arkada — x_fin merkezde
                corners_fin = [
                    (-fc/2, sign * r_root_up),  # root LE
                    ( fc/2, sign * r_root_up),  # root TE
                    ( fc/2, sign * r_tip_up),   # tip TE
                    (-fc/2, sign * r_tip_up),   # tip LE
                ]
                # Rotasyon — kanadın chord ekseni etrafında deflection
                fin_pts_world = []
                for (cx, cy) in corners_fin:
                    # δ rotasyonu chord etrafında: x değişir
                    x_rot = cx * cs_f - (cy - sign*r_root_up) * 0 * sn_f
                    # Basit: deflection görsel olarak chord eğilmesi gösterir
                    fin_pts_world.append((fin_x_abs + cx, cy))

                # Islak/kuru bölümleri farklı renkle göster
                if sign == +1:
                    # Üst kanat
                    cur_root = max(r_root_up, R_eff)   # ıslak bölge başlangıcı
                    cur_tip  = r_tip_up
                    dry_color = "#5a708a"  # kuru
                    wet_color = "#d4a548"  # ıslak (altın)
                else:
                    cur_root = -r_tip_up
                    cur_tip  = -max(r_root_up, R_eff)
                    dry_color = "#5a708a"
                    wet_color = "#d4a548"

                # Tam kanat (gri)
                ax.add_patch(Polygon(fin_pts_world,
                    facecolor=dry_color, edgecolor="#dce6f0",
                    lw=1.0, alpha=0.65, zorder=8, closed=True))

                # Islak kısım (altın renkli, üstüne)
                if sign == +1 and r_tip_up > R_eff:
                    wet_corners = [
                        (fin_x_abs - fc/2, R_eff),
                        (fin_x_abs + fc/2, R_eff),
                        (fin_x_abs + fc/2, r_tip_up),
                        (fin_x_abs - fc/2, r_tip_up),
                    ]
                    ax.add_patch(Polygon(wet_corners,
                        facecolor=wet_color, edgecolor="#dce6f0",
                        lw=1.0, alpha=0.85, zorder=9, closed=True))
                elif sign == -1 and r_tip_up > R_eff:
                    wet_corners = [
                        (fin_x_abs - fc/2, -r_tip_up),
                        (fin_x_abs + fc/2, -r_tip_up),
                        (fin_x_abs + fc/2, -R_eff),
                        (fin_x_abs - fc/2, -R_eff),
                    ]
                    ax.add_patch(Polygon(wet_corners,
                        facecolor=wet_color, edgecolor="#dce6f0",
                        lw=1.0, alpha=0.85, zorder=9, closed=True))

                # Deflection etiketi
                if abs(delta_deg) > 0.5:
                    side_label = "üst" if sign > 0 else "alt"
                    y_label = sign * (r_tip_up + y_half * 0.04)
                    ax.text(fin_x_abs, y_label,
                            f"δ_{side_label}={delta_deg:+.1f}°",
                            color=wet_color, fontsize=6, family="monospace",
                            ha="center", va="center", zorder=11)

            # Kavite-kanat kesişim işareti (R_eff seviyesi)
            if R_cav_at_fin > R_body_at_fin and R_cav_at_fin < r_tip_up:
                ax.plot([fin_x_abs - fc*0.6, fin_x_abs + fc*0.6],
                        [R_cav_at_fin, R_cav_at_fin],
                        color="#88c8ff", lw=0.8, ls=":", alpha=0.7, zorder=10)
                ax.plot([fin_x_abs - fc*0.6, fin_x_abs + fc*0.6],
                        [-R_cav_at_fin, -R_cav_at_fin],
                        color="#88c8ff", lw=0.8, ls=":", alpha=0.7, zorder=10)

        # Konik geçiş bölgesi - gri gölgeli alan ile vurgu
        if L_taper_v > 1e-6:
            taper_pts = [
                (x_body0, Rn_v),
                (x_body0 + L_taper_v, Rmax_v),
                (x_body0 + L_taper_v, -Rmax_v),
                (x_body0, -Rn_v),
            ]
            ax.add_patch(Polygon(taper_pts,
                facecolor="#3a4a60", edgecolor="none",
                alpha=0.35, zorder=6, closed=True))
            # Konik bitim noktaları işaretleri (silindirik bölge başlangıcı)
            ax.plot([x_body0 + L_taper_v, x_body0 + L_taper_v],
                    [Rmax_v, -Rmax_v], color="#7aa3c4", lw=0.8,
                    ls=":", alpha=0.6, zorder=7)
            # Konik açı etiketi
            taper_angle_deg = np.degrees(np.arctan(
                (Rmax_v - Rn_v) / L_taper_v))
            ax.annotate(
                f"Konik: L={L_taper_v*1000:.0f}mm, açı={taper_angle_deg:.1f}°",
                xy=(x_body0 + L_taper_v/2, Rmax_v),
                xytext=(x_body0 + L_taper_v/2, y_top * 0.55),
                color="#7aa3c4", fontsize=7, family="monospace",
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#7aa3c4",
                                lw=0.5, alpha=0.6),
                zorder=8)

        # Şimdi ıslak segmentleri üstüne kırmızı bantla işaretle
        # FİZİK ile aynı mantığı kullanır: kavite ekseni h_c (yerçekimi+α_eff),
        # kavite yarıçapı R_zarf² = R_logv² + R_v² (gövde superposition),
        # ıslaklık δ = R_v + h_c − R_zarf > 0
        wet_segments = []
        for k in range(n_seg):
            # Araç burnundan ölçülen yerel konum
            x_local = (k + 0.5) * seg_dx
            # Kavite koordinatındaki konum (kavitatör ön kenarı = 0)
            x_cav = x_local + x_body0
            # Kavite ekseninin bu noktadaki yüksekliği — saturating profile
            hc_seg = cavity_axis_y_view(x_cav, V_now)
            # Kavite yarıçapı — Logvinovich asimptotik formül (fizik ile uyumlu)
            if x_cav < Lc_i and Lc_i > 1e-4:
                Rn_v = Dn / 2.0
                Rmax_v = Dc_i / 2.0
                Cx_now = d["Cx"][i]
                x_open = max(10.0 * Rn_v / max(np.sqrt(Cx_now), 0.5), Rn_v * 2.0)
                x_mid_v = Lc_i * 0.5
                S = 1.0 - np.exp(-x_cav / x_open)
                if x_cav < x_mid_v:
                    D = 1.0
                else:
                    D = max(0.0, 1.0 - ((x_cav - x_mid_v) / max(Lc_i - x_mid_v, 1e-6)) ** 2)
                rc2_logv = Rn_v**2 + (Rmax_v**2 - Rn_v**2) * S * D
                rc_logv = np.sqrt(max(rc2_logv, 0.0))
            else:
                rc_logv = 0.0
            # Aracın bu kesitteki yarıçapı (konik burun + silindir)
            if L_taper_v > 1e-6 and x_local < L_taper_v:
                R_v_x = Rn_v + (Rmax_v - Rn_v) * (x_local / L_taper_v)
            else:
                R_v_x = Rmax_v
            # GÖVDE-KAVİTE REJİMİ — fizik ile aynı mantık:
            if rc_logv >= R_v_x:
                # FREE-STANDING: kavite gövdeden büyük, superposition geçerli
                rc_seg = np.sqrt(rc_logv**2 + R_v_x**2)
                # Klasik ıslaklık: alt veya üst yarımda
                delta_seg = R_v_x + abs(hc_seg) - rc_seg
                wet_here = (delta_seg > 0)
            else:
                # ATTACHED: kavite gövdeden küçük, gaz tabakası halka şeklinde
                # rc_logv pratik olarak yok ise tam ıslak
                if rc_logv < 0.05 * R_v_x:
                    wet_here = True   # tam ıslak
                else:
                    # Çevresel ıslaklık var ama görsel için 'yarı ıslak' say
                    # (kullanıcıya kavite gövdeyi tam sarmadığını göster)
                    wet_here = True
            if wet_here:
                # Plot koordinatında segment başlangıcı (cav_w ofsetli)
                wet_segments.append((x_body0 + x_local - seg_dx/2, seg_dx))

        # Islak segmentleri kırmızı bant olarak çiz
        for (xs_plot, dxs) in wet_segments:
            ax.add_patch(Rectangle(
                (xs_plot, -veh_h/2 - veh_h*0.05), dxs, veh_h*0.18,
                facecolor="#e85a4f", edgecolor="none",
                alpha=0.85, zorder=6))

        # ---- Kavitatör disk (gerçek Dn çapı, δ_c açısıyla EĞİK) ----
        # Kavitatör gövdenin önünde duran ince bir disk olarak çizilir.
        # δ_cav>0 ise üst yüzeyi öne yatık (lift yukarı yönlü).
        try:
            delta_cav_deg = self.params["delta_cav_deg"].get()
        except KeyError:
            try:
                delta_cav_deg = self.params["delta_cav"].get()
            except KeyError:
                delta_cav_deg = 0.0
        delta_cav_rad_view = np.radians(max(-30.0, min(30.0, delta_cav_deg)))
        cs = np.cos(delta_cav_rad_view)
        sn = np.sin(delta_cav_rad_view)

        # Kavitatör tipi: disk (dikdörtgen) veya konik (üçgen-yönlü)
        try:
            cav_type_view = self.cav_type_var.get()
        except (KeyError, AttributeError):
            cav_type_view = "disk"
        try:
            cone_apex_view = float(self.params["cone_apex_deg"].get())
        except KeyError:
            cone_apex_view = 90.0

        if cav_type_view == "cone" and cone_apex_view < 175.0:
            # Konik kavitatör — DOĞRU YÖNELİM:
            #   Sivri uç ÖNDE (akışa karşı, x<0)
            #   Geniş taban ARKADA (x=0, kavite ayrılma noktası)
            # β = tepe açısı (apex angle), tan(β/2) = (Dn/2) / L_cone
            half_apex = np.radians(cone_apex_view / 2.0)
            L_cone_real = (Dn / 2.0) / np.tan(half_apex)

            # GÖRSEL ASPECT RATIO DÜZELTMESİ:
            # Kavite görselinde x ekseni (boy) y ekseninden (çap) çok daha geniş.
            # Bu yüzden bir koninin gerçek açısı ekranda DEFORME görünür.
            # Pixel space'te gerçek tepe açısını koruyabilmek için L_cone'u
            # pixel oranıyla ölçeklendir:
            #   L_cone_view = L_cone_real · (px_per_unit_y / px_per_unit_x)
            try:
                p0 = ax.transData.transform((0.0, 0.0))
                p1 = ax.transData.transform((1.0, 0.0))
                p2 = ax.transData.transform((0.0, 1.0))
                px_per_x = abs(p1[0] - p0[0])
                px_per_y = abs(p2[1] - p0[1])
                if px_per_x > 1e-6 and px_per_y > 1e-6:
                    aspect_corr = px_per_y / px_per_x
                else:
                    aspect_corr = 1.0
            except Exception:
                aspect_corr = 1.0
            L_cone = L_cone_real * aspect_corr

            # Konik üçgen: sivri uç (-L_cone, 0) öne, geniş taban x=0'da
            corners_local = [
                (-L_cone, 0),       # sivri uç (önde, akışa karşı)
                (0, Dn/2),          # geniş taban üst (kavite ayrılma noktası)
                (0, -Dn/2),         # geniş taban alt
            ]
            cav_pts = [(x*cs - y*sn, x*sn + y*cs) for x, y in corners_local]
            ax.add_patch(Polygon(cav_pts,
                facecolor="#9aa8b8", edgecolor="#dce6f0", lw=1.2, zorder=7))
            # Konik tepe açısı etiketi — gerçek L_cone değeri gösterilir
            ax.text(-L_cone*1.3, Dn*0.9,
                    f"Konik: β={cone_apex_view:.0f}°, L={L_cone_real*1000:.1f}mm",
                    color="#dce6f0", fontsize=7, family="monospace",
                    fontweight="bold", ha="left", zorder=11,
                    bbox=dict(facecolor=COLORS["panel"],
                              edgecolor="#7aa3c4",
                              alpha=0.85, boxstyle="round,pad=0.2"))
        else:
            # DİSK (β ≥ 175° veya cav_type=disk):
            # Dikdörtgen — Dn x cav_w, ön yüzü x=0'da
            corners_local = [
                (0,       Dn/2),    # ön üst
                (cav_w,   Dn/2),    # arka üst
                (cav_w,  -Dn/2),    # arka alt
                (0,      -Dn/2),    # ön alt
            ]
            cav_pts = [(x*cs - y*sn, x*sn + y*cs) for x, y in corners_local]
            ax.add_patch(Polygon(cav_pts,
                facecolor="#a8bacd", edgecolor="#dce6f0", lw=1.2, zorder=7))
            # Disk etiketi (sadece cav_type cone seçili ve β=180° ise göster)
            if cav_type_view == "cone":
                ax.text(cav_w/2, Dn*0.9,
                        f"Disk: β={cone_apex_view:.0f}° (≥175°)",
                        color="#dce6f0", fontsize=7, family="monospace",
                        fontweight="bold", ha="center", zorder=11,
                        bbox=dict(facecolor=COLORS["panel"],
                                  edgecolor="#7aa3c4",
                                  alpha=0.85, boxstyle="round,pad=0.2"))

        # Ayrılma noktası işaretleri — kavite başlangıç noktaları
        # Disk: kavitatörün ön köşesi (akışın çarptığı kenar)
        # Konik: geniş tabanın arka köşesi (akışın ayrıldığı kenar)
        # Her iki durumda da plot koordinatında x=0, y=±Dn/2
        for x_sep_l, y_sep_l in [(0, Dn/2), (0, -Dn/2)]:
            xs = x_sep_l*cs - y_sep_l*sn
            ys = x_sep_l*sn + y_sep_l*cs
            ax.plot([xs], [ys], marker='o', markersize=4,
                    color="#ffcc66", markeredgecolor="#ffaa00",
                    markeredgewidth=0.8, zorder=10)

        # δ_c etiketi — sadece sıfırdan farklıysa
        if abs(delta_cav_deg) > 0.1:
            # Açı arc çizimi (kavitatörün dönüşünü görsel ifade)
            angle_text_y = (Dn/2) * cs * 1.3
            ax.text(cav_w * 0.5 + x_span*0.015,
                    angle_text_y + y_half*0.06,
                    f"δ_c = {delta_cav_deg:+.1f}°",
                    color="#ffaa66", fontsize=8, family="monospace",
                    fontweight="bold", ha="left", zorder=11,
                    bbox=dict(facecolor=COLORS["panel"],
                              edgecolor="#ffaa66",
                              alpha=0.8, boxstyle="round,pad=0.2"))

        # Hücum açısı (α) — akış vektörü etiket
        try:
            alpha_AoA_deg = self.params["alpha_aoa_deg"].get()
        except KeyError:
            alpha_AoA_deg = 0.0
        if abs(alpha_AoA_deg) > 0.1:
            # Akış vektörü oku (sol taraftan gelen)
            arrow_len = x_span * 0.05
            arrow_y_pos = -y_half * 0.55
            arrow_x_start = -x_span * 0.08
            ax.annotate("", xy=(arrow_x_start + arrow_len, arrow_y_pos),
                xytext=(arrow_x_start, arrow_y_pos),
                arrowprops=dict(arrowstyle="->", color="#5cd47b",
                                lw=2.0), zorder=11)
            ax.text(arrow_x_start, arrow_y_pos - y_half*0.05,
                    f"α = {alpha_AoA_deg:+.1f}°\nakış",
                    color="#5cd47b", fontsize=7, family="monospace",
                    fontweight="bold", ha="left", va="top", zorder=11)

        # Eğer kavite araçtan küçükse uyarı göster (süperkavitasyon yok)
        if Dc_i > 0 and Dc_i < D_veh * 0.95:
            ax.text(x_body0 + L_veh * 0.5, y_top * 0.88,
                    "⚠ Dc < Dv  →  süperkavitasyon koşulu sağlanmıyor",
                    color="#e85a4f", fontsize=8, ha="center",
                    family="monospace", zorder=20,
                    bbox=dict(facecolor=COLORS["panel"], edgecolor="#e85a4f",
                              alpha=0.85, boxstyle="round,pad=0.3"))

        # Araç etiketi
        ax.annotate("ARAÇ",
            xy=(x_body0 + L_veh * 0.5, -veh_h/2),
            xytext=(x_body0 + L_veh * 0.5, -y_half*0.85),
            color=COLORS["vehicle2"], fontsize=8, family="monospace",
            ha="center", zorder=8)

        # Kuyruk
        ax.add_patch(Polygon(
            [[x_body0 + L_veh, -veh_h/2],
             [x_body0 + L_veh + x_span*0.025, 0],
             [x_body0 + L_veh, veh_h/2]],
            facecolor=COLORS["vehicle"], edgecolor=COLORS["vehicle2"],
            lw=1.2, zorder=5))

        # ---- MOMENT REFERANS NOKTASI (kullanıcı tanımlı x_cg) ----
        # M_y bu noktaya göre hesaplanır. Araç burnundan ölçülen mesafe.
        x_cg_val = self.params["x_cg"].get()
        x_cg_val = max(0.0, min(x_cg_val, L_veh))
        x_cg_plot = x_body0 + x_cg_val

        # Kırmızı × işareti — araç merkezinde
        ax.plot([x_cg_plot], [0], marker='x', markersize=12,
                color="#ff5555", markeredgecolor="#ff5555",
                markeredgewidth=2.5, zorder=11)
        # İçi dolu küçük daire (kontrast için)
        ax.plot([x_cg_plot], [0], marker='o', markersize=4,
                color="#ffffff", markeredgecolor="#ff5555",
                markeredgewidth=1.5, zorder=12)
        # Etiket
        ax.annotate(f"x_cg = {x_cg_val:.3f} m",
            xy=(x_cg_plot, 0),
            xytext=(x_cg_plot, y_half * 0.45),
            color="#ff5555", fontsize=8, family="monospace",
            fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#ff5555", lw=0.8),
            zorder=11)

        # Rejim rozeti
        sigma_i = d["sigma"][i]
        regime_name, regime_color = regime_of(sigma_i)
        ax.text(x_left + x_span*0.02, y_top*0.92, "● " + regime_name.upper(),
                color=regime_color, fontsize=10, family="monospace",
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000",
                          edgecolor=regime_color, alpha=0.75, lw=1))

        # Anlık değerler
        info = f"t = {d['t'][i]:5.2f} s    V = {d['V'][i]:6.1f} m/s    σ = {sigma_i:6.3f}"
        ax.text(x_span*0.98, y_top*0.92, info,
                color=COLORS["text"], fontsize=9, family="monospace",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000",
                          edgecolor=COLORS["border"], alpha=0.7))

        # Kaplama yüzdesi
        if d["cover"][i] > 0.01:
            ax.text(x_body0 + L_veh/2, -y_half*0.72,
                    f"KAPLAMA  {d['cover'][i]*100:.0f}%",
                    color=regime_color, fontsize=9, family="monospace",
                    ha="center", fontweight="bold")

        # Başlık
        ax.text(-x_span*0.11, y_half*0.97,
                "YANAL KESİT · ARAÇ ve KAVİTE ZARFI  (ölçek 1:1)",
                color=COLORS["mute"], fontsize=8, family="monospace", ha="left")

    def _update_frame(self):
        try:
            self._update_frame_impl()
        except Exception as e:
            # Sessiz Tk callback'inde kaybolmasını önle
            import traceback
            print(f"[_update_frame HATA] {type(e).__name__}: {e}")
            traceback.print_exc()

    def _update_frame_impl(self):
        d = self.data
        i = min(self.idx, len(d["t"]) - 1)
        t_now = d["t"][i]

        self._draw_vehicle()

        for vline in [self.vline_vsigma, self.vline_cavity,
                      self.vline_drag, self.vline_cover, self.vline_vent,
                      self.vline_force]:
            vline.set_xdata([t_now, t_now])

        self.canvas.draw_idle()

        m = self.metric_labels
        m["t"].config(text=f"{d['t'][i]:.2f}")
        m["V"].config(text=f"{d['V'][i]:.1f}")
        m["sigma"].config(text=f"{d['sigma'][i]:.3f}")
        m["Lc"].config(text=f"{d['Lc'][i]:.2f}")
        m["Dc"].config(text=f"{d['Dc'][i]:.3f}")
        m["Fd"].config(text=f"{d['Fd'][i]:.0f}")
        m["Cx"].config(text=f"{d['Cx'][i]:.3f}")
        m["a"].config(text=f"{d['a'][i]:.1f}")
        m["x"].config(text=f"{d['x'][i]:.1f}")

        # Havalandırma metrikleri (hibrit modelden)
        pc_kPa = d["pc"][i] / 1000.0
        m["pc"].config(text=f"{pc_kPa:.2f}")
        m["Cq"].config(text=f"{d['Cq'][i]:.3f}")
        m["Fr"].config(text=f"{d['Fr'][i]:.1f}")
        m["beta"].config(text=f"{d['beta'][i]:.2f}")
        # Kavite sapması (yerçekimi) — m'den mm'ye çevir
        m["hg"].config(text=f"{d['hg'][i] * 1000:.2f}")

        # YENİ: Kavite dışında miktar = |hg| - (Dc-D_veh)/2
        # Pozitif → araç kavite dışına çıkmış (kuyrukta suyla temas)
        # Negatif → araç hala kavite içinde (fazlalık = -değer kadar)
        try:
            D_veh_val = float(self.params["veh_diam"].get())
            Dc_i = float(d["Dc"][i])
            cav_margin = (Dc_i - D_veh_val) / 2.0     # kavite fazlalık yarıçapı [m]
            hg_abs = abs(float(d["hg"][i]))            # kavite kayması [m]
            out_amt = (hg_abs - cav_margin) * 1000.0   # dışarıda miktarı [mm]
            m["cavity_clear"].config(text=f"{out_amt:+.1f}")
            if out_amt > 0:
                # ARAÇ KAVİTE DIŞINDA - kırmızı vurgu
                m["cavity_clear"].config(fg="#ff5040")
            else:
                # ARAÇ HALA İÇERİDE - yeşil
                m["cavity_clear"].config(fg="#40b070")
        except Exception:
            pass
        # Yerçekimi asimetrisinden doğan dikey kuvvet ve moment
        m["Fz"].config(text=f"{d['Fz'][i]:.0f}")
        m["F_planing"].config(text=f"{d['F_planing'][i]:.0f}")
        m["F_buoy"].config(text=f"{d['F_buoy'][i]:.0f}")
        m["F_cavz"].config(text=f"{d['F_cavz'][i]:.0f}")
        m["F_body_lift"].config(text=f"{d['F_body_lift'][i]:.0f}")
        m["F_body_drag"].config(text=f"{d['F_body_drag'][i]:.0f}")
        m["F_grav"].config(text=f"{d['F_grav'][i]:.0f}")
        m["V_sub"].config(text=f"{d['V_sub'][i]*1000:.2f}")
        m["My"].config(text=f"{d['My'][i]:.1f}")
        m["My_no_grav"].config(text=f"{d['My_no_grav'][i]:.1f}")
        # Kanat verileri (eğer label varsa)
        if "F_fin_total" in m and "F_fin_total" in d:
            m["F_fin_total"].config(text=f"{d['F_fin_total'][i]:.0f}")
        if "M_fin" in m and "M_fin" in d:
            m["M_fin"].config(text=f"{d['M_fin'][i]:.1f}")
        if "F_fin_drag" in m and "F_fin_drag" in d:
            m["F_fin_drag"].config(text=f"{d['F_fin_drag'][i]:.0f}")

        # β > 2.645 → Paryshev kararsızlık sınırı aşıldı
        # Uyarı olarak β rengini kırmızı yap
        if d["beta"][i] > 2.645:
            m["beta"].config(fg=COLORS["drag"])
        else:
            m["beta"].config(fg="#e07b3a")

        regime_name, regime_color = regime_of(d["sigma"][i])

        # Hibrit mod göstergesi: havalandırma aktif mi?
        gas_on = self.params["gas_flow"].get() > 1e-6
        if gas_on:
            # Kavite içi basınç buhar basıncından belirgin yüksekse yapay baskın
            if d["pc"][i] > P_VAP * 1.5:
                regime_name = f"{regime_name} · YAPAY"
            else:
                regime_name = f"{regime_name} · HİBRİT"
        self.regime_label.config(text=regime_name, fg=regime_color)

        # Geometrik durum — Lc/L_veh, ıslak alan, h_g/Dc oranları
        L_v = self.params["veh_len"].get()
        Lc_ratio = d["Lc"][i] / max(L_v, 0.01)
        wet_pct = (1.0 - d["cover"][i]) * 100
        Dc_i = d["Dc"][i]
        Lc_Dc = d["Lc"][i] / max(Dc_i, 0.001) if Dc_i > 0 else 0.0
        hg_over_Dc = d["hg"][i] / max(Dc_i, 0.001) * 100 if Dc_i > 0 else 0.0
        # Aktif geometri model adı
        gm_name = self.geom_model_var.get()
        self.geom_label.config(
            text=f"[{gm_name}]  Lc/Dc = {Lc_Dc:>4.1f}  ·  Lc/L = {Lc_ratio:>4.1f}  "
                 f"·  ıslak = %{wet_pct:>4.1f}  ·  h_g/Dc = %{hg_over_Dc:>4.1f}")

        # Slider pozisyonunu animasyona göre güncelle (re-entrance engelli)
        if self.playing:
            self._internal_slider_update = True
            try:
                self.time_var.set(i)
            finally:
                self._internal_slider_update = False

    def _animate(self):
        try:
            if self.playing and self.data is not None:
                if self.idx < len(self.data["t"]) - 1:
                    N = len(self.data["t"])
                    step = max(1, int(N / 300))
                    self.idx = min(self.idx + step, N - 1)
                    self._update_frame()
                else:
                    self.playing = False
                    self.btn_play.config(text="▶ BAŞLAT")
        except Exception as e:
            import traceback
            print(f"[_animate HATA] {type(e).__name__}: {e}")
            traceback.print_exc()
            # Animasyon hata yapsa bile zinciri kırmayalım

        # ÖNEMLİ: zinciri her durumda devam ettir
        self.root.after(40, self._animate)


# ============================================================================
# ÇALIŞTIR
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SupercavitationApp(root)
    root.mainloop()
