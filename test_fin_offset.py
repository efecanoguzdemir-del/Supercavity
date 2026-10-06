"""
Sapmis kavite ekseninde kanat islakligi ve K (roll) momenti.

  python test_fin_offset.py     (exit 0 = gecti)

Kapsam:
  A. fin_wetted_segment geometrisi (es merkezli dalla birebir uyum, iki parcali islaklik,
     elle hesaplanmis kesisimler, sinirlar, sureklilik)
  B. cavity_axis_offset_components (isaret, olcek, yercekimi anahtari)
  C. VehicleModel: K'nin simetri kurallari (tek eksen roll uretmez, caprazlar uretir),
     legacy_exact birebir korunur, tahminci ile ortak yol
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from configs import case1_config as cfg
from src.control.estimator import CavityEstimator
from src.dynamics.model import VehicleModel, ControlInput
from src.hydrodynamics.cavity import cavity_axis_offset_components
from src.hydrodynamics.fins import fin_immersion_ratio, fin_wetted_segment

PASS = 0
FAIL = 0


def check(name, ok, info=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  [OK]   " + name)
    else:
        FAIL += 1
        print("  [HATA] " + name + "   " + info)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


R_ROOT = 0.15
SPAN = 0.25
R_TIP = R_ROOT + SPAN

# ---------------------------------------------------------------- A. geometri
print("\nA. fin_wetted_segment geometrisi")

# A1 - h = 0: eski es merkezli yolla birebir ayni (tum kavite yariciplarinda)
worst = 0.0
for R_c in np.linspace(0.0, 0.6, 241):
    for az in (0.0, 45.0, 90.0, 180.0, 270.0):
        w, r = fin_wetted_segment(R_c, R_ROOT, SPAN, np.radians(az), 0.0, 0.0)
        w_ref = fin_immersion_ratio(max(R_c, R_ROOT), R_ROOT, SPAN)
        r_ref = R_TIP - 0.5 * w_ref
        worst = max(worst, abs(w - w_ref), abs(r - r_ref))
check("h=0 -> fin_immersion_ratio + legacy r_mid ile birebir", worst == 0.0,
      "en buyuk fark %.3e" % worst)

# A2 - kavite kanadin radyal dogrultusunda kayarsa: o kanat kurur, karsiti islanir
R_C = 0.20
w_in, _ = fin_wetted_segment(R_C, R_ROOT, SPAN, 0.0, 0.05, 0.0)       # az=0 (+y)
w_out, _ = fin_wetted_segment(R_C, R_ROOT, SPAN, np.pi, 0.05, 0.0)    # az=180 (-y)
w_sym, _ = fin_wetted_segment(R_C, R_ROOT, SPAN, 0.0, 0.0, 0.0)
check("radyal kayma: kavitenin gittigi kanat az, karsiti cok islanir",
      w_in < w_sym < w_out, "%.4f < %.4f < %.4f" % (w_in, w_sym, w_out))

# A3 - kayma kanada DIK ise islaklik simetrik kalir (roll uretmeyen yon)
w_a, _ = fin_wetted_segment(R_C, R_ROOT, SPAN, 0.0, 0.0, 0.05)
w_b, _ = fin_wetted_segment(R_C, R_ROOT, SPAN, np.pi, 0.0, 0.05)
check("dik kayma: karsit kanatlar esit islanir", close(w_a, w_b),
      "%.6f vs %.6f" % (w_a, w_b))

# A4 - elle hesap: az=0 (e = +y), C = (0.10, 0), R_c = 0.12
#      kuru aralik s in [-0.02, 0.22] -> islak [0.22, 0.40] = 0.18 m, merkez 0.31
w, r = fin_wetted_segment(0.12, R_ROOT, SPAN, 0.0, 0.10, 0.0)
check("elle hesap (tek parca, uc tarafi islak)", close(w, 0.18) and close(r, 0.31),
      "w=%.6f (0.18), r=%.6f (0.31)" % (w, r))

# A5 - iki parcali islaklik: kavite kanadin ORTA bandini orter
#      az=0, C = (0.25, 0), R_c = 0.05 -> kuru [0.20, 0.30]
#      islak [0.15, 0.20] + [0.30, 0.40] = 0.05 + 0.10 = 0.15
w, r = fin_wetted_segment(0.05, R_ROOT, SPAN, 0.0, 0.25, 0.0)
r_exp = (0.175 * 0.05 + 0.35 * 0.10) / 0.15
check("iki parcali islaklik (kok + uc suda, orta kuru)",
      close(w, 0.15) and close(r, r_exp), "w=%.6f (0.15), r=%.6f (%.6f)" % (w, r, r_exp))

# A6 - isin kaviteyi hic kesmiyor (dik uzaklik > R_c) -> tamamen islak
w, r = fin_wetted_segment(0.05, R_ROOT, SPAN, 0.0, 0.0, 0.20)
check("isin kaviteyi kesmiyor -> tam islak",
      close(w, SPAN) and close(r, R_TIP - 0.5 * SPAN), "w=%.6f" % w)

# A7 - sinirlar: her zaman 0 <= wet <= span, r > 0
bad = []
rng = np.random.default_rng(7)
for _ in range(20000):
    R_c = rng.uniform(0.0, 0.8)
    hy, hz = rng.uniform(-0.5, 0.5, 2)
    az = rng.uniform(0.0, 2.0 * np.pi)
    w, r = fin_wetted_segment(R_c, R_ROOT, SPAN, az, hy, hz)
    if not (-1e-15 <= w <= SPAN + 1e-15 and r > 0.0 and np.isfinite(r)):
        bad.append((R_c, hy, hz, az, w, r))
check("20000 rastgele durumda sinirlar korunur", not bad, "%d ihlal" % len(bad))

# A8 - sureklilik. Kanat isini kavite cemberine TEGET gecerken kuru aralik 2*sqrt(disc)
# ile acilir: fonksiyon surekli ama o noktada DIKEY TEGETLI (sonsuz egim) - cember
# kirisinin gercek geometrisi (projede kavite profilinde de ayni desen var).
# Dolayisiyla iddia Lipschitz degil "sicrama yok": izgara 4x incelirse en buyuk adim
# kuculmeli (sicrama olsa sabit kalirdi; sqrt tepesinde oran ~2, duzgun yerde ~4).
def max_step(az_deg, n, lo=-0.4, hi=0.4):
    hs = np.linspace(lo, hi, n)
    ws = np.array([fin_wetted_segment(0.20, R_ROOT, SPAN, np.radians(az_deg), h, 0.0)[0]
                   for h in hs])
    return np.abs(np.diff(ws)).max(), hs[1] - hs[0]


ratios = []
for az in (0.0, 30.0, 60.0, 90.0):
    a, _ = max_step(az, 4001)
    b, _ = max_step(az, 16001)
    ratios.append(a / max(b, 1e-18))
check("izgara 4x incelince adim kuculur -> sicrama yok (en kotu oran >= 1.8)",
      min(ratios) >= 1.8, "oranlar " + ", ".join("%.2f" % v for v in ratios))

# Calisma araliginda (|h| <= 0.1 m; gercek sapma 2 derecede 0.023 m) teget noktasina
# hic girilmez -> orada Lipschitz (|dw| <= dh) olmali.
worst_ratio = 0.0
for az in (0.0, 30.0, 60.0, 90.0, 135.0):
    a, dh = max_step(az, 4001, -0.1, 0.1)
    worst_ratio = max(worst_ratio, a / dh)
check("calisma araliginda (|h| <= 0.1 m) Lipschitz: |dw| <= dh", worst_ratio <= 1.0 + 1e-9,
      "en kotu |dw|/dh = %.3f" % worst_ratio)

# ---------------------------------------------------------------- B. sapma
print("\nB. cavity_axis_offset_components")

X, V_REF, K_DEV, X_OPEN = 3.8, 30.0, 0.4, 1.49
SHAPE = X_OPEN * (1.0 - np.exp(-X / X_OPEN))

hy, hz = cavity_axis_offset_components(X, 0.0, 0.0, V_REF, K_DEV, X_OPEN)
check("aci yok, yercekimi kapali -> sapma yok", close(hy, 0.0) and close(hz, 0.0))

hy, hz = cavity_axis_offset_components(X, np.radians(2.0), 0.0, V_REF, K_DEV, X_OPEN)
check("alpha_eff>0 -> kavite asagi (h_z>0), yan sapma yok",
      close(hz, K_DEV * np.radians(2.0) * SHAPE) and close(hy, 0.0), "h_z=%.6f" % hz)

hy2, hz2 = cavity_axis_offset_components(X, 0.0, np.radians(2.0), V_REF, K_DEV, X_OPEN)
check("beta>0 -> kavite sancaga (h_y>0); yaw pitch'in simetrigi",
      close(hy2, hz) and close(hz2, 0.0), "h_y=%.6f vs h_z(alpha)=%.6f" % (hy2, hz))

hy3, _ = cavity_axis_offset_components(X, 0.0, -np.radians(2.0), V_REF, K_DEV, X_OPEN)
check("beta isareti ters -> h_y isareti ters", close(hy3, -hy2))

_, hz_g = cavity_axis_offset_components(X, 0.0, 0.0, V_REF, K_DEV, X_OPEN,
                                        include_gravity=True)
check("yercekimi anahtari: h_z = g x^2/(2V^2)",
      close(hz_g, 9.81 * X * X / (2 * V_REF * V_REF)), "%.6f" % hz_g)
check("yercekimi sagi aci teriminin en az 3 kati (baskin, isareti cozulmedi)",
      hz_g > 3.0 * hz, "yercekimi %.4f vs aci %.4f" % (hz_g, hz))

_, hz0 = cavity_axis_offset_components(0.0, np.radians(5.0), 0.0, V_REF, K_DEV, X_OPEN,
                                       include_gravity=True)
check("x=0'da sapma yok (kavitator agzi)", close(hz0, 0.0))

hy4, hz4 = cavity_axis_offset_components(X, np.radians(2.0), 0.0, V_REF, K_DEV, None)
check("x_open yok -> aci terimi 0", close(hz4, 0.0) and close(hy4, 0.0))

# ---------------------------------------------------------------- C. model
print("\nC. VehicleModel: K (roll) simetri kurallari")

M_ON = VehicleModel(dict(cfg.VEHICLE))
M_OFF = VehicleModel(dict(cfg.VEHICLE, fin_cavity_offset=False))
M_LEG = VehicleModel(dict(cfg.VEHICLE, legacy_exact=True, gas_flow_ref_depth=None))


def ev(model, alpha_deg=0.0, beta_deg=0.0, de=0.0, dr=0.0, dc=0.0,
       V=30.0, Lc=6.0, Dc=0.5, pc=95000.0, p=0.0, q=0.0, r=0.0):
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    xs = np.zeros(15)
    xs[0] = V * np.cos(a) * np.cos(b)
    xs[1] = V * np.sin(b)
    xs[2] = V * np.sin(a) * np.cos(b)
    xs[3] = p
    xs[4] = q
    xs[5] = r
    xs[11] = 10.0
    xs[12] = Lc
    xs[13] = Dc
    xs[14] = pc
    u = ControlInput(delta_e=np.radians(de), delta_r=np.radians(dr),
                     delta_c=np.radians(dc), thrust=0.0, gas_flow=125 * 60)
    return model.evaluate(0.0, xs, u)[1]


check("varsayilan: surekli modelde acik, legacy_exact'te kapali",
      M_ON.fin_cav_offset and not M_LEG.fin_cav_offset and not M_OFF.fin_cav_offset)
check("yercekimi sagi varsayilan kapali", not M_ON.fin_cav_offset_gravity)

# C1 - es merkezli varsayim K'yi her zaman sondurur (eski davranis)
worst = 0.0
for ad in (-4.0, 0.0, 3.0):
    for bd in (-4.0, 0.0, 3.0):
        for de in (-3.0, 0.0, 2.0):
            for dr in (-3.0, 0.0, 2.0):
                worst = max(worst, abs(ev(M_OFF, ad, bd, de, dr)["M_body"][0]))
check("kapaliyken K ~ 0 (arti duzen simetrisi)", worst < 1e-9, "|K|max=%.3e" % worst)

# C2 - tek eksen uyarimi roll URETMEZ (pitch ve yaw duzlemleri birbirine dik)
singles = [("alpha", dict(alpha_deg=3.0)), ("beta", dict(beta_deg=3.0)),
           ("de", dict(de=3.0)), ("dr", dict(dr=3.0)), ("dc", dict(dc=3.0)),
           ("q", dict(q=0.5)), ("r", dict(r=0.5))]
bad = []
for n, kw in singles:
    K = ev(M_ON, **kw)["M_body"][0]
    if abs(K) > 1e-9:
        bad.append((n, K))
check("tek eksen (alpha|beta|de|dr|dc|q|r) roll uretmez", not bad, str(bad))

# C3 - capraz uyarim roll URETIR
K_ar = ev(M_ON, alpha_deg=2.0, dr=2.0)["M_body"][0]
K_bd = ev(M_ON, beta_deg=2.0, de=2.0)["M_body"][0]
check("alpha x dr caprazi roll uretir", abs(K_ar) > 1.0, "K=%.3f N m" % K_ar)
check("beta x de caprazi roll uretir", abs(K_bd) > 1.0, "K=%.3f N m" % K_bd)
check("iki caprazin isareti zit, buyuklugu ayni (pitch-yaw simetrisi)",
      close(K_ar, -K_bd, 1e-9), "%.6f vs %.6f" % (K_ar, K_bd))

# C4 - beta isaret cevrimi K'yi cevirir (ayna simetrisi)
K_p = ev(M_ON, alpha_deg=2.0, beta_deg=2.0, dr=2.0)["M_body"][0]
K_m = ev(M_ON, alpha_deg=2.0, beta_deg=-2.0, dr=-2.0)["M_body"][0]
check("(beta, dr) -> -(beta, dr): K -> -K", close(K_p, -K_m, 1e-9),
      "%.6f vs %.6f" % (K_p, K_m))

# C5 - kanat roll sonumu hala baskin ve dogru isaretli
K_p1 = ev(M_ON, p=1.0)["M_body"][0]
check("p=1 rad/s -> roll sonumu negatif ve guclu (|K|>100)", K_p1 < -100.0,
      "K=%.2f N m" % K_p1)

# C6 - kavite yoksa (tam islak) dort kanat tam islak -> sapma etkisiz
d_on = ev(M_ON, alpha_deg=2.0, dr=2.0, Lc=0.0, Dc=0.0, pc=2340.0, V=20.0)
d_off = ev(M_OFF, alpha_deg=2.0, dr=2.0, Lc=0.0, Dc=0.0, pc=2340.0, V=20.0)
check("kavite yok -> sapma farki yok (dort kanat tam islak)",
      close(d_on["M_body"][0], d_off["M_body"][0], 1e-9)
      and close(d_on["fin_wet_span"], M_ON.fin_span),
      "K %.3e vs %.3e" % (d_on["M_body"][0], d_off["M_body"][0]))

# C7 - legacy_exact birebir korunur (sapma kapali -> eski kod yolu)
M_LEG_OLD = VehicleModel(dict(cfg.VEHICLE, legacy_exact=True, gas_flow_ref_depth=None,
                              fin_cavity_offset=False))
worst = 0.0
for ad in (-3.0, 0.0, 2.0):
    for bd in (-3.0, 0.0, 2.0):
        a = ev(M_LEG, ad, bd, 1.0, -1.0)
        b = ev(M_LEG_OLD, ad, bd, 1.0, -1.0)
        for k in ("F_fin_total", "F_fin_side", "F_fin_drag", "fin_wet_span"):
            worst = max(worst, abs(a[k] - b[k]))
        worst = max(worst, np.abs(np.asarray(a["M_body"])
                                  - np.asarray(b["M_body"])).max())
check("legacy_exact: kanat kuvvet/moment ve islaklik birebir", worst == 0.0,
      "en buyuk fark %.3e" % worst)

# C8 - ortalama islak aciklik (otopilot eta'si) sapmasiz degerden cok sapmamali
d_on = ev(M_ON, alpha_deg=2.0, beta_deg=2.0)
d_off = ev(M_OFF, alpha_deg=2.0, beta_deg=2.0)
rel = abs(d_on["fin_wet_span"] - d_off["fin_wet_span"]) / d_off["fin_wet_span"]
check("ortalama islak aciklik (eta) %5'ten az kayar", rel < 0.05,
      "kayma %%%.2f" % (100 * rel))

# C9 - tani tutarliligi
d = ev(M_ON, alpha_deg=2.0, beta_deg=1.0)
check("tani: fin_wet_spans 4 eleman, ortalamasi fin_wet_span",
      len(d["fin_wet_spans"]) == 4
      and close(float(np.mean(d["fin_wet_spans"])), d["fin_wet_span"], 1e-12))
check("tani: cav_h_y/cav_h_z isaretleri (beta>0 -> h_y>0, alpha>0 -> h_z>0)",
      d["cav_h_y"] > 0.0 and d["cav_h_z"] > 0.0,
      "h_y=%.5f h_z=%.5f" % (d["cav_h_y"], d["cav_h_z"]))

# C10 - sureklilik: beta supurmesinde K ve kanat kuvveti ziplamaz
bs = np.linspace(-6.0, 6.0, 1201)
Ks = np.array([ev(M_ON, 2.0, b, 0.0, 2.0)["M_body"][0] for b in bs])
Fz = np.array([ev(M_ON, 2.0, b, 0.0, 2.0)["F_fin_total"] for b in bs])
dK = np.abs(np.diff(Ks))
dF = np.abs(np.diff(Fz))
check("beta supurulurken K surekli", dK.max() < 10.0 * np.median(dK) + 1e-6,
      "max adim %.4e, medyan %.4e" % (dK.max(), np.median(dK)))
check("beta supurulurken kanat dikey kuvveti surekli",
      dF.max() < 10.0 * np.median(dF) + 1e-6,
      "max adim %.4e, medyan %.4e" % (dF.max(), np.median(dF)))

# C11 - tahminci modelle AYNI geometri yolunu kullanir
est = CavityEstimator(cfg.VEHICLE)
est.Lc, est.Dc, est.pc = 6.0, 0.5, 95000.0
est.s_raw = 0.2
_, Cx = M_ON.cavitator_Cx_geometry(est.s_raw, np.hypot(np.radians(2.0), np.radians(1.0)))
spans_m, _, _, _ = M_ON.fin_wet_geometry(6.0, 0.5, np.radians(2.0), np.radians(1.0),
                                         30.0, Cx)
eta_est = est._wet_span(np.radians(2.0), np.radians(1.0), 30.0)
check("tahminci _wet_span = model fin_wet_geometry ortalamasi",
      close(eta_est, float(spans_m.mean()), 1e-12),
      "%.9f vs %.9f" % (eta_est, float(spans_m.mean())))

# ---------------------------------------------------------------- sonuc
print("\n" + "=" * 60)
print("SONUC: %d gecti, %d basarisiz  (%d/%d)" % (PASS, FAIL, PASS, PASS + FAIL))
print("=" * 60)
sys.exit(1 if FAIL else 0)
