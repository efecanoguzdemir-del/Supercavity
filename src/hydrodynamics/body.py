"""
================================================================================
 Body Wetting Model — gövde ıslanma, sürtünme, Arşimet ve gövde α-lift/drag
================================================================================
Legacy kaynağı: supercavitation_gui_LIVE_v10.py, simulate() içindeki kesit
döngüsü (satır ~686-946), tam-ıslak dalı (~1167-1238) ve gövde lift üzerine
yazma bloğu (~1350-1361). Global durum yoktur; tüm fonksiyonlar saftır.

İŞARET KONVANSİYONU (legacy ile BİREBİR, değiştirilmedi):
  * x: araç BURNUNDAN kuyruğa doğru ölçülür [m] (nose-referanslı, arkaya +).
  * Dikey kuvvetler YUKARI POZİTİF (F_buoy > 0 yukarı, F_body_lift > 0 yukarı).
    Bu gövde-ekseni z-aşağı DEĞİLDİR; dönüşüm (F_z_body = −F_up) başka modülün işi.
  * Moment: M = (x_cg − x_force) · F_up. x_force < x_cg (önde) ve F_up > 0
    → M > 0 (burun yukarı).
  * Drag bileşenleri (F_skin, F_body_drag) pozitif büyüklüktür (geri yönlü).
  * alpha (alpha_aoa) radyan, burun yukarı pozitif.

Gövde lift hakkında önemli not:
  Legacy kesit döngüsünde Munk + Hoerner crossflow integrali hesaplanır, ancak
  satır 1354'te bu değer basit Munk formülü ile ÜZERİNE YAZILIR:
      F_body_lift = CL_alpha_body · q · A_max · alpha · wet_frac
      F_body_drag = |F_body_lift · alpha|
      M_body_lift = (x_cg − L/2) · F_body_lift
  Legacy'nin çıktı dizilerindeki (F_body_lift, F_body_drag) değerler bu
  üzerine yazılmış formüldür. Bu modül:
      F_body_lift / F_body_drag / M_body_lift      → legacy efektif (üzerine yazılmış)
      F_body_lift_section / _drag_section / M_...  → kesit integrali (legacy ölü kod)
  döndürür.
================================================================================
"""

import numpy as np

from .constants import (
    RHO, G, CF_SKIN, CDC_CROSSFLOW_SECTION, N_BODY_SECTIONS,
    CL_ALPHA_BODY_DEFAULT, CDC_BODY_DEFAULT, K_DEV_DEFAULT, BODY_VOLUME_CORRECTION,
    ARC_FREE_BLEND, NU_WATER, FORM_FACTOR_DEFAULT, RE_MIN_FRICTION,
)
from .cavity import cavity_axis_offset
from .smooth import smoothstep

# Legacy eşikleri (fiziksel sabit değil, sayısal koruma)
_V_AXIS_MIN = 0.5          # bu hızın altında kavite ekseni sapması 0 alınır
_CAV_EXIST_MIN = 1e-4      # Lc, Dc bu değerin altında → kavite yok (tam ıslak)
_ATTACHED_CLOSED_FRAC = 0.05


# ============================================================================
# SÜRTÜNME KATSAYISI
# ============================================================================

def skin_friction_ittc(V, L_ref, form_factor=FORM_FACTOR_DEFAULT):
    """
    ITTC-57 korelasyon hattı + form/pürüzlülük payı:  Cf = (1+k)·0.075/(log10(Re) − 2)²

    Re = V·L_ref/ν (ν = NU_WATER ≈ 1.161e-6 m²/s, 15°C deniz suyu). Legacy sabit
    Cf = 0.003 Re bağımsızdır ve bu araçta belirgin yüksektir: L = 4 m'de V = 20 m/s
    (Re = 6.9e7) → ITTC 0.00220, V = 40 m/s (Re = 1.4e8) → 0.00199.

    (1+k) hem gövde formunu (Hoerner, L/D ≈ 13 için ~1.03) hem pürüzlülük payını
    (ITTC ΔCf ≈ 3e-4) kapsar; varsayılan 1.1. Re < RE_MIN_FRICTION'da formül
    RE_MIN_FRICTION'da dondurulur (V → 0 koruması; q → 0 olduğu için kuvvet yine 0'a
    sürekli iner).

    Args:
        V: Hız [m/s]
        L_ref: Referans uzunluk (gövde boyu) [m]
        form_factor: (1+k) form + pürüzlülük çarpanı

    Returns:
        Cf: Sürtünme katsayısı [-] (ıslak alana göre)
    """
    Re = max(abs(float(V)) * max(float(L_ref), 1e-6) / NU_WATER, RE_MIN_FRICTION)
    return float(form_factor) * 0.075 / (np.log10(Re) - 2.0) ** 2


# ============================================================================
# GEOMETRİ
# ============================================================================

def vehicle_radius(x_local, R_n, R_v_max, L_taper):
    """
    Araç burnundan x_local [m] uzakta gövde yarıçapı (konik burun + silindir).
      x < L_taper : R = R_n + (R_v_max − R_n)·x/L_taper
      x ≥ L_taper : R = R_v_max
    """
    if L_taper <= 1e-6:
        return R_v_max
    if x_local < L_taper:
        return R_n + (R_v_max - R_n) * (x_local / L_taper)
    return R_v_max


def body_surface_area(R_n, R_v_max, L_veh, L_taper):
    """Toplam yan yüzey alanı [m²]: kesik koni burun + silindir."""
    if L_taper > 1e-6:
        slant = np.sqrt(L_taper ** 2 + (R_v_max - R_n) ** 2)
        return np.pi * (R_n + R_v_max) * slant + 2.0 * np.pi * R_v_max * (L_veh - L_taper)
    return 2.0 * np.pi * R_v_max * L_veh


def body_volume(R_n, R_v_max, L_veh, L_taper):
    """Toplam gövde hacmi [m³]: kesik koni burun + silindir."""
    if L_taper > 1e-6:
        V_taper = (np.pi * L_taper / 3.0) * (R_n ** 2 + R_n * R_v_max + R_v_max ** 2)
        return V_taper + np.pi * R_v_max ** 2 * (L_veh - L_taper)
    return np.pi * R_v_max ** 2 * L_veh


def cavity_opening_length(R_n, Cx):
    """Logvinovich açılma ölçeği x_open = max(10·Rn/max(√Cx,0.5), 2·Rn)."""
    return max(10.0 * R_n / max(np.sqrt(max(Cx, 0.0)), 0.5), R_n * 2.0)


# ============================================================================
# KESİT KAVİTE YARIÇAPI
# ============================================================================

def cavity_radius_logvinovich(x_cav, Lc, Dc, R_n, Cx, smooth_closure=False):
    """
    Logvinovich asimptotik profil: R²(x) = Rn² + (Rmax² − Rn²)·S(x)·D(x)
      S = 1 − exp(−x/x_open)                       (σ-bağımsız açılma)
      D = 1 (x < Lc/2), 1 − ((x−Lc/2)/(Lc/2))² aksi (parabolik kapanma, ≥0)
    x_cav ≥ Lc ise 0 döner (kavite dışı).

    Legacy (smooth_closure=False): kapanma ucunda R → Rn, x = Lc'de 0'a SIÇRAR
    (her gövde kesitinde ıslaklık/sürtünme basamağı üretir).
    smooth_closure=True: kapanma yarısında R² = (Rn² + (Rmax² − Rn²)·S)·D → x→Lc'de R→0
    sürekli. Açılma yarısı (x < Lc/2) legacy ile aynı; fark sadece R ~ Rn olduğu uçta.
    """
    if x_cav >= Lc:
        return 0.0
    Rmax = Dc / 2.0
    x_open = cavity_opening_length(R_n, Cx)
    x_mid = 0.5 * Lc
    S = 1.0 - np.exp(-x_cav / x_open)
    if x_cav < x_mid:
        D = 1.0
    else:
        D = max(0.0, 1.0 - ((x_cav - x_mid) / max(Lc - x_mid, 1e-6)) ** 2)
    if smooth_closure:
        rc2 = (R_n ** 2 + (Rmax ** 2 - R_n ** 2) * S) * D
    else:
        rc2 = R_n ** 2 + (Rmax ** 2 - R_n ** 2) * S * D
    return float(np.sqrt(max(rc2, 0.0)))


def cavity_radius_at_section(x_cav, R_v, Lc, Dc, R_n, Cx,
                             body_volume_correction=BODY_VOLUME_CORRECTION,
                             smooth_closure=False):
    """
    Gövde varlığı düzeltmeli kesit kavite yarıçapı.

    Returns: (rc_x, rc_logv, regime)
      regime = "free"     : rc_logv ≥ R_v → rc_x = √(rc_logv² + R_v²) (düzeltme açıksa)
      regime = "attached" : rc_logv < R_v → rc_x = rc_logv
      regime = "none"     : x_cav ≥ Lc   → rc_x = rc_logv = 0
    """
    if x_cav >= Lc:
        return 0.0, 0.0, "none"
    rc_logv = cavity_radius_logvinovich(x_cav, Lc, Dc, R_n, Cx, smooth_closure)
    if rc_logv >= R_v:
        rc_x = np.sqrt(rc_logv ** 2 + R_v ** 2) if body_volume_correction else rc_logv
        return float(rc_x), rc_logv, "free"
    return rc_logv, rc_logv, "attached"


# ============================================================================
# ISLAK YAY AÇISI
# ============================================================================

def _free_arc(R_v, delta):
    """Serbest kavite içinde (rc ≥ R_v) dalma derinliğinden ıslak yay açısı."""
    if delta >= 2.0 * R_v:
        return 2.0 * np.pi
    if delta > 0:
        return 2.0 * np.arccos(max(min(1.0 - delta / R_v, 1.0), -1.0))
    return 0.0


def wetted_arc(R_v, rc_x, rc_logv, hc_x, smooth=False):
    """
    Kesitteki ıslak yay açısı.

    delta = R_v + |h_c| − rc_x
    Free-standing (rc_logv ≥ R_v):
        delta ≥ 2R_v → θ = 2π ; 0 < delta → θ = 2·arccos(1 − delta/R_v) ; aksi 0
    Attached (rc_logv < R_v):
        rc_logv < 0.05·R_v → θ = 2π ; aksi θ = 2π·(1 − rc_logv/R_v)
    Kavite yok (rc_x ≤ 1e-4): θ = 2π

    smooth=True (sürekli): attached θ = 2π·(1 − rc_logv/R_v) eşiksiz (rc→0'da 2π'ye
    sürekli); rc_logv ∈ [R_v, (1+ARC_FREE_BLEND)·R_v] aralığında attached (=0) ile free
    formül smoothstep ile harmanlanır. Legacy'de gövde hacmi düzeltmesi (rc_x = √(rc²+R_v²))
    sınırda δ'yı −0.41·R_v kaydırdığından |h| > 0.41·R_v iken burada sıçrama vardı.

    Returns: (theta_wet [rad], s_wet [m], delta_eff [m])
    """
    delta = R_v + abs(hc_x) - rc_x
    if smooth:
        if rc_logv <= 0.0 or R_v <= 1e-6:
            theta = 2.0 * np.pi
        elif rc_logv < R_v:
            theta = 2.0 * np.pi * (1.0 - rc_logv / R_v)
        else:
            w = smoothstep((rc_logv - R_v) / (ARC_FREE_BLEND * R_v))
            theta = w * _free_arc(R_v, delta)
        return theta, R_v * theta, R_v * (1.0 - np.cos(0.5 * theta))
    if rc_x > 1e-4 and R_v > 1e-6:
        if rc_logv >= R_v:
            if delta >= 2.0 * R_v:
                return 2.0 * np.pi, 2.0 * np.pi * R_v, 2.0 * R_v
            if delta > 0:
                arg = max(min(1.0 - delta / R_v, 1.0), -1.0)
                theta = 2.0 * np.arccos(arg)
                return theta, R_v * theta, delta
            return 0.0, 0.0, 0.0
        if rc_logv < _ATTACHED_CLOSED_FRAC * R_v:
            return 2.0 * np.pi, 2.0 * np.pi * R_v, 2.0 * R_v
        wet_fraction = max(0.0, 1.0 - rc_logv / R_v)
        theta = 2.0 * np.pi * wet_fraction
        return theta, R_v * theta, R_v - rc_logv
    return 2.0 * np.pi, 2.0 * np.pi * R_v, 2.0 * R_v


# ============================================================================
# TOPLAYICI
# ============================================================================

def _p(params, key, default):
    v = params.get(key, default)
    return default if v is None else v


def compute_body_forces(V, alpha, Lc, Dc, Cx, params):
    """
    Gövde ıslanma kuvvetleri (legacy simulate() ile aynı sayısal sonuç).

    Args:
        V:      Hız [m/s] (legacy'de kuvvet adımında kullanılan vi = V[i-1])
        alpha:  Hücum açısı alpha_aoa [rad], burun yukarı +
        Lc, Dc: Anlık (gecikmeli) kavite uzunluğu/çapı [m] (legacy Lc_r, Dc_r)
        Cx:     Kavitatör Cx (legacy Cx_i, cos² düzeltmesi SONRASI) — x_open için
        params: dict. Anahtarlar (legacy GUI isimleri):
            veh_len, veh_diam, diam_cav, L_taper, x_cg          [m]
            cavitator_type ("cone"|"disk")
            delta_cav_rad [rad]  veya  delta_cav [DERECE, legacy]  (alpha_eff için)
            alpha_eff [rad] (verilirse delta_cav yok sayılır)
            k_dev, CL_alpha_body, Cdc_body, body_volume_correction
            Cf, n_x (opsiyonel; varsayılan constants.CF_SKIN, N_BODY_SECTIONS)
            friction_model ("constant" varsayılan = legacy sabit Cf | "ittc" = Re bağımlı
                ITTC-57 korelasyon hattı, form_factor = (1+k) ile; bkz. skin_friction_ittc)
            form_factor (opsiyonel; varsayılan constants.FORM_FACTOR_DEFAULT = 1.1)
            smooth_transitions (bool, varsayılan False = legacy birebir):
                True → kesit döngüsü kavite olmasa da çalışır (legacy tam-ıslak dalı
                F_skin=0 hatası ve kavite oluşumundaki sıçrama yok), sürekli kapanma
                ve sürekli ıslak yay kullanılır.

    Returns dict (kuvvet YUKARI +, moment (x_cg − x)·F, x burundan):
        F_skin        ıslak yay sürtünmesi [N]
        F_body_lift   legacy efektif gövde lifti (CL·q·A_max·α·wet_frac) [N]
        F_body_drag   |F_body_lift·α| [N]
        M_body_lift   (x_cg − L/2)·F_body_lift [N·m]
        F_body_lift_section, F_body_drag_section, M_body_lift_section
                      kesit integrali Munk+Hoerner (legacy'de üzerine yazılan) /
                      kavite yoksa legacy tam-ıslak Munk+Allen-Perkins formülü
        F_buoy, M_buoy, V_submerged   Arşimet [N], [N·m], [m³]
        Cf [-]        kullanılan sürtünme katsayısı (ITTC modelinde V'ye bağlı)
        wet_area [m²], wet_len [m], total_area [m²], cover [-], wet_frac [-]
        fully_wet (bool): kavite yok (Lc veya Dc < 1e-4) dalı kullanıldı
    """
    L_veh = float(params["veh_len"])
    D_veh = float(params["veh_diam"])
    Dn = float(params["diam_cav"])
    L_taper = float(_p(params, "L_taper", 0.15 * L_veh))
    L_taper = max(0.0, min(L_taper, L_veh))
    x_cg = float(_p(params, "x_cg", L_veh / 2.0))
    x_cg = max(0.0, min(x_cg, L_veh))
    cav_type = str(_p(params, "cavitator_type", _p(params, "cav_type", "disk"))).lower()
    k_dev = max(0.0, min(float(_p(params, "k_dev", K_DEV_DEFAULT)), 1.0))
    CL_alpha_body = max(0.0, min(float(_p(params, "CL_alpha_body", CL_ALPHA_BODY_DEFAULT)), 6.28))
    body_vol_corr = bool(_p(params, "body_volume_correction", BODY_VOLUME_CORRECTION))
    # Sürtünme: "constant" → legacy sabit Cf (varsayılan, legacy_exact ile birebir),
    # "ittc" → ITTC-57 Re bağımlı (model.py legacy_exact=False iken bunu geçirir)
    if str(_p(params, "friction_model", "constant")).lower() == "ittc":
        Cf = skin_friction_ittc(V, L_veh,
                                form_factor=_p(params, "form_factor", FORM_FACTOR_DEFAULT))
    else:
        Cf = float(_p(params, "Cf", CF_SKIN))
    n_x = int(_p(params, "n_x", N_BODY_SECTIONS))
    smooth = bool(_p(params, "smooth_transitions", False))

    if "alpha_eff" in params and params["alpha_eff"] is not None:
        alpha_eff = float(params["alpha_eff"])
    elif "delta_cav_rad" in params:
        alpha_eff = alpha + float(params["delta_cav_rad"])
    else:
        delta_deg = max(-30.0, min(float(_p(params, "delta_cav", 0.0)), 30.0))
        alpha_eff = alpha + np.radians(delta_deg)

    vi = float(V)
    q = 0.5 * RHO * vi * vi
    R_v_max = D_veh / 2.0
    R_n = Dn / 2.0

    cav_w = max(Dn * 0.25, 0.01)
    x_body_start = 0.0 if cav_type == "cone" else cav_w

    total_area = body_surface_area(R_n, R_v_max, L_veh, L_taper)
    dx = L_veh / n_x

    F_skin = 0.0
    wet_len = 0.0
    wet_area = 0.0
    F_buoy = 0.0
    M_buoy = 0.0
    V_sub = 0.0
    FL_sec = 0.0
    FD_sec = 0.0
    ML_sec = 0.0

    cavity_exists = Lc > _CAV_EXIST_MIN and Dc > _CAV_EXIST_MIN

    if cavity_exists or smooth:
        x_open = cavity_opening_length(R_n, Cx)
        for xs_local in np.linspace(0.5 * dx, L_veh - 0.5 * dx, n_x):
            xs_cav = xs_local + x_body_start
            R_v = vehicle_radius(xs_local, R_n, R_v_max, L_taper)
            if vi > _V_AXIS_MIN:
                hc_x = cavity_axis_offset(xs_cav, alpha_eff, vi, k_dev=k_dev, x_open=x_open)
            else:
                hc_x = 0.0
            rc_x, rc_logv, _ = cavity_radius_at_section(xs_cav, R_v, Lc, Dc, R_n, Cx,
                                                        body_vol_corr, smooth_closure=smooth)
            theta_wet, s_wet, _ = wetted_arc(R_v, rc_x, rc_logv, hc_x, smooth=smooth)

            if s_wet > 0.0:
                wet_len += dx
            wet_area += s_wet * dx
            F_skin += q * Cf * s_wet * dx

            if theta_wet > 0:
                dV = 0.5 * theta_wet * R_v * R_v * dx
                dFb = RHO * G * dV
                F_buoy += dFb
                M_buoy += (x_cg - xs_local) * dFb
                V_sub += dV

            if theta_wet > 0 and abs(alpha) > 1e-6:
                f_wet = theta_wet / (2.0 * np.pi)
                sa = np.sin(alpha); ca = np.cos(alpha)
                s2a = np.sin(2.0 * alpha); sgn = np.sign(alpha)
                dFL = f_wet * (
                    RHO * vi * vi * np.pi * R_v * R_v * s2a * 0.5
                    + RHO * vi * vi * 2.0 * R_v * CDC_CROSSFLOW_SECTION * sa * sa * ca * sgn
                ) * dx
                FL_sec += dFL
                FD_sec += abs(dFL) * abs(np.tan(alpha))
                ML_sec += (x_cg - xs_local) * dFL
    else:
        # Kavite yok — tam ıslak (legacy satır ~1168-1228)
        wet_area = total_area
        wet_len = L_veh
        V_sub = body_volume(R_n, R_v_max, L_veh, L_taper)
        F_buoy = RHO * G * V_sub
        M_buoy = 0.0
        # NOT: legacy tam-ıslak dalında F_skin döngü dışında kaldığı için 0'dır.
        if abs(alpha) > 1e-6:
            Cdc_eff = max(0.0, min(float(_p(params, "Cdc_body", CDC_BODY_DEFAULT)), 2.0))
            sa = np.sin(alpha); ca = np.cos(alpha)
            s2a = np.sin(2.0 * alpha); sgn = np.sign(alpha)
            A_avg = V_sub / L_veh
            S_plan = (R_n + R_v_max) * L_taper + 2.0 * R_v_max * (L_veh - L_taper)
            munk_factor = CL_alpha_body / 2.0
            FL_sec = (munk_factor * RHO * vi * vi * A_avg * s2a
                      + RHO * vi * vi * S_plan * Cdc_eff * sa * sa * ca * sgn)
            FL_sec *= max(0.0, min(wet_area / total_area, 1.0)) if total_area > 1e-9 else 1.0
            FD_sec = abs(FL_sec) * abs(np.tan(alpha))
            x_lc = L_taper * 0.5 if L_taper > 1e-6 else L_veh * 0.5
            ML_sec = (x_cg - x_lc) * FL_sec

    cover = max(0.0, min(1.0 - wet_area / total_area, 1.0)) if total_area > 1e-9 else 0.0
    wet_frac = max(0.0, min(1.0 - cover, 1.0))

    # Legacy efektif gövde lifti (satır 1350-1361, döngü sonucunu ezer)
    A_max = np.pi * R_v_max ** 2
    F_body_lift = CL_alpha_body * q * A_max * alpha * wet_frac
    F_body_drag = abs(F_body_lift * alpha)
    M_body_lift = (x_cg - L_veh / 2.0) * F_body_lift

    return {
        "Cf": float(Cf),
        "F_skin": float(F_skin),
        "F_body_lift": float(F_body_lift),
        "F_body_drag": float(F_body_drag),
        "M_body_lift": float(M_body_lift),
        "F_body_lift_section": float(FL_sec),
        "F_body_drag_section": float(FD_sec),
        "M_body_lift_section": float(ML_sec),
        "F_buoy": float(F_buoy),
        "M_buoy": float(M_buoy),
        "V_submerged": float(V_sub),
        "wet_area": float(wet_area),
        "wet_len": float(wet_len),
        "total_area": float(total_area),
        "cover": float(cover),
        "wet_frac": float(wet_frac),
        "fully_wet": not cavity_exists,
    }
