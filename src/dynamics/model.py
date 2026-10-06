"""
6-DOF süperkavitasyon araç modeli:  (t, x, u) -> x_dot.

Tüm dış kuvvetleri (kavitatör, 4 kanat, gövde ıslanma/lift/Arşimet, transom
planing, yerçekimi, itki) gövde çerçevesinde toplar; katı cisim, kinematik ve
kavite (Lc, Dc, pc) türevlerini tek 15 elemanlı x_dot'ta birleştirir.

KONVANSİYON DÖNÜŞÜMÜ — TEK YER BURASI
  hydrodynamics/ fonksiyonları legacy konvansiyonunda döner: dikey/yan kuvvet
  "yukarı/yana pozitif" büyüklük, x burundan ölçülür. Burada:
    x_body = x_cg − x_nose                         (CG orijinli, ileri +)
    F_body = (−F_drag, −F_side, −F_up)             (y-sancak, z-aşağı)
    M_body = Σ r_body × F_body                     (K roll, M pitch, N yaw)
  Böylece M_pitch = (x_cg − x)·F_up (legacy My ile aynı işaret, + burun yukarı).

AKIŞ AÇILARI
  alpha = atan2(w, u)  (+ burun akışa göre yukarı → yukarı kuvvet)
  beta  = asin(v / V)  (+ araç sancağa kayıyor → iskele (−y) yönlü kuvvet)
  Yan (yaw) düzlem, pitch düzleminin simetriği olarak aynı formüllerle kurulur.

KANAT GEOMETRİSİ (legacy'den bilinçli sapma)
  Legacy azimut 90/270 (üst/alt) kanatlara DİKEY kuvvet yazdırıyor; fiziksel
  olarak dikey kuvveti yatay kanatlar (sağ/sol) üretir. Burada her kanadın
  normali azimuttan geometrik olarak türetilir:
    span yönü e_s = (0, cos az, −sin az),   normal n = (0, sin az, cos az)
    sağ (0°) → n=+z (elevator), üst (90°) → n=+y (rudder)
  Kanat-yerel sapma δ_i > 0, yerel hücum açısıyla aynı yönde (F = −L·n).
  Mixer: δ_i = δe·n_z + δr·n_y  → artı-düzende sağ=+δe, sol=−δe, üst=+δr, alt=−δr.
  Kanatlar eksenel simetrik kavite içinde olduğundan Case 1 sayısal olarak
  aynıdır: legacy fin1=fin3=−0.80°  ≡  δe = −0.80°.

KONTROL İŞARETLERİ
  δe > 0 → kuyrukta yukarı kuvvet → burun aşağı (M < 0)
  δr > 0 → kuyrukta iskele kuvvet → burun sancağa (N > 0)
  δc > 0 → kavitatörde yukarı kuvvet → burun yukarı
  Roll kontrol kanalı YOK (kullanıcı kararı); p, φ pasif + c_roll_damp sönümü.

REJİM GEÇİŞLERİ (legacy_exact)
  legacy_exact=True  → legacy simulate() birebir (regresyon testleri bunu kullanır).
  legacy_exact=False → (varsayılan) sürekli geçişler: kavite oluşumu (Lc=0, σ=1),
    gövde kesit kapanması, ıslak yay bitişik↔serbest, transom teması ve tamamen
    dışarıda rejimi, kavitatör Cx(σ ≥ 1). Ayrıntı: hydrodynamics/smooth.py kullanan
    fonksiyonların docstring'leri; test_continuity.py.
    Ayrıca TAM ISLAK (kavitasyon öncesi) rejim düzeltmeleri — süperkavitasyon
    korelasyonlarının kavite yokken geçersiz olduğu iki terim:
      * Burun direnci: Cx0·(1+σ) (Reichardt) gövdenin basınç direncini de kavitatöre
        yükler → ancak gövde kavitenin içindeyken geçerli. Kavite gövdeden geniş
        olana kadar (Dc/D ∈ [1, 1+w], w = cavity_estab_width) tam ıslak koni/disk
        basınç direnci (cavitator.wetted_cavitator_Cx); arada harmanlanır.
        Ölçüt σ değil kavite durumudur.
      * Sürtünme: legacy sabit Cf = 0.003 → ITTC-57(Re)·(1+k)
        (body.skin_friction_ittc).

BİLİNEN BASİTLEŞTİRMELER
  * Drag gövde ekseni boyunca (−x) uygulanır (legacy gibi), hız vektörüne ters değil.
  * Gövde ıslanması / planing kavite ekseni sapması sadece pitch düzleminde
    (yerçekimi + α_eff) hesaplanır; yan sapma (β) ıslanmayı etkilemez.
  * Kanat ıslaklığı eksenel simetrik kavite yarıçapıyla (ekseni sapmasız) hesaplanır.
"""

from dataclasses import dataclass

import numpy as np

from src.hydrodynamics.constants import (
    RHO, G, P_ATM, P_VAP, CX0_DISK, CAVITY_TAU, TAU_PC, A_V_DEFAULT,
    FIN_CD0, FIN_OSWALD_E, FIN_STALL_ANGLE, CD_BASE_FULLY_WET,
    CL_ALPHA_BODY_DEFAULT, K_DEV_DEFAULT, CAVITY_ESTAB_WIDTH,
)
from src.hydrodynamics.smooth import smoothstep
from src.hydrodynamics import body as hbody
from src.hydrodynamics import cavitator as hcav
from src.hydrodynamics import cavity as hcavity
from src.hydrodynamics import fins as hfins
from src.hydrodynamics import planing as hplaning

from .state import (
    N_STATES, SL_NU, SL_NU_LIN, SL_NU_ANG, SL_EULER, SL_POS,
    IDX_PHI, IDX_THETA, IDX_Z, IDX_LC, IDX_DC, IDX_PC,
)
from .kinematics import kinematics_derivative, gravity_body, flow_angles
from .rigid_body import cylinder_inertia, rigid_body_derivative

# Legacy kanat numaralandırması: 1-üst, 2-sağ, 3-alt, 4-sol
FIN_AZIMUTHS_DEG = (90.0, 0.0, 270.0, 180.0)

# Legacy: bu hızın altında kanat kuvveti, Cq ve kavite ekseni sapması 0 alınır [m/s]
V_FORCE_MIN = 0.5

FIN_DELTA_MAX = np.radians(20.0)
DELTA_CAV_MAX = np.radians(30.0)


@dataclass
class ControlInput:
    """Bir kontrol periyodu boyunca ZOH tutulan komutlar (SI, radyan)."""
    delta_e: float = 0.0    # elevator [rad]
    delta_r: float = 0.0    # rudder [rad]
    delta_c: float = 0.0    # kavitatör pitch açısı [rad]
    thrust: float = 0.0     # itki [N], gövde +x
    gas_flow: float = 0.0   # vent_mode "Q": L/min ; "Cq": boyutsuz Cq


def _clip(v, lo, hi):
    return max(lo, min(float(v), hi))


def fin_normal(az_rad):
    """Azimuttan kanat normali (gövde çerçevesi). 0=sağ, 90°=üst."""
    return np.array([0.0, np.sin(az_rad), np.cos(az_rad)])


def fin_mixer(delta_e, delta_r, azimuths_deg=FIN_AZIMUTHS_DEG):
    """(δe, δr) → kanat-yerel sapmalar [rad].  δ_i = δe·n_z + δr·n_y  (roll kanalı yok)."""
    out = np.empty(len(azimuths_deg))
    for i, az in enumerate(azimuths_deg):
        n = fin_normal(np.radians(az))
        out[i] = delta_e * n[2] + delta_r * n[1]
    return np.clip(out, -FIN_DELTA_MAX, FIN_DELTA_MAX)


class VehicleModel:
    """
    Parametreler legacy GUI anahtar adlarıyla verilir (bkz. legacy_reference.CASE1_PARAMS).
    6-DOF ekleri (opsiyonel):
      Ixx, Iyy, Izz [kg m²]   (yoksa homojen silindir)
      M_A (6x6)               sabit eklenmiş kütle
      c_roll_damp [N m s/rad] pasif roll sönümü
      fin_azimuths_deg        kanat azimutları (varsayılan artı düzen)
      fins_use_steady_cavity  True → kanat ıslaklığı Lc_ss/Dc_ss ile (legacy tuhaflığı),
                              False (varsayılan) → gecikmeli durum Lc/Dc ile
      legacy_exact            True → legacy rejim anahtarlamaları birebir; False (varsayılan)
                              → sürekli geçişler
      gas_flow_ref_depth [m]  Q-mod: gaz debisinin ölçüldüğü derinlik (hidrostatik p_ref);
                              gaz kavitede pc'ye genleşir. None (varsayılan) → legacy.
      cavitator_Cx_wet        Kavite yokken (σ → 1) burun direnci katsayısı (Sn'ye göre).
                              None (varsayılan) → burun geometrisinden türetilir
                              (cavitator.wetted_cavitator_Cx; 40° koni → 0.20).
                              Yalnız legacy_exact=False iken etkin.
      cavity_estab_width      Burun direnci harmanlama genişliği: kavite gövdeden geniş
                              olana kadar (Dc/D ∈ [1, 1+w]) ıslak Cx → Reichardt Cx0(1+σ)
                              (varsayılan 0.5). Yalnız legacy_exact=False iken etkin.
      friction_model          "ittc" (legacy_exact=False varsayılanı) → Cf = (1+k)·ITTC-57(Re);
                              "constant" (legacy_exact=True varsayılanı) → legacy sabit Cf.
      form_factor             (1+k) form + pürüzlülük çarpanı (varsayılan 1.1).
    Gövde orijini x_cg'dedir (kütle merkezi); x_cg_mass yok sayılır.
    """

    def __init__(self, params):
        p = dict(params)
        self.params = p
        self.smooth = not bool(p.get("legacy_exact", False))

        self.m = float(p["mass"])
        self.L = float(p["veh_len"])
        self.D = float(p["veh_diam"])
        self.Dn = float(p["diam_cav"])
        self.R_v = 0.5 * self.D
        self.R_n = 0.5 * self.Dn
        self.L_taper = _clip(p.get("L_taper", 0.15 * self.L), 0.0, self.L)
        self.x_cg = _clip(p.get("x_cg", 0.5 * self.L), 0.0, self.L)
        self.S_n = np.pi * self.R_n ** 2
        self.S_b = np.pi * self.R_v ** 2

        self.I = cylinder_inertia(self.m, self.L, self.D,
                                  Ixx=p.get("Ixx"), Iyy=p.get("Iyy"), Izz=p.get("Izz"))
        self.M_A = p.get("M_A")
        self.c_p = float(p.get("c_roll_damp", 0.0))

        # Kavitatör
        self.cav_type = str(p.get("cavitator_type", p.get("cav_type", "disk"))).lower()
        self.cone_apex = np.radians(_clip(p.get("cone_apex_deg", 180.0), 20.0, 180.0))
        if self.cav_type == "cone":
            K_v = _clip(p.get("cone_drag_visc", 0.35), 0.0, 0.50)
            self.Cx0 = hcav.conical_cavitator_Cx(CX0_DISK, self.cone_apex, K_v)
        else:
            self.Cx0 = CX0_DISK
        # Kavite yokken (σ → 1) burun direnci: Reichardt Cx0·(1+σ) yerine tam ıslak
        # koni/disk basınç direnci (bkz. cavitator.wetted_cavitator_Cx). Sadece
        # legacy_exact=False iken kullanılır.
        # Kavite "kurulmuş" sayılma genişliği: Dc/D ∈ [1, 1+w] (kalibrasyon parametresi)
        self.cav_estab_w = max(float(p.get("cavity_estab_width", CAVITY_ESTAB_WIDTH)), 1e-3)
        Cx_wet = p.get("cavitator_Cx_wet")
        self.Cx_wet = (float(Cx_wet) if Cx_wet is not None else
                       hcav.wetted_cavitator_Cx(
                           self.cav_type,
                           self.cone_apex if self.cav_type == "cone" else None))
        self.K_slender = _clip(p.get("K_slender", 1.0), 0.0, 2.0)
        self.K_cone_lift = _clip(p.get("K_cone_lift", 0.0), 0.0, 3.0)
        self.cone_lift_gain = _clip(p.get("cone_lift_gain", 0.0), 0.0, 2.0)
        self.x_body_start = 0.0 if self.cav_type == "cone" else max(self.Dn * 0.25, 0.01)

        # Kavite / havalandırma
        self.geom_model = p.get("geom_model", "savchenko")
        self.k_g = _clip(p.get("k_g_cavity", 0.78), 0.50, 1.20)
        self.K_Dc = _clip(p.get("K_Dc_factor", 1.0), 0.5, 2.0)
        self.K_Lc = _clip(p.get("K_Lc_factor", 1.0), 0.3, 2.0)
        self.A_v = _clip(p.get("A_v", A_V_DEFAULT), 0.005, 0.30)
        self.vent_mode = p.get("vent_mode", "Q")
        # Q-mod: gaz debisinin ölçüldüğü derinlik [m] (None → legacy: kavite basıncında hacim)
        ref_depth = p.get("gas_flow_ref_depth")
        self.gas_p_ref = (None if ref_depth is None or self.vent_mode != "Q"
                          else P_ATM + RHO * G * max(float(ref_depth), 0.0))
        self.tau_cav = float(p.get("cavity_tau", CAVITY_TAU))
        self.tau_pc = float(p.get("tau_pc", TAU_PC))
        self.k_dev = _clip(p.get("k_dev", K_DEV_DEFAULT), 0.0, 1.0)

        # Gövde
        self.CL_alpha_body = _clip(p.get("CL_alpha_body", CL_ALPHA_BODY_DEFAULT), 0.0, 6.28)
        self._body_params = dict(p, smooth_transitions=self.smooth)
        # Sürtünme: sürekli modelde Re bağımlı ITTC-57, legacy_exact'te legacy sabit Cf.
        # Config'teki açık "friction_model" anahtarı bu seçimi ezer.
        if p.get("friction_model") is None:
            self._body_params["friction_model"] = "ittc" if self.smooth else "constant"
        self.planing_enable = bool(p.get("planing_enable", False))

        # Kanatlar
        self.fins_enabled = bool(p.get("fins_enabled", True))
        self.fin_chord = float(p.get("fin_chord", 0.040))
        self.fin_span = float(p.get("fin_span", 0.060))
        self.fin_x = _clip(p.get("fin_x_pos", 0.85 * self.L), 0.0, self.L)
        self.fin_az = tuple(p.get("fin_azimuths_deg", FIN_AZIMUTHS_DEG))
        self.fin_CL_alpha = hfins.fin_3d_lift_slope(self.fin_chord, self.fin_span)
        self.fins_use_steady_cavity = bool(p.get("fins_use_steady_cavity", False))
        self.R_v_fin = hbody.vehicle_radius(self.fin_x, self.R_n, self.R_v, self.L_taper)
        self.fin_x_body = self.x_cg - self.fin_x
        self._fin_normals = [fin_normal(np.radians(a)) for a in self.fin_az]
        self._fin_r_force = np.zeros(len(self.fin_az))
        # Kavite ekseni sapmasının kanat ıslaklığına etkisi (K roll momentinin kaynağı).
        # Varsayılan: sürekli modelde açık, legacy_exact'te kapalı (legacy birebir).
        fco = p.get("fin_cavity_offset")
        self.fin_cav_offset = self.smooth if fco is None else bool(fco)
        # Yerçekimi sagı ayrı anahtar: işareti çözülmemiş ve baskın (bkz.
        # cavity.cavity_axis_offset_components docstring'i) → varsayılan KAPALI.
        self.fin_cav_offset_gravity = bool(p.get("fin_cavity_offset_gravity", False))

    # ------------------------------------------------------------------
    def ambient_pressure(self, depth):
        return P_ATM + RHO * G * max(depth, 0.0)

    def gas_coefficient(self, gas_flow, V):
        """Cq = Q/(V·Dn²) (Q-mod, L/min) veya doğrudan Cq."""
        if self.vent_mode == "Cq":
            return float(gas_flow)
        Q = gas_flow / 60000.0
        if Q > 1e-10 and V > V_FORCE_MIN and self.Dn > 1e-4:
            return Q / (V * self.Dn ** 2)
        return 0.0

    def cavitator_Cx_geometry(self, s_raw, a_tot=0.0):
        """
        Kavite GEOMETRİSİ için kavitatör direnç katsayısı (legacy ile aynı; kuvvet için
        kullanılan Cx_force'tan farklı — bkz. evaluate). Model ve tahminci ortak yolu.

        Returns: (Cx_sc, Cx_i) — σ'ya bağlı katsayı ve cos²(a_tot) ile ölçeklenmişi.
        """
        if self.smooth:
            # Cx0·(1+σ) σ ≥ 1'de doygun (legacy: σ ≥ 1'de Cx0'a yarıya SIÇRAR)
            Cx_sc = self.Cx0 * (1.0 + min(s_raw, 1.0))
        else:
            Cx_sc = self.Cx0 * (1.0 + s_raw) if s_raw < 1.0 else self.Cx0
        return Cx_sc, Cx_sc * np.cos(a_tot) ** 2

    def fin_wet_geometry(self, Lc, Dc, alpha_eff, beta, V, Cx):
        """
        Kanat başına ıslak açıklık ve kuvvet yarıçapı — model ve tahminci ortak yolu.

        Kavite ekseni akış doğrultusunu izler (α_eff, β ile gövde ekseninden kayar) →
        kanatlar farklı ıslanır → K (roll) momenti doğar. `fin_cavity_offset` kapalıysa
        eş merkezli (eski) geometri kullanılır ve sonuç birebir eskisiyle aynıdır.

        Returns: (spans, radii, h_y, h_z) — açıklıklar [m], kuvvet yarıçapları [m], sapma [m]
        """
        x_fin = self.fin_x + self.x_body_start
        R_c = hfins.cavity_radius_ellipse(x_fin, Lc, Dc)
        if self.fin_cav_offset:
            h_y, h_z = hcavity.cavity_axis_offset_components(
                x_fin, alpha_eff, beta, V, k_dev=self.k_dev,
                x_open=hbody.cavity_opening_length(self.R_n, Cx),
                include_gravity=self.fin_cav_offset_gravity)
        else:
            h_y = h_z = 0.0
        n = len(self.fin_az)
        spans = np.empty(n)
        radii = np.empty(n)
        for i, az in enumerate(self.fin_az):
            spans[i], radii[i] = hfins.fin_wetted_segment(
                R_c, self.R_v_fin, self.fin_span, np.radians(az), h_y, h_z)
        return spans, radii, h_y, h_z

    def cavity_derivative(self, Lc, Dc, pc, V, depth, gas_flow):
        """(Cq, kavite türev sözlüğü) — model ve tahminci (control/estimator.py) ortak yolu."""
        Cq = self.gas_coefficient(gas_flow, V)
        cav = hcavity.cavity_state_derivative(
            Lc, Dc, pc, V, self.ambient_pressure(depth), Cq, self.Dn, model=self.geom_model,
            k_g=self.k_g, K_Dc=self.K_Dc, K_Lc=self.K_Lc, A_v=self.A_v,
            tau_cav=self.tau_cav, tau_pc=self.tau_pc, smooth=self.smooth,
            gas_p_ref=self.gas_p_ref)
        return Cq, cav

    # ------------------------------------------------------------------
    def evaluate(self, t, x, u):
        """(x_dot, diag). diag: tanı büyüklükleri (legacy adlarıyla, yukarı +)."""
        x = np.asarray(x, dtype=float)
        u_b, v_b, w_b = x[SL_NU_LIN]
        omega = x[SL_NU_ANG]
        phi, theta = x[IDX_PHI], x[IDX_THETA]
        Lc, Dc, pc = x[IDX_LC], x[IDX_DC], x[IDX_PC]

        V, alpha, beta = flow_angles(u_b, v_b, w_b)
        q = 0.5 * RHO * V * V
        p_inf = self.ambient_pressure(x[IDX_Z])

        F = np.zeros(3)
        M = np.zeros(3)

        def add(force, r):
            F[:] += force
            M[:] += np.cross(r, force)

        # ---- Kavite durumu ----
        Cq, cav = self.cavity_derivative(Lc, Dc, pc, V, x[IDX_Z], u.gas_flow)
        s_raw = cav["sigma_raw"]
        sigma = cav["sigma"]

        # ---- Kavitatör (burun, x_nose = 0) ----
        delta_c = _clip(u.delta_c, -DELTA_CAV_MAX, DELTA_CAV_MAX)
        a_z = alpha + delta_c
        a_y = beta
        a_tot = np.hypot(a_z, a_y)
        Cx_sc, Cx_i = self.cavitator_Cx_geometry(s_raw, a_tot)   # Cx_i: kavite GEOMETRİSİ
        cos2 = np.cos(a_tot) ** 2
        if self.smooth:
            # Kuvvet için: Reichardt Cx0·(1+σ) (gövdenin basınç direncini de kavitatöre
            # yükleyen süperkavite yasası) ancak gövde kavitenin içindeyken geçerlidir →
            # kavite gövdeden geniş olana kadar (Dc/D ∈ [1, 1+w], w = cavity_estab_width)
            # tam ıslak burun basınç direnci (Cx_wet) kullanılır; arada harmanlanır.
            # Ölçüt σ DEĞİL kavite DURUMUDUR: σ hem V arttıkça kavite yokken düşer
            # (V = 25 m/s'de t = 0'da σ = 0.62) hem de havalandırma başlar başlamaz ms'ler
            # içinde düşer → σ tabanlı eşik düzeltmeyi kavite oluşmadan kapatıyordu.
            w_cav = smoothstep((Dc / self.D - 1.0) / self.cav_estab_w)
            Cx_force = (w_cav * Cx_sc + (1.0 - w_cav) * self.Cx_wet) * cos2
        else:
            w_cav = 1.0
            Cx_force = Cx_i
        F_cav = q * self.S_n * Cx_force * np.cos(a_tot)
        CL_cav = hcav.cavitator_lift_coefficient(
            max(sigma, 0.001), self.cav_type,
            self.cone_apex if self.cav_type == "cone" else None,
            K_slender=self.K_slender, K_cone_lift=self.K_cone_lift,
            cone_lift_gain=self.cone_lift_gain)
        F_cavz = q * self.S_n * CL_cav * a_z
        F_cavy = q * self.S_n * CL_cav * a_y
        add(np.array([-F_cav, -F_cavy, -F_cavz]), np.array([self.x_cg, 0.0, 0.0]))

        # ---- Gövde: ıslanma, sürtünme, Arşimet, gövde lifti ----
        bp = self._body_params
        bp["alpha_eff"] = a_z
        bf = hbody.compute_body_forces(V, alpha, Lc, Dc, Cx_i, bp)
        wet_frac = bf["wet_frac"]
        F_body_lift = bf["F_body_lift"]
        F_body_side = self.CL_alpha_body * q * self.S_b * beta * wet_frac
        F_body_drag = bf["F_body_drag"] + abs(F_body_side * beta)
        r_body_mid = np.array([self.x_cg - 0.5 * self.L, 0.0, 0.0])
        add(np.array([-F_body_drag, -F_body_side, -F_body_lift]), r_body_mid)
        add(np.array([-bf["F_skin"], 0.0, 0.0]), np.zeros(3))

        F_buoy = bf["F_buoy"]
        if F_buoy > 1e-9:
            x_b_buoy = bf["M_buoy"] / F_buoy
            g_dir = gravity_body(1.0, phi, theta) / G
            add(-F_buoy * g_dir, np.array([x_b_buoy, 0.0, 0.0]))

        # ---- Transom planing / taban basınç sürüklemesi ----
        planing_regime = "fully_wet"
        F_planing = 0.0
        x_open = hbody.cavity_opening_length(self.R_n, Cx_i)
        if self.smooth or (Lc > 1e-4 and Dc > 1e-4):
            x_tail = self.L + self.x_body_start
            rc_tail = hbody.cavity_radius_logvinovich(x_tail, Lc, Dc, self.R_n, Cx_i,
                                                      smooth_closure=self.smooth)
            if V > V_FORCE_MIN:
                h_tail = hcavity.cavity_axis_offset(x_tail, a_z, V, k_dev=self.k_dev,
                                                    x_open=x_open)
                alpha_p = hplaning.planing_angle_from_cavity_slope(
                    x_tail, V, k_dev=self.k_dev, x_open=x_open, alpha_eff=a_z)
            else:
                h_tail = 0.0
                alpha_p = self.k_dev * a_z * np.exp(-x_tail / x_open)
            transom = (hplaning.transom_forces_smooth if self.smooth
                       else hplaning.transom_forces_legacy)
            tr = transom(V, self.R_v, rc_tail, h_tail, alpha_p, self.S_b, self.planing_enable)
            F_planing, F_press, planing_regime = tr["F_planing"], tr["F_press"], tr["regime"]
        else:
            F_press = q * self.S_b * CD_BASE_FULLY_WET
        r_tail = np.array([self.x_cg - self.L, 0.0, 0.0])
        add(np.array([-F_press, 0.0, -F_planing]), r_tail)

        # ---- Kanatlar ----
        deltas = fin_mixer(u.delta_e, u.delta_r, self.fin_az)
        F_fin_up = F_fin_side = F_fin_drag = 0.0
        wet_span = 0.0
        h_y = h_z = 0.0
        spans = np.zeros(len(self.fin_az))
        if self.fins_enabled and self.fin_chord > 1e-6 and self.fin_span > 1e-6 and V > V_FORCE_MIN:
            Lc_f, Dc_f = (cav["Lc_ss"], cav["Dc_ss"]) if self.fins_use_steady_cavity else (Lc, Dc)
            # Kavite ekseni sapması: kanat başına farklı ıslaklık → K (roll) momenti.
            # Tahminci (control/estimator.py) aynı yolu kullanır (kendi kavite tahminiyle).
            spans, radii, h_y, h_z = self.fin_wet_geometry(Lc_f, Dc_f, a_z, beta, V, Cx_i)
            self._fin_r_force[:] = radii
            v_lin = x[SL_NU_LIN]
            # Tanıda/otopilot etkinliğinde (eta) tek skaler açıklık: kanat ortalaması
            # (sapma yokken dördü de aynı → eski değer birebir korunur)
            wet_span = float(spans.mean())
            if spans.max() > 1e-6:
                for i, (n, az, d_i) in enumerate(zip(self._fin_normals, self.fin_az, deltas)):
                    w_i = spans[i]
                    if w_i <= 1e-6:
                        continue
                    a = np.radians(az)
                    r_i = self._fin_r_force[i]
                    r = np.array([self.fin_x_body, r_i * np.cos(a), -r_i * np.sin(a)])
                    v_loc = v_lin + np.cross(omega, r)
                    a_loc = np.arctan2(v_loc @ n, max(v_loc[0], 1e-6))
                    a_fin = _clip(a_loc + d_i, -FIN_STALL_ANGLE, FIN_STALL_ANGLE)
                    L_fin, D_fin = hfins.fin_lift_and_drag(
                        V, self.fin_chord, w_i, self.fin_CL_alpha, a_fin,
                        span_for_AR=self.fin_span)
                    f = -L_fin * n + np.array([-D_fin, 0.0, 0.0])
                    add(f, r)
                    F_fin_up -= f[2]
                    F_fin_side -= f[1]
                    F_fin_drag += D_fin

        # ---- Yerçekimi ve itki ----
        F_grav_body = gravity_body(self.m, phi, theta)
        add(F_grav_body, np.zeros(3))
        add(np.array([u.thrust, 0.0, 0.0]), np.zeros(3))

        # ---- Türevler ----
        tau = np.concatenate([F, M])
        nu_dot = rigid_body_derivative(x[SL_NU], tau, self.m, self.I,
                                       M_A=self.M_A, c_p=self.c_p)
        pos_dot, euler_dot = kinematics_derivative(x)

        xdot = np.zeros(N_STATES)
        xdot[SL_NU] = nu_dot
        xdot[SL_EULER] = euler_dot
        xdot[SL_POS] = pos_dot
        xdot[IDX_LC] = cav["dLc"]
        xdot[IDX_DC] = cav["dDc"]
        xdot[IDX_PC] = 0.0 if (pc <= P_VAP and cav["dpc"] < 0.0) else cav["dpc"]

        diag = {
            "V": V, "alpha": alpha, "beta": beta, "q": q, "p_inf": p_inf,
            "sigma": sigma, "sigma_vapor": cav["sigma_vapor"], "Cq": Cq,
            "pc_target": cav["pc_target"], "Lc_ss": cav["Lc_ss"], "Dc_ss": cav["Dc_ss"],
            "Cx": Cx_i, "Cx_force": Cx_force, "cav_estab": w_cav,
            "alpha_eff": a_z, "Cf": bf["Cf"],
            # sürükleme bileşenleri (geri yönlü +)
            "F_cav": F_cav, "F_skin": bf["F_skin"], "F_press": F_press,
            "F_body_drag": F_body_drag, "F_fin_drag": F_fin_drag,
            "Fd": F_cav + bf["F_skin"] + F_press + F_body_drag,
            # dikey (yukarı +) — legacy adları
            "F_cavz": F_cavz, "F_planing": F_planing, "F_body_lift": F_body_lift,
            "F_fin_total": F_fin_up, "F_buoy": F_buoy, "F_grav": -self.m * G,
            # yan (iskele +)
            "F_cavy": F_cavy, "F_body_side": F_body_side, "F_fin_side": F_fin_side,
            "cover": bf["cover"], "wet_frac": wet_frac, "fin_wet_span": wet_span,
            # kanat başına ıslak açıklık + kavite ekseni sapması (K roll kaynağı)
            "fin_wet_spans": tuple(float(v) for v in spans),
            "cav_h_y": float(h_y), "cav_h_z": float(h_z),
            "planing_regime": planing_regime,
            "delta_fins": deltas,
            # gövde çerçevesi toplamları
            "F_body": F.copy(), "M_body": M.copy(),
        }
        return xdot, diag

    def derivatives(self, t, x, u):
        return self.evaluate(t, x, u)[0]
