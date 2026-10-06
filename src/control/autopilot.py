"""
Pitch + yaw duruş otopilotu (ayrık, kapalı çevrim). Roll kontrolü YOK (kullanıcı kararı).

Simülatör update(t, x)'i her dt_control (1 ms) başında bir kez çağırır; komut periyot
boyunca ZOH tutulur. PID durumu (integratör, rate limit belleği) sadece update'te ilerler.

Eksen başına kontrol yasası (SI, radyan):
    e      = ref − ölçüm                   (yaw: [−π, π]'ye sarılır)
    ω_meas = Euler açı hızı (θ̇ veya ψ̇)   — türev ÖLÇÜMDEN, hatadan değil (derivative kick yok)
    v      = k_s·(Kp·(b·ref − ölçüm) + I − Kd·ω_meas)   "burnu ref yönüne çevirme" isteği [rad]
             b: referans ağırlığı (0..1). b<1 referans adımında P sıçramasını ve aşımı azaltır;
             kalıcı hatayı integratör (tam hata e ile) sıfırlar.
    δ      = trim + sign·v                  → |δ| ≤ limit, |Δδ| ≤ rate_limit·dt
    I     += Ki·e·dt   (koşullu: çıkış doymuşken hata doymayı derinleştiriyorsa DONDUR; |I| ≤ i_limit)

İşaretler (model.py): δe > 0 → burun aşağı ⇒ pitch sign = −1 ;  δr > 0 → burun sancak (ψ↑) ⇒ yaw sign = +1.

Kazanç çizelgesi (kanat kuvveti ∝ V²·wet_span):
    k_s = clip((V_ref/V)² · eta_ref/eta, k_s_min, k_s_max)
    eta = kanat ıslak açıklık oranı, estimator.CavityEstimator'dan (kavite boyu ölçülmez;
    V, derinlik ve gaz debisinden model tabanlı). estimator=None → eta = eta_ref (sadece V²).
    V_ref=None → k_s = 1.

Derinlik tutma (opsiyonel dış döngü, depth_ref_m verilirse):
    e_z    = Z − Z_ref                     (Z+ derinlik; fazla derin → burun yukarı)
    θ_cmd  = θ_ref(program) + clip(Kp·e_z + I_z + Kd·Ż, ±theta_limit)
    I_z   += Ki·e_z·dt  (|I_z| ≤ i_limit; çıkış doymuşken doymayı derinleştiren artış donar)
    Ż NED dikey hız (kinematik), türev ölçümden. Birimler config'te derece ve metre.

Trim ileri beslemesi (opsiyonel): δ = δ_ff + PID;  δ_ff = −M0/Mδ, tahmin edilen kavite
durumunda ve p=q=r=0 iken (AttitudeAutopilot.trim_feedforward). Referans ön filtresi
(opsiyonel): birinci derece, θ/ψ program referansına.

Birimler: Kp [rad/rad], Ki [1/s], Kd [s]; config'te limitler derece.
"""

import dataclasses

import numpy as np

from src.dynamics.blocks import Controller, schedule_value
from src.dynamics.model import ControlInput
from src.dynamics.kinematics import flow_angles, kinematics_derivative
from src.dynamics.state import (IDX_P, IDX_Q, IDX_R, IDX_PHI, IDX_THETA, IDX_PSI, IDX_Z,
                                IDX_LC, IDX_DC, IDX_PC, SL_NU_LIN)

GAIN_SCALE_MIN, GAIN_SCALE_MAX = 0.25, 4.0
COS_THETA_MIN = 1e-3
FF_DELTA_STEP = np.radians(1.0)      # trim ileri besleme: Mδ sayısal türev adımı
FF_MIN_CONTROL_POWER = 1.0           # |Mδ| [N·m/rad] altında ileri besleme yok (kanat kuru)


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class AxisPID:
    """Tek eksen: anti-windup (koşullu integrasyon + |I| sınırı), ölçümden türev, doyum, hız sınırı."""

    def __init__(self, Kp, Ki=0.0, Kd=0.0, b=1.0, limit_deg=20.0, i_limit_deg=None,
                 rate_limit_deg_s=None, sign=1.0, trim_deg=0.0):
        self.Kp, self.Ki, self.Kd, self.b = float(Kp), float(Ki), float(Kd), float(b)
        self.limit = np.radians(limit_deg)
        self.i_limit = np.radians(i_limit_deg) if i_limit_deg is not None else self.limit
        self.rate_limit = np.radians(rate_limit_deg_s) if rate_limit_deg_s else None
        self.sign = float(sign)
        self.trim = np.radians(trim_deg)
        self.reset()

    def reset(self):
        self.I = 0.0
        self.cmd = self.trim
        self.saturated = False

    def update(self, e, omega_meas, dt, k_s=1.0, ref=0.0, ff=0.0):
        """e = ref − ölçüm (sarılmış olabilir); ref sadece b<1 ise P terimini etkiler.
        ff: ileri besleme [rad], doyumdan önce eklenir (anti-windup toplam çıkışa bakar)."""
        e_p = e - (1.0 - self.b) * ref

        def command(I):
            raw = (self.trim + ff
                   + self.sign * k_s * (self.Kp * e_p + I - self.Kd * omega_meas))
            c = min(max(raw, -self.limit), self.limit)
            if self.rate_limit is not None and dt > 0.0:
                step = self.rate_limit * dt
                c = min(max(c, self.cmd - step), self.cmd + step)
            return raw, c

        I_new = self.I
        if dt > 0.0 and self.Ki != 0.0:
            I_new = float(np.clip(self.I + self.Ki * e * dt, -self.i_limit, self.i_limit))
        raw, c = command(I_new)
        # Çıkış kırpıldıysa ve integratör artışı kırpma yönüne itiyorsa integrasyonu dondur
        pushing = self.sign * (I_new - self.I) * (raw - c) > 0.0
        if c != raw and pushing:
            I_new = self.I
            raw, c = command(I_new)
        self.I = I_new
        self.saturated = c != raw
        self.cmd = c
        return c


class DepthHold:
    """Derinlik → θ_ref dış döngüsü (PID, türev Ż ölçümünden, anti-windup)."""

    def __init__(self, Kp_deg_m, Ki_deg_m_s=0.0, Kd_deg_s_m=0.0, theta_limit_deg=5.0,
                 i_limit_deg=None):
        self.Kp = np.radians(Kp_deg_m)
        self.Ki = np.radians(Ki_deg_m_s)
        self.Kd = np.radians(Kd_deg_s_m)
        self.limit = np.radians(theta_limit_deg)
        self.i_limit = np.radians(i_limit_deg) if i_limit_deg is not None else self.limit
        self.reset()

    def reset(self):
        self.I = 0.0
        self.saturated = False

    def update(self, e_z, z_dot, dt):
        def out(I):
            raw = self.Kp * e_z + I + self.Kd * z_dot
            return raw, min(max(raw, -self.limit), self.limit)

        I_new = self.I
        if dt > 0.0 and self.Ki != 0.0:
            I_new = float(np.clip(self.I + self.Ki * e_z * dt, -self.i_limit, self.i_limit))
        raw, c = out(I_new)
        if c != raw and (I_new - self.I) * (raw - c) > 0.0:
            I_new = self.I
            raw, c = out(I_new)
        self.I = I_new
        self.saturated = c != raw
        return c


class AttitudeAutopilot(Controller):
    """
    θ_ref, ψ_ref (derece; sayı veya [(t, değer), ...]) takibi → δe, δr.
    Açık çevrim kanallar (schedule): delta_c_deg, thrust, gas_flow.

    estimator : None veya dict(param_scale=None, eta_min=0.1, tau_pc_obs, k_tail, lc_push)
                — kanat etkinliği çizelgesi. vehicle (araç parametreleri) gerekir;
                run_simulation.build_controller verir.
    sensors   : None veya dict(use_pc=True, use_tail=True, pc_noise_pa=0, tail_margin=0, seed=0)
                — kavite basınç ve kuyruk gaz sensörleri tahminciyi düzeltir (estimator gerekli).
    trim_ff   : None veya dict(gain=1.0, every=10) — model tabanlı trim ileri beslemesi:
                tahmin edilen kavite durumunda, p=q=r=0 ve δ=0 iken toplam momenti sıfırlayan
                δ_ff = −gain·M0/Mδ (Mδ sayısal, 1°). `every` kontrol periyodunda bir hesaplanır
                (3 model değerlendirmesi), arada ZOH. Statik (α, yerçekimi, planing,
                kavitatör) momentleri iptal eder; PID kalan dinamiği ve model hatasını taşır.
    ref_tau_s : θ/ψ program referansı için birinci derece ön filtre [s] (None → yok). Adım
                anındaki P sıçramasını (k_s·Kp·b·Δref) yumuşatır; kapalı çevrim kararlılığını
                etkilemez. Derinlik döngüsü çıkışı filtrelenmez.
    depth_hold: None veya DepthHold argümanları; depth_ref_m verilirse etkin.
    """

    def __init__(self, pitch, yaw, theta_ref_deg=0.0, psi_ref_deg=0.0, V_ref=None,
                 delta_c_deg=0.0, thrust=0.0, gas_flow=0.0, V_min=5.0,
                 eta_ref=1.0, k_s_min=GAIN_SCALE_MIN, k_s_max=GAIN_SCALE_MAX,
                 estimator=None, vehicle=None, sensors=None, trim_ff=None, ref_tau_s=None,
                 depth_hold=None, depth_ref_m=None):
        self.pitch = AxisPID(sign=-1.0, **pitch)
        self.yaw = AxisPID(sign=+1.0, **yaw)
        self.theta_ref = theta_ref_deg
        self.psi_ref = psi_ref_deg
        self.V_ref = V_ref
        self.V_min = float(V_min)
        self.eta_ref = float(eta_ref)
        self.k_s_min, self.k_s_max = float(k_s_min), float(k_s_max)
        self.open_loop = dict(delta_c_deg=delta_c_deg, thrust=thrust, gas_flow=gas_flow)
        self.ref_tau = float(ref_tau_s) if ref_tau_s else None

        self.estimator = None
        self.eta_min = 0.1
        if estimator is not None and estimator is not False:
            if vehicle is None:
                raise ValueError("estimator için araç parametreleri (vehicle) gerekli")
            from src.control.estimator import CavityEstimator
            est = dict(estimator) if isinstance(estimator, dict) else {}
            self.eta_min = float(est.pop("eta_min", 0.1))
            self.estimator = CavityEstimator(vehicle, **est)

        self.sensors = None
        self.use_pc = self.use_tail = False
        if sensors is not None and sensors is not False:
            if self.estimator is None:
                raise ValueError("sensors için estimator gerekli")
            from src.control.sensors import CavitySensors
            sp = dict(sensors) if isinstance(sensors, dict) else {}
            self.use_pc = bool(sp.pop("use_pc", True))
            self.use_tail = bool(sp.pop("use_tail", True))
            self.sensors = CavitySensors(vehicle, **sp)   # ölçüm GERÇEK araçtan

        self.ff = None
        if trim_ff is not None and trim_ff is not False:
            if self.estimator is None:
                raise ValueError("trim_ff için estimator (iç model) gerekli")
            ff = dict(trim_ff) if isinstance(trim_ff, dict) else {}
            self.ff = dict(gain=float(ff.get("gain", 1.0)), every=max(int(ff.get("every", 10)), 1))

        self.depth = (DepthHold(**depth_hold)
                      if (depth_hold is not None and depth_ref_m is not None) else None)
        self.depth_ref = depth_ref_m
        self.reset()

    def reset(self):
        self.pitch.reset()
        self.yaw.reset()
        if self.estimator is not None:
            self.estimator.reset()
        if self.sensors is not None:
            self.sensors.reset()
        if self.depth is not None:
            self.depth.reset()
        self.t_prev = None
        self.n_calls = 0
        self.ff_e = self.ff_r = 0.0
        self.th_ref_f = self.ps_ref_f = None
        self.log = {}

    def gain_scale(self, V, eta=None):
        if not self.V_ref:
            return 1.0
        eta = self.eta_ref if eta is None else max(eta, self.eta_min)
        k = (self.V_ref / max(V, self.V_min)) ** 2 * self.eta_ref / eta
        return float(np.clip(k, self.k_s_min, self.k_s_max))

    def trim_feedforward(self, t, x, u_ol):
        """(δe_ff, δr_ff) [rad] — tahmin edilen kavite durumunda moment dengesi."""
        est = self.estimator
        m = est.m
        xh = np.array(x, dtype=float)
        xh[IDX_LC], xh[IDX_DC], xh[IDX_PC] = est.Lc, est.Dc, est.pc
        xh[IDX_P] = xh[IDX_Q] = xh[IDX_R] = 0.0
        u0 = ControlInput(delta_e=0.0, delta_r=0.0, **u_ol)
        M0 = m.evaluate(t, xh, u0)[1]["M_body"]
        h = FF_DELTA_STEP
        Me = m.evaluate(t, xh, dataclasses.replace(u0, delta_e=h))[1]["M_body"][1]
        Mr = m.evaluate(t, xh, dataclasses.replace(u0, delta_r=h))[1]["M_body"][2]
        g = self.ff["gain"]
        lim = max(self.pitch.limit, self.yaw.limit)

        def solve(M_0, M_d):
            dMd = (M_d - M_0) / h
            if abs(dMd) < FF_MIN_CONTROL_POWER:
                return 0.0
            return float(np.clip(-g * M_0 / dMd, -lim, lim))

        return solve(M0[1], Me), solve(M0[2], Mr)

    def _filter_ref(self, attr, r, dt, wrap=False):
        rf = getattr(self, attr)
        if self.ref_tau is None or rf is None:
            rf = r
        else:
            diff = wrap_angle(r - rf) if wrap else r - rf
            rf = rf + min(dt / self.ref_tau, 1.0) * diff
        setattr(self, attr, rf)
        return rf

    def update(self, t, x):
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t

        phi, theta, psi = x[IDX_PHI], x[IDX_THETA], x[IDX_PSI]
        q, r = x[IDX_Q], x[IDX_R]
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth = np.cos(theta)
        cth = np.copysign(max(abs(cth), COS_THETA_MIN), cth)
        theta_dot = q * cphi - r * sphi
        psi_dot = (q * sphi + r * cphi) / cth

        V = float(np.linalg.norm(x[SL_NU_LIN]))
        ol = self.open_loop
        gas = schedule_value(t, ol["gas_flow"])
        u_ol = dict(delta_c=np.radians(schedule_value(t, ol["delta_c_deg"])),
                    thrust=schedule_value(t, ol["thrust"]), gas_flow=gas)
        meas = self.sensors.measure(x) if self.sensors is not None else {}
        eta = None
        if self.estimator is not None:
            # Akış açıları: kavite ekseni sapmasının kanat ıslaklığına etkisi için
            # (modelle aynı geometri). Yeni ölçüm değil — durum zaten elde.
            _, alpha_m, beta_m = flow_angles(*x[SL_NU_LIN])
            eta = self.estimator.update(
                dt, V, x[IDX_Z], gas,
                pc_meas=meas.get("pc") if self.use_pc else None,
                tail_gas=meas.get("tail_gas") if self.use_tail else None,
                alpha_eff=alpha_m + u_ol["delta_c"], beta=beta_m)
        k_s = self.gain_scale(V, eta)

        if self.ff is not None and self.n_calls % self.ff["every"] == 0:
            self.ff_e, self.ff_r = self.trim_feedforward(t, x, u_ol)
        self.n_calls += 1

        th_prog = self._filter_ref("th_ref_f", np.radians(schedule_value(t, self.theta_ref)), dt)
        ps_ref = self._filter_ref("ps_ref_f", np.radians(schedule_value(t, self.psi_ref)), dt,
                                  wrap=True)
        th_ref = th_prog
        th_depth = 0.0
        z_ref = np.nan
        if self.depth is not None:
            z_ref = schedule_value(t, self.depth_ref)
            z_dot = kinematics_derivative(x)[0][2]
            th_depth = self.depth.update(x[IDX_Z] - z_ref, z_dot, dt)
            th_ref = th_ref + th_depth
        e_th = th_ref - theta
        e_ps = wrap_angle(ps_ref - psi)

        de = self.pitch.update(e_th, theta_dot, dt, k_s, ref=th_ref, ff=self.ff_e)
        dr = self.yaw.update(e_ps, psi_dot, dt, k_s, ref=ps_ref, ff=self.ff_r)

        est = self.estimator
        self.log = dict(theta_ref=th_ref, psi_ref=ps_ref, e_theta=e_th, e_psi=e_ps,
                        I_pitch=self.pitch.I, I_yaw=self.yaw.I, gain_scale=k_s,
                        eta_hat=np.nan if eta is None else eta,
                        Lc_hat=np.nan if est is None else est.Lc,
                        pc_hat=np.nan if est is None else est.pc,
                        s_L=np.nan if est is None else est.s_L,
                        tail_gas=float(meas["tail_gas"]) if "tail_gas" in meas else np.nan,
                        ff_e=self.ff_e, ff_r=self.ff_r,
                        depth_ref=z_ref, theta_depth=th_depth,
                        sat_pitch=self.pitch.saturated, sat_yaw=self.yaw.saturated)

        return ControlInput(delta_e=de, delta_r=dr, **u_ol)
