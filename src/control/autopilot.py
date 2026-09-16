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
Kazanç çizelgesi: k_s = clip((V_ref/V)², 0.25, 4) — kanat etkinliği ~V² (V_ref=None → k_s=1).

Birimler: Kp [rad/rad], Ki [1/s], Kd [s]; config'te limitler derece.
"""

import numpy as np

from src.dynamics.blocks import Controller, schedule_value
from src.dynamics.model import ControlInput
from src.dynamics.state import IDX_PHI, IDX_THETA, IDX_PSI, IDX_Q, IDX_R, SL_NU_LIN

GAIN_SCALE_MIN, GAIN_SCALE_MAX = 0.25, 4.0
COS_THETA_MIN = 1e-3


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

    def update(self, e, omega_meas, dt, k_s=1.0, ref=0.0):
        """e = ref − ölçüm (sarılmış olabilir); ref sadece b<1 ise P terimini etkiler."""
        e_p = e - (1.0 - self.b) * ref

        def command(I):
            raw = self.trim + self.sign * k_s * (self.Kp * e_p + I - self.Kd * omega_meas)
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


class AttitudeAutopilot(Controller):
    """
    θ_ref, ψ_ref (derece; sayı veya [(t, değer), ...]) takibi → δe, δr.
    Açık çevrim kanallar (schedule): delta_c_deg, thrust, gas_flow.
    """

    def __init__(self, pitch, yaw, theta_ref_deg=0.0, psi_ref_deg=0.0, V_ref=None,
                 delta_c_deg=0.0, thrust=0.0, gas_flow=0.0, V_min=5.0):
        self.pitch = AxisPID(sign=-1.0, **pitch)
        self.yaw = AxisPID(sign=+1.0, **yaw)
        self.theta_ref = theta_ref_deg
        self.psi_ref = psi_ref_deg
        self.V_ref = V_ref
        self.V_min = float(V_min)
        self.open_loop = dict(delta_c_deg=delta_c_deg, thrust=thrust, gas_flow=gas_flow)
        self.reset()

    def reset(self):
        self.pitch.reset()
        self.yaw.reset()
        self.t_prev = None
        self.log = {}

    def gain_scale(self, V):
        if not self.V_ref:
            return 1.0
        return float(np.clip((self.V_ref / max(V, self.V_min)) ** 2, GAIN_SCALE_MIN, GAIN_SCALE_MAX))

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
        k_s = self.gain_scale(V)

        th_ref = np.radians(schedule_value(t, self.theta_ref))
        ps_ref = np.radians(schedule_value(t, self.psi_ref))
        e_th = th_ref - theta
        e_ps = wrap_angle(ps_ref - psi)

        de = self.pitch.update(e_th, theta_dot, dt, k_s, ref=th_ref)
        dr = self.yaw.update(e_ps, psi_dot, dt, k_s, ref=ps_ref)

        self.log = dict(theta_ref=th_ref, psi_ref=ps_ref, e_theta=e_th, e_psi=e_ps,
                        I_pitch=self.pitch.I, I_yaw=self.yaw.I, gain_scale=k_s,
                        sat_pitch=self.pitch.saturated, sat_yaw=self.yaw.saturated)

        ol = self.open_loop
        return ControlInput(
            delta_e=de, delta_r=dr,
            delta_c=np.radians(schedule_value(t, ol["delta_c_deg"])),
            thrust=schedule_value(t, ol["thrust"]),
            gas_flow=schedule_value(t, ol["gas_flow"]),
        )
