"""
3-DOF Attitude Autopilot — Kapalı çevrim yaw, pitch, roll servo.

PIDController: tek-eksen PID (yaw, pitch, roll için)
AttitudeAutopilot: 3-DOF servo, durum vektöründen feedback alır
"""

import numpy as np


class PIDController:
    """Tek-eksen PID kontrolör — anti-windup ve angle wrapping desteği."""

    def __init__(self, Kp, Ki=0.0, Kd=0.0, dt=0.001, output_max=20.0, wrap_error=False):
        """
        Args:
            Kp, Ki, Kd: PID kazançları
            dt: zaman adımı [s]
            output_max: çıkış satürasyon sınırı [deg] (kayançlar bu limitte kırpılır)
            wrap_error: True ise hata -π to π aralığında sarılır (yaw için)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_max = output_max
        self.wrap_error = wrap_error

        self.error_integral = 0.0
        self.error_prev = 0.0

    def step(self, error):
        """
        PID adımı: u = Kp·e + Ki·∫e·dt + Kd·de/dt

        Args:
            error: desired - current (hatanın yönü)

        Returns:
            output: [-output_max, +output_max] satürasyonlu kontrol komut [deg]
        """
        if self.wrap_error:
            # Yaw hatası -π to π aralığında
            error = (error + np.pi) % (2 * np.pi) - np.pi

        # İntegral terim
        self.error_integral += error * self.dt

        # Türev terim
        d_error = (error - self.error_prev) / self.dt if self.dt > 0 else 0.0
        self.error_prev = error

        # PID hesabı
        output = self.Kp * error + self.Ki * self.error_integral + self.Kd * d_error

        # Satürasyon
        return np.clip(output, -self.output_max, self.output_max)

    def reset(self):
        """Durumu sıfırla (simülasyon başında)."""
        self.error_integral = 0.0
        self.error_prev = 0.0


class AttitudeAutopilot:
    """
    3-DOF attitude servo: yaw (ψ) → δr, pitch (θ) → δe, roll (φ) → δa.

    ControlInputBlock arayüzünü uygular.
    """

    def __init__(self, pid_yaw, pid_pitch, pid_roll=None, dt=0.001):
        """
        Args:
            pid_yaw: {"Kp": ..., "Ki": ..., "Kd": ..., "output_max": ...}
            pid_pitch: {"Kp": ..., ...}
            pid_roll: {"Kp": ..., ...} (opsiyonel, None ise roll kontrol off)
            dt: zaman adımı [s]
        """
        self.dt = dt

        self.pid_yaw = PIDController(**pid_yaw, dt=dt, wrap_error=True)
        self.pid_pitch = PIDController(**pid_pitch, dt=dt, wrap_error=False)

        if pid_roll is not None:
            self.pid_roll = PIDController(**pid_roll, dt=dt, wrap_error=False)
        else:
            self.pid_roll = None

    def step(self, t, x_state, attitude_refs):
        """
        Kontrol komutları hesapla.

        Args:
            t: mevcut zaman [s]
            x_state: durum vektörü [u,v,w, p,q,r, φ,θ,ψ, X,Y,Z, Lc,pc]
            attitude_refs: {
                "ψ_desired": float [rad],
                "θ_desired": float [rad],
                "φ_desired": float [rad] (opsiyonel, default 0),
                "δc_fixed": float [deg] (kavitatör, kontrol edilmez)
            }

        Returns:
            (δc, δe, δr, δa): kontrol komutları [deg]
                δc: kavitatör (açık çevrim veya sabit)
                δe: elevator (pitch servo)
                δr: rudder (yaw servo)
                δa: aileron (roll servo)
        """
        # Durum vektöründen attitude al
        φ_current = x_state[6]   # roll
        θ_current = x_state[7]   # pitch
        ψ_current = x_state[8]   # yaw

        # İstenilen attitude'lar
        ψ_desired = attitude_refs.get("ψ_desired", ψ_current)
        θ_desired = attitude_refs.get("θ_desired", 0.0)
        φ_desired = attitude_refs.get("φ_desired", 0.0)

        # Hatalar
        ψ_error = ψ_desired - ψ_current
        θ_error = θ_desired - θ_current
        φ_error = φ_desired - φ_current

        # PID çıkışları
        δr = self.pid_yaw.step(ψ_error)
        δe = self.pid_pitch.step(θ_error)
        δa = self.pid_roll.step(φ_error) if self.pid_roll else 0.0

        # δc (kavitatör) otopilot tarafından kontrol edilmez
        δc = attitude_refs.get("δc_fixed", 0.0)

        return δc, δe, δr, δa

    def reset(self):
        """Tüm PID state'lerini sıfırla."""
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        if self.pid_roll:
            self.pid_roll.reset()
