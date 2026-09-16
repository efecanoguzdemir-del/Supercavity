"""
Kontrol girdisi sözleşmesi (simulation/simulator.py bunu kullanır).

  Controller          ayrık kontrolcü tabanı: update(t, x) -> ControlInput
  ScheduleController  açık çevrim ZOH zaman programı
  schedule_value      ZOH yardımcı

Kapalı çevrim otopilot: src/control/autopilot.py::AttitudeAutopilot (Controller alt sınıfı).
"""

from abc import ABC, abstractmethod

import numpy as np

from .model import ControlInput


def schedule_value(t, spec, default=0.0):
    """ZOH değer. spec: sayı, None veya [(t0, v0), (t1, v1), ...] (t < t0 → v0)."""
    if spec is None:
        return default
    if np.isscalar(spec):
        return float(spec)
    if len(spec) == 0:
        return default
    current = spec[0][1]
    for tk, vk in spec:
        if t >= tk:
            current = vk
        else:
            break
    return float(current)


class Controller(ABC):
    """
    Ayrık kontrolcü. Simülatör update()'i her dt_control periyodunun başında
    BİR kez çağırır; dönen komut RK4 alt aşamaları boyunca sabit (ZOH) tutulur.
    İç durum (integratör vb.) sadece update() içinde ilerletilmelidir.
    """

    def reset(self):
        """Koşum başında çağrılır."""

    @abstractmethod
    def update(self, t, x):
        """t [s], x (15 durum, ölçüm) -> ControlInput."""


class ScheduleController(Controller):
    """
    Açık çevrim zaman programı. Her kanal sayı veya [(t, değer), ...] listesi:
      delta_e_deg, delta_r_deg, delta_c_deg [derece], thrust [N],
      gas_flow [L/min veya Cq, vent_mode'a göre]
    """

    def __init__(self, delta_e_deg=0.0, delta_r_deg=0.0, delta_c_deg=0.0,
                 thrust=0.0, gas_flow=0.0):
        self.spec = dict(delta_e_deg=delta_e_deg, delta_r_deg=delta_r_deg,
                         delta_c_deg=delta_c_deg, thrust=thrust, gas_flow=gas_flow)

    def update(self, t, x):
        s = self.spec
        return ControlInput(
            delta_e=np.radians(schedule_value(t, s["delta_e_deg"])),
            delta_r=np.radians(schedule_value(t, s["delta_r_deg"])),
            delta_c=np.radians(schedule_value(t, s["delta_c_deg"])),
            thrust=schedule_value(t, s["thrust"]),
            gas_flow=schedule_value(t, s["gas_flow"]),
        )
