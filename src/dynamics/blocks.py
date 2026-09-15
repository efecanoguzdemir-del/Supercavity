"""
Kontrol girdisi sözleşmesi ve implementasyonları.

ControlInputBlock: Abstract kontrol bloğu
OpenLoopController: Zaman-tabanlı schedule'lar
ClosedLoopController: AttitudeAutopilot + açık çevrim δc/Cq
"""

from abc import ABC, abstractmethod
import numpy as np


class ControlInputBlock(ABC):
    """Kontrol girdisi sözleşmesi."""

    @abstractmethod
    def step(self, t, x_state, params):
        """
        Kontrol komutları hesapla.

        Args:
            t: zaman [s]
            x_state: durum vektörü
            params: konfigürasyon dict'i

        Returns:
            (δc, δe, δr, δa, Cq): kontrol komutları
        """
        pass


class OpenLoopController(ControlInputBlock):
    """Zaman-tabanlı açık çevrim schedule'lar."""

    def __init__(self, δc_schedule, δe_schedule, δr_schedule, δa_schedule, Cq_schedule):
        """
        Args:
            *_schedule: [( (t0, val0), (t1, val1), ... )]
                Zero-order hold: t < t0 → val0, t0 ≤ t < t1 → val0, vs.
        """
        self.δc_schedule = δc_schedule
        self.δe_schedule = δe_schedule
        self.δr_schedule = δr_schedule
        self.δa_schedule = δa_schedule
        self.Cq_schedule = Cq_schedule

    def step(self, t, x_state, params):
        δc = self._get_schedule_value(t, self.δc_schedule)
        δe = self._get_schedule_value(t, self.δe_schedule)
        δr = self._get_schedule_value(t, self.δr_schedule)
        δa = self._get_schedule_value(t, self.δa_schedule)
        Cq = self._get_schedule_value(t, self.Cq_schedule)
        return δc, δe, δr, δa, Cq

    @staticmethod
    def _get_schedule_value(t_now, schedule):
        """
        Schedule listesinden t_now anındaki değeri ZOH döndürür.

        schedule: [(t0, v0), (t1, v1), ...]
        t < t0 ise v0, t0 ≤ t < t1 ise v0, t1 ≤ t < t2 ise v1, vs.
        """
        if not schedule:
            return 0.0

        current_val = schedule[0][1]  # t < t0 ise ilk değer
        for (t_step, val) in schedule:
            if t_now >= t_step:
                current_val = val
            else:
                break
        return current_val


class ClosedLoopController(ControlInputBlock):
    """3-DOF AttitudeAutopilot + açık çevrim δc/Cq."""

    def __init__(self, attitude_autopilot, δc_schedule, Cq_schedule):
        """
        Args:
            attitude_autopilot: AttitudeAutopilot instance
            δc_schedule, Cq_schedule: açık çevrim schedule'lar
        """
        self.autopilot = attitude_autopilot
        self.δc_schedule = δc_schedule
        self.Cq_schedule = Cq_schedule

    def step(self, t, x_state, params):
        # AttitudeAutopilot'tan δe, δr, δa al
        attitude_refs = params.get("attitude_refs", {})
        δc_ap, δe_ap, δr_ap, δa_ap = self.autopilot.step(t, x_state, attitude_refs)

        # δc ve Cq açık çevrim schedule'dan
        δc = self._get_schedule_value(t, self.δc_schedule)
        Cq = self._get_schedule_value(t, self.Cq_schedule)

        # δe, δr, δa otopilot tarafından sağlanır
        return δc, δe_ap, δr_ap, δa_ap, Cq

    @staticmethod
    def _get_schedule_value(t_now, schedule):
        """Schedule ZOH değeri."""
        if not schedule:
            return 0.0

        current_val = schedule[0][1]
        for (t_step, val) in schedule:
            if t_now >= t_step:
                current_val = val
            else:
                break
        return current_val
