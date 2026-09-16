"""
Model tabanlı kavite ve kanat etkinliği tahmincisi (otopilot kazanç çizelgesi için).

Kavite boyu doğrudan ölçülmez. Tahminci, modelin kavite ODE'sini
(VehicleModel.cavity_derivative) ölçülen/bilinen büyüklüklerle kendi içinde entegre eder:
    girdiler : V (hız), Z (derinlik), gaz debisi (komut)
    ölçümler : pc (kavite basınç sensörü), tail_gas (kuyruk gaz sensörü, ikili) — opsiyonel
    durum    : Lc_hat, Dc_hat, pc_hat, s_L (kavite boyu düzeltme çarpanı)
    çıktı    : kanat ıslak açıklığı → etkinlik  eta = wet_span / fin_span  ∈ [0, 1]

Düzeltmeler (gözlemci):
    pc  : dpc_hat += (pc_meas − pc_hat)/tau_pc_obs
          σ = f(pc) kavite hedef boyutlarını belirlediği için havalandırma parametresi (A_v)
          hatasını büyük ölçüde giderir.
    kuyruk gazı: sensör "gaz var" derken tahmin "yok" diyorsa kavite kısa tahmin ediliyor
          → s_L artar (tersinde azalır):  ds_L/dt = k_tail·(g_meas − g_hat),  s_L ∈ [0.5, 2].
          Lc hedefi s_L ile ölçeklenir; uyuşmazlıkta Lc_hat doğrudan da itilir
          (dLc += lc_push·(g_meas − g_hat)). Geometri modeli hatasını (k_g, K_Lc) kuyruk
          geçişinde yakalar; kavite kuyruğun çok ötesindeyken bilgi vermez.

Modelde kanat kaldırması  L = q·c·wet_span·CLα(tam açıklık)·α  olduğundan kontrol etkinliği
ıslak açıklıkla doğrusal orantılıdır; eta bu yüzden doğrudan kazanç bölenidir.

Tahmincinin araç parametreleri gerçek modelinkinden farklı verilebilir (param_scale) —
sağlamlık testleri bunu kullanır. Entegrasyon: açık Euler, dt = kontrol periyodu.
"""

import numpy as np

from src.dynamics.model import VehicleModel, V_FORCE_MIN
from src.hydrodynamics import fins as hfins
from src.hydrodynamics.constants import P_VAP

S_L_MIN, S_L_MAX = 0.5, 2.0


class CavityEstimator:
    def __init__(self, vehicle, Lc0=0.0, Dc0=0.0, pc0=None, param_scale=None,
                 tau_pc_obs=0.02, k_tail=1.0, lc_push=2.0):
        """
        vehicle     : araç parametre sözlüğü (VEHICLE) veya VehicleModel
        param_scale : {anahtar: çarpan} — tahmincinin parametre hatası (ör. {"A_v": 1.2})
        tau_pc_obs  : basınç düzeltme zaman sabiti [s]
        k_tail      : kuyruk sensörü çarpan düzeltme kazancı [1/s]
        lc_push     : kuyruk uyuşmazlığında Lc_hat itme hızı [m/s]
        """
        params = dict(vehicle.params) if isinstance(vehicle, VehicleModel) else dict(vehicle)
        for k, s in (param_scale or {}).items():
            params[k] = params[k] * s
        self.m = VehicleModel(params)
        self.x0 = (float(Lc0), float(Dc0), P_VAP if pc0 is None else float(pc0))
        self.tau_pc_obs = float(tau_pc_obs)
        self.k_tail = float(k_tail)
        self.lc_push = float(lc_push)
        from src.control.sensors import CavitySensors
        self._tail = CavitySensors(self.m)           # sadece geometri (tail_gas_from)
        self.reset()

    def reset(self):
        self.Lc, self.Dc, self.pc = self.x0
        self.s_L = 1.0
        self.tail_gas_hat = False
        self.wet_span = self._wet_span()

    def _wet_span(self):
        m = self.m
        if not m.fins_enabled or m.fin_span <= 1e-6:
            return 0.0
        R_c = hfins.cavity_radius_ellipse(m.fin_x + m.x_body_start, self.Lc, self.Dc)
        return hfins.fin_immersion_ratio(max(R_c, m.R_v_fin), m.R_v_fin, m.fin_span)

    @property
    def eta(self):
        """Kanat kontrol etkinliği (tam ıslak = 1)."""
        return self.wet_span / self.m.fin_span if self.m.fin_span > 0 else 0.0

    def update(self, dt, V, depth, gas_flow, pc_meas=None, tail_gas=None):
        if dt > 0.0:
            _, d = self.m.cavity_derivative(self.Lc, self.Dc, self.pc, max(V, V_FORCE_MIN),
                                            depth, gas_flow)
            tau = max(self.m.tau_cav, 1e-6)
            dLc = (self.s_L * d["Lc_ss"] - self.Lc) / tau
            dDc = d["dDc"]
            dpc = d["dpc"]
            if pc_meas is not None:
                dpc += (pc_meas - self.pc) / self.tau_pc_obs
            self.tail_gas_hat = self._tail.tail_gas_from(self.Lc, self.Dc)
            if tail_gas is not None:
                err = float(tail_gas) - float(self.tail_gas_hat)
                self.s_L = float(np.clip(self.s_L + dt * self.k_tail * err, S_L_MIN, S_L_MAX))
                dLc += self.lc_push * err
            self.Lc = max(self.Lc + dt * dLc, 0.0)
            self.Dc = max(self.Dc + dt * dDc, 0.0)
            self.pc = self.pc + dt * dpc
            self.wet_span = self._wet_span()
        return self.eta
