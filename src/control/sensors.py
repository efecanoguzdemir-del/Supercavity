"""
Otopilotun kullandığı sensör modelleri (simülasyonda gerçek durumdan üretilir).

  pc        kavite basınç sensörü [Pa]      — opsiyonel Gauss gürültüsü
  tail_gas  kuyruk gaz sensörü (ikili)      — araç sonundaki gövde yüzeyi (x = L) gaz
            içinde mi: kavite zarfı yarıçapı R_c(L) > gövde yarıçapı R(L)·(1 + margin).
            Kavite ekseni sapması (yerçekimi, α) ihmal edilir — kanat ıslaklığı hesabıyla
            aynı eksenel simetrik zarf.

Kontrolcü tam durum vektörünü alır (simülatör sözleşmesi); kavite durumlarına (Lc, Dc, pc)
yalnızca bu sensörler üzerinden erişmelidir.
"""

import numpy as np

from src.dynamics.model import VehicleModel
from src.dynamics.state import IDX_LC, IDX_DC, IDX_PC
from src.hydrodynamics import body as hbody
from src.hydrodynamics import fins as hfins


class CavitySensors:
    def __init__(self, vehicle, pc_noise_pa=0.0, tail_margin=0.0, seed=0):
        self.m = vehicle if isinstance(vehicle, VehicleModel) else VehicleModel(vehicle)
        self.pc_noise = float(pc_noise_pa)
        self.tail_margin = float(tail_margin)
        self.x_tail = self.m.L
        self.R_tail = hbody.vehicle_radius(self.x_tail, self.m.R_n, self.m.R_v, self.m.L_taper)
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def tail_gas_from(self, Lc, Dc):
        """Verilen kavite boyutlarında kuyruk sensörünün göreceği değer (tahminci de kullanır)."""
        R_c = hfins.cavity_radius_ellipse(self.x_tail + self.m.x_body_start, Lc, Dc)
        return bool(R_c > self.R_tail * (1.0 + self.tail_margin))

    def measure(self, x):
        pc = float(x[IDX_PC])
        if self.pc_noise > 0.0:
            pc += self.pc_noise * self.rng.standard_normal()
        return dict(pc=pc, tail_gas=self.tail_gas_from(x[IDX_LC], x[IDX_DC]))
