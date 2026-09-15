"""
Legacy altın referans: supercavitation_gui_LIVE_v10.py içindeki simulate()
fonksiyonunu GUI varsayılanlarıyla (Case 1) GERÇEKTEN çalıştırır.

Legacy dosyaya dokunulmaz. Modül import edilirken matplotlib/tkinter GUI
parçaları geçici olarak MagicMock ile değiştirilir, import bittikten sonra
sys.modules eski haline döndürülür (böylece aynı süreçte gerçek matplotlib
ile grafik çizmek bozulmaz).

Kullanım:
    from src.validation.legacy_reference import run_legacy, CASE1_PARAMS
    r = run_legacy()               # Case 1, t_max=0.5, dt=0.001
    r = run_legacy(steady_v_mode=True)

Legacy'nin bilinen tuhaflıkları (port sırasında tespit edildi):
  * Hız güncellemesi V[i+1] = V[i-1] + a_i*dt  (vi = V[i-1]). Tek/çift
    indeksler ayrışır ve V fiziksel hızın ~yarısı oranında değişir.
    Aynı a[] ile tam oranlı Euler t=0.5'te 33.88 m/s verirken legacy 36.93.
  * i. adımdaki kuvvetler V[i-1] ile hesaplanır (kayıtlı V[i] ile değil).
  * Kanat ıslaklığı gecikmeli Lc/Dc yerine anlık Lc_ss/Dc_ss ile hesaplanır.
  * Döngü içindeki gövde α-lift integrali, satır 1354'te basit Munk formülü
    ile ÜZERİNE YAZILIR (ölü kod).
  * Fd, kanat drag'ını İÇERMEZ; ivmede ayrıca çıkarılır.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_MOCKED = ("matplotlib", "matplotlib.figure", "matplotlib.backends",
           "matplotlib.backends.backend_tkagg", "matplotlib.patches")

_LEGACY = None

CASE1_PARAMS = dict(
    depth=10.0, diam_cav=0.20, mass=350.0, thrust=6000.0, veh_len=4.0, veh_diam=0.30,
    L_taper=0.60, gas_flow=15000.0, t_max=0.5, dt=0.001, x_cg=2.4, x_mg=2.4, x_cg_mass=2.4,
    V_init=40.0, alpha_aoa=-1.0, delta_cav=2.0, CL_alpha_body=2.5, Cdc_body=0.4, k_dev=0.4,
    A_v=0.020, K_slender=0.5, cone_lift_gain=0.50, K_cone_lift=0.0, cone_apex_deg=40.0,
    cone_drag_visc=0.18, k_g_cavity=0.78, K_Dc_factor=1.0, K_Lc_factor=1.0,
    fin_chord=0.10, fin_span=0.25, fin_x_pos=3.80,
    fin_delta_1=-0.80, fin_delta_2=0.0, fin_delta_3=-0.80, fin_delta_4=0.0,
    geom_model="savchenko", vent_mode="Q", cavitator_type="cone", cav_type="cone",
    body_volume_correction=True, fins_enabled=True, steady_v_mode=False, planing_enable=True,
)


def load_legacy():
    """Legacy modülü GUI bağımlılıkları mock'lanmış olarak import eder."""
    global _LEGACY
    if _LEGACY is not None:
        return _LEGACY
    saved = {m: sys.modules.get(m) for m in _MOCKED}
    try:
        for m in _MOCKED:
            sys.modules[m] = MagicMock()
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        import importlib
        _LEGACY = importlib.import_module("supercavitation_gui_LIVE_v10")
    finally:
        for m, mod in saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod
    return _LEGACY


def run_legacy(**overrides):
    """Case 1 parametreleriyle (üzerine yazılabilir) legacy simulate() çalıştırır."""
    L = load_legacy()
    p = dict(CASE1_PARAMS)
    p.update(overrides)
    r = L.simulate(p)
    return {k: (np.asarray(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in r.items()}


def legacy_force_inputs(r, i=-1):
    """
    i. adımda legacy'nin kuvvet hesabında GERÇEKTEN kullandığı durum.
    (V için V[i-1]; Lc, Dc, pc, sigma için i. kayıt.)
    """
    n = len(r["t"])
    i = i % n
    V_used = r["V"][i - 1] if i > 0 else r["V"][0]
    return dict(V=float(V_used), Lc=float(r["Lc"][i]), Dc=float(r["Dc"][i]),
                pc=float(r["pc"][i]), sigma=float(r["sigma"][i]), Cq=float(r["Cq"][i]),
                t=float(r["t"][i]))


CASE1_COMPONENT_KEYS = ("V", "sigma", "Lc", "Dc", "pc", "Cq", "Fd", "F_cav", "F_skin",
                        "F_press", "F_body_drag", "F_fin_drag", "Fz", "F_cavz",
                        "F_planing", "F_body_lift", "F_fin_total", "F_buoy", "F_grav",
                        "My", "My_no_grav", "M_fin")


if __name__ == "__main__":
    r = run_legacy()
    for k in CASE1_COMPONENT_KEYS:
        print(f"{k:12s} = {r[k][-1]:12.2f}")
    comb = r["F_cavz"][-1] + r["F_body_lift"][-1] + r["F_fin_total"][-1]
    print("F_cavz+F_body_lift+F_fin_total =", round(comb, 2),
          "(dokümandaki -2109 N buna karşılık geliyor)")
