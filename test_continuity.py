"""
Kuvvet sürekliliği testi (düz script, exit 0 = geçti).

Tasarım kriteri: rejim geçişlerinde (kavite oluşumu, kesit kapanması, ıslak yay
bitişik↔serbest, transom teması/çıkışı, σ = 1) kuvvet eğrisinde sıçrama olmamalı.

Yöntem: her tarama N, 2N ve 4N noktayla yapılır; en büyük komşu farkın inceltmeyle oranı:
  düzgün (Lipschitz)            → ≈ 2
  √ tipi sonsuz eğim (sürekli)   → ≈ 1.41   Logvinovich kapanma profili R ∝ √(Lc − x) dikey
                                           teğetle kapanır; kavite çok kısayken ilk kesit bu
                                           geçişi mm'ler içinde yapar — fiziksel, sıçrama değil.
  sıçrama                        → ≈ 1
Geçer: maks_fark(4N) ≤ mutlak_eşik  veya  iki inceltme oranı da ≥ 1.25.
Geçemezse adaptif yakınlaştırma: en büyük farkın olduğu aralık (±1 adım) yeniden N/2N/4N
noktayla taranır (çözünürlük her seferinde ~200×), 3 kez. Sıçrama her ölçekte sıçrama
kalır; tarama adımından dar ama sürekli bir geçiş (ör. kavite 5 cm iken ilk kesitin
mm ölçeğinde kapanması) yakınlaştırınca çözülür.

Test duyarlılığı: aynı taramalar legacy_exact=True ile de çalıştırılır; bilinen legacy
sıçramalarının (F_skin, F_press, F_planing, Lc_ss ...) YAKALANMASI beklenir.
Çalıştır:  python test_continuity.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dynamics.model import VehicleModel, ControlInput
from src.dynamics.state import make_state, IDX_LC
from src.simulation.run_simulation import load_config

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                               "case1_config.py"))
RESULTS = []
N_BASE = 400
RATIO_MIN = 1.25

# büyüklük -> mutlak eşik (bu kadar küçük farklar sıçrama sayılmaz)
ABS_TOL = {"F_x": 2.0, "F_z": 2.0, "M_y": 2.0, "F_skin": 2.0, "F_press": 2.0,
           "F_planing": 2.0, "F_buoy": 1.0, "F_body_lift": 1.0, "F_fin_total": 2.0,
           "cover": 1e-3, "F_cav": 2.0, "Cx": 1e-4, "Cx_force": 1e-4, "Lc_ss": 1e-3,
           "Dc_ss": 1e-4, "dLc": 1e-2}


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")


def quantities(model, x, u):
    xdot, d = model.evaluate(0.0, x, u)
    return {"F_x": d["F_body"][0], "F_z": d["F_body"][2], "M_y": d["M_body"][1],
            "F_skin": d["F_skin"], "F_press": d["F_press"], "F_planing": d["F_planing"],
            "F_buoy": d["F_buoy"], "F_body_lift": d["F_body_lift"],
            "F_fin_total": d["F_fin_total"], "cover": d["cover"], "F_cav": d["F_cav"],
            "Cx": d["Cx"], "Cx_force": d["Cx_force"], "Lc_ss": d["Lc_ss"],
            "Dc_ss": d["Dc_ss"], "dLc": xdot[IDX_LC]}


def sweep(model, lo, hi, n, make_x, u):
    vals = [quantities(model, make_x(s), u) for s in np.linspace(lo, hi, n)]
    return {k: np.array([v[k] for v in vals]) for k in vals[0]}


def max_step(arr):
    da = np.abs(np.diff(arr))
    i = int(np.argmax(da))
    return da[i], i


def refinement(model, lo, hi, make_x, u, key, n):
    """[lo, hi] aralığında N, 2N, 4N tarama → (geçti mi, m1, m4, r1, r2, konum, 4N adımı)."""
    lv = [sweep(model, lo, hi, m * n + 1, make_x, u)[key] for m in (1, 2, 4)]
    grid = np.linspace(lo, hi, 4 * n + 1)
    m1, _ = max_step(lv[0])
    m2, _ = max_step(lv[1])
    m4, i = max_step(lv[2])
    r1, r2 = m1 / max(m2, 1e-300), m2 / max(m4, 1e-300)
    ok = m4 <= ABS_TOL[key] or (r1 >= RATIO_MIN and r2 >= RATIO_MIN)
    return ok, m1, m4, r1, r2, grid[i], grid[1] - grid[0]


def analyze(model, label, lo, hi, make_x, u, keys, zooms=3, n_zoom=50):
    jumps = {}
    for k in keys:
        res = refinement(model, lo, hi, make_x, u, k, N_BASE)
        ok, at, h = res[0], res[5], res[6]
        z = 0
        while not ok and z < zooms:
            res = refinement(model, at - h, at + h, make_x, u, k, n_zoom)
            ok, at, h = res[0], res[5], res[6]
            z += 1
        jumps[k] = res[:6] + (z,)
    return jumps


def lc_state(V, alpha_deg):
    a = np.radians(alpha_deg)

    def make(L):
        Dc = 0.123 * L if L < 4.2 else 0.5166 + 0.02 * (L - 4.2)
        return make_state(u=V * np.cos(a), w=V * np.sin(a), Z=10.0, Lc=L, Dc=Dc, pc=94000.0)
    return make


SWEEPS = [
    ("Lc 0→8 m (V=36, α=+1°): kavite oluşumu, kesit kapanması, transom",
     0.0, 8.0, lc_state(36.0, 1.0),
     ["F_x", "F_z", "M_y", "F_skin", "F_press", "F_planing", "F_buoy", "F_body_lift",
      "F_fin_total", "cover"]),
    ("Lc 0→8 m (V=25, α=−2°): büyük kavite ekseni sapması, ıslak yay sınırı",
     0.0, 8.0, lc_state(25.0, -2.0),
     ["F_x", "F_z", "M_y", "F_skin", "F_press", "F_planing", "F_buoy", "cover"]),
    ("α −4°→+4° (olgun kavite, V=36): |h| işaret değişimi, planing girişi/çıkışı",
     -4.0, 4.0,
     lambda a: make_state(u=36.0 * np.cos(np.radians(a)), w=36.0 * np.sin(np.radians(a)),
                          Z=10.0, Lc=4.3, Dc=0.52, pc=94000.0),
     ["F_x", "F_z", "M_y", "F_press", "F_planing", "F_skin", "cover"]),
    ("V 12→60 m/s (Lc=3.5, Dc=0.45): kavite ekseni sapması h ∝ 1/V²",
     12.0, 60.0,
     lambda V: make_state(u=V, Z=10.0, Lc=3.5, Dc=0.45, pc=94000.0),
     ["F_x", "F_z", "M_y", "F_skin", "F_press", "F_planing", "cover"]),
]


def sigma_sweep_state(model, V=12.0):
    p_inf = model.ambient_pressure(10.0)
    return lambda s: make_state(u=V, Z=10.0, Lc=1.0, Dc=0.2, pc=p_inf - 0.5 * 1025.0 * V * V * s)


def run_all(model, u, u_nogas):
    out = []
    for label, lo, hi, make_x, keys in SWEEPS:
        out.append((label, analyze(model, label, lo, hi, make_x, u, keys)))
    out.append(("σ 0.6→1.6 (V=12, gazsız): kavite oluşum eşiği σ=1",
                analyze(model, "sigma", 0.6, 1.6, sigma_sweep_state(model), u_nogas,
                        ["Lc_ss", "Dc_ss", "Cx", "Cx_force", "F_cav", "dLc"])))
    return out


def main():
    u = ControlInput(delta_c=np.radians(2.0), thrust=6000.0, gas_flow=15000.0)
    u_nogas = ControlInput(delta_c=np.radians(2.0))

    print("\nSürekli model (legacy_exact=False)")
    smooth = VehicleModel(dict(CFG.VEHICLE, legacy_exact=False))
    for label, jumps in run_all(smooth, u, u_nogas):
        print(f"\n  {label}")
        for k, (ok, m1, m4, r1, r2, at, z) in jumps.items():
            check(f"{k:12s} sürekli", ok,
                  f"maks fark N={m1:.4g} 4N={m4:.4g} oranlar={r1:.2f},{r2:.2f} @{at:.5g}"
                  + (f" (yakınlaştırma {z})" if z else ""))

    print("\nTest duyarlılığı — legacy_exact=True (sıçramalar YAKALANMALI)")
    legacy = VehicleModel(dict(CFG.VEHICLE, legacy_exact=True))
    detected = set()
    for label, jumps in run_all(legacy, u, u_nogas):
        bad = [(k, m4, at) for k, (ok, m1, m4, r1, r2, at, z) in jumps.items() if not ok]
        detected |= {k for k, _, _ in bad}
        print(f"  {label}: " + (", ".join(f"{k} ({mf:.3g} @{at:.3f})" for k, mf, at in bad) or "—"))
    expected = {"F_skin", "F_press", "F_planing", "Lc_ss", "Cx"}
    check("legacy sıçramaları test tarafından yakalandı", expected <= detected,
          f"yakalanan={sorted(detected)}")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} kontrol geçti")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
