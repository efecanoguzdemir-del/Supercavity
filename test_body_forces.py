"""
Gövde ıslanma modeli (src/hydrodynamics/body.py) — legacy karşılaştırma testi.

Legacy simulate() Case 1 ile çalıştırılır; seçilen adımlarda legacy'nin kuvvet
hesabında kullandığı girdiler (V[i-1], Lc[i], Dc[i], Cx[i]) compute_body_forces'a
verilir ve çıktı dizileriyle karşılaştırılır.
Tolerans: |fark| ≤ max(%5·|legacy|, 0.5 N).
Çalıştır:  python test_body_forces.py   (düz script, pytest yok)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hydrodynamics.body import compute_body_forces
from src.validation.legacy_reference import run_legacy, legacy_force_inputs, CASE1_PARAMS

REL_TOL = 0.05
ABS_TOL = 0.5

# (anahtar_legacy, anahtar_yeni, birim, mutlak tolerans)
COMPONENTS = [
    ("F_skin", "F_skin", "N", ABS_TOL),
    ("F_body_lift", "F_body_lift", "N", ABS_TOL),
    ("F_body_drag", "F_body_drag", "N", ABS_TOL),
    ("F_buoy", "F_buoy", "N", ABS_TOL),
    ("V_sub", "V_submerged", "m3", 1e-6),
    ("cover", "cover", "-", 1e-4),
]


def check_step(r, i):
    inp = legacy_force_inputs(r, i)
    alpha = np.radians(CASE1_PARAMS["alpha_aoa"])
    Cx = float(r["Cx"][i])
    out = compute_body_forces(inp["V"], alpha, inp["Lc"], inp["Dc"], Cx, CASE1_PARAMS)

    print(f"\n--- t = {inp['t']:.3f} s  (i={i % len(r['t'])})  V_used={inp['V']:.3f} m/s  "
          f"Lc={inp['Lc']:.4f} m  Dc={inp['Dc']:.4f} m  Cx={Cx:.4f}  "
          f"{'FULLY-WET' if out['fully_wet'] else ''}")
    print(f"  {'bileşen':12s} {'legacy':>14s} {'yeni':>14s} {'fark %':>9s}  sonuç")
    ok_all = True
    for kl, kn, unit, abs_tol in COMPONENTS:
        lv = float(r[kl][i]); nv = out[kn]
        diff = abs(nv - lv)
        pct = 100.0 * diff / abs(lv) if abs(lv) > 1e-12 else (0.0 if diff < 1e-12 else np.inf)
        ok = diff <= max(REL_TOL * abs(lv), abs_tol)
        ok_all &= ok
        print(f"  {kl:12s} {lv:14.6g} {nv:14.6g} {pct:9.3f}  {'OK' if ok else 'FAIL'} [{unit}]")
    print(f"  (bilgi) M_buoy={out['M_buoy']:.4g} N·m  M_body_lift={out['M_body_lift']:.4g} N·m  "
          f"wet_area={out['wet_area']:.4g} m²  wet_len={out['wet_len']:.3f} m  "
          f"F_body_lift_section={out['F_body_lift_section']:.4g} N")
    return ok_all


def main():
    r = run_legacy()
    t = r["t"]
    n = len(t)
    steps = [int(round(0.005 / 0.001)), int(round(0.05 / 0.001)),
             int(round(0.2 / 0.001)), int(round(0.35 / 0.001)), n - 1]
    steps = sorted(set(s for s in steps if 0 < s < n))

    results = [check_step(r, i) for i in steps]

    # Belgelenmiş son değerler ile ek kontrol
    print("\nDokümante legacy son değerleri: F_skin=7.68, F_body_lift=-2.15, "
          "F_body_drag=0.04, F_buoy=2.76")
    print("Legacy çıktı (t=%.3f): F_skin=%.2f F_body_lift=%.2f F_body_drag=%.2f F_buoy=%.2f"
          % (t[-1], r["F_skin"][-1], r["F_body_lift"][-1], r["F_body_drag"][-1], r["F_buoy"][-1]))

    # Birim sanity: kavite yok → tam ıslak, tam Arşimet
    out0 = compute_body_forces(40.0, np.radians(-1.0), 0.0, 0.0, 0.5, CASE1_PARAMS)
    assert out0["fully_wet"] and abs(out0["cover"]) < 1e-12 and out0["F_buoy"] > 0
    print("\nSanity (kavite yok): F_buoy=%.1f N, cover=%.3f  OK" % (out0["F_buoy"], out0["cover"]))

    if all(results):
        print("\nSONUÇ: TÜM ADIMLAR GEÇTİ")
        return 0
    print("\nSONUÇ: BAŞARISIZ ADIM VAR")
    return 1


if __name__ == "__main__":
    sys.exit(main())
