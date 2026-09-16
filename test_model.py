"""
src/dynamics/model.py testleri (düz script, exit 0 = geçti).

(A) Statik regresyon: legacy Case 1'in kaydettiği durumlarda (V[i-1], Lc, Dc, pc)
    model kuvvet bileşenleri legacy çıktılarıyla karşılaştırılır.
    Tolerans: |fark| ≤ max(%2·|legacy|, 1 N)  (moment için 1 N·m).
(S) İşaret/ayrışma: δe sadece pitch, δr sadece yaw; kanat sönümleri; kavite
    denge noktasında türev ~0; kavitesiz/düşük hızda NaN yok.
Çalıştır:  python test_model.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dynamics.model import VehicleModel, ControlInput, fin_mixer
from src.dynamics.state import make_state, IDX_LC, IDX_DC, IDX_PC, SL_NU
from src.hydrodynamics.cavity import cavity_geometry
from src.hydrodynamics.constants import CX0_DISK
from src.validation.legacy_reference import run_legacy, legacy_force_inputs, CASE1_PARAMS

REL_TOL = 0.02
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")


def case1_model(**extra):
    p = dict(CASE1_PARAMS)
    p.update(fins_use_steady_cavity=True, legacy_exact=True)
    p.update(extra)
    return VehicleModel(p)


def case1_control():
    P = CASE1_PARAMS
    return ControlInput(delta_e=np.radians(P["fin_delta_1"]), delta_r=0.0,
                        delta_c=np.radians(P["delta_cav"]), thrust=P["thrust"],
                        gas_flow=P["gas_flow"])


def state_from_legacy(inp, alpha_deg):
    a = np.radians(alpha_deg)
    return make_state(u=inp["V"] * np.cos(a), w=inp["V"] * np.sin(a),
                      Z=CASE1_PARAMS["depth"], Lc=inp["Lc"], Dc=inp["Dc"], pc=inp["pc"])


# ---------------------------------------------------------------------------
def test_static_regression():
    print("\n(A) Statik regresyon — legacy Case 1")
    r = run_legacy()
    n = len(r["t"])
    model = case1_model()
    u = case1_control()
    keys = ["sigma", "Cx", "F_cav", "F_skin", "F_press", "F_body_drag", "F_fin_drag", "Fd",
            "F_cavz", "F_planing", "F_body_lift", "F_fin_total", "F_buoy", "F_grav"]
    for i in (5, 50, 200, 350, n - 1):
        inp = legacy_force_inputs(r, i)
        x = state_from_legacy(inp, CASE1_PARAMS["alpha_aoa"])
        _, d = model.evaluate(inp["t"], x, u)
        print(f"  --- t={inp['t']:.3f}  V={inp['V']:.3f}  Lc={inp['Lc']:.3f}  "
              f"Dc={inp['Dc']:.4f}  pc={inp['pc']:.0f}  rejim={d['planing_regime']}")
        rows = [(k, float(r[k][i]), float(d[k]), 1.0 if k != "sigma" and k != "Cx" else 1e-4)
                for k in keys]
        rows.append(("Fz(yukarı+)", float(r["Fz"][i]), -float(d["F_body"][2]), 1.0))
        rows.append(("My", float(r["My"][i]), float(d["M_body"][1]), 1.0))
        for k, lv, nv, abs_tol in rows:
            diff = abs(nv - lv)
            ok = diff <= max(REL_TOL * abs(lv), abs_tol)
            RESULTS.append(ok)
            pct = 100 * diff / abs(lv) if abs(lv) > 1e-12 else 0.0
            flag = "OK  " if ok else "FAIL"
            print(f"    {flag} {k:12s} legacy={lv:12.4f}  model={nv:12.4f}  fark%={pct:7.3f}")
        # yan eksen sızıntısı olmamalı
        check("pitch-only durumda Fy, K, N ≈ 0",
              abs(d["F_body"][1]) < 1e-6 and abs(d["M_body"][0]) < 1e-6
              and abs(d["M_body"][2]) < 1e-6)


# ---------------------------------------------------------------------------
def test_signs_and_decoupling():
    print("\n(S) İşaret / ayrışma kontrolleri")
    model = case1_model(fins_use_steady_cavity=False, planing_enable=False)
    sig = 0.15
    Lc, Dc, _ = cavity_geometry(sig, 0.20, model="savchenko", Cx0=CX0_DISK, k_g_val=0.78)
    V = 40.0
    base_x = make_state(u=V, Z=10.0, Lc=Lc, Dc=Dc, pc=94000.0)

    def FM(x, **kw):
        c = dict(delta_e=0.0, delta_r=0.0, delta_c=0.0, thrust=0.0, gas_flow=15000.0)
        c.update(kw)
        _, d = model.evaluate(0.0, x, ControlInput(**c))
        return d["F_body"], d["M_body"], d

    F0, M0, d0 = FM(base_x)
    print(f"    (bilgi) fin_wet_span={d0['fin_wet_span']:.3f} m, cover={d0['cover']:.3f}")

    F1, M1, _ = FM(base_x, delta_e=np.radians(2.0))
    dF, dM = F1 - F0, M1 - M0
    check("δe>0 → M azalır (burun aşağı)", dM[1] < 0, f"dM={dM[1]:.1f}")
    check("δe → Fy, K, N değişmez",
          abs(dF[1]) < 1e-6 and abs(dM[0]) < 1e-6 and abs(dM[2]) < 1e-6)

    F2, M2, _ = FM(base_x, delta_r=np.radians(2.0))
    dF, dM = F2 - F0, M2 - M0
    check("δr>0 → N artar (burun sancak)", dM[2] > 0, f"dN={dM[2]:.1f}")
    check("δr → Fz, K, M değişmez",
          abs(dF[2]) < 1e-6 and abs(dM[0]) < 1e-6 and abs(dM[1]) < 1e-6)
    check("δe ve δr simetrik büyüklük", abs(abs(M1[1] - M0[1]) - abs(M2[2] - M0[2])) < 1e-6)

    xw = base_x.copy(); xw[2] = V * np.tan(np.radians(1.0))
    _, _, dw = FM(xw)
    xv = base_x.copy(); xv[1] = xw[2]
    _, _, dv = FM(xv)
    check("w>0 (α>0) → kanat yukarı kuvvet", dw["F_fin_total"] > 0, f"{dw['F_fin_total']:.1f}")
    check("v>0 (β>0) → kanat iskele kuvvet, pitch ile simetrik",
          dv["F_fin_side"] > 0 and abs(dv["F_fin_side"] - dw["F_fin_total"]) < 1e-3 * abs(dw["F_fin_total"]),
          f"{dv['F_fin_side']:.1f}")

    xq = base_x.copy(); xq[4] = 0.5
    _, Mq, _ = FM(xq)
    check("q>0 → kanat pitch sönümü (M azalır)", Mq[1] < M0[1], f"dM={Mq[1]-M0[1]:.1f}")
    xr = base_x.copy(); xr[5] = 0.5
    _, Mr, _ = FM(xr)
    check("r>0 → kanat yaw sönümü (N azalır)", Mr[2] < M0[2], f"dN={Mr[2]-M0[2]:.1f}")
    xp = base_x.copy(); xp[3] = 2.0
    _, Mp, _ = FM(xp)
    check("p>0 → kanat roll sönümü (K < 0)", Mp[0] < 0, f"K={Mp[0]:.2f}")

    de = fin_mixer(np.radians(1.0), 0.0)
    check("mixer artı düzen: δe → sağ=+, sol=−, üst/alt=0",
          np.allclose(np.degrees(de), [0.0, 1.0, 0.0, -1.0]))


# ---------------------------------------------------------------------------
def test_cavity_equilibrium_and_robustness():
    print("\n(K) Kavite dengesi ve sağlamlık")
    model = case1_model(fins_use_steady_cavity=False)
    u = case1_control()
    x = make_state(u=40.0, Z=10.0, Lc=1.0, Dc=0.1, pc=3000.0)
    _, d = model.evaluate(0.0, x, u)
    x[IDX_PC] = d["pc_target"]
    _, d = model.evaluate(0.0, x, u)
    x[IDX_LC], x[IDX_DC] = d["Lc_ss"], d["Dc_ss"]
    xdot, d = model.evaluate(0.0, x, u)
    check("denge noktasında dLc, dDc, dpc ≈ 0",
          abs(xdot[IDX_LC]) < 1e-9 and abs(xdot[IDX_DC]) < 1e-9 and abs(xdot[IDX_PC]) < 1e-6,
          f"Lc_ss={d['Lc_ss']:.3f} Dc_ss={d['Dc_ss']:.4f} σ={d['sigma']:.4f}")

    cases = {
        "kavite yok, V=40": make_state(u=40.0, Z=10.0),
        "kavite yok, V=1": make_state(u=1.0, Z=10.0),
        "V≈0": make_state(u=1e-3, Z=10.0),
        "büyük açılar": make_state(u=30.0, v=3.0, w=-4.0, p=1.0, q=0.5, r=-0.5,
                                   phi=0.3, theta=0.2, psi=1.0, Z=5.0, Lc=2.0, Dc=0.3,
                                   pc=20000.0),
    }
    for name, xs in cases.items():
        xdot = model.derivatives(0.0, xs, u)
        check(f"NaN/inf yok: {name}", np.all(np.isfinite(xdot)),
              f"u_dot={xdot[0]:.2f}")


def main():
    test_static_regression()
    test_signs_and_decoupling()
    test_cavity_equilibrium_and_robustness()
    n_ok = sum(RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} kontrol geçti")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
