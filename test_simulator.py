"""
src/simulation/simulator.py testleri (düz script, exit 0 = geçti).

(B) Dinamik regresyon: pitch/heave/lateral + ileri hız kilitli model koşumu,
    legacy steady_v_mode=True (V=40 sabit) zaman serisiyle karşılaştırılır:
    Lc, Dc, pc, σ, Fd, Fz, My.  (Legacy'nin V güncellemesi hatalı olduğundan —
    V[i+1]=V[i-1]+a·dt — serbest-V karşılaştırması sadece bilgi amaçlıdır.)
(C) Açık çevrim serbest 6-DOF: NaN yok, status ok.
(D) Ayrık kontrolcü: çağrı sayısı = t_max/dt_control, çağrı zamanları tam
    periyot katları, dt < dt_control iken komut periyot içinde sabit (ZOH).
(E) Süreklilik: gaz adımında Lc/Dc/pc sıçramaz, sadece eğim değişir.
(F) RK4 yakınsama: dt=1 ms ile dt=0.25 ms kilitli koşum farkı küçük.
Çalıştır:  python test_simulator.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dynamics.blocks import Controller, ScheduleController
from src.dynamics.model import VehicleModel, ControlInput
from src.simulation.run_simulation import load_config, build_run
from src.simulation.simulator import Simulator
from src.validation.legacy_reference import run_legacy

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "case1_config.py")
CFG = load_config(CFG_PATH)
SCEN = {s["name"]: s for s in CFG.SCENARIOS}
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")


def scenario(name, **over):
    sc = dict(SCEN[name])
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(sc.get(k), dict):
            merged = dict(sc[k]); merged.update(v); sc[k] = merged
        else:
            sc[k] = v
    return sc


# ---------------------------------------------------------------------------
def test_locked_regression():
    print("\n(B) Kilitli dinamik regresyon — legacy steady_v_mode (V=40)")
    leg = run_legacy(steady_v_mode=True)
    sc = scenario("case1_kilitli",
                  locked_states=CFG.LOCK_ALL_BUT_SURGE + ["u"],
                  vehicle=dict(fins_use_steady_cavity=True, legacy_exact=True))
    _, sim, x0, _ = build_run(CFG, sc)
    res = sim.run(x0)
    print(f"    duvar saati {res.wall_time:.1f} s, status={res.status}")

    # Legacy i. adımda pc, Lc, Dc'yi ÖNCE günceller sonra kaydeder → kaydı t_i + dt anına
    # aittir. Model bu yüzden t_i + dt'de örneklenir (kaymasız fark erken t'de %5-17).
    series = [("Lc", res.state("Lc"), 0.01, 1e-3), ("Dc", res.state("Dc"), 0.01, 1e-4),
              ("pc", res.state("pc"), 0.01, 50.0), ("sigma", res.diag["sigma"], 0.01, 1e-4),
              ("Fd", res.diag["Fd"], 0.01, 5.0), ("Fz", -res.diag["F_body_2"], 0.01, 5.0),
              ("My", res.diag["M_body_1"], 0.01, 5.0)]
    t_leg = leg["t"]
    dt_leg = 0.001
    for name, arr, rel, abs_tol in series:
        idx = [int(round(tq / dt_leg)) for tq in (0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.498)]
        worst = 0.0
        ok = True
        for i in idx:
            lv = float(leg[name][i]); mv = float(np.interp(t_leg[i] + dt_leg, res.t, arr))
            err = abs(mv - lv)
            ok &= err <= max(rel * abs(lv), abs_tol)
            worst = max(worst, err / max(abs(lv), 1e-12))
        check(f"{name}(t) legacy ile ≤%{rel*100:.0f}", ok, f"en kötü bağıl fark %{100*worst:.2f}")


def test_free_surge_info():
    print("\n(B') Serbest ileri hız (bilgi): legacy V güncellemesi hatalı")
    leg = run_legacy()
    _, sim, x0, _ = build_run(CFG, scenario("case1_kilitli"))
    res = sim.run(x0)
    print(f"    V(0.499): legacy={leg['V'][-1]:.2f}  model={np.interp(0.499, res.t, res.diag['V']):.2f}"
          f"  (legacy_reference notu: tutarlı Euler ≈ 33.88)")
    check("serbest ileri hız koşumu tamamlandı", res.status == "ok")


# ---------------------------------------------------------------------------
def test_free_6dof():
    print("\n(C) Açık çevrim serbest 6-DOF")
    for name in ("case1_serbest_6dof", "dusuk_hiz_baslangic"):
        _, sim, x0, _ = build_run(CFG, scenario(name))
        res = sim.run(x0)
        finite = np.all(np.isfinite(res.x))
        th = np.degrees(res.state("theta"))
        check(f"{name}: NaN yok, status ok", finite and res.status == "ok",
              f"θ son={th[-1]:.1f}° V son={res.diag['V'][-1]:.1f} Lc son={res.state('Lc')[-1]:.2f}"
              f"  ({res.wall_time:.1f}s)")


# ---------------------------------------------------------------------------
class CountingController(Controller):
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    def update(self, t, x):
        self.calls.append(t)
        # her çağrıda farklı komut → ZOH kontrolü için
        return ControlInput(delta_e=np.radians(0.1 * (len(self.calls) % 5)), thrust=6000.0,
                            gas_flow=15000.0)


def test_discrete_controller():
    print("\n(D) Ayrık kontrolcü / ZOH")
    model = VehicleModel(CFG.VEHICLE)
    ctrl = CountingController()
    _, _, x0, _ = build_run(CFG, SCEN["case1_kilitli"])
    t_max, dt, dtc = 0.1, 0.00025, 0.001
    sim = Simulator(model, ctrl, t_max=t_max, dt=dt, dt_control=dtc)
    res = sim.run(x0)
    expected = int(round(t_max / dtc))
    check("çağrı sayısı = t_max/dt_control", res.n_controller_calls == expected == len(ctrl.calls),
          f"{res.n_controller_calls} (beklenen {expected})")
    calls = np.array(ctrl.calls)
    check("çağrı zamanları k·dt_control", np.allclose(calls, np.arange(expected) * dtc))
    de = res.u["delta_e"][:-1]
    k_step = np.arange(len(de))
    changes = np.nonzero(np.diff(de))[0] + 1
    check("komut sadece periyot sınırlarında değişir (ZOH)",
          len(changes) > 0 and np.all(k_step[changes] % 4 == 0))
    try:
        Simulator(model, ctrl, t_max=0.01, dt=0.0003, dt_control=0.001)
        check("dt_control tam kat değilse hata", False)
    except ValueError:
        check("dt_control tam kat değilse hata", True)


# ---------------------------------------------------------------------------
def test_continuity():
    print("\n(E) Gaz adımında durum sürekliliği")
    _, sim, x0, _ = build_run(CFG, scenario("case1_gaz_adimi"))
    res = sim.run(x0)
    i = int(np.searchsorted(res.t, 0.25))
    ok = True
    info = []
    for name in ("Lc", "Dc", "pc"):
        s = res.state(name)
        steps = np.abs(np.diff(s))
        jump = steps[i - 1:i + 2].max()
        typical = np.median(steps[i - 20:i + 20]) + 1e-12
        ok &= jump < 5.0 * typical
        info.append(f"{name}: adım={jump:.3g} tipik={typical:.3g}")
    check("t=0.25 s'de Lc/Dc/pc sıçramıyor", ok, "; ".join(info))
    dLc = np.gradient(res.state("pc"), res.t)
    check("adımdan sonra pc azalma yönüne döner", dLc[i + 20] < dLc[i - 20],
          f"dpc/dt önce={dLc[i-20]:.0f} sonra={dLc[i+20]:.0f}")
    kinds = {k for _, k, _ in res.events}
    print(f"    olaylar: {[(round(t, 3), k, s) for t, k, s in res.events if not s.startswith('başlangıç')]}")
    check("olay gözlemcisi başlangıç durumlarını kaydetti", {"kanat", "planing", "kavite"} <= kinds)


# ---------------------------------------------------------------------------
def test_convergence():
    print("\n(F) RK4 yakınsama (kilitli, t_max=0.3)")
    finals = {}
    for dt in (0.001, 0.00025):
        sc = scenario("case1_kilitli", simulation=dict(t_max=0.3, dt=dt, dt_control=0.001))
        _, sim, x0, _ = build_run(CFG, sc)
        res = sim.run(x0)
        finals[dt] = (res.state("u")[-1], res.state("Lc")[-1], res.state("pc")[-1])
    rel = [abs(a - b) / abs(b) for a, b in zip(finals[0.001], finals[0.00025])]
    check("dt=1ms vs 0.25ms bağıl fark < %0.1 (u, Lc, pc)", max(rel) < 1e-3,
          f"{[f'{100*r:.4f}%' for r in rel]}")


def main():
    test_locked_regression()
    test_free_surge_info()
    test_free_6dof()
    test_discrete_controller()
    test_continuity()
    test_convergence()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} kontrol geçti")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
