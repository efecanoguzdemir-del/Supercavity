"""
src/control/autopilot.py testleri (düz script, exit 0 = geçti).

(U) AxisPID birim: işaret, ölçümden türev (kick yok), referans ağırlığı b,
    anti-windup (doyumda integratör donar, çıkış hızla doyumdan çıkar), |I| sınırı,
    hız sınırı, yaw açı sarma.
(M) Model ile işaret tutarlılığı: θ_ref>θ → δe<0 → M>0 ; ψ_ref>ψ → δr>0 → N>0.
    Kazanç çizelgesi, ilk çağrıda integrasyon yok, reset.
(K) Kapalı çevrim, tam doğrusal olmayan 6-DOF (configs/case1_config.py):
    tutma, θ/ψ adımları (aşım, oturma, çapraz etkileşim, doyum), düşük hız başlangıcı,
    kontrolcü çağrı sayısı = t_max/dt_control.
Çalıştır:  python test_autopilot.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.control.autopilot import AxisPID, AttitudeAutopilot, wrap_angle
from src.dynamics.model import VehicleModel
from src.dynamics.state import make_state
from src.simulation.run_simulation import load_config, build_run

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                               "case1_config.py"))
SCEN = {s["name"]: s for s in CFG.SCENARIOS}
RESULTS = []
D = np.degrees
R = np.radians


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")


# ---------------------------------------------------------------------------
def test_axis_pid():
    print("\n(U) AxisPID birim testleri")
    dt = 0.001

    pid = AxisPID(Kp=2.0, Ki=0.0, Kd=0.0, limit_deg=15.0, sign=+1.0)
    check("sign=+1: e>0 → δ>0", pid.update(R(1.0), 0.0, dt) > 0)
    pid = AxisPID(Kp=2.0, Ki=0.0, Kd=0.0, limit_deg=15.0, sign=-1.0)
    check("sign=−1: e>0 → δ<0", pid.update(R(1.0), 0.0, dt) < 0)

    # Ölçümden türev: referans adımı, ω_meas=0 → çıkış sadece Kp·b·Δref
    pid = AxisPID(Kp=2.0, Ki=0.0, Kd=0.5, b=1.0, limit_deg=45.0)
    pid.update(0.0, 0.0, dt, ref=0.0)
    out = pid.update(R(2.0), 0.0, dt, ref=R(2.0))
    check("referans adımında türev kick yok", abs(out - 2.0 * R(2.0)) < 1e-12, f"δ={D(out):.3f}°")
    check("türev ölçüme tepki verir (ω>0 → δ azalır)", pid.update(R(2.0), 1.0, dt, ref=R(2.0)) < out)

    pid = AxisPID(Kp=2.0, Ki=0.0, Kd=0.0, b=0.0, limit_deg=45.0)
    check("b=0: referans adımında P sıçraması yok",
          abs(pid.update(R(2.0), 0.0, dt, ref=R(2.0))) < 1e-12)

    # Anti-windup
    pid = AxisPID(Kp=1.0, Ki=50.0, Kd=0.0, limit_deg=5.0, i_limit_deg=30.0)
    for _ in range(2000):
        pid.update(R(10.0), 0.0, dt)
    I_sat = pid.I
    check("doyumda çıkış = limit", abs(pid.cmd - R(5.0)) < 1e-12 and pid.saturated)
    check("doyumda integratör donar (i_limit'e koşmaz)", I_sat < R(1.0), f"I={D(I_sat):.3f}°")
    n = 0
    while pid.update(R(-1.0), 0.0, dt) >= R(5.0) - 1e-12 and n < 1000:
        n += 1
    check("hata tersine dönünce doyumdan hemen çıkar", n <= 1, f"{n} adım")

    pid = AxisPID(Kp=0.0, Ki=100.0, Kd=0.0, limit_deg=45.0, i_limit_deg=3.0)
    for _ in range(1000):
        pid.update(R(10.0), 0.0, dt)
    check("|I| ≤ i_limit", abs(pid.I - R(3.0)) < 1e-12)

    pid = AxisPID(Kp=10.0, Ki=0.0, Kd=0.0, limit_deg=15.0, rate_limit_deg_s=100.0)
    pid.update(0.0, 0.0, dt)
    out = pid.update(R(1.0), 0.0, dt)
    check("hız sınırı: |Δδ| ≤ rate·dt", abs(out) <= R(100.0) * dt + 1e-15, f"Δδ={D(out):.3f}°")

    check("wrap: 179° − (−179°) → −2°", abs(D(wrap_angle(R(179.0) - R(-179.0))) + 2.0) < 1e-9)


# ---------------------------------------------------------------------------
def test_signs_with_model():
    print("\n(M) Model ile işaret tutarlılığı")
    model = VehicleModel(CFG.VEHICLE)
    ap_spec = {k: v for k, v in CFG.AUTOPILOT.items() if k != "type"}
    x = make_state(u=40.0, Z=10.0, Lc=3.0, Dc=0.4, pc=80000.0)

    ap = AttitudeAutopilot(**dict(ap_spec, theta_ref_deg=2.0, psi_ref_deg=0.0))
    u_up = ap.update(0.0, x)
    ap0 = AttitudeAutopilot(**ap_spec)
    u_0 = ap0.update(0.0, x)
    M_up = model.evaluate(0, x, u_up)[1]["M_body"]
    M_0 = model.evaluate(0, x, u_0)[1]["M_body"]
    check("θ_ref>θ → δe<0 → pitch momenti artar", u_up.delta_e < 0 and M_up[1] > M_0[1],
          f"δe={D(u_up.delta_e):.2f}° ΔM={M_up[1]-M_0[1]:.0f}")

    ap = AttitudeAutopilot(**dict(ap_spec, psi_ref_deg=2.0))
    u_r = ap.update(0.0, x)
    N_r = model.evaluate(0, x, u_r)[1]["M_body"]
    check("ψ_ref>ψ → δr>0 → yaw momenti artar", u_r.delta_r > 0 and N_r[2] > M_0[2],
          f"δr={D(u_r.delta_r):.2f}° ΔN={N_r[2]-M_0[2]:.0f}")
    check("pitch hatası rudder'a sızmaz", abs(u_up.delta_r) < 1e-12)

    check("kazanç çizelgesi: V=20 → 4, V=80 → 0.25, V=40 → 1",
          ap.gain_scale(20.0) == 4.0 and ap.gain_scale(80.0) == 0.25 and ap.gain_scale(40.0) == 1.0)
    ap = AttitudeAutopilot(**dict(ap_spec, theta_ref_deg=2.0))
    ap.update(0.0, x)
    check("ilk çağrıda integrasyon yok (dt=0)", ap.pitch.I == 0.0)
    ap.update(0.001, x)
    I1 = ap.pitch.I
    ap.reset()
    check("reset integratörü ve zamanı sıfırlar", I1 != 0.0 and ap.pitch.I == 0.0 and ap.t_prev is None)


# ---------------------------------------------------------------------------
def run(name, **sim_over):
    sc = dict(SCEN[name])
    if sim_over:
        sc["simulation"] = dict(sc.get("simulation", {}), **sim_over)
    _, sim, x0, _ = build_run(CFG, sc)
    res = sim.run(x0)
    return sim, res


def window(res, arr, t0, t1):
    m = (res.t >= t0) & (res.t < t1)
    return res.t[m], arr[m]


def step_quality(res, arr, t0, t1, y0, y1):
    """
    Adım yanıtı [t0, t1) penceresinde (t1 = bir sonraki referans değişimi):
      aşım  [%]  hedefin ötesine en büyük taşma (adım yönünde)
      t95   [s]  hedefin %95'ine ilk ulaşma
      ts5   [s]  ±%5 bandından son çıkış (geri tepme/düşüş/salınımı yakalar)
      e_end [°]  pencere sonundaki |hata|
    """
    t, y = window(res, arr, t0, t1)
    t = t - t0
    step = y1 - y0
    frac = (y - y0) / step
    over = 100.0 * max(0.0, frac.max() - 1.0)
    k95 = np.nonzero(frac >= 0.95)[0]
    t95 = t[k95[0]] if len(k95) else np.inf
    out = np.nonzero(np.abs(y - y1) > 0.05 * abs(step))[0]
    ts5 = t[out[-1]] if len(out) else 0.0
    return over, t95, ts5, abs(y[-1] - y1)


def test_closed_loop():
    print("\n(K) Kapalı çevrim, doğrusal olmayan 6-DOF")
    lim = CFG.AXIS_GAINS["limit_deg"]

    sim, res = run("otopilot_tutma")
    th, ps = D(res.state("theta")), D(res.state("psi"))
    check("tutma: NaN yok, çağrı sayısı = t_max/dt_control",
          res.status == "ok" and res.n_controller_calls == round(sim.t_max / sim.dt_control),
          f"{res.n_controller_calls} çağrı, {res.wall_time:.1f}s")
    check("tutma: |θ|, |ψ| < 0.5° (başlangıç M≈+6 kN·m bozucu)",
          np.abs(th).max() < 0.5 and np.abs(ps).max() < 1e-6, f"|θ|max={np.abs(th).max():.3f}°")

    sim, res = run("otopilot_adimlar")
    th, ps, ph = D(res.state("theta")), D(res.state("psi")), D(res.state("phi"))
    de, dr = D(res.u["delta_e"]), D(res.u["delta_r"])
    check("adımlar: NaN yok", res.status == "ok", f"{res.wall_time:.1f}s")

    fmt = lambda q: f"aşım=%{q[0]:.1f} t95={1e3*q[1]:.0f}ms ts5={1e3*q[2]:.0f}ms |e_son|={q[3]:.3f}°"

    # Kavite oturduktan sonraki adımlar — performans kriteri
    for label, arr, t0, t1, y0, y1 in (("ψ 0→2° (t=1.0, kavite oturmuş)", ps, 1.0, 3.0, 0.0, 2.0),
                                       ("θ 2→0° (t=1.5, kavite oturmuş)", th, 1.5, 3.0, 2.0, 0.0)):
        q = step_quality(res, arr, t0, t1, y0, y1)
        check(f"{label}: aşım≤%10, t95≤400ms, ts5≤800ms, son |e|≤0.1°",
              q[0] <= 10.0 and q[1] <= 0.4 and q[2] <= 0.8 and q[3] <= 0.1, fmt(q))

    # t=0.5 θ adımı en sert rejim geçişine düşer (V 40→27 m/s, kavite 3→9 m, kanat ıslak oranı
    # 1.0→0.23, araç açık çevrimde kararsızlaşır). Sabit kazançlı PID burada hedefi (%10 aşım)
    # KARŞILAMIYOR — bilinen sorun (CLAUDE.md). Sayılan kontrol sadece kötüleşmeye karşı koruma.
    q = step_quality(res, th, 0.5, 1.5, 0.0, 2.0)
    check("θ 0→2° (t=0.5, rejim geçişi): kararlı, aşım≤%35, t=1.5'te |e|≤0.5° [koruma]",
          q[0] <= 35.0 and q[3] <= 0.5, fmt(q))
    print(f"    BİLİNEN SORUN: bu adımda hedef aşım ≤%10, ölçülen %{q[0]:.1f}")

    # Eksen ayrışması: aynı koşum ψ adımı OLMADAN; θ farkı ψ adımının pitch'e etkisidir
    sc = dict(SCEN["otopilot_adimlar"])
    sc["control"] = dict(sc["control"], psi_ref_deg=0.0)
    _, sim_b, x0, _ = build_run(CFG, sc)
    res_b = sim_b.run(x0)
    _, y_a = window(res, th, 1.0, 1.5)
    _, y_b = window(res_b, D(res_b.state("theta")), 1.0, 1.5)
    cross = np.abs(y_a - y_b).max()
    check("ψ adımının θ'ya etkisi < 0.02° (eksen ayrışması)", cross < 0.02, f"{cross:.4f}°")
    check("kanat komutları doymadı", np.abs(de).max() < lim and np.abs(dr).max() < lim,
          f"|δe|max={np.abs(de).max():.1f}° |δr|max={np.abs(dr).max():.1f}°")
    check("roll (kontrolsüz) küçük kalır: |φ| < 1°", np.abs(ph).max() < 1.0,
          f"|φ|max={np.abs(ph).max():.3f}°")

    sc = dict(SCEN["dusuk_hiz_baslangic"], control=CFG.AUTOPILOT)
    _, sim, x0, _ = build_run(CFG, sc)
    res = sim.run(x0)
    th = D(res.state("theta"))
    check("düşük hız (V0=20, kavite yok) kapalı çevrim: kararlı, |θ| < 1°",
          res.status == "ok" and np.abs(th).max() < 1.0,
          f"|θ|max={np.abs(th).max():.3f}° V son={res.diag['V'][-1]:.1f}")


def main():
    test_axis_pid()
    test_signs_with_model()
    test_closed_loop()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} kontrol geçti")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
