"""
src/control/autopilot.py testleri (düz script, exit 0 = geçti).

(U) AxisPID birim: işaret, ölçümden türev (kick yok), referans ağırlığı b,
    anti-windup (doyumda integratör donar, çıkış hızla doyumdan çıkar), |I| sınırı,
    hız sınırı, yaw açı sarma.
(M) Model ile işaret tutarlılığı: θ_ref>θ → δe<0 → M>0 ; ψ_ref>ψ → δr>0 → N>0.
    Kazanç çizelgesi, ilk çağrıda integrasyon yok, reset.
(E) Kavite tahmincisi (Lc → Lc_ss, parametre hatası, reset), kanat etkinliği çizelgesi,
    derinlik döngüsü (işaret, anti-windup, kapalıyken devre dışı), sensörler (kuyruk gazı,
    basınç gürültüsü), pc ve kuyruk düzeltmeleri, trim ileri beslemesi, referans ön filtresi.
(K) Kapalı çevrim, tam doğrusal olmayan 6-DOF (configs/case1_config.py, çalışma noktası
    8000 N / V0=20 m/s / 125 L/s @5 m): tutma, θ/ψ adımları (geçiş sırasında ve kavite
    oturmuşken aşım/oturma, çapraz etkileşim, doyum), düşük hız başlangıcı, çağrı sayısı.
(D) Derinlik tutma (kavite geçişi boyunca, Z_ref adımı + örtülmenin korunması, derinlik
    tutarken ψ adımı), tahminci doğruluğu, sağlamlık: tahminci A_v ±%25 ve K_Lc ±%15
    hatası, basınç sensörü gürültüsü.
Çalıştır:  python test_autopilot.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.control.autopilot import AxisPID, AttitudeAutopilot, DepthHold, wrap_angle
from src.control.estimator import CavityEstimator
from src.control.sensors import CavitySensors
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
    ap_spec = dict({k: v for k, v in CFG.AUTOPILOT.items() if k != "type"}, vehicle=CFG.VEHICLE)
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


T_END = 5.5     # otopilot_adimlar t_max


def test_closed_loop():
    print("\n(K) Kapalı çevrim, doğrusal olmayan 6-DOF")
    lim = CFG.AXIS_GAINS["limit_deg"]

    sim, res = run("otopilot_tutma")
    th, ps = D(res.state("theta")), D(res.state("psi"))
    check("tutma: NaN yok, çağrı sayısı = t_max/dt_control",
          res.status == "ok" and res.n_controller_calls == round(sim.t_max / sim.dt_control),
          f"{res.n_controller_calls} çağrı, {res.wall_time:.1f}s")
    check("tutma (kavite geçişi dahil): |θ| < 0.3°, |ψ| ≈ 0",
          np.abs(th).max() < 0.3 and np.abs(ps).max() < 1e-6, f"|θ|max={np.abs(th).max():.3f}°")

    sim, res = run("otopilot_adimlar")
    th, ps, ph = D(res.state("theta")), D(res.state("psi")), D(res.state("phi"))
    de, dr = D(res.u["delta_e"]), D(res.u["delta_r"])
    check("adımlar: NaN yok", res.status == "ok", f"{res.wall_time:.1f}s")

    fmt = lambda q: f"aşım=%{q[0]:.1f} t95={1e3*q[1]:.0f}ms ts5={1e3*q[2]:.0f}ms |e_son|={q[3]:.3f}°"

    # Kavite oturduktan sonraki adımlar
    for label, arr, t0, t1, y0, y1 in (("ψ 0→2° (t=3.0, kavite oturmuş)", ps, 3.0, T_END, 0.0, 2.0),
                                       ("θ 2→0° (t=4.0, kavite oturmuş)", th, 4.0, T_END, 2.0, 0.0)):
        q = step_quality(res, arr, t0, t1, y0, y1)
        check(f"{label}: aşım≤%10, t95≤400ms, ts5≤800ms, son |e|≤0.05°",
              q[0] <= 10.0 and q[1] <= 0.4 and q[2] <= 0.8 and q[3] <= 0.05, fmt(q))

    # t=1.0 θ adımı rejim geçişine düşer (kanatlar kaviteye girer, gövde örtülür, araç açık
    # çevrimde kararsızlaşır). Trim ileri beslemesiyle hedef karşılanır (önce %40 → %14).
    eta = res.diag["fin_wet_span"] / CFG.VEHICLE["fin_span"]
    t_in = res.t[np.argmax(eta < 0.999)]
    t_out = res.t[np.argmax(res.diag["cover"] >= 0.999)]
    # Alt sınır 0.8 s: kavite ekseni sapmasıyla (fin_cavity_offset) kaviteye yakın taraftaki
    # kanat önce kapanır → ortalama eta 0.999'un altına ~90 ms daha erken iner (1.05 → 0.96 s).
    # Kontrolün niyeti "t=1.0 adımı rejim geçişine düşsün": geçiş [1, 3] penceresiyle
    # örtüşmeli (t_out > 1.0), kanat girişi de adımın hemen çevresinde olmalı.
    check("adım penceresi geçişi kapsıyor (kanat kaviteye girişi ve tam örtülme 1-3 s içinde)",
          0.8 < t_in < 3.0 and 1.0 < t_out < 3.0, f"kanat girişi t={t_in:.2f}s, tam örtülme t={t_out:.2f}s")
    q = step_quality(res, th, 1.0, 3.0, 0.0, 2.0)
    check("θ 0→2° (t=1.0, rejim geçişi sırasında): aşım≤%10, t95≤450ms, son |e|≤0.05°",
          q[0] <= 10.0 and q[1] <= 0.45 and q[3] <= 0.05, fmt(q))

    # Eksen ayrışması: aynı koşum ψ adımı OLMADAN; θ farkı ψ adımının pitch'e etkisidir
    sc = dict(SCEN["otopilot_adimlar"])
    sc["control"] = dict(sc["control"], psi_ref_deg=0.0)
    _, sim_b, x0, _ = build_run(CFG, sc)
    res_b = sim_b.run(x0)
    _, y_a = window(res, th, 3.0, 4.0)
    _, y_b = window(res_b, D(res_b.state("theta")), 3.0, 4.0)
    cross = np.abs(y_a - y_b).max()
    check("ψ adımının θ'ya etkisi < 0.02° (eksen ayrışması)", cross < 0.02, f"{cross:.4f}°")
    # Referans ön filtresi adım anındaki P sıçramasını yumuşatır → doyum yok, %20 pay
    t_sat = sim.dt_control * np.count_nonzero((np.abs(de) >= lim - 1e-9) | (np.abs(dr) >= lim - 1e-9))
    d_max = max(np.abs(de).max(), np.abs(dr).max())
    check("kanat komutları doymuyor ve |δ| ≤ 0.8·limit", t_sat == 0.0 and d_max <= 0.8 * lim,
          f"|δe|max={np.abs(de).max():.1f}° |δr|max={np.abs(dr).max():.1f}° doyum={1e3 * t_sat:.0f} ms")
    check("roll (kontrolsüz) küçük kalır: |φ| < 1°", np.abs(ph).max() < 1.0,
          f"|φ|max={np.abs(ph).max():.3f}°")

    sc = dict(SCEN["dusuk_hiz_baslangic"], control=CFG.AUTOPILOT)
    _, sim, x0, _ = build_run(CFG, sc)
    res = sim.run(x0)
    th = D(res.state("theta"))
    check("düşük hız (V0=20, kavite yok) kapalı çevrim: kararlı, |θ| < 1°",
          res.status == "ok" and np.abs(th).max() < 1.0,
          f"|θ|max={np.abs(th).max():.3f}° V son={res.diag['V'][-1]:.1f}")


def test_estimator_and_depth_units():
    print("\n(E) Kavite tahmincisi ve derinlik döngüsü birim")
    est = CavityEstimator(CFG.VEHICLE)
    check("tahminci başlangıç: kavite yok → eta = 1", est.eta == 1.0 and est.Lc == 0.0)
    for _ in range(3000):
        est.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW)
    model = VehicleModel(CFG.VEHICLE)
    _, ss = model.cavity_derivative(est.Lc, est.Dc, est.pc, 40.0, 10.0, CFG.GAS_FLOW)
    check("tahminci 3 s sabit V=40: Lc → Lc_ss (±%1)", abs(est.Lc - ss["Lc_ss"]) < 0.01 * ss["Lc_ss"],
          f"Lc={est.Lc:.3f} Lc_ss={ss['Lc_ss']:.3f} eta={est.eta:.3f}")
    check("uzun kavite → kanat kısmen kuru: 0 < eta < 1", 0.0 < est.eta < 1.0, f"eta={est.eta:.3f}")
    est_hi = CavityEstimator(CFG.VEHICLE, param_scale={"A_v": 0.8})
    for _ in range(3000):
        est_hi.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW)
    check("param_scale: A_v×0.8 → daha uzun kavite, daha düşük eta",
          est_hi.Lc > est.Lc and est_hi.eta < est.eta, f"Lc={est_hi.Lc:.3f} eta={est_hi.eta:.3f}")
    est.reset()
    check("tahminci reset", est.Lc == 0.0 and est.eta == 1.0)

    ap = AttitudeAutopilot(**{k: v for k, v in CFG.AUTOPILOT.items() if k != "type"},
                           vehicle=CFG.VEHICLE)
    check("çizelge: V=V_ref, eta=eta_ref → 1; eta yarıya → 2",
          abs(ap.gain_scale(40.0, 0.5) - 1.0) < 1e-12 and abs(ap.gain_scale(40.0, 0.25) - 2.0) < 1e-12)
    check("çizelge: eta_min tabanı ve k_s_max sınırı",
          ap.gain_scale(40.0, 0.0) == ap.gain_scale(40.0, ap.eta_min)
          and ap.gain_scale(5.0, 0.05) == ap.k_s_max)

    dh = DepthHold(Kp_deg_m=2.0, Ki_deg_m_s=1.0, Kd_deg_s_m=1.0, theta_limit_deg=5.0)
    check("derinlik: fazla derin (e_z>0) → θ_ref > 0 (burun yukarı)", dh.update(1.0, 0.0, 0.0) > 0)
    check("derinlik: batma hızı (Ż>0) → θ_ref > 0", dh.update(0.0, 1.0, 0.0) > 0)
    dh.reset()
    for _ in range(20000):
        dh.update(100.0, 0.0, 1e-3)
    check("derinlik: doyumda integratör donar (anti-windup)",
          dh.saturated and abs(D(dh.I)) < 5.0 + 1e-9, f"I={D(dh.I):.2f}°")
    ap = AttitudeAutopilot(**{k: v for k, v in CFG.AUTOPILOT.items() if k != "type"},
                           vehicle=CFG.VEHICLE)
    check("depth_ref_m=None → derinlik döngüsü kapalı", ap.depth is None)

    # Sensörler
    sens = CavitySensors(CFG.VEHICLE)
    L = CFG.VEHICLE["veh_len"]
    x_short = make_state(u=40.0, Z=10.0, Lc=2.0, Dc=0.4, pc=90000.0)
    x_long = make_state(u=40.0, Z=10.0, Lc=3.0 * L, Dc=0.8, pc=90000.0)
    m_s, m_l = sens.measure(x_short), sens.measure(x_long)
    check("kuyruk gaz sensörü: kısa kavite → yok, uzun kavite → var",
          (not m_s["tail_gas"]) and m_l["tail_gas"])
    check("basınç sensörü gürültüsüz = pc", m_s["pc"] == 90000.0)
    noisy = CavitySensors(CFG.VEHICLE, pc_noise_pa=500.0, seed=1)
    vals = np.array([noisy.measure(x_short)["pc"] for _ in range(2000)])
    check("basınç sensörü gürültüsü: ortalama ≈ pc, std ≈ 500 Pa",
          abs(vals.mean() - 9e4) < 50 and abs(vals.std() - 500) < 50, f"std={vals.std():.0f}")

    # Basınç düzeltmesi: yanlış A_v'li tahminci ölçülen pc'ye yakınsar
    model = VehicleModel(CFG.VEHICLE)
    est_true = CavityEstimator(CFG.VEHICLE)
    est_bad = CavityEstimator(CFG.VEHICLE, param_scale={"A_v": 1.3})
    est_fix = CavityEstimator(CFG.VEHICLE, param_scale={"A_v": 1.3})
    for _ in range(3000):
        est_true.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW)
        est_bad.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW)
        est_fix.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW, pc_meas=est_true.pc)
    check("pc sensörü A_v×1.3 hatasını düzeltir: |Δeta| küçülür",
          abs(est_fix.eta - est_true.eta) < 0.2 * abs(est_bad.eta - est_true.eta),
          f"düzeltmesiz {abs(est_bad.eta - est_true.eta):.3f} → {abs(est_fix.eta - est_true.eta):.4f}")

    # Kuyruk sensörü: kısa tahmin eden tahminci (K_Lc×0.6) kuyrukta gaz görünce kaviteyi uzatır
    est_g = CavityEstimator(CFG.VEHICLE, param_scale={"K_Lc_factor": 0.6})
    est_n = CavityEstimator(CFG.VEHICLE, param_scale={"K_Lc_factor": 0.6})
    for _ in range(3000):
        est_g.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW, tail_gas=True)
        est_n.update(1e-3, 40.0, 10.0, CFG.GAS_FLOW)
    check("kuyruk sensörü: 'gaz var' → s_L > 1 ve Lc_hat uzar, tahmin kuyruğu geçer",
          est_g.s_L > 1.0 and est_g.Lc > est_n.Lc and est_g.tail_gas_hat,
          f"s_L={est_g.s_L:.2f} Lc {est_n.Lc:.2f}→{est_g.Lc:.2f} m")

    # Trim ileri beslemesi: δ_ff uygulanınca (p=q=r=0) moment ≈ 0
    spec = dict({k: v for k, v in CFG.AUTOPILOT.items() if k != "type"}, vehicle=CFG.VEHICLE)
    ap = AttitudeAutopilot(**spec)
    x = make_state(u=40.0, w=1.0, v=-0.5, Z=10.0, Lc=6.0, Dc=0.55, pc=95000.0)
    ap.estimator.Lc, ap.estimator.Dc, ap.estimator.pc = 6.0, 0.55, 95000.0
    u_ol = dict(delta_c=R(2.0), thrust=CFG.THRUST_N, gas_flow=CFG.GAS_FLOW)
    de_ff, dr_ff = ap.trim_feedforward(0.0, x, u_ol)
    from src.dynamics.model import ControlInput
    M = model.evaluate(0.0, x, ControlInput(delta_e=de_ff, delta_r=dr_ff, **u_ol))[1]["M_body"]
    M0 = model.evaluate(0.0, x, ControlInput(**u_ol))[1]["M_body"]
    check("trim ileri besleme: M_pitch, N_yaw ≈ 0 (|M| < %2·|M0|)",
          abs(M[1]) < 0.02 * abs(M0[1]) and abs(M[2]) < 0.02 * abs(M0[2]),
          f"δe_ff={D(de_ff):.2f}° δr_ff={D(dr_ff):.2f}° M0=({M0[1]:.0f},{M0[2]:.0f}) → ({M[1]:.1f},{M[2]:.1f})")

    # Referans ön filtresi
    ap = AttitudeAutopilot(**dict(spec, theta_ref_deg=[(0.0, 0.0), (0.01, 2.0)]))
    x0 = make_state(u=40.0, Z=10.0)
    refs = []
    for k in range(200):
        ap.update(k * 1e-3, x0)
        refs.append(D(ap.log["theta_ref"]))
    refs = np.array(refs)
    tau = CFG.AUTOPILOT["ref_tau_s"]
    k_tau = 10 + int(round(tau / 1e-3))
    check("ön filtre: adımda sıçrama yok, τ'da ~%63",
          refs[10] < 0.1 and abs(refs[k_tau] / 2.0 - 0.63) < 0.03, f"r(τ)={refs[k_tau]:.3f}°")


def test_depth_and_robustness():
    print("\n(D) Kapalı çevrim derinlik tutma ve tahminci sağlamlığı")
    sim, res = run("otopilot_derinlik")
    Z, t = res.state("Z"), res.t
    th = D(res.state("theta"))
    ps = D(res.state("psi"))
    check("derinlik: NaN yok", res.status == "ok", f"{res.wall_time:.1f}s")
    m = t < 5.0
    check("derinlik tutma (0-5 s, kavite geçişi dahil): |Z − 10| < 0.5 m",
          np.abs(Z[m] - 10.0).max() < 0.5, f"|ΔZ|max={np.abs(Z[m] - 10.0).max():.3f} m")
    q = step_quality(res, Z, 5.0, 8.0, 10.0, 11.0)
    check("Z_ref 10→11 m: aşım≤%15, t95≤2.5 s, t=8 s'de |e|≤0.05 m",
          q[0] <= 15.0 and q[1] <= 2.5 and q[3] <= 0.05,
          f"aşım=%{q[0]:.1f} t95={q[1]:.2f}s ts5={q[2]:.2f}s |e_son|={q[3]:.3f} m")
    check("derinlik: |θ| ≤ 5° (θ_ref sınırı)", np.abs(th).max() <= 5.0 + 0.5,
          f"|θ|max={np.abs(th).max():.2f}°")
    m = t >= 11.0
    check("derinlik: son 1 s |Z − 11| < 0.05 m", np.abs(Z[m] - 11.0).max() < 0.05,
          f"{np.abs(Z[m] - 11.0).max():.3f} m")
    cov = res.diag["cover"]
    check("11 m'de de gövde tamamen örtülü kalır (t ≥ 5 s)", cov[t >= 5.0].min() >= 0.999,
          f"min örtülme={cov[t >= 5.0].min():.4f}")
    qy = step_quality(res, ps, 8.0, 12.0, 0.0, 2.0)
    check("derinlik tutarken ψ 0→2°: aşım≤%10, t95≤400ms",
          qy[0] <= 10.0 and qy[1] <= 0.4, f"aşım=%{qy[0]:.1f} t95={1e3 * qy[1]:.0f}ms")
    eta_err = np.abs(res.diag["ctrl_eta_hat"] - res.diag["fin_wet_span"] / CFG.VEHICLE["fin_span"])
    check("tahminci (model birebir): |eta_hat − eta| < 0.02", eta_err.max() < 0.02,
          f"{eta_err.max():.4f}")

    # Sağlamlık: tahmincinin iç modeli gerçek araçtan farklı (sensörler açık), basınç gürültüsü
    def robust_run(estimator=None, sensors=None):
        sc = dict(SCEN["otopilot_adimlar"])
        ctrl = dict(sc["control"])
        if estimator:
            ctrl["estimator"] = dict(ctrl["estimator"], **estimator)
        if sensors:
            ctrl["sensors"] = dict(ctrl["sensors"], **sensors)
        sc["control"] = ctrl
        sc["simulation"] = dict(sc.get("simulation", {}), t_max=3.0)
        _, sim_b, x0, _ = build_run(CFG, sc)
        rb = sim_b.run(x0)
        err = np.abs(rb.diag["ctrl_eta_hat"] - rb.diag["fin_wet_span"] / CFG.VEHICLE["fin_span"])
        return rb, step_quality(rb, D(rb.state("theta")), 1.0, 3.0, 0.0, 2.0), err.max()

    cases = [("A_v×0.75", dict(param_scale={"A_v": 0.75}), None, 0.05),
             ("A_v×1.25", dict(param_scale={"A_v": 1.25}), None, 0.05),
             ("K_Lc×0.85", dict(param_scale={"K_Lc_factor": 0.85}), None, None),
             ("K_Lc×1.15", dict(param_scale={"K_Lc_factor": 1.15}), None, None),
             ("pc gürültüsü 2 kPa", None, dict(pc_noise_pa=2000.0), 0.05)]
    for label, est_over, sens_over, eta_tol in cases:
        rb, qb, e_max = robust_run(est_over, sens_over)
        ok = rb.status == "ok" and qb[0] <= 15.0 and qb[3] <= 0.05
        crit = "aşım≤%15, |e_son|≤0.05°"
        if eta_tol is not None:
            ok = ok and e_max < eta_tol
            crit += f", |Δeta|<{eta_tol}"
        check(f"sağlamlık {label}: kararlı, geçiş adımı {crit}", ok,
              f"aşım=%{qb[0]:.1f} |e_son|={qb[3]:.3f}° |Δeta|max={e_max:.3f}")

    # Sensörsüz karşılaştırma: basınç sensörü A_v hatasını gideriyor mu
    _, _, e_nos = robust_run(dict(param_scale={"A_v": 1.25}), dict(use_pc=False, use_tail=False))
    _, _, e_s = robust_run(dict(param_scale={"A_v": 1.25}))
    check("sensörler A_v×1.25 tahmin hatasını en az 5× azaltır", e_s * 5 <= e_nos,
          f"|Δeta|max sensörsüz={e_nos:.3f} → sensörlü={e_s:.3f}")


def main():
    test_axis_pid()
    test_signs_with_model()
    test_estimator_and_depth_units()
    test_closed_loop()
    test_depth_and_robustness()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} kontrol geçti")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
