"""
Katı cisim + kinematik doğrulama testleri (düz script, pytest yok).

Çalıştırma:
  C:\\Users\\oguz.demir\\PyCharmMiscProject\\.venv\\Scripts\\python.exe test_rigid_body.py
"""

import sys
import warnings

import numpy as np

from src.dynamics.state import (make_state, N_STATES, SL_NU, SL_EULER, SL_POS,
                                IDX_PSI, IDX_R)
from src.dynamics.kinematics import (rotation_body_to_ned, euler_rate_matrix,
                                     kinematics_derivative, gravity_body, flow_angles,
                                     nose_to_body_x, check_pitch_limit,
                                     reset_pitch_warning)
from src.dynamics.rigid_body import (cylinder_inertia, M_RB, C_RB, C_A,
                                     rigid_body_derivative, thrust_body)
from src.hydrodynamics.constants import G

# Case 1
M_KG, L_M, D_M, X_CG = 350.0, 4.0, 0.30, 2.4

RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# Basit RK4 ve 6-DOF (kuvvetsiz) türev
# ---------------------------------------------------------------------------
def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_free_dynamics(m, I, M_A=None, tau=None):
    tau = np.zeros(6) if tau is None else tau

    def f(x):
        xd = np.zeros(N_STATES)
        xd[SL_NU] = rigid_body_derivative(x[SL_NU], tau, m, I, M_A=M_A)
        pos_dot, eul_dot = kinematics_derivative(x)
        xd[SL_POS] = pos_dot
        xd[SL_EULER] = eul_dot
        return xd
    return f


def integrate(f, x0, t_end, dt):
    x = x0.copy()
    n = int(round(t_end / dt))
    for _ in range(n):
        x = rk4_step(f, x, dt)
    return x


# ---------------------------------------------------------------------------
# 1) Kuvvetsiz dönen cisim: enerji ve açısal momentum korunumu
# ---------------------------------------------------------------------------
def test_torque_free():
    dt, T = 1e-3, 5.0
    cases = {
        "generic I=[10,20,30]": (M_KG, np.array([10.0, 20.0, 30.0])),
        "Case1 cyl (Izz*1.2)": (M_KG, np.array(cylinder_inertia(M_KG, L_M, D_M))
                                * np.array([1.0, 1.0, 1.2])),
    }
    nu0_ang = np.array([1.5, -0.8, 0.6])
    for label, (m, I) in cases.items():
        x0 = make_state(u=40.0, v=0.5, w=-0.3, p=nu0_ang[0], q=nu0_ang[1], r=nu0_ang[2])
        f = make_free_dynamics(m, I)
        x1 = integrate(f, x0, T, dt)

        def energy(x):
            nu = x[SL_NU]
            return 0.5 * nu @ M_RB(m, I) @ nu

        def ang_mom(x):
            return np.linalg.norm(I * x[SL_NU][3:])

        def lin_mom(x):
            return np.linalg.norm(m * x[SL_NU][:3])

        eE = abs(energy(x1) - energy(x0)) / energy(x0)
        eH = abs(ang_mom(x1) - ang_mom(x0)) / ang_mom(x0)
        eP = abs(lin_mom(x1) - lin_mom(x0)) / lin_mom(x0)
        ok = eE < 1e-6 and eH < 1e-6
        report(f"torque-free {label}", ok,
               f"rel err KE={eE:.3e}, |h_body|={eH:.3e} (|p_lin|={eP:.3e})")


def test_torque_free_added_mass():
    """Ek: sabit diyagonal M_A ile KE = 0.5 nu^T (M_RB+M_A) nu korunmalı.

    Not: ilk sürümde I=[10,20,30] kullanıldı; Munk momenti bu küçük inertiada
    ~40 rad/s dönüşler üretip RK4 kesme hatası 1.37e-6 verdi (dt yarılanınca
    ~30x azalıyor -> fizik değil entegratör hatası). Case 1 inertiası kullanılıyor.
    """
    dt, T = 1e-3, 5.0
    m = M_KG
    I = np.array(cylinder_inertia(M_KG, L_M, D_M)) * np.array([1.0, 1.0, 1.2])
    M_A = np.diag([5.0, 80.0, 90.0, 0.5, 15.0, 12.0])
    x0 = make_state(u=40.0, v=0.5, w=-0.3, p=1.5, q=-0.8, r=0.6)
    f = make_free_dynamics(m, I, M_A=M_A)
    x1 = integrate(f, x0, T, dt)
    Mt = M_RB(m, I) + M_A
    E0 = 0.5 * x0[SL_NU] @ Mt @ x0[SL_NU]
    E1 = 0.5 * x1[SL_NU] @ Mt @ x1[SL_NU]
    eE = abs(E1 - E0) / E0
    report("torque-free with M_A (extra)", eE < 1e-6, f"rel err KE={eE:.3e}")


# ---------------------------------------------------------------------------
# 2) Saf yaw hızı: psi(t) = r t
# ---------------------------------------------------------------------------
def test_pure_yaw():
    dt, T, r = 1e-3, 5.0, 0.3
    I = np.array(cylinder_inertia(M_KG, L_M, D_M))
    x0 = make_state(r=r)
    x1 = integrate(make_free_dynamics(M_KG, I), x0, T, dt)
    err_psi = abs(x1[IDX_PSI] - r * T)
    err_r = abs(x1[IDX_R] - r)
    ok = err_psi < 1e-9 and err_r < 1e-12
    report("pure yaw psi=r*t", ok,
           f"psi(5s)={x1[IDX_PSI]:.12f}, r*t={r*T:.12f}, |err|={err_psi:.3e}, |dr|={err_r:.3e}")


# ---------------------------------------------------------------------------
# 3) Konum türevi
# ---------------------------------------------------------------------------
def test_position_rates():
    pd0, _ = kinematics_derivative(make_state(u=40.0, theta=0.0))
    ok0 = abs(pd0[0] - 40.0) < 1e-12 and np.all(np.abs(pd0[1:]) < 1e-12)
    report("theta=0, u=40 -> Xdot=40", ok0, f"pos_dot={pd0}")

    th = np.radians(10.0)
    pd1, _ = kinematics_derivative(make_state(u=40.0, theta=th))
    Zexp = -40.0 * np.sin(th)
    Xexp = 40.0 * np.cos(th)
    ok1 = abs(pd1[2] - Zexp) < 1e-12 and abs(pd1[0] - Xexp) < 1e-12
    report("theta=10deg, u=40 -> Zdot=-40 sin10", ok1,
           f"Zdot={pd1[2]:.10f} (exp {Zexp:.10f}), Xdot={pd1[0]:.10f} (exp {Xexp:.10f})")


# ---------------------------------------------------------------------------
# 4) Yerçekimi
# ---------------------------------------------------------------------------
def test_gravity():
    W = M_KG * G
    g0 = gravity_body(M_KG, 0.0, 0.0)
    ok0 = np.allclose(g0, [0.0, 0.0, W], atol=1e-9)
    report("gravity theta=0 -> [0,0,mg]", ok0, f"{g0} (mg={W})")

    th = np.radians(89.999)
    g90 = gravity_body(M_KG, 0.0, th)
    relx = abs(g90[0] + W) / W
    ok90 = relx < 1e-6 and abs(g90[2]) < 1e-3 * W
    report("gravity theta~90deg -> Fx=-mg", ok90,
           f"{g90}, rel err Fx={relx:.3e}")

    # tutarlılık: R^T [0,0,mg]
    phi, th = 0.4, -0.7
    ref = rotation_body_to_ned(phi, th, 1.1).T @ np.array([0.0, 0.0, W])
    okc = np.allclose(gravity_body(M_KG, phi, th), ref, atol=1e-9)
    report("gravity == R^T [0,0,mg] (extra)", okc,
           f"max diff={np.max(np.abs(gravity_body(M_KG, phi, th) - ref)):.3e}")


# ---------------------------------------------------------------------------
# 5) Euler rate matrisi gimbal lock civarı
# ---------------------------------------------------------------------------
def test_euler_rate_singular():
    thetas = [np.radians(89.9), np.pi / 2, -np.pi / 2, np.radians(90.0 + 1e-7),
              np.pi / 2 + 1e-12]
    finite = True
    maxabs = 0.0
    for th in thetas:
        for phi in (0.0, 0.3, -2.0):
            T = euler_rate_matrix(phi, th)
            finite &= bool(np.all(np.isfinite(T)))
            maxabs = max(maxabs, float(np.max(np.abs(T))))
    report("euler rate no NaN/inf near 90deg", finite, f"max|T|={maxabs:.3e}")

    # T, R ile tutarlı mı? (theta=0.5): omega_skew = R^T Rdot
    phi, th, psi = 0.3, 0.5, -0.8
    omega = np.array([0.2, -0.4, 0.7])
    etad = euler_rate_matrix(phi, th) @ omega
    h = 1e-6
    Rp = rotation_body_to_ned(phi + h * etad[0], th + h * etad[1], psi + h * etad[2])
    Rm = rotation_body_to_ned(phi - h * etad[0], th - h * etad[1], psi - h * etad[2])
    S = rotation_body_to_ned(phi, th, psi).T @ (Rp - Rm) / (2 * h)
    om_num = np.array([S[2, 1], S[0, 2], S[1, 0]])
    err = np.max(np.abs(om_num - omega))
    report("T consistent with R (extra)", err < 1e-7, f"max|omega_num-omega|={err:.3e}")


# ---------------------------------------------------------------------------
# Ekler: uyarı, akış açıları, yardımcılar
# ---------------------------------------------------------------------------
def test_misc():
    reset_pitch_warning()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for th in (np.radians(85), np.radians(-85), np.radians(88)):
            check_pitch_limit(th)
        check_pitch_limit(0.1)
    n = sum(1 for wi in w if issubclass(wi.category, RuntimeWarning))
    report("theta>80deg warns once (extra)", n == 1, f"warnings={n}")
    reset_pitch_warning()

    V, a, b = flow_angles(40.0, 2.0, 3.0)
    okf = (abs(V - np.sqrt(1613.0)) < 1e-12 and abs(a - np.arctan2(3, 40)) < 1e-15
           and abs(b - np.arcsin(2 / np.sqrt(1613.0))) < 1e-15)
    V0, a0, b0 = flow_angles(0.0, 0.0, 0.0)
    ok0 = all(np.isfinite([V0, a0, b0]))
    report("flow_angles (extra)", okf and ok0,
           f"(40,2,3)->V={V:.6f}, a={np.degrees(a):.4f}deg, b={np.degrees(b):.4f}deg; "
           f"(0,0,0)->({V0},{a0},{b0})")

    Ixx, Iyy, Izz = cylinder_inertia(M_KG, L_M, D_M)
    okx = nose_to_body_x(0.0, X_CG) == 2.4 and abs(nose_to_body_x(4.0, X_CG) + 1.6) < 1e-12
    oks = np.allclose(C_RB(M_KG, [Ixx, Iyy, Izz], np.arange(1, 7.0)),
                      -C_RB(M_KG, [Ixx, Iyy, Izz], np.arange(1, 7.0)).T)
    MA = np.diag([5.0, 80.0, 90.0, 0.5, 15.0, 12.0])
    oka = np.allclose(C_A(MA, np.arange(1, 7.0)), -C_A(MA, np.arange(1, 7.0)).T)
    okt = np.allclose(thrust_body(1000.0), [1000, 0, 0, 0, 0, 0])
    report("helpers (extra)", okx and oks and oka and okt,
           f"Case1 I=({Ixx:.4f}, {Iyy:.4f}, {Izz:.4f}) kg m^2; nose->x_body ok={okx}; "
           f"C_RB skew={oks}; C_A skew={oka}; thrust ok={okt}")


if __name__ == "__main__":
    test_torque_free()
    test_torque_free_added_mass()
    test_pure_yaw()
    test_position_rates()
    test_gravity()
    test_euler_rate_singular()
    test_misc()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
