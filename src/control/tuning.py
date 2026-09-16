"""
Otopilot kazanç ayarı: kapalı çevrim yörünge boyunca donmuş doğrusallaştırma + en kötü durum
optimizasyonu (Nelder-Mead, çoklu başlangıç).

Yöntem
  1. Referans koşum (config senaryosu, mevcut kazançlar) → çalışma noktaları (t, x, u).
     Noktalar denge değildir (araç hızlanıyor, kavite büyüyor); sapma dinamiği donmuş
     zamanda doğrusallaştırılır. Kavite durumları (Lc, Dc, pc) sabit tutulur.
  2. Sayısal Jacobian: boyuna [w, q, θ, Z] / δe, yanal [v, r, ψ] / δr
     (φ, p çıkarıldı: roll nötr; u çıkarıldı: itki açık çevrim, hız yarı-statik parametre —
     hızlanan yörüngede donmuş u modu planing başlangıcında s≈+0.06 rad/s artefaktı veriyor).
  3. Ayrık (dt = kontrol periyodu, ZOH) kapalı çevrim: AxisPID yasası birebir
     (b ağırlığı, Ki·dt ön-integrasyon, ölçümden türev) ve noktadaki GERÇEK kazanç ölçeği
     k_s(V, eta). Tahmin hatasına karşı k_s × {0.7, 1, 1.4} de denenir.
  4. İç döngü maliyeti (en kötü nokta × çarpan):
        ITAE + aşım + ts5 + komut genliği + ζ_min < 0.6 cezası + |s|max > 80 rad/s cezası
        + giriş bozucusu reddi (kanatta 1° eşdeğer adım bozucu → tepe |θ|, geçişte trim kayması)
     Derinlik döngüsü (iç döngü sabit): Z_ref adımı 1 m — ITAE, aşım, θ_ref tepe değeri,
     ζ_min < 0.5 cezası + dikey kuvvet bozucusu (ẇ kanalına 0.5 m/s² adım: tepe |ΔZ| ve
     kalıcı hata — integratörsüz derinlik döngüsü trim kaldırmasını karşılayamaz).
  5. Trim ileri beslemesi (config trim_ff.gain = κ) doğrusal modelde durum geri beslemesidir:
     δ_ff = −κ·Σ_j A[ω,j]·x_j / B[ω]   (ω = q veya r satırı; açısal hız sütunu hariç —
     kontrolcü momenti p=q=r=0'da hesaplar). Hem iç hem derinlik döngüsüne eklenir.
  6. Sonuç doğrusal olmayan simülasyonla doğrulanmalıdır (test_autopilot.py).

Dersler (CLAUDE.md): adım metriği tek başına yetmez → kutup sönümü ayrıca kısıtlanır;
"%90 sonrası sapma" metriği kullanılmaz; aşım, t95, ts5 ayrı ölçülür.

Kullanım:
  python src/control/tuning.py [--config configs/case1_config.py] [--scenario otopilot_tutma]
                               [--starts 4] [--skip-inner] [--skip-depth]
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dynamics.state import (IDX_V, IDX_W, IDX_Q, IDX_R, IDX_THETA, IDX_PSI,  # noqa: E402
                                IDX_Z)
from src.simulation.run_simulation import load_config, build_run  # noqa: E402

LON = [IDX_W, IDX_Q, IDX_THETA, IDX_Z]
LON_INNER = [0, 1, 2]          # Z'siz alt küme (iç döngü Z'yi görmez)
LAT = [IDX_V, IDX_R, IDX_PSI]
GAIN_MULTS = (0.7, 1.0, 1.4)
ZETA_MIN_INNER = 0.6
ZETA_MIN_DEPTH = 0.5
OMEGA_MAX = 80.0          # rad/s — modellenmemiş servo/gecikmeye karşı bant genişliği sınırı
STEP_DEG = 2.0
REF_MAX_DEG = 5.0         # en büyük beklenen θ/ψ referansı (derinlik döngüsü sınırı)
LIMIT_DEG = 15.0
I_LIMIT_DEG = 40.0        # AxisPID i_limit_deg (config ile aynı tutulmalı)
B_MIN = 0.3               # b→0'da referansı yalnız integratör taşır: I_ss = Kp·(1−b)·ref
T_INNER = 1.5
T_DEPTH = 12.0
PARAM_KEYS = ("Kp", "Ki", "Kd", "b")
DEPTH_KEYS = ("Kp_deg_m", "Ki_deg_m_s", "Kd_deg_s_m")


# ---------------------------------------------------------------------------
# Çalışma noktaları ve doğrusallaştırma
# ---------------------------------------------------------------------------
def operating_points(cfg, scenario, times):
    model, sim, x0, _ = build_run(cfg, scenario)
    res = sim.run(x0)
    if res.status != "ok":
        raise RuntimeError(f"referans koşum ıraksadı: {scenario['name']}")
    ctrl = sim.controller
    pts = []
    for tp in times:
        k = int(np.argmin(np.abs(res.t - tp)))
        x = res.x[k].copy()
        u = _control_at(res, k)
        V = float(res.diag["V"][k])
        eta = float(res.diag["fin_wet_span"][k] / model.fin_span)
        pts.append(dict(t=float(res.t[k]), x=x, u=u, V=V, eta=eta,
                        k_s=ctrl.gain_scale(V, eta if ctrl.estimator is not None else None)))
    return model, pts, res


def _control_at(res, k):
    from src.dynamics.model import ControlInput
    return ControlInput(**{key: float(arr[k]) for key, arr in res.u.items()})


def jacobian(model, x, u, states, input_key, eps_x=1e-5, eps_u=1e-5):
    def f(xx, uu):
        return model.evaluate(0.0, xx, uu)[0][states]

    n = len(states)
    A = np.zeros((n, n))
    for j, s in enumerate(states):
        h = eps_x * max(1.0, abs(x[s]))
        xp, xm = x.copy(), x.copy()
        xp[s] += h
        xm[s] -= h
        A[:, j] = (f(xp, u) - f(xm, u)) / (2.0 * h)
    up = dataclasses.replace(u, **{input_key: getattr(u, input_key) + eps_u})
    um = dataclasses.replace(u, **{input_key: getattr(u, input_key) - eps_u})
    B = (f(x, up) - f(x, um)) / (2.0 * eps_u)
    return A, B


def discretize(A, B, dt):
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n]


def linear_models(model, pts, dt):
    out = []
    for p in pts:
        A, B = jacobian(model, p["x"], p["u"], LON, "delta_e")
        Phi, Gam = discretize(A, B, dt)
        Al, Bl = jacobian(model, p["x"], p["u"], LAT, "delta_r")
        Phil, Gaml = discretize(Al, Bl, dt)
        out.append(dict(p, lon=(Phi, Gam, A), lat=(Phil, Gaml, Al), B_lon=B, B_lat=Bl))
    return out


# ---------------------------------------------------------------------------
# Kapalı çevrim yapıları
# ---------------------------------------------------------------------------
def ff_row(A, B, i_rate, kappa):
    """Trim ileri beslemesinin doğrusal karşılığı: δ_ff = Kff·x (açısal hız sütunu hariç)."""
    n = A.shape[0]
    if kappa == 0.0 or abs(B[i_rate]) < 1e-9:
        return np.zeros(n)
    K = -kappa * A[i_rate, :] / B[i_rate]
    K[i_rate] = 0.0
    return K


def inner_closed_loop(Phi, Gam, g, k_s, sign, i_ang, i_rate, rate_gain, dt, Kff=None):
    """z = [x; I]. δ_k = sign·k_s·(Kp·(b·r − y) + I + Ki·dt·(r − y) − Kd·ẏ)."""
    n = Phi.shape[0]
    Kp, Ki, Kd, b = g
    c = sign * k_s
    Kx = np.zeros(n)
    Kx[i_ang] -= c * (Kp + Ki * dt)
    Kx[i_rate] -= c * Kd * rate_gain
    if Kff is not None:
        Kx += Kff
    KI = c
    Kr = c * (Kp * b + Ki * dt)
    Acl = np.zeros((n + 1, n + 1))
    Acl[:n, :n] = Phi + np.outer(Gam, Kx)
    Acl[:n, n] = Gam * KI
    Acl[n, i_ang] = -Ki * dt
    Acl[n, n] = 1.0
    Bcl = np.zeros(n + 1)
    Bcl[:n] = Gam * Kr
    Bcl[n] = Ki * dt
    return Acl, Bcl, Kx, KI, Kr


def depth_closed_loop(Phi, Gam, A, g, gz, k_s, dt, Kff=None):
    """z = [x_lon; I_θ; I_z], r = Z_ref. θ_r = Kpz·(Z − Zr) + I_z + Kdz·Ż."""
    n = Phi.shape[0]
    iZ, iT, iQ = LON.index(IDX_Z), LON.index(IDX_THETA), LON.index(IDX_Q)
    Kp, Ki, Kd, b = g
    Kpz, Kiz, Kdz = (np.radians(v) for v in gz)
    c = -k_s
    # θ_r = Tx·x + I_z + Tr·Zr
    Tx = Kdz * A[iZ, :].copy()
    Tx[iZ] += Kpz
    Tr = -Kpz
    # δ = c·(Kp·(b·θ_r − θ) + I_θ + Ki·dt·(θ_r − θ) − Kd·q)
    a_r = Kp * b + Ki * dt
    Dx = c * a_r * Tx
    Dx[iT] -= c * (Kp + Ki * dt)
    Dx[iQ] -= c * Kd
    if Kff is not None:
        Dx += Kff
    DIt, DIz, Dr = c, c * a_r, c * a_r * Tr
    m = n + 2
    Acl = np.zeros((m, m))
    Acl[:n, :n] = Phi + np.outer(Gam, Dx)
    Acl[:n, n] = Gam * DIt
    Acl[:n, n + 1] = Gam * DIz
    Acl[n, :n] = Ki * dt * Tx
    Acl[n, iT] -= Ki * dt
    Acl[n, n] = 1.0
    Acl[n, n + 1] = Ki * dt
    Acl[n + 1, iZ] = Kiz * dt
    Acl[n + 1, n + 1] = 1.0
    Bcl = np.zeros(m)
    Bcl[:n] = Gam * Dr
    Bcl[n] = Ki * dt * Tr
    Bcl[n + 1] = -Kiz * dt
    theta_r_row = np.concatenate([Tx, [0.0, 1.0]])
    return Acl, Bcl, theta_r_row, Tr


def step_response(Acl, Bcl, C, N):
    """Birim adım: z_k = z_ss − V·diag(λ^k)·V⁻¹·z_ss (z_0 = 0). C: (m, nz) çıkış satırları."""
    lam, V = np.linalg.eig(Acl)
    if np.max(np.abs(lam)) >= 1.0:
        return None, lam
    z_ss = np.linalg.solve(np.eye(len(lam)) - Acl, Bcl)
    w = np.linalg.solve(V, z_ss)
    k = np.arange(N)[:, None]
    trans = (lam[None, :] ** k) * w[None, :]            # (N, nz) modal
    Y = (C @ z_ss)[None, :] - np.real(trans @ (C @ V).T)
    return Y, lam


def pole_metrics(lam, dt):
    s = np.log(lam.astype(complex)) / dt
    mag = np.abs(s)
    zeta = np.where(mag > 1e-6, -s.real / np.maximum(mag, 1e-12), 1.0)
    osc = np.abs(s.imag) > 1e-6
    zmin = float(zeta[osc].min()) if osc.any() else 1.0
    return zmin, float(mag.max()), s


def step_metrics(y, t, target=1.0):
    frac = y / target
    over = max(0.0, float(frac.max()) - 1.0)
    k95 = np.nonzero(frac >= 0.95)[0]
    t95 = t[k95[0]] if len(k95) else np.inf
    out = np.nonzero(np.abs(frac - 1.0) > 0.05)[0]
    ts5 = t[out[-1]] if len(out) else 0.0
    itae = float(np.sum(t * np.abs(1.0 - frac)) * (t[1] - t[0]))
    return over, t95, ts5, itae


# ---------------------------------------------------------------------------
# Maliyetler
# ---------------------------------------------------------------------------
def inner_eval(g, lin, dt, detail=False, kappa=0.0):
    N = int(T_INNER / dt)
    t = np.arange(N) * dt
    # Doğrusal analiz integratör sınırını görmez: kalıcı durumda |I| ≥ Kp·(1−b)·ref gerekir
    # (+ trim/k_s payı için yarısı ayrılır)
    i_need = g[0] * (1.0 - g[3]) * REF_MAX_DEG
    worst = 100.0 * max(0.0, i_need / (0.5 * I_LIMIT_DEG) - 1.0) ** 2
    rows = []
    for p in lin:
        for mult in GAIN_MULTS:
            k_s = p["k_s"] * mult
            for axis in ("lon", "lat"):
                if axis == "lon":
                    Phi, Gam, _ = p["lon"]
                    # θ ekseni: Z çıkarılır (iç döngü Z'yi görmez)
                    keep = LON_INNER
                    Phi, Gam = Phi[np.ix_(keep, keep)], Gam[keep]
                    sign, i_ang, i_rate, rg = -1.0, 2, 1, 1.0
                    Kff = ff_row(p["lon"][2][np.ix_(keep, keep)], p["B_lon"][keep], 1, kappa)
                else:
                    Phi, Gam, _ = p["lat"]
                    sign, i_ang, i_rate = 1.0, 2, 1
                    rg = 1.0 / max(np.cos(p["x"][IDX_THETA]), 1e-3)
                    Kff = ff_row(p["lat"][2], p["B_lat"], 1, kappa)
                Acl, Bcl, Kx, KI, Kr = inner_closed_loop(Phi, Gam, g, k_s, sign, i_ang, i_rate,
                                                         rg, dt, Kff=Kff)
                n = Phi.shape[0]
                C = np.zeros((2, n + 1))
                C[0, i_ang] = 1.0
                C[1, :n], C[1, n] = Kx, KI           # δ (Kr·r ayrıca eklenir)
                Y, lam = step_response(Acl, Bcl, C, N)
                if Y is None:
                    c = 1e3 + 1e3 * float(np.max(np.abs(lam)) - 1.0)
                    worst = max(worst, c)
                    rows.append((p["t"], mult, axis, "KARARSIZ", c))
                    continue
                y = Y[:, 0]
                delta = np.abs(Y[:, 1] + Kr) * STEP_DEG      # derece
                over, t95, ts5, itae = step_metrics(y, t)
                zmin, smax, _ = pole_metrics(lam, dt)
                d_pk = float(delta.max())
                # giriş bozucusu: plant girişine 1° (δ eşdeğeri) adım, r = 0
                Yd, _ = step_response(Acl, np.concatenate([Gam * np.radians(1.0), [0.0]]), C[:1], N)
                dist_pk = float(np.degrees(np.max(np.abs(Yd[:, 0]))))
                dist_end = float(np.degrees(abs(Yd[-1, 0])))
                c = (itae / 0.02
                     + (dist_pk / 0.10) ** 2
                     + 10.0 * (dist_end / 0.02) ** 2
                     + 4.0 * (over / 0.10) ** 2
                     + (ts5 / 0.6) ** 2
                     + 50.0 * max(0.0, ZETA_MIN_INNER - zmin) ** 2 / 0.01
                     + 10.0 * max(0.0, smax / OMEGA_MAX - 1.0) ** 2
                     + 10.0 * max(0.0, d_pk / (0.6 * LIMIT_DEG) - 1.0) ** 2)
                worst = max(worst, c)
                if detail:
                    rows.append((p["t"], mult, axis, dict(over=100 * over, t95=t95, ts5=ts5,
                                                          zeta=zmin, smax=smax, dpk=d_pk,
                                                          dist_pk=dist_pk, dist_end=dist_end), c))
    return (worst, rows) if detail else worst


def depth_eval(gz, g, lin, dt, detail=False, kappa=0.0):
    N = int(T_DEPTH / dt)
    t = np.arange(N) * dt
    worst, rows = 0.0, []
    for p in lin:
        for mult in GAIN_MULTS:
            Phi, Gam, A = p["lon"]
            Kff = ff_row(A, p["B_lon"], LON.index(IDX_Q), kappa)
            Acl, Bcl, th_row, Tr = depth_closed_loop(Phi, Gam, A, g, gz, p["k_s"] * mult, dt,
                                                     Kff=Kff)
            C = np.zeros((2, Acl.shape[0]))
            C[0, LON.index(IDX_Z)] = 1.0
            C[1] = th_row
            Y, lam = step_response(Acl, Bcl, C, N)
            if Y is None:
                c = 1e3 + 1e3 * float(np.max(np.abs(lam)) - 1.0)
                worst = max(worst, c)
                rows.append((p["t"], mult, "KARARSIZ", c))
                continue
            over, t95, ts5, itae = step_metrics(Y[:, 0], t)
            th_pk = float(np.degrees(np.max(np.abs(Y[:, 1] + Tr))))   # 1 m adım başına
            Bd = np.zeros(Acl.shape[0])
            Bd[:Phi.shape[0]] = _dist_column(p["lon"][2], dt, LON.index(IDX_W)) * W_DIST
            Yd, _ = step_response(Acl, Bd, C[:1], N)
            zd_pk = float(np.max(np.abs(Yd[:, 0])))
            zd_end = float(abs(Yd[-1, 0]))
            zmin, smax, s = pole_metrics(lam, dt)
            # sadece yavaş (dış döngü) kutupların sönümü: |s| < 10 rad/s
            slow = (np.abs(s) < 10.0) & (np.abs(s.imag) > 1e-6)
            zslow = float((-s.real / np.abs(s))[slow].min()) if slow.any() else 1.0
            c = (itae / 2.0
                 + 4.0 * (over / 0.10) ** 2
                 + (ts5 / 4.0) ** 2
                 + 50.0 * max(0.0, ZETA_MIN_DEPTH - zslow) ** 2 / 0.01
                 + 10.0 * max(0.0, th_pk / 3.0 - 1.0) ** 2
                 + (zd_pk / 0.2) ** 2
                 + 10.0 * (zd_end / 0.02) ** 2)
            worst = max(worst, c)
            if detail:
                rows.append((p["t"], mult, dict(over=100 * over, t95=t95, ts5=ts5, zeta=zslow,
                                                th_pk=th_pk, zd_pk=zd_pk, zd_end=zd_end), c))
    return (worst, rows) if detail else worst


W_DIST = 0.5              # dikey kuvvet bozucusu [m/s²] (≈175 N, 350 kg)


def _dist_column(A, dt, i):
    """Sürekli zamanda x_i kanalına birim sabit girişin ZOH ayrık giriş sütunu."""
    e = np.zeros(A.shape[0])
    e[i] = 1.0
    return discretize(A, e, dt)[1]


def _unpack_inner(v):
    Kp, Ki, Kd = np.exp(v[:3])
    b = B_MIN + (1.0 - B_MIN) / (1.0 + np.exp(-v[3]))
    return (Kp, Ki, Kd, b)


def _pack_inner(g):
    Kp, Ki, Kd, b = g
    x = min(max((b - B_MIN) / (1.0 - B_MIN), 1e-3), 1 - 1e-3)
    return np.array([np.log(Kp), np.log(Ki), np.log(Kd), np.log(x / (1 - x))])


def optimize(fun, starts, unpack, pack, maxiter=400, label=""):
    best = None
    for i, g0 in enumerate(starts):
        t0 = time.perf_counter()
        r = minimize(lambda v: fun(unpack(v)), pack(g0), method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-4))
        g = unpack(r.x)
        print(f"  {label} başlangıç {i + 1}/{len(starts)}: J={r.fun:.4f} "
              f"{np.round(g, 4).tolist()} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if best is None or r.fun < best[0]:
            best = (r.fun, g)
    return best


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Otopilot kazanç ayarı (doğrusallaştırma)")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "case1_config.py"))
    ap.add_argument("--scenario", default="otopilot_tutma")
    ap.add_argument("--times", default="0.05,0.3,0.55,0.7,0.8,0.9,1.0,1.2,1.5,2.5,5.0")
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--skip-inner", action="store_true")
    ap.add_argument("--skip-depth", action="store_true")
    ap.add_argument("--out", default=None, help="sonuç JSON dosyası")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    scen = {s["name"]: s for s in cfg.SCENARIOS}[args.scenario]
    times = [float(v) for v in args.times.split(",")]
    dt = float(getattr(cfg, "SIMULATION", {}).get("dt_control", 1e-3))

    print(f"Referans koşum: {args.scenario}")
    model, pts, _ = operating_points(cfg, scen, times)
    lin = linear_models(model, pts, dt)
    for p in lin:
        ev = np.linalg.eigvals(p["lon"][2][:3, :3])
        print(f"  t={p['t']:.2f} V={p['V']:.1f} eta={p['eta']:.2f} k_s={p['k_s']:.2f} "
              f"açık çevrim boyuna max Re(λ)={ev.real.max():+.2f}")

    ctrl = scen["control"]
    g = tuple(float(ctrl["pitch"][k]) for k in PARAM_KEYS)
    ff = ctrl.get("trim_ff")
    kappa = float(ff.get("gain", 1.0)) if isinstance(ff, dict) else (1.0 if ff else 0.0)
    print(f"Trim ileri besleme kazancı κ = {kappa}")
    result = {}
    if not args.skip_inner:
        J0 = inner_eval(g, lin, dt, kappa=kappa)
        print(f"\nİç döngü, mevcut kazançlar {g}: J={J0:.4f}")
        rng = np.random.default_rng(0)
        starts = [g] + [(g[0] * np.exp(rng.normal(0, .5)), g[1] * np.exp(rng.normal(0, .7)),
                         g[2] * np.exp(rng.normal(0, .5)), float(rng.uniform(.4, .9)))
                        for _ in range(args.starts - 1)]
        J, g = optimize(lambda gg: inner_eval(gg, lin, dt, kappa=kappa), starts, _unpack_inner,
                        _pack_inner, label="iç")
        print(f"İç döngü sonucu: J={J:.4f}  " + ", ".join(f"{k}={v:.4g}" for k, v in zip(PARAM_KEYS, g)))
        _, rows = inner_eval(g, lin, dt, detail=True, kappa=kappa)
        for r in rows:
            if r[1] == 1.0 or isinstance(r[3], str):
                print("   ", r[0], r[1], r[2], r[3] if isinstance(r[3], str) else
                      " ".join(f"{k}={v:.3g}" for k, v in r[3].items()), f"J={r[4]:.3f}")
        result["inner"] = dict(zip(PARAM_KEYS, map(float, g)), J=float(J))

    if not args.skip_depth:
        dh = ctrl.get("depth_hold") or dict(Kp_deg_m=2.0, Ki_deg_m_s=0.5, Kd_deg_s_m=1.0)
        gz = tuple(float(dh[k]) for k in DEPTH_KEYS)
        J0 = depth_eval(gz, g, lin, dt, kappa=kappa)
        print(f"\nDerinlik döngüsü, başlangıç {gz}: J={J0:.4f}")
        starts = [gz, (1.0, 0.3, 2.0), (4.0, 1.0, 3.0), (2.0, 0.2, 0.5)][:max(args.starts, 1)]
        unpack = lambda v: tuple(np.exp(v))
        pack = lambda gg: np.log(np.maximum(gg, 1e-4))
        J, gz = optimize(lambda gg: depth_eval(gg, g, lin, dt, kappa=kappa), starts, unpack, pack,
                         maxiter=250, label="derinlik")
        print(f"Derinlik sonucu: J={J:.4f}  " + ", ".join(f"{k}={v:.4g}" for k, v in zip(DEPTH_KEYS, gz)))
        _, rows = depth_eval(gz, g, lin, dt, detail=True, kappa=kappa)
        for r in rows:
            if r[1] == 1.0 or isinstance(r[2], str):
                print("   ", r[0], r[1], r[2] if isinstance(r[2], str) else
                      " ".join(f"{k}={v:.3g}" for k, v in r[2].items()), f"J={r[3]:.3f}")
        result["depth"] = dict(zip(DEPTH_KEYS, map(float, gz)), J=float(J))

    print("\n" + json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
