"""
Tablo çıktıları: zaman serisi CSV, olay listesi CSV, özet metni.

Açılar CSV'de hem rad (durum sütunları) hem derece (_deg sütunları) verilir.
"""

import csv

import numpy as np

from src.dynamics.state import STATE_NAMES

_ANGLE_STATES = ("phi", "theta", "psi", "p", "q", "r")
_ANGLE_DIAG = ("alpha", "beta", "alpha_eff")


def timeseries_columns(result):
    """(başlıklar, 2B dizi) — durumlar, kontroller, sayısal tanılar."""
    cols = {"t": result.t}
    for i, name in enumerate(STATE_NAMES):
        cols[name] = result.x[:, i]
    for name in _ANGLE_STATES:
        cols[f"{name}_deg"] = np.degrees(result.state(name))
    for key, arr in result.u.items():
        cols[f"u_{key}"] = arr
    for key in ("delta_e", "delta_r", "delta_c"):
        cols[f"u_{key}_deg"] = np.degrees(result.u[key])
    for key, arr in result.diag.items():
        if arr.dtype.kind in "fi":
            cols[key] = arr
    for key in _ANGLE_DIAG:
        if key in result.diag:
            cols[f"{key}_deg"] = np.degrees(result.diag[key])
    headers = list(cols)
    return headers, np.column_stack([cols[h] for h in headers])


def write_timeseries_csv(result, path):
    headers, data = timeseries_columns(result)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in data:
            w.writerow([f"{v:.9g}" for v in row])
    return path


def write_events_csv(result, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "tur", "aciklama"])
        for t, kind, text in result.events:
            w.writerow([f"{t:.6f}", kind, text])
    return path


def summary_lines(result, name=""):
    d = result.diag
    s = lambda n: result.state(n)
    lines = [
        f"Koşum: {name}",
        f"Durum: {result.status}   süre (sim): {result.t[-1]:.4f} s   duvar saati: {result.wall_time:.1f} s",
        f"dt={result.meta['dt']}  dt_control={result.meta['dt_control']}  "
        f"kontrolcü çağrısı={result.n_controller_calls}  kilitli={result.meta['locked_states']}",
        "",
        f"{'büyüklük':14s} {'başlangıç':>12s} {'son':>12s} {'min':>12s} {'max':>12s}",
    ]
    rows = [
        ("V [m/s]", d["V"]), ("u [m/s]", s("u")), ("w [m/s]", s("w")), ("v [m/s]", s("v")),
        ("alpha [deg]", np.degrees(d["alpha"])), ("beta [deg]", np.degrees(d["beta"])),
        ("phi [deg]", np.degrees(s("phi"))), ("theta [deg]", np.degrees(s("theta"))),
        ("psi [deg]", np.degrees(s("psi"))), ("q [deg/s]", np.degrees(s("q"))),
        ("r [deg/s]", np.degrees(s("r"))), ("derinlik Z [m]", s("Z")), ("X [m]", s("X")),
        ("Lc [m]", s("Lc")), ("Dc [m]", s("Dc")), ("pc [Pa]", s("pc")), ("sigma", d["sigma"]),
        ("cover", d["cover"]), ("Fd [N]", d["Fd"]), ("F_fin_up [N]", d["F_fin_total"]),
        ("F_planing [N]", d["F_planing"]), ("M_pitch [Nm]", d["M_body_1"]),
        ("N_yaw [Nm]", d["M_body_2"]),
    ]
    for label, arr in rows:
        lines.append(f"{label:14s} {arr[0]:12.4g} {arr[-1]:12.4g} {np.min(arr):12.4g} {np.max(arr):12.4g}")
    lines += ["", "Olaylar:"]
    lines += [f"  t={t:8.4f}  {kind:8s} {text}" for t, kind, text in result.events]
    return lines


def write_summary(result, path, name=""):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines(result, name)) + "\n")
    return path
