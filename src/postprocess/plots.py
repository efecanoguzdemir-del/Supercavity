"""
Zaman serisi grafikleri (matplotlib, Agg — ekran açmaz).

Her koşum için PNG seti:
  01_hiz_akis.png     V, u/v/w, α/β
  02_durus.png        φ/θ/ψ, p/q/r
  03_yorunge.png      X–derinlik, X–Y
  04_kavite.png       Lc, Dc (durum + hedef), σ, pc, örtülme
  05_kuvvetler.png    sürükleme bileşenleri, dikey bileşenler, gövde F, momentler
  06_kontrol.png      δe/δr/δc, kanat sapmaları, itki, gaz debisi
Olaylar (kanat/planing/kavite geçişleri) dikey kesik çizgiyle işaretlenir.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_EVENT_KINDS = ("kanat", "planing", "kavite", "gövde", "ıraksama")


def _mark_events(ax, result):
    for t, kind, text in result.events:
        if kind in _EVENT_KINDS and not text.startswith("başlangıç"):
            ax.axvline(t, color="0.6", lw=0.7, ls="--", zorder=0)


def _style(ax, ylabel, result, events=True):
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if events:
        _mark_events(ax, result)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")


def _fig(nrows, title, sharex=True):
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.6 * nrows), sharex=sharex,
                             squeeze=False)
    fig.suptitle(title, fontsize=11)
    return fig, axes.ravel()


def _save(fig, out_dir, name, xlabel_axes=()):
    for ax in xlabel_axes:
        ax.set_xlabel("t [s]")
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_all_plots(result, out_dir, title=""):
    os.makedirs(out_dir, exist_ok=True)
    t, d, u = result.t, result.diag, result.u
    s = result.state
    deg = np.degrees
    files = []

    fig, ax = _fig(3, f"{title} — hız ve akış açıları")
    ax[0].plot(t, d["V"], label="V")
    _style(ax[0], "V [m/s]", result)
    for n in ("u", "v", "w"):
        ax[1].plot(t, s(n), label=n)
    _style(ax[1], "hız [m/s]", result)
    ax[2].plot(t, deg(d["alpha"]), label="α")
    ax[2].plot(t, deg(d["beta"]), label="β")
    ax[2].plot(t, deg(d["alpha_eff"]), label="α_eff = α+δc", ls=":")
    _style(ax[2], "açı [deg]", result)
    files.append(_save(fig, out_dir, "01_hiz_akis.png", [ax[-1]]))

    fig, ax = _fig(2, f"{title} — duruş")
    for n, lab in (("phi", "φ"), ("theta", "θ"), ("psi", "ψ")):
        ax[0].plot(t, deg(s(n)), label=lab)
    _style(ax[0], "Euler [deg]", result)
    for n in ("p", "q", "r"):
        ax[1].plot(t, deg(s(n)), label=n)
    _style(ax[1], "açısal hız [deg/s]", result)
    files.append(_save(fig, out_dir, "02_durus.png", [ax[-1]]))

    fig, ax = _fig(2, f"{title} — yörünge (NED)", sharex=False)
    fig.set_size_inches(10, 7)
    ax[0].plot(s("X"), s("Z"))
    ax[0].invert_yaxis()
    ax[0].set_xlabel("X [m]")
    _style(ax[0], "derinlik Z [m]", result, events=False)
    ax[1].plot(s("X"), s("Y"))
    ax[1].set_xlabel("X [m]")
    _style(ax[1], "Y (sancak) [m]", result, events=False)
    files.append(_save(fig, out_dir, "03_yorunge.png"))

    fig, ax = _fig(5, f"{title} — kavite")
    ax[0].plot(t, s("Lc"), label="Lc (durum)")
    ax[0].plot(t, d["Lc_ss"], ls="--", label="Lc_ss (hedef)")
    _style(ax[0], "Lc [m]", result)
    ax[1].plot(t, s("Dc"), label="Dc (durum)")
    ax[1].plot(t, d["Dc_ss"], ls="--", label="Dc_ss (hedef)")
    _style(ax[1], "Dc [m]", result)
    ax[2].plot(t, d["sigma"], label="σ")
    ax[2].plot(t, np.minimum(d["sigma_vapor"], 3.0), ls=":", label="σ_buhar")
    _style(ax[2], "σ", result)
    ax[3].plot(t, s("pc") / 1e3, label="pc")
    ax[3].plot(t, d["pc_target"] / 1e3, ls="--", label="pc_hedef")
    _style(ax[3], "pc [kPa]", result)
    ax[4].plot(t, d["cover"], label="gövde örtülme")
    ax[4].plot(t, d["fin_wet_span"] / max(result.meta.get("fin_span", 1.0), 1e-9),
               label="kanat ıslak oranı")
    _style(ax[4], "oran [-]", result)
    files.append(_save(fig, out_dir, "04_kavite.png", [ax[-1]]))

    fig, ax = _fig(4, f"{title} — kuvvetler ve momentler")
    for k, lab in (("F_cav", "kavitatör"), ("F_skin", "sürtünme"), ("F_press", "taban/planing"),
                   ("F_body_drag", "gövde α"), ("F_fin_drag", "kanat")):
        ax[0].plot(t, d[k], label=lab)
    _style(ax[0], "sürükleme [N]", result)
    for k, lab in (("F_cavz", "kavitatör"), ("F_planing", "planing"), ("F_body_lift", "gövde"),
                   ("F_fin_total", "kanat"), ("F_buoy", "Arşimet")):
        ax[1].plot(t, d[k], label=lab)
    ax[1].axhline(-d["F_grav"][0], color="k", lw=0.8, ls=":", label="ağırlık (m·g)")
    _style(ax[1], "dikey, yukarı+ [N]", result)
    for j, lab in enumerate(("Fx", "Fy", "Fz")):
        ax[2].plot(t, d[f"F_body_{j}"], label=lab)
    _style(ax[2], "gövde toplam F [N]", result)
    for j, lab in enumerate(("K (roll)", "M (pitch)", "N (yaw)")):
        ax[3].plot(t, d[f"M_body_{j}"], label=lab)
    _style(ax[3], "moment [N·m]", result)
    files.append(_save(fig, out_dir, "05_kuvvetler.png", [ax[-1]]))

    fig, ax = _fig(4, f"{title} — kontrol girdileri")
    ax[0].step(t, deg(u["delta_e"]), where="post", label="δe")
    ax[0].step(t, deg(u["delta_r"]), where="post", label="δr")
    ax[0].step(t, deg(u["delta_c"]), where="post", label="δc")
    _style(ax[0], "komut [deg]", result)
    for j, lab in enumerate(("1-üst", "2-sağ", "3-alt", "4-sol")):
        key = f"delta_fins_{j}"
        if key in d:
            ax[1].step(t, deg(d[key]), where="post", label=lab)
    _style(ax[1], "kanat-yerel δ [deg]", result)
    ax[2].step(t, u["thrust"], where="post", label="itki")
    _style(ax[2], "T [N]", result)
    ax[3].step(t, u["gas_flow"], where="post", label="gaz debisi")
    _style(ax[3], "Q [L/min] / Cq", result)
    files.append(_save(fig, out_dir, "06_kontrol.png", [ax[-1]]))

    return files
