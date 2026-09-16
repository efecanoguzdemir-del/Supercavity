"""
Sabit adımlı RK4 simülatörü + ayrık kontrolcü + olay gözlemcisi.

Kurallar:
  * Tek koşum = tek kesintisiz entegrasyon; durum hiçbir zaman sıfırlanmaz.
  * Kontrolcü AYRIK: her dt_control başında (t_k = k·dt_control) bir kez
    update(t, x) çağrılır, komut RK4'ün k1..k4 aşamalarında ZOH tutulur.
    dt_control, dt'nin tam katı olmalıdır.
  * Olaylar (kanat ıslanması, planing rejimi, kavite oluşumu ...) sadece
    loglanır; entegratörü durdurmaz / durumu değiştirmez.
  * Tek durdurma koşulu: durum veya türev NaN/inf olursa (ıraksama).
  * locked_states: bu durumların türevi 0'a sabitlenir (regresyon için
    serbestlik kilitleme, ör. pitch/heave/lateral).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from src.dynamics.state import N_STATES, STATE_NAMES, IDX_THETA, IDX_Z
from src.dynamics.kinematics import reset_pitch_warning

CONTROL_KEYS = ("delta_e", "delta_r", "delta_c", "thrust", "gas_flow")


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray                      # (n, 15)
    u: dict                            # kanal -> (n,)
    diag: dict                         # tanı -> (n,)
    events: list                       # [(t, tür, açıklama)]
    n_controller_calls: int
    status: str                        # "ok" | "diverged"
    wall_time: float
    meta: dict = field(default_factory=dict)

    def state(self, name):
        return self.x[:, STATE_NAMES.index(name)]


# ---------------------------------------------------------------------------
# Olay gözlemcisi
# ---------------------------------------------------------------------------
class EventObserver:
    """Adım adım tanı değerlerinden durum değişimlerini yakalar (sadece log)."""

    THETA_LIMIT = np.radians(80.0)

    def __init__(self, fin_span):
        self.fin_span = fin_span
        self.prev = None
        self.events = []

    def _fin_state(self, d):
        w = d["fin_wet_span"]
        if w <= 1e-6:
            return "kuru"
        if w >= self.fin_span - 1e-6:
            return "tam ıslak"
        return "kısmi ıslak"

    def _cover_state(self, d):
        c = d["cover"]
        return "tam örtülü" if c >= 0.99 else ("kısmi örtülü" if c > 0.01 else "ıslak")

    def check(self, t, x, d):
        cur = {
            "kanat": self._fin_state(d),
            "planing": d["planing_regime"],
            "kavite": "var" if d["Lc_ss"] > 0.0 else "yok",
            "gövde": self._cover_state(d),
            "pitch": "limit aşıldı" if abs(x[IDX_THETA]) > self.THETA_LIMIT else "normal",
            "yüzey": "yüzeyde" if x[IDX_Z] < 0.0 else "batık",
        }
        if self.prev is None:
            for k, v in cur.items():
                self.events.append((t, k, f"başlangıç: {v}"))
        else:
            for k, v in cur.items():
                if v != self.prev[k]:
                    self.events.append((t, k, f"{self.prev[k]} → {v}"))
        self.prev = cur


# ---------------------------------------------------------------------------
# Simülatör
# ---------------------------------------------------------------------------
class Simulator:
    def __init__(self, model, controller, t_max, dt=0.001, dt_control=0.001,
                 log_every=1, locked_states=(), verbose=False):
        self.model = model
        self.controller = controller
        self.t_max = float(t_max)
        self.dt = float(dt)
        self.dt_control = float(dt_control)
        ratio = self.dt_control / self.dt
        self.ctrl_every = int(round(ratio))
        if self.ctrl_every < 1 or abs(ratio - self.ctrl_every) > 1e-9:
            raise ValueError(f"dt_control ({dt_control}) dt'nin ({dt}) tam katı olmalı")
        self.log_every = max(int(log_every), 1)
        self.lock_mask = np.ones(N_STATES)
        for name in locked_states:
            self.lock_mask[STATE_NAMES.index(name)] = 0.0
        self.locked_states = tuple(locked_states)
        self.verbose = verbose

    def _f(self, t, x, u):
        return self.model.derivatives(t, x, u) * self.lock_mask

    def run(self, x0):
        reset_pitch_warning()
        self.controller.reset()
        observer = EventObserver(self.model.fin_span)

        n_steps = int(round(self.t_max / self.dt))
        dt = self.dt
        x = np.array(x0, dtype=float)
        u = None
        n_calls = 0

        t_log, x_log, u_log, d_log = [], [], [], []
        status = "ok"
        wall0 = time.perf_counter()
        report_every = max(n_steps // 10, 1)

        for k in range(n_steps + 1):
            t = k * dt
            if k < n_steps and k % self.ctrl_every == 0:
                u = self.controller.update(t, x.copy())
                n_calls += 1

            k1, d = self.model.evaluate(t, x, u)
            k1 = k1 * self.lock_mask
            observer.check(t, x, d)

            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(k1))):
                observer.events.append((t, "ıraksama", "durum/türev NaN veya inf — koşum durdu"))
                status = "diverged"
                break

            if k % self.log_every == 0 or k == n_steps:
                t_log.append(t)
                x_log.append(x.copy())
                u_log.append(u)
                ctrl_log = getattr(self.controller, "log", None)
                if ctrl_log:
                    d = dict(d, **{f"ctrl_{key}": val for key, val in ctrl_log.items()})
                d_log.append(d)

            if k == n_steps:
                break

            h2 = 0.5 * dt
            k2 = self._f(t + h2, x + h2 * k1, u)
            k3 = self._f(t + h2, x + h2 * k2, u)
            k4 = self._f(t + dt, x + dt * k3, u)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if self.verbose and k % report_every == 0:
                print(f"    t={t:.3f}s  V={d['V']:.2f}  Lc={x[12]:.3f}  "
                      f"({time.perf_counter() - wall0:.1f}s)")

        return SimResult(
            t=np.array(t_log),
            x=np.array(x_log),
            u={key: np.array([getattr(ui, key) for ui in u_log]) for key in CONTROL_KEYS},
            diag=_stack_diag(d_log),
            events=observer.events,
            n_controller_calls=n_calls,
            status=status,
            wall_time=time.perf_counter() - wall0,
            meta=dict(t_max=self.t_max, dt=self.dt, dt_control=self.dt_control,
                      log_every=self.log_every, locked_states=list(self.locked_states),
                      fin_span=self.model.fin_span,
                      vent_mode=getattr(self.model, "vent_mode", "Q")),
        )


def _stack_diag(d_log):
    """Tanı sözlüklerini sütunlara çevirir; vektörler _0, _1 ... ile açılır."""
    if not d_log:
        return {}
    out = {}
    for key, val in d_log[0].items():
        if isinstance(val, str):
            out[key] = np.array([d[key] for d in d_log])
        elif np.ndim(val) == 0:
            out[key] = np.array([float(d[key]) for d in d_log])
        else:
            arr = np.array([np.asarray(d[key], dtype=float) for d in d_log])
            for j in range(arr.shape[1]):
                out[f"{key}_{j}"] = arr[:, j]
    return out
