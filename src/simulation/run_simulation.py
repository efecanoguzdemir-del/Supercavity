"""
CLI giriş noktası: config oku → her senaryoyu ayrı koşum olarak simüle et →
outputs/<run_id>/ altına CSV + PNG + özet + config_used.py yaz.

Kullanım:
  python src/simulation/run_simulation.py --config configs/case1_config.py
  python src/simulation/run_simulation.py --config configs/case1_config.py --scenario case1_kilitli
  seçenekler: --out outputs  --no-plots  --verbose

Config dosyası (Python modülü) şunları tanımlar:
  VEHICLE     dict — legacy GUI anahtar adlarıyla araç/hidrodinamik parametreleri
  SIMULATION  dict — t_max, dt, dt_control, log_every (varsayılanlar)
  SCENARIOS   list[dict] — her biri:
      name, description
      initial:  V, alpha_deg, beta_deg, phi_deg, theta_deg, psi_deg, p/q/r_deg_s,
                depth, X, Y, Lc, Dc, pc   (eksikler 0; pc yoksa buhar basıncı)
      control:  {"type": "schedule", delta_e_deg, delta_r_deg, delta_c_deg, thrust, gas_flow}
                {"type": "autopilot", pitch={Kp,Ki,Kd,limit_deg,...}, yaw={...},
                 theta_ref_deg, psi_ref_deg, V_ref, delta_c_deg, thrust, gas_flow}
                (her kanal/referans sayı veya [(t, değer), ...])
      vehicle:  VEHICLE üzerine yazılacak anahtarlar (opsiyonel)
      simulation: SIMULATION üzerine yazılacaklar (opsiyonel)
      locked_states: türevi 0 tutulacak durum adları (opsiyonel)
"""

import argparse
import datetime as _dt
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dynamics.blocks import ScheduleController  # noqa: E402
from src.dynamics.model import VehicleModel  # noqa: E402
from src.dynamics.state import make_state  # noqa: E402
from src.hydrodynamics.constants import P_VAP  # noqa: E402
from src.simulation.simulator import Simulator  # noqa: E402

SIM_DEFAULTS = dict(t_max=1.0, dt=0.001, dt_control=0.001, log_every=1)


def load_config(path):
    spec = importlib.util.spec_from_file_location("sim_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("VEHICLE", "SCENARIOS"):
        if not hasattr(mod, name):
            raise ValueError(f"config '{path}' içinde {name} tanımlı değil")
    return mod


def initial_state(ic):
    """Senaryo başlangıç sözlüğü → 15 durum. Hız vektörü akış açılarından."""
    V = float(ic.get("V", 0.0))
    a = np.radians(ic.get("alpha_deg", 0.0))
    b = np.radians(ic.get("beta_deg", 0.0))
    return make_state(
        u=V * np.cos(a) * np.cos(b), v=V * np.sin(b), w=V * np.sin(a) * np.cos(b),
        p=np.radians(ic.get("p_deg_s", 0.0)), q=np.radians(ic.get("q_deg_s", 0.0)),
        r=np.radians(ic.get("r_deg_s", 0.0)),
        phi=np.radians(ic.get("phi_deg", 0.0)), theta=np.radians(ic.get("theta_deg", 0.0)),
        psi=np.radians(ic.get("psi_deg", 0.0)),
        X=ic.get("X", 0.0), Y=ic.get("Y", 0.0), Z=ic.get("depth", 0.0),
        Lc=ic.get("Lc", 0.0), Dc=ic.get("Dc", 0.0),
        pc=P_VAP if ic.get("pc") is None else ic["pc"],
    )


def build_controller(spec):
    spec = dict(spec or {})
    kind = spec.pop("type", "schedule")
    if kind == "schedule":
        return ScheduleController(**spec)
    if kind == "autopilot":
        from src.control.autopilot import AttitudeAutopilot
        return AttitudeAutopilot(**spec)
    raise ValueError(f"bilinmeyen kontrolcü tipi: {kind!r} ('schedule' | 'autopilot')")


def build_run(cfg, scenario):
    """(model, simulator, x0, çözümlenmiş senaryo sözlüğü)."""
    vehicle = dict(cfg.VEHICLE)
    vehicle.update(scenario.get("vehicle", {}))
    sim = dict(SIM_DEFAULTS)
    sim.update(getattr(cfg, "SIMULATION", {}))
    sim.update(scenario.get("simulation", {}))

    model = VehicleModel(vehicle)
    controller = build_controller(scenario.get("control"))
    simulator = Simulator(model, controller, t_max=sim["t_max"], dt=sim["dt"],
                          dt_control=sim["dt_control"], log_every=sim["log_every"],
                          locked_states=scenario.get("locked_states", ()))
    x0 = initial_state(scenario.get("initial", {}))
    resolved = dict(name=scenario["name"], description=scenario.get("description", ""),
                    vehicle=vehicle, simulation=sim, initial=scenario.get("initial", {}),
                    control=scenario.get("control"),
                    locked_states=list(scenario.get("locked_states", ())))
    return model, simulator, x0, resolved


def run_scenario(cfg, cfg_path, scenario, out_root, plots=True, verbose=False, stamp=None):
    from src.postprocess import tables
    model, simulator, x0, resolved = build_run(cfg, scenario)
    simulator.verbose = verbose
    name = scenario["name"]
    stamp = stamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{stamp}_{name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {name}: t_max={simulator.t_max}s dt={simulator.dt}s "
          f"dt_control={simulator.dt_control}s → {out_dir}")
    result = simulator.run(x0)

    tables.write_timeseries_csv(result, os.path.join(out_dir, "timeseries.csv"))
    tables.write_events_csv(result, os.path.join(out_dir, "events.csv"))
    tables.write_summary(result, os.path.join(out_dir, "summary.txt"), name)
    shutil.copyfile(cfg_path, os.path.join(out_dir, "config_used.py"))
    with open(os.path.join(out_dir, "scenario_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False, default=str)
    if plots:
        from src.postprocess import plots as P
        P.make_all_plots(result, out_dir, title=name)

    print("\n".join(tables.summary_lines(result, name)[:3]))
    return result, out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="Süperkavitasyon 6-DOF simülasyonu")
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario", action="append",
                    help="sadece bu senaryo(lar)ı çalıştır (tekrarlanabilir)")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    scenarios = cfg.SCENARIOS
    if args.scenario:
        scenarios = [s for s in scenarios if s["name"] in args.scenario]
        missing = set(args.scenario) - {s["name"] for s in scenarios}
        if missing:
            ap.error(f"config'de olmayan senaryo: {sorted(missing)}")

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    statuses = []
    for sc in scenarios:
        result, _ = run_scenario(cfg, args.config, sc, args.out, plots=not args.no_plots,
                                 verbose=args.verbose, stamp=stamp)
        statuses.append(result.status)
    return 0 if all(s == "ok" for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
