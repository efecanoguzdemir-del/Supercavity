"""
Kapalı çevrim otopilot testi — basit demo.

Bu script:
1. AttitudeAutopilot oluştur
2. Basit RK4 adımlarıyla durum güncelle
3. Doğrulama testlerini çalıştır
"""

import sys
import numpy as np
sys.path.insert(0, ".")

from src.control.autopilot import AttitudeAutopilot
from src.dynamics.blocks import OpenLoopController, ClosedLoopController
from src.validation.sanity_checks import run_all_checks
from configs.example_config import SIM_PARAMS, CONTROL_MODE, AUTOPILOT, CONTROL, LEGACY_CASE_1


def test_attitude_autopilot():
    """Basit otopilot testi."""

    print("=" * 80)
    print("KAPALΙ ÇEVRIM OTOPILOT TESTI")
    print("=" * 80)

    dt = SIM_PARAMS["dt"]
    t_max = SIM_PARAMS["t_max"]
    n_steps = int(t_max / dt)

    # Başlangıç durum vektörü (minimal)
    x_state = np.array([
        30.0,   # u (body-frame longitudinal speed)
        0.0,    # v (lateral)
        0.0,    # w (vertical)
        0.0,    # p (roll rate)
        0.0,    # q (pitch rate)
        0.0,    # r (yaw rate)
        0.0,    # φ (roll angle)
        0.0,    # θ (pitch angle)
        0.0,    # ψ (yaw angle)
        0.0,    # X (inertial position)
        0.0,    # Y
        0.0,    # Z
        0.1,    # Lc (cavity length)
        101325.0,  # pc (cavity pressure)
    ])

    # Tutarlı durum vektörü boyutu: 14
    assert len(x_state) == 14, f"Durum vektörü boyutu hata: {len(x_state)}"

    # Autopilot oluştur
    if CONTROL_MODE == "closed_loop":
        print(f"\n✓ Kapalı çevrim modu seçildi")
        autopilot = AttitudeAutopilot(
            **AUTOPILOT,
            dt=dt
        )

        # ClosedLoopController
        δc_sched = CONTROL.get("δc_schedule", [(0, 0)])
        Cq_sched = CONTROL.get("Cq_schedule", [(0, 0)])
        controller = ClosedLoopController(autopilot, δc_sched, Cq_sched)

        print(f"  Yaw PID: Kp={AUTOPILOT['pid_yaw']['Kp']}, Ki={AUTOPILOT['pid_yaw']['Ki']}, Kd={AUTOPILOT['pid_yaw']['Kd']}")
        print(f"  Pitch PID: Kp={AUTOPILOT['pid_pitch']['Kp']}, Ki={AUTOPILOT['pid_pitch']['Ki']}, Kd={AUTOPILOT['pid_pitch']['Kd']}")
    else:
        print(f"\n✓ Açık çevrim modu seçildi")
        δc_sched = CONTROL.get("δc_schedule", [(0, 0)])
        δe_sched = CONTROL.get("δe_schedule", [(0, 0)])
        δr_sched = CONTROL.get("δr_schedule", [(0, 0)])
        δa_sched = CONTROL.get("δa_schedule", [(0, 0)])
        Cq_sched = CONTROL.get("Cq_schedule", [(0, 0)])
        controller = OpenLoopController(δc_sched, δe_sched, δr_sched, δa_sched, Cq_sched)

    # Simülasyon döngüsü
    print(f"\n✓ Simülasyon başlıyor: t_max={t_max}s, dt={dt}s, n={n_steps} adım")

    # Çıkış dizileri
    results = {
        "t": [],
        "ψ": [],
        "θ": [],
        "φ": [],
        "δr": [],
        "δe": [],
        "δa": [],
        "δc": [],
    }

    for i in range(n_steps):
        t = i * dt

        # Kontrol komutları
        if CONTROL_MODE == "closed_loop":
            δc, δe, δr, δa, Cq = controller.step(t, x_state, {
                "attitude_refs": CONTROL.get("attitude_refs", {}),
                "δc_schedule": CONTROL.get("δc_schedule", [(0, 0)]),
                "Cq_schedule": CONTROL.get("Cq_schedule", [(0, 0)]),
            })
        else:
            δc, δe, δr, δa, Cq = controller.step(t, x_state, CONTROL)

        # Minimal durum güncelleme (demo için)
        # Gerçek simülasyon hydrodynamics modülü kuvvetleri hesaplar ve RK4 ile integre eder
        # Burada sadece pitch/yaw açılarını döngüyle güncelliyoruz

        # Basit integrasyon: dθ/dt = q, dψ/dt = r
        # Elevator ve rudder kuvvetlerinden q/r hızları üretir (simplified)
        q = 0.1 * δe / 15.0  # 0.1 rad/s @ max elevator
        r = 0.05 * δr / 20.0  # 0.05 rad/s @ max rudder

        x_state[4] = q  # pitch rate
        x_state[5] = r  # yaw rate
        x_state[7] += q * dt  # pitch açısı
        x_state[8] += r * dt  # yaw açısı

        # Angle wrap
        x_state[8] = (x_state[8] + np.pi) % (2*np.pi) - np.pi  # ψ: -π to π

        # Çıkışları kaydet
        results["t"].append(t)
        results["ψ"].append(x_state[8])
        results["θ"].append(x_state[7])
        results["φ"].append(x_state[6])
        results["δr"].append(δr)
        results["δe"].append(δe)
        results["δa"].append(δa)
        results["δc"].append(δc)

    print(f"✓ Simülasyon tamamlandı")

    # Doğrulama testleri
    print(f"\n✓ Doğrulama testleri çalıştırılıyor...")
    checks = run_all_checks(results, LEGACY_CASE_1)

    print(f"\n{'=' * 80}")
    print(f"DOĞRULAMA SONUÇLARI")
    print(f"{'=' * 80}")

    for check_name, check_result in checks["checks"].items():
        status = "✓ PASS" if check_result["passed"] else "✗ FAIL"
        print(f"\n{status} — {check_name.upper()}")
        print(f"  {check_result['summary']}")

    overall = "✓ TÜM TESTLER GEÇTÎ" if checks["all_passed"] else "✗ BAZÎ TESTLER BAŞARISIZ"
    print(f"\n{'=' * 80}")
    print(f"{overall}")
    print(f"{'=' * 80}\n")

    return checks["all_passed"]


if __name__ == "__main__":
    success = test_attitude_autopilot()
    sys.exit(0 if success else 1)
