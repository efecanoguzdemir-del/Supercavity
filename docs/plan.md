# Kapalı Çevrim Yapay Kavitasyon Simülasyonu — 3-DOF Attitude Autopilot

## Bağlam

Mevcut 4 ajanlı mimarisi (GeçişFazı.txt) 6-DOF simülasyonun modüler yapısını tanımlıyor: 14 durumlu vektör, hydrodynamics/dynamics/simulation/ bileşenleri. Referans kod (supercavitation_gui_LIVE_v10.py) ampirik kuvvet modellerini ve kavite dinamiğini sağlıyor.

Yeni hedef: **kapalı çevrim 3-DOF attitude otopilot** eklemek — yaw (ψ), pitch (θ), roll (φ) kontrol, 1 ms güncelleme hızı (dt=0.001). Su altı aracını doğrultuda tutması için delta komutlarını (δe, δr, δa) gerçek zamanlı feedback'ten hesaplamak.

## Mimarı — Temiz 5-Agent Model

| Agent | Sorumluluk | Çıktı |
|-------|-----------|-------|
| **Hidrodinamik Tasarımcı** | Kuvvet modelleri, kavite, ventilasyon | `hydrodynamics/` |
| **Modelleme Simülasyoncu** | 14-DOF integrator, RK4, state management | `dynamics/`, `simulation/` |
| **Autopilot Engineer** | 3-DOF PID servo, feedback control | `control/` |
| **Optimizasyoncu** | Case 1 kalibrasyonu, legacy doğrulaması | `validation/calibration.py` |
| **Kontrolcü/Doğrulayıcı** | Decoupling, cross-coupling, işaret kontrol | `validation/sanity_checks.py` |

## Teknik Tasarım

### 1. Autopilot Yapısı (`src/control/autopilot.py`)

```python
class PIDController:
    """Tek-eksen PID (yaw, pitch, roll için)."""
    def __init__(self, Kp, Ki=0.0, Kd=0.0, dt=0.001, output_max=20.0, wrap_error=False):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.output_max = output_max
        self.wrap_error = wrap_error  # yaw için True
        self.error_integral = 0.0
        self.error_prev = 0.0
    
    def step(self, error):
        """PID: u = Kp·e + Ki·∫e·dt + Kd·de/dt, [-output_max, +output_max] satürasyonlu."""
        if self.wrap_error:
            error = (error + np.pi) % (2*np.pi) - np.pi  # -π to π
        
        self.error_integral += error * self.dt
        d_error = (error - self.error_prev) / self.dt if self.dt > 0 else 0
        
        output = self.Kp * error + self.Ki * self.error_integral + self.Kd * d_error
        return np.clip(output, -self.output_max, self.output_max)
    
    def reset(self):
        """Durum sıfırla."""
        self.error_integral = 0.0
        self.error_prev = 0.0


class AttitudeAutopilot(ControlInputBlock):
    """3-DOF attitude servo: yaw (ψ) → δr, pitch (θ) → δe, roll (φ) → δa."""
    
    def __init__(self, pid_yaw, pid_pitch, pid_roll=None, dt=0.001):
        self.pid_yaw = PIDController(**pid_yaw, dt=dt, wrap_error=True)
        self.pid_pitch = PIDController(**pid_pitch, dt=dt, wrap_error=False)
        self.pid_roll = (PIDController(**pid_roll, dt=dt, wrap_error=False) 
                         if pid_roll else None)
    
    def step(self, t, x_state, attitude_refs):
        """
        x_state: [u,v,w, p,q,r, φ,θ,ψ, X,Y,Z, Lc,pc]
        attitude_refs: {ψ_desired, θ_desired, φ_desired, δc_fixed, ...}
        
        Returns: (δc, δe, δr, δa)
        """
        φ_current = x_state[6]
        θ_current = x_state[7]
        ψ_current = x_state[8]
        
        ψ_desired = attitude_refs.get("ψ_desired", ψ_current)
        θ_desired = attitude_refs.get("θ_desired", 0.0)
        φ_desired = attitude_refs.get("φ_desired", 0.0)
        
        δr = self.pid_yaw.step(ψ_desired - ψ_current)
        δe = self.pid_pitch.step(θ_desired - θ_current)
        δa = self.pid_roll.step(φ_desired - φ_current) if self.pid_roll else 0.0
        
        δc = attitude_refs.get("δc_fixed", 0.0)
        
        return δc, δe, δr, δa
```

### 2. Kontrol Girdileri (`src/dynamics/blocks.py`)

ControlInputBlock sözleşmesi:
```python
class ControlInputBlock:
    def step(self, t, x_state, params) -> (δc, δe, δr, δa, Cq):
        pass
```

**Açık çevrim** (mevcut, schedule-tabanlı):
```python
class OpenLoopController(ControlInputBlock):
    def step(self, t, x_state, params):
        return (schedule(t, params["δc_schedule"]),
                schedule(t, params["δe_schedule"]),
                schedule(t, params["δr_schedule"]),
                schedule(t, params["δa_schedule"]),
                schedule(t, params["Cq_schedule"]))
```

**Kapalı çevrim** (yeni, otopilot):
```python
class ClosedLoopController(ControlInputBlock):
    def __init__(self, attitude_autopilot, δc_schedule, Cq_schedule):
        self.autopilot = attitude_autopilot
        self.δc_schedule = δc_schedule
        self.Cq_schedule = Cq_schedule
    
    def step(self, t, x_state, attitude_refs):
        δc_ap, δe_ap, δr_ap, δa_ap = self.autopilot.step(t, x_state, attitude_refs)
        δc = self.δc_schedule(t)
        Cq = self.Cq_schedule(t)
        return δc, δe_ap, δr_ap, δa_ap, Cq
```

### 3. Konfigürasyon (`configs/example_config.py`)

```python
# Simülasyon parametreleri
SIM_PARAMS = {
    "t_max": 10.0,
    "dt": 0.001,        # 1 ms otopilot rate
    "depth": 10.0,
    "mass": 100.0,
    "veh_len": 2.5,
    "veh_diam": 0.15,
    "diam_cav": 0.05,
    # ... (legacy hydrodynamics sabitleri)
}

CONTROL_MODE = "closed_loop"  # "open_loop" veya "closed_loop"

if CONTROL_MODE == "closed_loop":
    # 3-DOF PID kazançları
    AUTOPILOT = {
        "pid_yaw": {
            "Kp": 0.5, "Ki": 0.1, "Kd": 0.2, "output_max": 20.0
        },
        "pid_pitch": {
            "Kp": 1.0, "Ki": 0.05, "Kd": 0.3, "output_max": 15.0
        },
        "pid_roll": {
            "Kp": 0.3, "Ki": 0.02, "Kd": 0.15, "output_max": 15.0
        },
    }
    
    # Açık çevrim girdiler (δc, Cq)
    CONTROL = {
        "δc_schedule": [(0, 2.0), (10, 2.0)],
        "Cq_schedule": [(0, 0.02), (10, 0.02)],
        # Attitude referansları
        "ψ_desired": 0.0,    # hedef yaw [rad]
        "θ_desired": 0.0,    # hedef pitch [rad]
        "φ_desired": 0.0,    # hedef roll [rad]
    }
else:
    # Açık çevrim schedule'lar
    CONTROL = {
        "δc_schedule": [(0, 2.0), (10, 2.0)],
        "δe_schedule": [(0, 0), (10, 0)],
        "δr_schedule": [(0, 0), (10, 0)],
        "δa_schedule": [(0, 0), (10, 0)],
        "Cq_schedule": [(0, 0.02), (10, 0.02)],
    }
```

### 4. Entegrasyon (`src/simulation/simulator.py`)

RK4 entegratör içinde:
```python
def run_simulation(config):
    # Config'den controller seç
    if config["CONTROL_MODE"] == "closed_loop":
        controller = ClosedLoopController(
            AttitudeAutopilot(**config["AUTOPILOT"]),
            schedule_from(config["CONTROL"]["δc_schedule"]),
            schedule_from(config["CONTROL"]["Cq_schedule"])
        )
    else:
        controller = OpenLoopController()
    
    x = x_init  # durum vektörü
    
    for i in range(N_steps):
        t = i * dt
        
        # Kontrol komutları (açık veya kapalı çevrim)
        δc, δe, δr, δa = controller.step(t, x, config["CONTROL"])
        Cq = ...
        
        # Kuvvetler hesapla (hydrodynamics modülleri)
        F_drag, Fz, My, ... = compute_forces(x, δc, δe, δr, δa, Cq)
        
        # RK4 adım: durum güncelle
        x_dot = state_derivatives(x, F_drag, Fz, My, ...)
        x += dt * x_dot  # (gerçekte RK4 4 adım)
```

## Klasör Yapısı

```
src/
├── hydrodynamics/
│   ├── constants.py, cavity.py, cavitator.py, fins.py, planing.py
├── dynamics/
│   ├── blocks.py        (ControlInputBlock sözleşmesi)
│   ├── rigid_body.py    (14-DOF EOM)
│   ├── kinematics.py    (Euler açıları)
│   └── model.py
├── control/             ← YENİ
│   ├── autopilot.py     (PIDController, AttitudeAutopilot)
│   └── controllers.py   (OpenLoop, ClosedLoop wrappers)
├── simulation/
│   ├── simulator.py     (RK4, controller.step() çağrısı)
│   └── run_simulation.py
├── postprocess/
│   ├── plots.py, tables.py
└── validation/
    ├── calibration.py   (Case 1 kalibrasyonu)
    └── sanity_checks.py (decoupling, işaret kontrol)

configs/
└── example_config.py    (kontrol modu, PID kazançları, referanslar)

outputs/
└── <run_id>/            (grafikler, tablolar)
```

## Doğrulama

1. **Açık çevrim regresyon:** mevcut legacy Case 1 (V=40 m/s, δc=2°, α=-1°) match doğruluğu ±%5
2. **Kapalı çevrim yaw:** ψ(t) → ψ_desired, steady-state hata < 1°
3. **Kapalı çevrim pitch:** θ(t) → 0°, steady-state hata < 0.5°
4. **Decoupling:** δr→ψ (Fz < 100 N), δe→θ (Mψ < 50 N·m)
5. **Hız dayanıklılığı:** V=20,40,60 m/s, her hızda ψ/θ izleme
6. **Edge cases:** σ→0, V→0, NaN/ıraksama yok
7. **Anti-windup:** δ satürasyondayken integral windup yok

## Kritik Dosyalar

| Dosya | Amaç |
|-------|------|
| `src/control/autopilot.py` | AttitudeAutopilot implementation |
| `src/dynamics/blocks.py` | ControlInputBlock + OpenLoop/ClosedLoop |
| `src/simulation/simulator.py` | Controller entegrasyonu |
| `configs/example_config.py` | Control mode, PID gains, references |
| `src/validation/sanity_checks.py` | Decoupling testleri |
