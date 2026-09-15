# Süperkavitasyon Kapalı Çevrim Simülatörü

Yapay süperkavitasyon oluşumu sırasında bir su altı aracının 6-DOF kinetiğini/kinematiğini hesaplayan, kapalı çevrim otopilotlu (1 ms'de bir komut; yaw + pitch, roll opsiyonel) modüler Python simülatörü. Referans: `supercavitation_gui_LIVE_v10.py` (legacy Tkinter GUI — **değiştirme**, sadece referans). Orijinal plan: `docs/plan.md` (bazı kısımları eskidi; çelişki olursa bu dosya geçerli).

## Ortam
- Python + `pip install -r requirements.txt` (numpy, scipy, matplotlib). pytest yok; testler proje kökünde düz script (`python test_xxx.py`, exit 0 = geçti).
- Legacy GUI'yi GUI açmadan çalıştırmak için: `python src/validation/legacy_reference.py` (matplotlib'i mock'layıp `simulate()`'i Case 1 ile çağırır).

## Durum (son commit itibarıyla)
| Modül | Durum | Test |
|---|---|---|
| `src/hydrodynamics/` cavity, cavitator, fins, planing | Hazır | `test_hydrodynamics_units.py` 6/6 |
| `src/hydrodynamics/body.py` gövde ıslanma/skin/buoyancy/gövde lift | Hazır, legacy ile birebir | `test_body_forces.py` |
| `src/dynamics/state.py` | Hazır (tek indeks kaynağı) | — |
| `src/dynamics/rigid_body.py`, `kinematics.py` | Hazır | `test_rigid_body.py` 14/14 |
| `src/validation/legacy_reference.py` | Altın referans | — |
| `src/control/autopilot.py`, `src/dynamics/blocks.py`, `configs/example_config.py`, `src/validation/sanity_checks.py`, `test_autopilot.py` | **İskelet** — entegre değil, sorunlu (aşağıya bkz.) | — |
| `src/dynamics/model.py`, `src/simulation/`, `src/postprocess/` | **Yok** | — |
| `test_hydrodynamics_legacy.py` | **Başarısız** — yanlış geometri; `legacy_reference.py` ile değiştirilmeli | — |

## Sıradaki adımlar
1. `src/dynamics/model.py`: `(t, x, u_cmd) → ẋ`. Tüm hidrodinamik kuvvetleri topla; legacy "yukarı pozitif" → gövde z-aşağı dönüşümünü **tek yerde** yap. Kavite durumları (Lc, Dc, pc) hydrodynamics fonksiyonlarından. Fin mixer: (δe, δr, δa) → 4 fin sapması; Case 1 trimi (fin1=fin3=−0.80°) mixer üzerinden üretilebilmeli.
2. `src/simulation/simulator.py`: sabit adımlı RK4. Kontrolcü **ayrık**: her `dt_control`=1 ms başında bir kez çağrılır, komut RK4 alt aşamalarında ZOH tutulur (k1..k4'te çağırmak PID durumunu bozar). Olay tespiti sadece loglar, durumu sıfırlamaz. `run_simulation.py --config` → `outputs/<run_id>/` (CSV, PNG, config_used.py); grafikler `src/postprocess/plots.py`.
3. Doğrulama: (A) legacy t=0.5 durumunda statik bileşen regresyonu, (B) pitch/heave/lateral kilitli dinamik regresyon (V, Lc, pc, Fd zaman serisi), (C) açık çevrim 6-DOF NaN'sız, (D) kapalı çevrim çalışıyor + kontrolcü çağrı sayısı = t_max/dt_control.
4. Autopilot: anti-windup, türevi ölçümden (p, q, r) al, birimleri düzelt, kazanç ayarı.
5. Kalibrasyon (scipy least_squares) ve sanity_checks yeniden yazımı (zaman ortalaması değil, eşleşen durumda karşılaştırma).

## Konvansiyonlar
- Durum (15): `[u,v,w,p,q,r,phi,theta,psi,X,Y,Z,Lc,Dc,pc]` — indeksler **sadece** `src/dynamics/state.py` sabitlerinden (`IDX_*`). Dc, legacy'de Lc ile aynı τ=0.15 s gecikmeyle entegre edildiği için ayrı durum.
- Gövde ekseni x-ileri, y-sancak, z-aşağı; eylemsiz NED (Z+ = derinlik); Euler ZYX; SI + radyan (derece sadece config/grafik sınırında). Kod tanımlayıcıları ASCII (δ gibi Unicode kullanma).
- Gövde çerçevesi orijini CG'de; legacy konumları burundan ölçülür → `kinematics.nose_to_body_x(x_from_nose, x_cg)`.
- **Dikkat — işaret tutarsızlığı:** `hydrodynamics/__init__.py` ve `planing.py` başlıkları "Fz pozitif = aşağı" diyor, ama `fins`, `planing`, `body` fonksiyonları lift'i **yukarı pozitif** döndürüyor (legacy konvansiyonu). Legacy moment: `M_y = (x_cg − x)·F_up`, pozitif = burun yukarı. Dönüşüm model.py'de yapılmalı; docstring başlıkları düzeltilmeli.
- Legacy fin azimutu: 90°=üst, 0°=sağ, 270°=alt, 180°=sol; legacy'de azimut 90/270 fin'ler **dikey** kuvvet üretiyor (fiziksel olarak şüpheli — netleştir).

## Legacy Case 1 (altın referans)
GUI varsayılanları (`supercavitation_gui_LIVE_v10.py` ~satır 1605-1663): depth=10, diam_cav=0.20, mass=350, thrust=6000, veh_len=4.0, veh_diam=0.30, L_taper=0.60, gas_flow=15000 L/min (vent_mode Q), x_cg=2.4, V_init=40, alpha_aoa=−1°, delta_cav=+2°, CL_alpha_body=2.5, A_v=0.020, cavitator_type="cone", cone_apex_deg=40, cone_drag_visc=0.18, k_g=0.78, geom_model="savchenko", fin_chord=0.10, fin_span=0.25, fin_x_pos=3.80, fin_delta_1=fin_delta_3=−0.80°, steady_v_mode=False, planing_enable=True, t_max=0.5, dt=0.001.

t≈0.5 s sonucu: V=36.9, σ=0.15, Lc=4.19, Dc=0.52, pc=94173 · **Fd=5764.1 N**, **My=3343.8 N·m** · Fz(toplam, yukarı+)=−4851.2 = F_cavz 395.0 + F_planing 688.9 + F_body_lift −2.2 + F_fin_total −2502.3 + F_buoy 2.8 + F_grav −3433.5.
- Dokümandaki **Fz=−2109 N = F_cavz + F_body_lift + F_fin_total** (planing, buoyancy, ağırlık hariç).
- Legacy döngüde kuvvetleri önceki adımın hızıyla (`vi = V[i-1]`) hesaplar; son örnek t=0.499.
- Legacy gövde lift'i döngü integralini (~satır 1354'te) basit formülle ezer: `CL_alpha_body·q·A_max·α·wet_frac`. `body.py` ikisini de döndürür (`F_body_lift` = legacy, `F_body_lift_section` = integral).

## Bilinen sorunlar
- `src/hydrodynamics/constants.py::LEGACY_CASE_1` **yanlış geometri** (Dn=0.05, mass=100, L=2.0). Yukarıdaki değerlerle düzeltilmeli.
- `configs/example_config.py` uydurma araç değerleri (mass=100, L=2.5); açık çevrim modunda `AUTOPILOT` tanımsız → import hatası.
- `autopilot.py`: anti-windup yok, türev hata üzerinden (derivative kick), hata rad / çıkış derece karışık → kazançlar anlamsız; sabit indeks (6/7/8) kullanıyor. Araçta aileron yok — roll ancak 4 fin diferansiyel sapmasıyla.
- `blocks.py`: `ControlInputBlock.step` imzası/dönüş tuple'ı ile `ClosedLoopController` tutarsız.
- `cavity.cavity_axis_offset` docstring'i "h>0 = kavite aşağı", legacy yorumu "yukarı" (hesapta |h| kullanıldığı için sonucu etkilemiyor). V≤0.5 m/s'de legacy sapmayı 0 alır.
- `cylinder_inertia` homojen silindir, geometrik merkez etrafında (CG 2.4 m, geometrik merkez 2.0 m); gerçek inertia config'den verilmeli.

## Agent ile çalışma
- Agent roller: Hidrodinamik Tasarımcı (`hydrodynamics/`), Modelleme Simülasyoncu (`dynamics/`, `simulation/`, `postprocess/`), Autopilot Engineer (`control/`), Optimizasyoncu (`validation/calibration.py`), Kontrolcü/Doğrulayıcı (`validation/sanity_checks.py`). Her agent kendi klasörü dışına dokunmaz; eksik fizik hydrodynamics'e eklenir, dynamics'e gömülmez.
- Uzun tek-parça agent görevleri API bağlantı kopmalarıyla (ECONNRESET) iş kaybetti. Görevleri küçük, bağımsız parçalara böl; agent'a dosyaları erken diske yazdır; paralel agent'lar farklı dosyalara dokunsun.
- Agent "testler geçti" dese de testleri kendin çalıştırıp doğrula (ilk hidrodinamik agent başarısız legacy regresyonunu özetinde belirtmemişti).
