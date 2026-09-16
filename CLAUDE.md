# Süperkavitasyon Kapalı Çevrim Simülatörü

Yapay süperkavitasyon oluşumu sırasında bir su altı aracının 6-DOF kinetiğini/kinematiğini hesaplayan, kapalı çevrim otopilotlu (1 ms'de bir komut; sadece yaw + pitch) modüler Python simülatörü. Referans: `supercavitation_gui_LIVE_v10.py` (legacy Tkinter GUI — **değiştirme**, sadece referans). Orijinal plan: `docs/plan.md` ve `GeçişFazı.txt` (eskidi: 14 durum, pasif roll vb.; çelişki olursa bu dosya geçerli — kullanıcı teyit etti).

**Kullanıcı kararları (2026-09-15):** Roll kontrolü **yok/gereksiz** — otopilot yalnızca pitch ve yaw komutlar; mixer (δe, δr) → 4 fin. p ve φ yine de durum olarak kalır (pasif, hafif sönümlü; lateral kuvvetlerin roll'a etkisi görünsün). Dc ayrı durum (15 durum) onaylandı. Otopilot ayrık, 1 ms periyotlu, kapalı çevrim.

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
| `src/validation/sanity_checks.py` | **İskelet** — eski, hiçbir yerde kullanılmıyor (adım 5'te yeniden yazılacak) | — |
| `src/control/autopilot.py` (`AxisPID`, `AttitudeAutopilot`) | Hazır; pitch+yaw, 1 ms, doğrusal olmayan 6-DOF'ta doğrulandı — **rejim geçişi sırasında adımda %28 aşım (bilinen sorun)** | `test_autopilot.py` 27/27 |
| Sürekli rejim geçişleri (`legacy_exact=False`, varsayılan): `hydrodynamics/smooth.py`, body/planing/cavity smooth dalları | Hazır | `test_continuity.py` 38/38 |
| `src/dynamics/model.py` (`VehicleModel.evaluate(t,x,u)→(ẋ,diag)`, `ControlInput`, `fin_mixer`) | Hazır; legacy Case 1 kuvvet/momentleriyle ≤%0.9 | `test_model.py` 101/101 |
| `src/simulation/simulator.py` (RK4, ayrık kontrolcü ZOH, `EventObserver`, `locked_states`), `run_simulation.py` (CLI), `src/postprocess/plots.py`, `tables.py`, `configs/case1_config.py` | Hazır | `test_simulator.py` 18/18 |
| `src/dynamics/blocks.py`: `Controller.update(t,x)→ControlInput`, `ScheduleController` | Hazır (eski iskelet sınıflar ve `configs/example_config.py` silindi) | — |
| `test_hydrodynamics_legacy.py` | **Başarısız** — yanlış geometri; `legacy_reference.py` ile değiştirilmeli | — |

## Sıradaki adımlar
1. ~~`src/dynamics/model.py`~~ tamamlandı (bkz. "Model kararları").
2. ~~Simülatör~~ ve 3. ~~doğrulama A–D~~ tamamlandı. Çalıştır: `python src/simulation/run_simulation.py --config configs/case1_config.py [--scenario ad] [--no-plots]` → `outputs/<zaman>_<senaryo>/` (timeseries.csv, events.csv, summary.txt, 01-06 PNG, config_used.py, scenario_resolved.json). Hız: ~3.5 s duvar / 1 s simülasyon (dt=1 ms; darboğaz `body.compute_body_forces` 40 kesit döngüsü).
4. ~~Autopilot~~ tamamlandı (bkz. "Otopilot kararları").
5. Kalibrasyon (scipy least_squares) ve sanity_checks yeniden yazımı (zaman ortalaması değil, eşleşen durumda karşılaştırma).

## Konvansiyonlar
- Durum (15): `[u,v,w,p,q,r,phi,theta,psi,X,Y,Z,Lc,Dc,pc]` — indeksler **sadece** `src/dynamics/state.py` sabitlerinden (`IDX_*`). Dc, legacy'de Lc ile aynı τ=0.15 s gecikmeyle entegre edildiği için ayrı durum.
- Gövde ekseni x-ileri, y-sancak, z-aşağı; eylemsiz NED (Z+ = derinlik); Euler ZYX; SI + radyan (derece sadece config/grafik sınırında). Kod tanımlayıcıları ASCII (δ gibi Unicode kullanma).
- Gövde çerçevesi orijini CG'de; legacy konumları burundan ölçülür → `kinematics.nose_to_body_x(x_from_nose, x_cg)`.
- `hydrodynamics/` fonksiyonları legacy konvansiyonunda döner (lift **yukarı pozitif**, x burundan, `M_y = (x_cg − x)·F_up`). Gövde çerçevesine dönüşüm **sadece** `model.py`'de (`F_body = (−drag, −side, −up)`, `M = Σ r×F`). Docstring başlıkları buna göre düzeltildi.

## Model kararları (`model.py`)
- Akış açıları durumdan: α=atan2(w,u), β=asin(v/V). Yaw düzlemi pitch'in simetriği (kavitatör, gövde lifti, kanatlar).
- **Kanat geometrisi legacy'den bilinçli sapma:** legacy üst/alt (90/270) kanatlara dikey kuvvet yazdırıyordu; modelde normal azimuttan geometrik türetilir → sağ/sol = elevator, üst/alt = rudder. Mixer `δ_i = δe·n_z + δr·n_y` (kanat-yerel sapma; sol kanat −δe). Kavite eksenel simetrik olduğundan Case 1 sayısal olarak aynı: legacy fin1=fin3=−0.80° ≡ δe=−0.80°.
- İşaretler: δe>0 → burun aşağı; δr>0 → burun sancak (N>0); δc>0 → burun yukarı. Kanat kuvvetleri yerel hızla (v + ω×r) → q, r, p sönümü doğal olarak çıkar.
- Kavite: `hydrodynamics/cavity.py::cavity_state_derivative` (legacy ODE formu). p∞ derinliği Z durumundan.
- `fins_use_steady_cavity=True` legacy tuhaflığını (kanat ıslaklığı Lc_ss/Dc_ss ile) taklit eder; varsayılan False (gecikmeli durum). Regresyon testi True kullanır.
- Kalan legacy farkı: legacy kanat kavite koordinatına sabit `+0.013 m` ekliyor, model `x_body_start` (cone=0) kullanıyor → t≥0.35'te kanat kuvvetinde ~%0.6-0.9 fark (doğrulandı: 0.013 ile birebir).
- Basitleştirmeler: drag gövde ekseni boyunca; gövde ıslanması/planing kavite ekseni sapması sadece pitch düzleminde (yerçekimi + α_eff); kanat ıslaklığı sapmasız kavite yarıçapıyla. `x_cg_mass` yok sayılır (orijin = kütle merkezi).
- **`legacy_exact` anahtarı:** `True` → legacy rejim anahtarlamaları birebir (TÜM legacy regresyon testleri bunu kullanır: `test_model`, `test_simulator` (B), `case1_kilitli` senaryosu). `False` (varsayılan, `case1_config.VEHICLE`) → sürekli geçişler. Legacy'de bulunan sıçramalar ve çözümleri:
  - Kavite oluşumu (Lc=0): legacy tam-ıslak dalında F_skin=0 (7.3 kN sıçrama) ve taban sürüklemesi 0.15·q·S ↔ 0 → kesit döngüsü her zaman çalışır; taban sürüklemesi transom kaplamasıyla harmanlanır.
  - Kesit kapanması: Logvinovich profili ucunda R→Rn sonra 0'a sıçrıyordu (40 basamak) → kapanma yarısında R² = (Rn² + (Rmax²−Rn²)S)·D, x→Lc'de R→0. Kalan: profil R ∝ √(Lc−x) dikey teğetli (sonsuz eğim, sürekli) — kavite ~5 cm iken ilk kesit mm ölçeğinde kapanır.
  - Islak yay bitişik↔serbest sınırı (|h| > 0.41·R_v iken sıçrama) → rc ∈ [R_v, 1.2R_v] harmanlama; <%5 bitişik eşiği kaldırıldı.
  - Transom: kavite ucu transomu geçerken planing 0→660 N, δ=2R_v eşiği → `transom_forces_smooth` (kaplama ağırlığı smoothstep(rc_tail/R_v), çıkış harmanlaması δ/R_v ∈ [1.5, 2.5]).
  - σ=1: Lc_ss 1.3 m→0, Cx0(1+σ)→Cx0 (F_cav yarıya) → hedef boyutlar smoothstep((1−σ)/0.2) ile 0'a iner; Cx0·(1+min(σ,1)).
  - Sonuç: sürekli modelde Case 1 inception daha fazla sürükleme görür (legacy F_skin=0 hatası yok) → V daha çok düşer (0.7 s'de ~27 m/s), kavite daha uzar (~9 m).
- Legacy'ye uyum için hydrodynamics'e eklenenler: `cavity_state_derivative`, `fins.cavity_radius_ellipse`, `fin_lift_and_drag(span_for_AR=)`, `cavitator_lift_coefficient(cone_lift_gain=)`, `planing.transom_forces_legacy` (ρV², işaretli α_p — eski `planing_force_dzielski_kurdila` ½ρV² kullanıyor, legacy ile uyumsuz), `constants.CD_BASE_EXPOSED/FULLY_WET`.

## Legacy Case 1 (altın referans)
GUI varsayılanları (`supercavitation_gui_LIVE_v10.py` ~satır 1605-1663): depth=10, diam_cav=0.20, mass=350, thrust=6000, veh_len=4.0, veh_diam=0.30, L_taper=0.60, gas_flow=15000 L/min (vent_mode Q), x_cg=2.4, V_init=40, alpha_aoa=−1°, delta_cav=+2°, CL_alpha_body=2.5, A_v=0.020, cavitator_type="cone", cone_apex_deg=40, cone_drag_visc=0.18, k_g=0.78, geom_model="savchenko", fin_chord=0.10, fin_span=0.25, fin_x_pos=3.80, fin_delta_1=fin_delta_3=−0.80°, steady_v_mode=False, planing_enable=True, t_max=0.5, dt=0.001.

t≈0.5 s sonucu: V=36.9, σ=0.15, Lc=4.19, Dc=0.52, pc=94173 · **Fd=5764.1 N**, **My=3343.8 N·m** · Fz(toplam, yukarı+)=−4851.2 = F_cavz 395.0 + F_planing 688.9 + F_body_lift −2.2 + F_fin_total −2502.3 + F_buoy 2.8 + F_grav −3433.5.
- Dokümandaki **Fz=−2109 N = F_cavz + F_body_lift + F_fin_total** (planing, buoyancy, ağırlık hariç).
- Legacy döngüde kuvvetleri önceki adımın hızıyla (`vi = V[i-1]`) hesaplar; son örnek t=0.499.
- Legacy gövde lift'i döngü integralini (~satır 1354'te) basit formülle ezer: `CL_alpha_body·q·A_max·α·wet_frac`. `body.py` ikisini de döndürür (`F_body_lift` = legacy, `F_body_lift_section` = integral).

## Simülatör notları
- Legacy i. adımda pc/Lc/Dc'yi önce güncelleyip sonra kaydeder → legacy kaydı t_i + dt anına aittir. Zaman serisi kıyasında model t_i + dt'de örneklenmeli (yoksa erken t'de %5-17 sahte fark). Kaymalı kıyasta kilitli Case 1 (steady V) tüm seriler ≤%0.7.
- Serbest ileri hızda V(0.5): legacy 36.93 (hatalı V[i-1] güncellemesi), model 34.51 — beklenen fark, regresyon hedefi değil.
- Açık çevrim serbest Case 1 dengeli trim değil (başlangıçta M≈+6400 N·m); legacy fiziğiyle θ 0.5 s'de ~6.7°'ye çıkıyordu (sürekli fizikte değerler farklı).

## Otopilot kararları (`control/autopilot.py`)
- Eksen başına `δ = trim + sign·k_s·(Kp·(b·ref − y) + I − Kd·ẏ)`; pitch sign=−1 (δe>0 burun aşağı), yaw sign=+1. Türev Euler açı hızından (θ̇ = q cosφ − r sinφ, ψ̇ = (q sinφ + r cosφ)/cosθ) — hatadan değil. Yaw hatası [−π, π]'ye sarılır.
- Anti-windup: çıkış kırpılmışken integratör artışı kırpma yönündeyse integrasyon donar + |I| ≤ i_limit. Opsiyonel `rate_limit_deg_s` (servo hız sınırı). İlk çağrıda dt=0 (integrasyon yok).
- Kazanç çizelgesi `k_s = clip((V_ref/V)², 0.25, 4)`, V_ref=40. **Kanat ıslaklığıyla çizelgeleme yok**: kavite kanatları geçince (ıslak oran 1.0→~0.3) kontrol etkinliği ~3.6× düşer ve kanat α-kararlılığı neredeyse kaybolur — kazançlar bu aralığa dayanıklı seçildi.
- Kazançlar (`configs/case1_config.py::AXIS_GAINS`, `legacy_exact=False` fiziğiyle): Kp=2.9, Ki=10.5, Kd=0.19, b=0.5, limit ±15°. Yöntem: kapalı çevrim koşumdan alınan 5 çalışma noktasında (kavite yok → 9 m kavite, kanat ıslak oranı 1.0→0.23, etkinlik ~6× düşük; son iki noktada araç **açık çevrimde statik kararsız**, özdeğer +1.8 boyuna / +1.6 yanal) sayısal Jacobian ile boyuna [u,w,q,θ] / yanal [v,r,ψ] (φ, p çıkarıldı: roll nötr, sıfır özdeğer) doğrusallaştırma; en kötü durum maliyeti = ITAE + aşım + komut genliği + ζ<0.6 cezası + rate döngüsü ωc>80 rad/s cezası; Nelder-Mead çoklu başlangıç. En kötü ζ≈0.54. (Önceki legacy-fizik kazançları Kp=4.4/Ki=25 sürekli fizikte ζ=0.13'e düşüyordu.) Ayar betiği projede değil (oturum scratchpad'i) — gerekirse `src/control/tuning.py` olarak eklenmeli.
- Dersler: (1) adım yanıtı metriği tek başına yetmez — yavaş reel kutup düşük sönümlü kompleks çifti maskeleyebilir → **kutup sönümünü ayrıca kısıtla**; (2) adım testi penceresi bir sonraki referans değişimine kadar uzanmalı (0.5 s pencere %27 aşımı kaçırmıştı); (3) "%90'a ulaştıktan sonraki sapma" metriği tanım gereği ≥%10 çıkar — kullanma; aşım (hedef ötesi), t95, ts5 (±%5 bandından son çıkış) ayrı ölç.
- Doğrusal olmayan performans (otopilot_adimlar, t_max=3 s): kavite oturmuş rejimde ψ adımı aşım %0, t95=286 ms; θ 2→0 adımı aşım %3.2, t95=270 ms, ts5=592 ms. Pitch↔yaw etkileşimi 0.001°. Maks kanat komutu 8.8°.
- **Bilinen sorun — rejim geçişinde adım:** t=0.5 s θ 0→2° adımı en sert geçişe denk gelir (V 40→27 m/s, kavite 3→9 m, kanatlar kavite içine girer, araç açık çevrim kararsızlaşır) → **%27.6 aşım**, 1 s'de oturmuyor. Sabit kazançlı PID'nin sınırı: Ki/b düşürmek aşımı %8'e indiriyor ama oturmuş rejimde t95'i 0.3 s → 1.2-1.6 s'ye çıkarıyor (denendi). Çözüm adayları: kanat etkinliği/kavite tahminiyle kazanç çizelgesi (kavite boyu doğrudan ölçülmez; gaz debisi, V, derinlikten model tabanlı tahmin), trim ileri besleme, veya daha gelişmiş (LQR/gain-scheduled) kontrolcü. Kullanıcı kararı bekliyor.
- Derinlik tutma yok: θ tutulurken araç derinliği serbestçe kayar (Case 1'de 2 s'de ~0.5 m). İstenirse dış döngü (Z → θ_ref) eklenebilir.

## Bilinen sorunlar
- ~~Kuvvet süreksizlikleri~~ → `legacy_exact=False` ile giderildi (bkz. Model kararları, `test_continuity.py`). `legacy_exact=True` modunda bilerek duruyor.
- Otopilot rejim geçişi sırasında adımda %27.6 aşım (bkz. Otopilot kararları).
- `src/hydrodynamics/constants.py::LEGACY_CASE_1` **yanlış geometri** (Dn=0.05, mass=100, L=2.0). Yukarıdaki değerlerle düzeltilmeli.
- `cavity.cavity_axis_offset` docstring'i "h>0 = kavite aşağı", legacy yorumu "yukarı" (hesapta |h| kullanıldığı için sonucu etkilemiyor). V≤0.5 m/s'de legacy sapmayı 0 alır.
- `cylinder_inertia` homojen silindir, geometrik merkez etrafında (CG 2.4 m, geometrik merkez 2.0 m); gerçek inertia config'den verilmeli.

## Agent ile çalışma
- Agent roller: Hidrodinamik Tasarımcı (`hydrodynamics/`), Modelleme Simülasyoncu (`dynamics/`, `simulation/`, `postprocess/`), Autopilot Engineer (`control/`), Optimizasyoncu (`validation/calibration.py`), Kontrolcü/Doğrulayıcı (`validation/sanity_checks.py`). Her agent kendi klasörü dışına dokunmaz; eksik fizik hydrodynamics'e eklenir, dynamics'e gömülmez.
- Uzun tek-parça agent görevleri API bağlantı kopmalarıyla (ECONNRESET) iş kaybetti. Görevleri küçük, bağımsız parçalara böl; agent'a dosyaları erken diske yazdır; paralel agent'lar farklı dosyalara dokunsun.
- Agent "testler geçti" dese de testleri kendin çalıştırıp doğrula (ilk hidrodinamik agent başarısız legacy regresyonunu özetinde belirtmemişti).
