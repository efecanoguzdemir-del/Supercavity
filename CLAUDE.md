# Süperkavitasyon Kapalı Çevrim Simülatörü

Yapay süperkavitasyon oluşumu sırasında bir su altı aracının 6-DOF kinetiğini/kinematiğini hesaplayan, kapalı çevrim otopilotlu (1 ms'de bir komut; sadece yaw + pitch) modüler Python simülatörü. Referans: `supercavitation_gui_LIVE_v10.py` (legacy Tkinter GUI — **değiştirme**, sadece referans). Orijinal plan: `docs/plan.md` ve `GeçişFazı.txt` (eskidi: 14 durum, pasif roll vb.; çelişki olursa bu dosya geçerli — kullanıcı teyit etti).

**Kullanıcı kararları (2026-09-15):** Roll kontrolü **yok/gereksiz** — otopilot yalnızca pitch ve yaw komutlar; mixer (δe, δr) → 4 fin. p ve φ yine de durum olarak kalır (pasif, hafif sönümlü; lateral kuvvetlerin roll'a etkisi görünsün). Dc ayrı durum (15 durum) onaylandı. Otopilot ayrık, 1 ms periyotlu, kapalı çevrim.

## Ortam
- Python + `pip install -r requirements.txt` (numpy, scipy, matplotlib). pytest yok; testler proje kökünde düz script (`python test_xxx.py`, exit 0 = geçti). Windows'ta çıktı yönlendirilirken `PYTHONUTF8=1` gerekir (yoksa Unicode print'te UnicodeEncodeError).
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
| `src/control/autopilot.py` (`AxisPID`, `DepthHold`, `AttitudeAutopilot`), `estimator.py` (`CavityEstimator`), `sensors.py` (`CavitySensors`), `tuning.py` | Hazır; pitch+yaw, 1 ms, kanat etkinliği çizelgesi, trim ileri besleme, referans ön filtresi, pc + kuyruk gaz sensörlü tahminci, derinlik tutma | `test_autopilot.py` 60/60 |
| Sürekli rejim geçişleri (`legacy_exact=False`, varsayılan): `hydrodynamics/smooth.py`, body/planing/cavity smooth dalları | Hazır | `test_continuity.py` 38/38 |
| `src/dynamics/model.py` (`VehicleModel.evaluate(t,x,u)→(ẋ,diag)`, `cavity_derivative`, `ControlInput`, `fin_mixer`) | Hazır; legacy Case 1 kuvvet/momentleriyle ≤%0.9; gaz debisi referans derinliği | `test_model.py` 111/111 |
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
- **Gaz debisi referansı** (`gas_flow_ref_depth` [m], kullanıcı 2026-09-16: 5 m): Q o derinliğin hidrostatik basıncında (p_ref) hacim debisi; kavitede izotermal genleşir, Cq_eff = Cq·p_ref/pc. Cebirsel döngü denge basıncında kapalı formda çözülür: pc_t = p∞/(1 + ½ρV²·A_v/(Cq·p_ref)), σ_vent = A_v·pc_t/(Cq·p_ref) (`cavity_state_derivative(gas_p_ref=)`). None → legacy. Sonuç: derinlik arttıkça σ_vent artar (aynı debiyle kavite kısalır).
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

## Otopilot kararları (`control/autopilot.py`, `estimator.py`, `sensors.py`, `tuning.py`)
- Eksen başına `δ = δ_ff + trim + sign·k_s·(Kp·(b·ref − y) + I − Kd·ẏ)`; pitch sign=−1 (δe>0 burun aşağı), yaw sign=+1. Türev Euler açı hızından — hatadan değil. Yaw hatası [−π, π]'ye sarılır.
- Anti-windup: çıkış (ileri besleme dahil) kırpılmışken integratör artışı kırpma yönündeyse donar + |I| ≤ i_limit (40°; b<1'de kalıcı I ≈ Kp·(1−b)·ref + trim/k_s — küçük i_limit referansı kilitler). Opsiyonel `rate_limit_deg_s`. İlk çağrıda dt=0.
- **Kazanç çizelgesi** `k_s = clip((V_ref/V)²·eta_ref/eta, 0.25, 8)`, V_ref=40, eta_ref=0.5; eta = kanat ıslak açıklık oranı (kanat kuvveti wet_span ile doğrusal).
- **Tahminci** `CavityEstimator`: modelin kavite ODE'si (`VehicleModel.cavity_derivative`, model ile ortak yol) V, Z, gaz debisiyle kontrolcü içinde entegre edilir; düzeltmeler: **pc sensörü** (dpc += (pc_ölçüm − pc_hat)/0.02 s) ve **kuyruk gaz sensörü** (x=L gövde yüzeyi gazda mı; uyuşmazlıkta Lc hedef çarpanı s_L ∈ [0.5, 2] 1/s ile ve Lc_hat 2 m/s ile itilir). `sensors.CavitySensors` ölçümleri gerçek durumdan üretir (opsiyonel pc gürültüsü); kontrolcü kavite durumuna yalnız sensörlerle erişmeli. Sonuç: A_v ×1.25'te |Δeta| 0.35 → 0.034; K_Lc ±%15'te 0.16-0.24 kalır (kuyruk sensörü kavite kuyruğun çok ötesindeyken bilgi vermez — kanat hizasına ikinci sensör önerilebilir).
- **Trim ileri beslemesi** `trim_ff=dict(gain=1, every=10)`: tahmin edilen kavitede, p=q=r=0 ve δ=0 iken δ_ff = −M0/Mδ (Mδ sayısal 1°), 10 ms'de bir (3 model değerlendirmesi). Toplam δe'nin neredeyse tamamı ileri beslemeden gelir; geçişteki "aşım"ın kaynağı olan trim kayması ortadan kalktı (%13.8 → %1.1). tuning.py'de doğrusal karşılığı durum geri beslemesi (`ff_row`).
- **Referans ön filtresi** `ref_tau_s=0.05` (θ/ψ program referansı; derinlik çıkışı hariç): adım anındaki k_s·Kp·b·Δref sıçramasını ve anlık doyumu giderir (|δe|max 15° → 3.7°), t95 ~250 → 320 ms.
- **Derinlik tutma** `DepthHold` (senaryoda `depth_ref_m` verilirse): θ_ref += clip(Kp·(Z−Z_ref) + I_z + Kd·Ż, ±5°); Kp=8 °/m, Ki=1.0 °/(m·s), Kd=2.5 °·s/m.
- **Kazançlar** `AXIS_GAINS` Kp=4.445, Ki=27.69, Kd=0.2501, b=0.30 (250 L/s legacy-Q noktasında `tuning.py` ile bulundu; 125 L/s + ileri besleme + sensörlerle yeniden doğrulandı: doğrusal en kötü ζ=0.77). tuning.py yeni noktada Kp 7.3/Ki 53 önerdi — derinlik adımında δe 14.3°'ye (sınır 15°) çıktığı için seçilmedi. `python src/control/tuning.py --scenario otopilot_derinlik --times ...`: donmuş doğrusallaştırma boyuna [w,q,θ,Z] / yanal [v,r,ψ] (u yarı-statik: hızlanan yörüngede planing başlangıcında s≈+0.06 artefaktı), ayrık 1 ms, k_s×{0.7,1,1.4} en kötü durum, ITAE + aşım + ts5 + ζ≥0.6 + |s|≤80 + 1° giriş bozucusu + integratör payı, b≥0.3; derinlik: Z_ref adımı + 0.5 m/s² ẇ bozucusu. Derinlik kazançları doğrusal olmayan ızgara (scratchpad) + doğrusal ζ≥0.5 kontrolüyle seçildi.
- Dersler: (1) kutup sönümünü ayrıca kısıtla; (2) adım penceresi bir sonraki referans değişimine kadar; (3) "%90 sonrası sapma" kullanma — aşım, t95, ts5 ayrı; (4) doğrusal optimum doğrusal olmayanda kırılabilir (b→0, Ki=159 integratör sınırında kilitlendi) — her adayı doğrusal olmayan koşumla doğrula; (5) geçişteki "aşım"ın çoğu trim kayması (bozucu) — ileri besleme en etkili çözüm; (6) test eşiği kayan nokta yuvarlamasıyla geçmesin (doyum kontrolü süre ölçer).
- Doğrusal olmayan performans (`test_autopilot.py` 60/60, 125 L/s): tutma |θ|max 0.004°; `otopilot_adimlar` (θ 0→2 t=1.0 — kanat girişi 1.05 s, tam örtülme 1.73 s; ψ 0→2 t=3.0; θ 2→0 t=4.0) aşım %1.1-1.2, t95 ≈ 320 ms, |δe|max 3.7°, doyum yok; eksen etkileşimi 0.0001°; derinlik: geçişte |ΔZ| 0.37 m, Z_ref 10→11 aşım %2.1, t95 1.22 s, 11 m'de örtülme korunur; sağlamlık (A_v ±%25, K_Lc ±%15, pc gürültüsü 2 kPa) aşım ≤ %3.4.

## Çalışma noktası (kullanıcı, 2026-09-16)
- İtki 8000 N, V0=20 m/s (kavite yok), gaz debisi = gövdeyi tam örten minimum (kullanıcı tahmini 250-400 L/s). **Q, 5 m derinlikte ölçülmüş hacim debisi** (kullanıcı kararı; `VEHICLE.gas_flow_ref_depth=5.0`, bkz. Model kararları). `case1_config.py`: `THRUST_N`, `GAS_FLOW_LPS=125` (model girdisi L/min = ×60), `V0`, `OP_CONTROL`; `AUTOPILOT`, `otopilot_*`, `dusuk_hiz_baslangic` bunu kullanır. `case1_kilitli/serbest/gaz_adimi` legacy Case 1 (6000 N, 40 m/s, `gas_flow_ref_depth=None`).
- Debi taraması (otopilot + derinlik tutma, 15 s; scratchpad `sweep_gas.py`): tam örtülme için en düşük debi 5 m 55-85, 10 m 85-100, 20 m 200-215 L/s (10 m'de pc ~107 kPa < 150 kPa → gaz genleşir). 10 m için 125 L/s (~%25 pay; eta ~0.47) seçildi; daha derin çalışma için debi artırılmalı (debi planlaması yok). Eşiğin altında V↔kavite limit çevrimi / düşük hızda takılma. Debi arttıkça kanat ıslaklığı düşer → kontrol etkinliği düşer.
- Önceki legacy-Q taraması (derinlik tutmasız): ≥215 L/s tam örtülme, 250 seçilmişti — artık geçersiz.
- Otopilotsuz (`dusuk_hiz_baslangic`, 125 L/s): burun sürekli düşer (θ 3.5 s'de −14°), kavite gövdeden ayrılır (örtülme 0.45). Otopilot zorunlu. `otopilot_tutma` derinlik döngüsü kapalı (bilerek batar); `otopilot_derinlik` 12 s derinlik tutma + Z_ref adımı + ψ adımı.
- Özet/kontrol grafiği gaz debisini L/s ve Cq ile gösterir. Simülatör kontrolcü `log` sözlüğünü `ctrl_*` tanı sütunları olarak kaydeder (referanslar, k_s, eta_hat, Lc_hat, pc_hat, s_L, tail_gas, ff_e/ff_r); 02_durus.png bunları çizer.
- Simülasyon hızı: otopilotlu koşumlar ~2-4 s duvar / 1 s simülasyon (ileri besleme 10 ms'de 3 ek model değerlendirmesi); `test_autopilot.py` ~5-10 dk.

## Bilinen sorunlar
- ~~Kuvvet süreksizlikleri~~ → `legacy_exact=False` ile giderildi (bkz. Model kararları, `test_continuity.py`). `legacy_exact=True` modunda bilerek duruyor.
- ~~Otopilot geçiş aşımı~~ → trim ileri beslemesiyle %1.1 (bkz. Otopilot kararları). Kalan: tahmincinin kavite boyu modeli hatasına (K_Lc) duyarlılığı (|Δeta| ≤ 0.24; kontrol hedefleri yine de karşılanıyor).
- Gaz debisi derinliğe göre planlanmıyor: 125 L/s yalnız ~12 m'ye kadar tam örtülme sağlar (20 m için ≥215 L/s).
- `src/hydrodynamics/constants.py::LEGACY_CASE_1` **yanlış geometri** (Dn=0.05, mass=100, L=2.0). Yukarıdaki değerlerle düzeltilmeli.
- `cavity.cavity_axis_offset` docstring'i "h>0 = kavite aşağı", legacy yorumu "yukarı" (hesapta |h| kullanıldığı için sonucu etkilemiyor). V≤0.5 m/s'de legacy sapmayı 0 alır.
- `cylinder_inertia` homojen silindir, geometrik merkez etrafında (CG 2.4 m, geometrik merkez 2.0 m); gerçek inertia config'den verilmeli.

## Agent ile çalışma
- Agent roller: Hidrodinamik Tasarımcı (`hydrodynamics/`), Modelleme Simülasyoncu (`dynamics/`, `simulation/`, `postprocess/`), Autopilot Engineer (`control/`), Optimizasyoncu (`validation/calibration.py`), Kontrolcü/Doğrulayıcı (`validation/sanity_checks.py`). Her agent kendi klasörü dışına dokunmaz; eksik fizik hydrodynamics'e eklenir, dynamics'e gömülmez.
- Uzun tek-parça agent görevleri API bağlantı kopmalarıyla (ECONNRESET) iş kaybetti. Görevleri küçük, bağımsız parçalara böl; agent'a dosyaları erken diske yazdır; paralel agent'lar farklı dosyalara dokunsun.
- Agent "testler geçti" dese de testleri kendin çalıştırıp doğrula (ilk hidrodinamik agent başarısız legacy regresyonunu özetinde belirtmemişti).
