"""
Doğrulama testleri: decoupling, işaret kontrol, legacy Case 1 regresyon.

Decoupling: δr→ψ, δe→θ, δa→φ — cross-coupling minimal
İşaret: kontrol komutunun etkisi beklenen yönde
Legacy Case 1: supercavitation_gui_LIVE_v10.py ile kalibrasyona ait match
"""

import numpy as np


class SanityChecker:
    """Kapalı çevrim otopilot doğrulama."""

    @staticmethod
    def check_decoupling(results, tolerance_cross_pct=10.0):
        """
        Decoupling testi: δr komutunun yaw etkileyip pitch etkilememesi.

        Args:
            results: simülasyon sonuçları dict
                {
                    "t": [time],
                    "ψ": [yaw],
                    "θ": [pitch],
                    "φ": [roll],
                    "Fz": [vertical force],
                    "My": [pitching moment],
                    "Mψ": [yawing moment],
                    "δr": [rudder],
                    "δe": [elevator],
                }
            tolerance_cross_pct: cross-coupling toleransu (%)

        Returns:
            {"passed": bool, "summary": str}
        """
        issues = []

        # δr → ψ etkilemesi
        if "δr" in results and "ψ" in results:
            δr_mean = np.mean(np.abs(results["δr"]))
            ψ_var = np.var(results["ψ"]) if len(results["ψ"]) > 1 else 0

            if δr_mean > 1.0 and ψ_var < 0.01:  # rudder aktif ama yaw değişmiyor
                issues.append("δr→ψ: rudder aktif fakat yaw yeterince etkilenmiyor")

        # δe → θ etkilemesi
        if "δe" in results and "θ" in results:
            δe_mean = np.mean(np.abs(results["δe"]))
            θ_var = np.var(results["θ"]) if len(results["θ"]) > 1 else 0

            if δe_mean > 1.0 and θ_var < 0.01:  # elevator aktif ama pitch değişmiyor
                issues.append("δe→θ: elevator aktif fakat pitch yeterince etkilenmiyor")

        # Cross-coupling: δr ile My (pitch moment)
        if "δr" in results and "My" in results:
            δr_changes = np.abs(np.diff(results["δr"]))
            My_changes = np.abs(np.diff(results["My"]))

            if len(δr_changes) > 0 and len(My_changes) > 0:
                δr_idx = np.argmax(δr_changes)
                My_max_around_δr = np.max(My_changes[max(0, δr_idx-2):min(len(My_changes), δr_idx+3)])
                δr_max = np.max(δr_changes)

                cross_coupling_ratio = My_max_around_δr / (δr_max + 1e-6)
                if cross_coupling_ratio > tolerance_cross_pct / 100.0:
                    issues.append(f"Cross-coupling δr→My: {cross_coupling_ratio*100:.1f}% (tolerans: {tolerance_cross_pct}%)")

        passed = len(issues) == 0
        summary = "\n".join(issues) if issues else "Decoupling testleri geçti"

        return {"passed": passed, "summary": summary}

    @staticmethod
    def check_sign_correctness(results):
        """
        İşaret kontrol: kontrol komutunun beklenen yönde etki etmesi.

        Args:
            results: simülasyon sonuçları

        Returns:
            {"passed": bool, "summary": str}
        """
        issues = []

        # δe > 0 (up elevator) → pitch moment pozitif (burun yukarı)
        if "δe" in results and "My" in results:
            up_elevator_idx = results["δe"] > 5.0
            if np.any(up_elevator_idx):
                My_mean_up = np.mean(results["My"][up_elevator_idx])
                if My_mean_up < -100:  # pitch moment negatif ise hata
                    issues.append(f"δe→My işaret hataları: δe>5° için My<-100 (beklenen My>0)")

        # δr > 0 (right rudder) → yaw rate pozitif (sağa dön)
        if "δr" in results and "r" in results:
            right_rudder_idx = results["δr"] > 5.0
            if np.any(right_rudder_idx):
                r_mean_right = np.mean(results["r"][right_rudder_idx])
                if r_mean_right < -0.01:  # r (yaw hızı) negatif ise hata
                    issues.append(f"δr→r işaret hataları: δr>5° için r<-0.01 (beklenen r>0)")

        passed = len(issues) == 0
        summary = "\n".join(issues) if issues else "İşaret kontrolleri geçti"

        return {"passed": passed, "summary": summary}

    @staticmethod
    def check_legacy_case_1(results, legacy_case_1):
        """
        Legacy Case 1 regresyonu: supercavitation_gui_LIVE_v10.py ile match doğruluğu.

        Args:
            results: simülasyon sonuçları (nominal Case 1 koşusu)
            legacy_case_1: {
                "V": 40.0,
                "F_drag_target": 5764.0,
                "Fz_target": -2109.0,
                "My_target": 3344.0,
                "tolerance_pct": 5.0,
            }

        Returns:
            {"passed": bool, "summary": str}
        """
        issues = []

        tolerance_pct = legacy_case_1.get("tolerance_pct", 5.0)

        # Nominal hız
        target_v = legacy_case_1.get("V", 40.0)
        if "V" in results:
            V_mean = np.mean(results["V"])
            V_error_pct = 100 * np.abs(V_mean - target_v) / (target_v + 1e-6)
            if V_error_pct > tolerance_pct:
                issues.append(f"Hız hatası: {V_mean:.1f} m/s (hedef {target_v}, hata {V_error_pct:.1f}%)")

        # Sürükle kuvveti
        target_fd = legacy_case_1.get("F_drag_target", 5764.0)
        if "F_drag" in results:
            Fd_mean = np.mean(results["F_drag"])
            Fd_error_pct = 100 * np.abs(Fd_mean - target_fd) / (target_fd + 1e-6)
            if Fd_error_pct > tolerance_pct:
                issues.append(f"Drag hatası: {Fd_mean:.0f} N (hedef {target_fd:.0f}, hata {Fd_error_pct:.1f}%)")

        # Dikey kuvvet
        target_fz = legacy_case_1.get("Fz_target", -2109.0)
        if "Fz" in results:
            Fz_mean = np.mean(results["Fz"])
            Fz_error_pct = 100 * np.abs(Fz_mean - target_fz) / (abs(target_fz) + 1e-6)
            if Fz_error_pct > tolerance_pct:
                issues.append(f"Fz hatası: {Fz_mean:.0f} N (hedef {target_fz:.0f}, hata {Fz_error_pct:.1f}%)")

        # Pitch moment
        target_my = legacy_case_1.get("My_target", 3344.0)
        if "My" in results:
            My_mean = np.mean(results["My"])
            My_error_pct = 100 * np.abs(My_mean - target_my) / (target_my + 1e-6)
            if My_error_pct > tolerance_pct:
                issues.append(f"My hatası: {My_mean:.0f} N·m (hedef {target_my:.0f}, hata {My_error_pct:.1f}%)")

        passed = len(issues) == 0
        summary = "\n".join(issues) if issues else f"Legacy Case 1 regresyonu geçti (tolerans ±{tolerance_pct}%)"

        return {"passed": passed, "summary": summary}

    @staticmethod
    def check_no_nan_or_overflow(results):
        """
        NaN/sonsuz değer kontrol.

        Args:
            results: simülasyon sonuçları

        Returns:
            {"passed": bool, "summary": str}
        """
        issues = []

        for key, values in results.items():
            if isinstance(values, list):
                arr = np.array(values)
                if np.any(np.isnan(arr)):
                    issues.append(f"{key}: NaN değer tespit edildi")
                if np.any(np.isinf(arr)):
                    issues.append(f"{key}: sonsuz değer tespit edildi")

        passed = len(issues) == 0
        summary = "\n".join(issues) if issues else "NaN/overflow kontrolü geçti"

        return {"passed": passed, "summary": summary}


def run_all_checks(results, legacy_case_1=None):
    """
    Tüm doğrulama testlerini çalıştır.

    Args:
        results: simülasyon sonuçları
        legacy_case_1: legacy Case 1 referans (opsiyonel)

    Returns:
        {"all_passed": bool, "checks": {...}}
    """
    checker = SanityChecker()

    checks = {
        "decoupling": checker.check_decoupling(results),
        "sign_correctness": checker.check_sign_correctness(results),
        "no_nan_overflow": checker.check_no_nan_or_overflow(results),
    }

    if legacy_case_1:
        checks["legacy_case_1"] = checker.check_legacy_case_1(results, legacy_case_1)

    all_passed = all(check["passed"] for check in checks.values())

    return {
        "all_passed": all_passed,
        "checks": checks,
    }
