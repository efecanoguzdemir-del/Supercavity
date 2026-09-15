"""
================================================================================
 Hydrodynamics Module — Unit Tests
================================================================================
Her modülün temel fonksiyonlarının doğruluğu ve singularity koruması.

Test kategorileri:
  1. Singularity guards (σ→0, V→0, h→0)
  2. Monotonicity (V↑→F↑, σ↓→Lc↑)
  3. Boundary conditions (immersion=0→force=0)
  4. Physical reasonableness (signs, magnitudes)
================================================================================
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.hydrodynamics import (
    constants,
    cavity,
    cavitator,
    fins,
    planing,
)


def test_cavity_singularities():
    """Test cavity geometry with singularity edge cases."""
    print("\n" + "="*80)
    print("TEST: Cavity Geometry Singularities")
    print("="*80)

    # σ → 0 (very small cavitation number)
    sigma_tiny = 1e-5
    Lc, Dc, Cx = cavity.cavity_geometry(sigma_tiny, Dn=0.05, model="savchenko")
    print(f"\n✓ σ={sigma_tiny:.0e}: Lc={Lc:.4f} m, Dc={Dc:.4f} m (no division by zero)")
    assert Lc > 0 and Dc > 0, "Cavity dimensions must be positive"

    # σ outside SIGMA_MIN should be clipped
    sigma_negative = -0.5  # Invalid
    Lc, Dc, Cx = cavity.cavity_geometry(sigma_negative, Dn=0.05)
    print(f"✓ σ={sigma_negative}: Handled gracefully (clipped to SIGMA_MIN)")
    assert Lc > 0 and Dc > 0, "Should not produce zero/negative dimensions"

    # Test all 6 models
    models = ["garabedian", "savchenko", "semenenko", "may", "vasin_serebr", "logvinovich"]
    sigma_ref = 0.02
    print(f"\n6-Model Comparison (σ={sigma_ref}):")
    for model in models:
        Lc, Dc, Cx = cavity.cavity_geometry(sigma_ref, Dn=0.05, model=model)
        print(f"  {model:15s}: Lc={Lc:.4f} m, Dc={Dc:.4f} m, Lc/Dc={Lc/Dc:.1f}")
        assert Lc > 0 and Dc > 0, f"Model {model} failed"

    print("\n✓ PASSED: Cavity geometry")
    return True


def test_cavity_monotonicity():
    """Test cavity physics: V↑→F↑, σ↓→Lc↑."""
    print("\n" + "="*80)
    print("TEST: Cavity Monotonicity")
    print("="*80)

    # σ↓ → Lc↑ (smaller σ = larger cavity)
    Lc_high_sigma, _, _ = cavity.cavity_geometry(0.1, Dn=0.05)
    Lc_low_sigma, _, _ = cavity.cavity_geometry(0.01, Dn=0.05)

    print(f"\nσ=0.10: Lc={Lc_high_sigma:.4f} m")
    print(f"σ=0.01: Lc={Lc_low_sigma:.4f} m")
    assert Lc_low_sigma > Lc_high_sigma, "σ↓ should increase Lc"
    print("✓ σ↓ ⇒ Lc↑ (correct)")

    # Ventilation: Cq↑ → σ_vent↓ (more gas = smaller σ)
    sigma_vent_low_Cq = cavity.compute_ventilation_sigma(Cq_in=0.001)
    sigma_vent_high_Cq = cavity.compute_ventilation_sigma(Cq_in=0.01)

    print(f"\nCq=0.001: σ_vent={sigma_vent_low_Cq:.4f}")
    print(f"Cq=0.010: σ_vent={sigma_vent_high_Cq:.4f}")
    assert sigma_vent_high_Cq < sigma_vent_low_Cq, "Cq↑ should decrease σ_vent"
    print("✓ Cq↑ ⇒ σ_vent↓ (correct)")

    print("\n✓ PASSED: Cavity monotonicity")
    return True


def test_cavitator_forces():
    """Test cavitator drag and lift."""
    print("\n" + "="*80)
    print("TEST: Cavitator Forces")
    print("="*80)

    # V=0 case (should not crash)
    F_drag = cavitator.cavitator_drag(V=0.0, Dn=0.05, Cx=0.82, alpha_eff=0.0)
    print(f"\nV=0: F_drag={F_drag:.1f} N (no crash)")
    assert F_drag >= 0, "Drag must be non-negative"

    # V↑ → F_drag↑ (monotonicity)
    F_drag_40 = cavitator.cavitator_drag(V=40.0, Dn=0.05, Cx=0.82, alpha_eff=0.0)
    F_drag_20 = cavitator.cavitator_drag(V=20.0, Dn=0.05, Cx=0.82, alpha_eff=0.0)

    print(f"V=20 m/s: F_drag={F_drag_20:.1f} N")
    print(f"V=40 m/s: F_drag={F_drag_40:.1f} N")
    assert F_drag_40 > F_drag_20, "V↑ should increase drag"
    assert F_drag_40 / F_drag_20 > 3.5, "Drag ∝ V² (4×)"  # Should be ~4× for 2× velocity
    print("✓ V↑ ⇒ F_drag↑ ∝ V² (correct)")

    # α_eff=0 → max drag; α_eff≠0 → drag reduces (cos²)
    F_drag_0 = cavitator.cavitator_drag(V=40.0, Dn=0.05, Cx=0.82, alpha_eff=0.0)
    F_drag_15deg = cavitator.cavitator_drag(V=40.0, Dn=0.05, Cx=0.82,
                                           alpha_eff=np.radians(15.0))

    print(f"\nα_eff=0°: F_drag={F_drag_0:.1f} N")
    print(f"α_eff=15°: F_drag={F_drag_15deg:.1f} N")
    assert F_drag_15deg < F_drag_0, "Angle should reduce drag (cos² factor)"
    cos2_15 = np.cos(np.radians(15.0))**2
    expected_ratio = cos2_15
    actual_ratio = F_drag_15deg / F_drag_0
    print(f"  Expected ratio: cos²(15°)={expected_ratio:.4f}")
    print(f"  Actual ratio: {actual_ratio:.4f}")
    assert abs(actual_ratio - expected_ratio) < 0.01, "cos² model mismatch"
    print("✓ α_eff effect ∝ cos²(α) (correct)")

    # Lift
    CL_α = cavitator.cavitator_lift_coefficient(sigma=0.02, cav_type="disk")
    print(f"\nDisk cavitator, σ=0.02: CL_α={CL_α:.4f} [1/rad]")
    assert CL_α > 0, "Lift coefficient must be positive"

    F_lift = cavitator.cavitator_lift(V=40.0, Dn=0.05, CL_alpha=CL_α, alpha_eff=0.035)
    print(f"V=40 m/s, α_eff=2°: F_lift={F_lift:.1f} N")
    assert F_lift > 0, "Lift must be positive for positive angle"

    print("\n✓ PASSED: Cavitator forces")
    return True


def test_fin_immersion():
    """Test fin immersion calculations."""
    print("\n" + "="*80)
    print("TEST: Fin Immersion")
    print("="*80)

    # Full immersion (cavity inside vehicle)
    wet_span = fins.fin_immersion_ratio(R_cavity=0.05, R_vehicle=0.10, fin_span=0.08)
    print(f"\nR_cav=0.05, R_veh=0.10, span=0.08: wet_span={wet_span:.4f} m")
    assert wet_span == 0.08, "Should be fully immersed"
    print("✓ Fully immersed case")

    # No immersion (cavity outside)
    wet_span = fins.fin_immersion_ratio(R_cavity=0.20, R_vehicle=0.10, fin_span=0.05)
    print(f"\nR_cav=0.20, R_veh=0.10, span=0.05: wet_span={wet_span:.4f} m")
    assert wet_span == 0.0, "Should be fully dry (cavity outside)"
    print("✓ Fully dry case")

    # Partial immersion
    wet_span = fins.fin_immersion_ratio(R_cavity=0.12, R_vehicle=0.10, fin_span=0.05)
    expected_wet = (0.10 + 0.05) - 0.12
    print(f"\nR_cav=0.12, R_veh=0.10, span=0.05: wet_span={wet_span:.4f} m")
    print(f"  Expected: {expected_wet:.4f} m")
    assert abs(wet_span - expected_wet) < 1e-6, "Partial immersion mismatch"
    print("✓ Partial immersion case")

    print("\n✓ PASSED: Fin immersion")
    return True


def test_fin_forces():
    """Test fin lift and drag."""
    print("\n" + "="*80)
    print("TEST: Fin Forces")
    print("="*80)

    # Fin with zero immersion → zero force
    result = fins.compute_fin_forces(
        V=40.0, fin_chord=0.04, fin_span=0.06,
        R_cavity=0.20, R_vehicle=0.075,  # Cavity outside fin
        alpha_aoa=0.0, delta_fin=0.0, azimuth=np.radians(90),
        x_fin=1.5, x_cg=1.0
    )
    print(f"\nZero immersion: F_L={result['F_L']:.1f} N, F_D={result['F_D']:.1f} N")
    assert result["F_L"] == 0.0 and result["F_D"] == 0.0, "Zero immersion should give zero force"
    print("✓ Zero immersion → zero force")

    # Fin with positive angle → positive lift
    result = fins.compute_fin_forces(
        V=40.0, fin_chord=0.04, fin_span=0.06,
        R_cavity=0.05, R_vehicle=0.075,  # Fully immersed
        alpha_aoa=np.radians(2.0), delta_fin=0.0, azimuth=np.radians(90),
        x_fin=1.5, x_cg=1.0
    )
    print(f"\nTop fin, α=2°: F_L={result['F_L']:.1f} N, immersion={result['immersion']:.2%}")
    assert result["F_L"] > 0, "Positive angle should give positive lift"
    print("✓ Positive angle → positive lift")

    # Vertical component (top fin sin(90°)=1)
    assert abs(result["F_L_z"] - result["F_L"]) < 1e-6, "Top fin: F_L_z = F_L"
    print("✓ Top fin vertical projection correct")

    # Bottom fin (azimuth=270°, sin(270°)=-1)
    # For symmetric pair: alpha_local_fin = alpha_world * sin(azim)
    # Bottom: alpha_local = 2° * (-1) = -2° → negative CL, negative F_L
    # F_L_z = F_L_fin * sin(azim) = (negative) * (-1) = positive
    # This is correct per wing theory (both fins lift in same world z direction for pitch)
    result_bottom = fins.compute_fin_forces(
        V=40.0, fin_chord=0.04, fin_span=0.06,
        R_cavity=0.05, R_vehicle=0.075,
        alpha_aoa=np.radians(2.0), delta_fin=0.0, azimuth=np.radians(270),
        x_fin=1.5, x_cg=1.0
    )
    print(f"\nBottom fin, α=2°: F_L={result_bottom['F_L']:.1f} N, F_L_z={result_bottom['F_L_z']:.1f} N")
    # Bottom fin sees inverted angle, but projection also inverts, so F_L_z same sign as top
    assert result_bottom["F_L_z"] > 0, "Bottom fin F_L_z should be positive (symmetric response)"
    print("✓ Bottom fin symmetric response (correct)")

    print("\n✓ PASSED: Fin forces")
    return True


def test_planing():
    """Test transom planing force."""
    print("\n" + "="*80)
    print("TEST: Planing Force")
    print("="*80)

    # h_imm=0 → F_pz=0 (no planing)
    F_pz = planing.planing_force_dzielski_kurdila(V=40.0, R_v=0.075, R_c=0.1,
                                                  alpha_p=0.01, h_immersion=0.0)
    print(f"\nh_imm=0: F_pz={F_pz:.1f} N")
    assert F_pz == 0.0, "No immersion should give zero force"
    print("✓ h_imm=0 → F_pz=0")

    # Positive h_imm → positive F_pz
    F_pz_positive = planing.planing_force_dzielski_kurdila(V=40.0, R_v=0.075, R_c=0.1,
                                                           alpha_p=0.01, h_immersion=0.01)
    print(f"h_imm=0.01 m: F_pz={F_pz_positive:.1f} N")
    assert F_pz_positive > 0, "Positive immersion should give positive force"
    print("✓ h_imm>0 → F_pz>0")

    # V↑ → F_pz↑
    F_pz_40 = planing.planing_force_dzielski_kurdila(V=40.0, R_v=0.075, R_c=0.1,
                                                     alpha_p=0.01, h_immersion=0.02)
    F_pz_20 = planing.planing_force_dzielski_kurdila(V=20.0, R_v=0.075, R_c=0.1,
                                                     alpha_p=0.01, h_immersion=0.02)
    print(f"\nV=20 m/s: F_pz={F_pz_20:.1f} N")
    print(f"V=40 m/s: F_pz={F_pz_40:.1f} N")
    assert F_pz_40 > F_pz_20, "V↑ should increase planing force"
    print("✓ V↑ ⇒ F_pz↑ ∝ V²")

    print("\n✓ PASSED: Planing force")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print(" HYDRODYNAMICS MODULE — UNIT TESTS")
    print("="*80)

    tests = [
        test_cavity_singularities,
        test_cavity_monotonicity,
        test_cavitator_forces,
        test_fin_immersion,
        test_fin_forces,
        test_planing,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            results.append((test_func.__name__, False))
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")
    print("="*80)

    return all(r for _, r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
