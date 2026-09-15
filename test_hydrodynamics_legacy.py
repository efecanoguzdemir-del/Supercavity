"""
================================================================================
 Hydrodynamics Module — Legacy Case 1 Regression Test
================================================================================
Referans GUI v10 nominal scenario'ya karşı doğrulama.

Test amacı: ±%5 tolerans içinde F_drag, Fz, My eşleşmesi.

Expected values (from GUI v10):
  - F_drag = 5764 N (±%5 = 5476-6052 N)
  - Fz = -2109 N (±%5 = -2214 to -2004 N)
  - My = 3344 N·m (±%5 = 3177-3511 N·m)

Burada: steady-state (V constant), kanat devre dışı, planing devre dışı
Test yalnızca kavitatör + gövde drag/lift'i doğrular.
================================================================================
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.hydrodynamics import (
    constants,
    cavity,
    cavitator,
    LEGACY_CASE_1, LEGACY_CASE_1_TOL,
)


def test_legacy_case_1():
    """
    Test Legacy Case 1 (nominal supercavitation scenario).

    Validates: F_drag, Fz, My within ±5% of reference values.
    """

    print("=" * 80)
    print(" LEGACY CASE 1 REGRESSION TEST")
    print("=" * 80)
    print()

    # Extract parameters
    case = LEGACY_CASE_1
    V = case["V"]
    alpha_aoa_deg = case["alpha_aoa_deg"]
    delta_cav_deg = case["delta_cav_deg"]
    depth = case["depth"]
    Dn = case["diam_cav"]
    mass = case["mass"]
    L_veh = case["veh_len"]
    D_veh = case["veh_diam"]
    x_cg = case["x_cg"]

    # Constants
    p_inf = constants.compute_ambient_pressure(depth)
    sigma_vapor = constants.compute_sigma_vapor(p_inf, V)

    # Nominal (no ventilation)
    sigma_eff = sigma_vapor
    Cx0 = constants.CX0_DISK  # disk cavitator
    k_g = case["k_g_cavity"]

    print("INPUT PARAMETERS:")
    print(f"  V = {V} m/s")
    print(f"  α (AoA) = {alpha_aoa_deg}°")
    print(f"  δ_c (cavitator pitch) = {delta_cav_deg}°")
    print(f"  depth = {depth} m")
    print(f"  Dn (cavitator diameter) = {Dn} m")
    print(f"  L_vehicle = {L_veh} m")
    print(f"  D_vehicle = {D_veh} m")
    print(f"  x_cg = {x_cg} m")
    print()

    print("AMBIENT CONDITIONS:")
    print(f"  p_∞ = {p_inf:.0f} Pa")
    print(f"  σ_vapor = {sigma_vapor:.4f}")
    print(f"  σ_eff = {sigma_eff:.4f}")
    print()

    # === CAVITY GEOMETRY ===
    Lc, Dc, Cx = cavity.cavity_geometry(sigma_eff, Dn, model=case["geom_model"],
                                        Cx0=Cx0, k_g_val=k_g,
                                        K_Dc_val=case["K_Dc_factor"],
                                        K_Lc_val=case["K_Lc_factor"])

    print("CAVITY GEOMETRY:")
    print(f"  Model: {case['geom_model']}")
    print(f"  Lc (length) = {Lc:.4f} m")
    print(f"  Dc (diameter) = {Dc:.4f} m")
    print(f"  Lc/Dc = {Lc/max(Dc, 1e-6):.1f}")
    print(f"  Cx (drag coefficient) = {Cx:.4f}")
    print()

    # === CAVITATOR FORCES ===
    alpha_aoa = np.radians(alpha_aoa_deg)
    delta_cav = np.radians(delta_cav_deg)

    forces_cav = cavitator.compute_cavitator_forces(
        V, Dn, sigma_eff,
        alpha_aoa, delta_cav, x_cg,
        cav_type="disk", Cx0=Cx0, k_g=k_g
    )

    F_drag_cav = forces_cav["F_drag"]
    F_lift_cav = forces_cav["F_lift"]
    M_cav = forces_cav["M_y"]

    print("CAVITATOR FORCES:")
    print(f"  F_drag = {F_drag_cav:.1f} N")
    print(f"  F_lift (raw) = {F_lift_cav:.1f} N")
    print(f"  M_y = {M_cav:.1f} N·m")
    print()

    # === GÖVDE (BODY) EFFECTS ===
    # Nominal (no ventilation, natural cavitation only)
    q = 0.5 * constants.RHO * V * V
    Sn = np.pi * (Dn / 2.0)**2
    Sb = np.pi * (D_veh / 2.0)**2
    Sw = np.pi * D_veh * L_veh

    # Tüm kavitatör drag'ı toplam drag (steady-state: kanat yok, planing yok)
    F_drag_total = F_drag_cav  # Plus skin friction (tipik çok küçük)

    # Body lift (if α != 0)
    if abs(alpha_aoa) > 1e-6:
        A_max = Sb
        wet_frac = 1.0  # Fully wet, no cavity yet (nominal doğal σ)
        CL_α_body = case["CL_alpha_body"]
        F_body_lift = CL_α_body * q * A_max * alpha_aoa * wet_frac
    else:
        F_body_lift = 0.0

    print("BODY EFFECTS:")
    print(f"  A_max (crosssection) = {A_max:.4f} m²")
    print(f"  CL_α (body) = {case['CL_alpha_body']:.2f}")
    print(f"  F_body_lift = {F_body_lift:.1f} N")
    print()

    # === BUOYANCY + WEIGHT ===
    # Nominal (araç tam suya batık, tüm gövde hacmi dış basıncı hisseder)
    if case.get("L_taper", 0.3) > 1e-6:
        R_n = Dn / 2.0
        R_v = D_veh / 2.0
        L_taper = case.get("L_taper", 0.3)
        V_taper = (np.pi * L_taper / 3.0) * (R_n**2 + R_n*R_v + R_v**2)
        V_cyl = np.pi * R_v**2 * (L_veh - L_taper)
        V_total = V_taper + V_cyl
    else:
        V_total = np.pi * (D_veh/2.0)**2 * L_veh

    F_buoy = constants.RHO * constants.G * V_total
    F_grav = -mass * constants.G

    print("BUOYANCY & WEIGHT:")
    print(f"  V_body (volume) = {V_total:.4f} m³")
    print(f"  F_buoy = {F_buoy:.1f} N")
    print(f"  F_gravity = {F_grav:.1f} N")
    print()

    # === TOTAL FORCES ===
    # F_z = F_planing + F_cav_z + F_body_lift + F_fin + F_buoy + F_grav
    #     = 0 + F_lift_cav + F_body_lift + 0 + F_buoy + F_grav
    # Nominal: F_cav_z is lift (positive up), body_lift can be up/down

    F_cav_z = F_lift_cav  # Positive = upward (but in body-fixed z down)
    # In body-fixed convention, upward lift contributes negatively to F_z
    # But let's compute in standard aeronautical form first:
    F_z_aero = F_lift_cav + F_body_lift + F_buoy + F_grav

    # For moment: everything about x_cg
    M_total = M_cav  # Only cavitator moment in nominal case

    print("TOTAL FORCES & MOMENTS:")
    print(f"  F_drag = {F_drag_total:.1f} N")
    print(f"  F_z (aero convention, up positive) = {F_z_aero:.1f} N")
    print(f"  M_y (pitch) = {M_total:.1f} N·m")
    print()

    # === COMPARISON WITH LEGACY ===
    exp_F_drag = case["expected_F_drag_N"]
    exp_Fz = case["expected_Fz_N"]  # Note: -2109 suggests down-positive convention
    exp_My = case["expected_My_Nm"]

    tol = LEGACY_CASE_1_TOL

    # Convert to body-fixed convention if needed (aeronautical Fz → body Fz)
    # Body-fixed z (down) vs aeronautical (up): multiply by -1
    F_z_body = -F_z_aero

    # Check against tolerance
    F_drag_ok = abs(F_drag_total - exp_F_drag) < exp_F_drag * tol
    Fz_ok = abs(F_z_body - exp_Fz) < abs(exp_Fz) * tol
    My_ok = abs(M_total - exp_My) < exp_My * tol

    print("REGRESSION CHECK (±%5):")
    print()
    print(f"  F_drag:")
    print(f"    Computed: {F_drag_total:8.1f} N")
    print(f"    Expected: {exp_F_drag:8.1f} N (±{exp_F_drag*tol:6.0f})")
    print(f"    Range: [{exp_F_drag*(1-tol):8.0f}, {exp_F_drag*(1+tol):8.0f}]")
    print(f"    Status: {'✓ PASS' if F_drag_ok else '✗ FAIL'}")
    print()

    print(f"  Fz:")
    print(f"    Computed: {F_z_body:8.1f} N")
    print(f"    Expected: {exp_Fz:8.1f} N (±{abs(exp_Fz)*tol:6.0f})")
    print(f"    Range: [{exp_Fz*(1-tol):8.0f}, {exp_Fz*(1+tol):8.0f}]")
    print(f"    Status: {'✓ PASS' if Fz_ok else '✗ FAIL'}")
    print()

    print(f"  My:")
    print(f"    Computed: {M_total:8.1f} N·m")
    print(f"    Expected: {exp_My:8.1f} N·m (±{exp_My*tol:6.0f})")
    print(f"    Range: [{exp_My*(1-tol):8.0f}, {exp_My*(1+tol):8.0f}]")
    print(f"    Status: {'✓ PASS' if My_ok else '✗ FAIL'}")
    print()

    # Overall pass/fail
    all_pass = F_drag_ok and Fz_ok and My_ok

    print("=" * 80)
    if all_pass:
        print(" OVERALL: ✓ ALL TESTS PASSED")
    else:
        print(" OVERALL: ✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_pass


if __name__ == "__main__":
    success = test_legacy_case_1()
    sys.exit(0 if success else 1)
