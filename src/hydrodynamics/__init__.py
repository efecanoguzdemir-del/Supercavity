"""
================================================================================
 Supercavitation Hydrodynamics Module
================================================================================
Modular hydrodynamics library for supercavitating vehicles.

Komponenter:
  - constants: SI units, physical constants, reference cases
  - cavity: Cavity geometry (6 models), ventilation, pressure dynamics
  - cavitator: Normal force, drag, lift (disk/conical cavitators)
  - fins: Control surface forces (NACA 16-009, finite wing theory)
  - planing: Transom planing force (Dzielski-Kurdila 2003)

Sign Conventions (Body-Fixed, Aeronautical):
  x: Forward (nose direction)
  y: Starboard (right)
  z: Down (gravity)

  α: Angle of attack (positive = nose-up relative to flow)
  δ: Control deflection (positive = nose-up deflection)
  β: Cone apex angle

  DİKKAT — bu paketin fonksiyonları legacy konvansiyonunu kullanır:
  Drag: pozitif büyüklük (geri yönlü)
  Lift / dikey kuvvet (F_lift, F_L_z, F_pz, F_planing, F_body_lift, F_buoy):
        YUKARI POZİTİF (gövde z-aşağı DEĞİL)
  Moment: M_y = (x_cg − x_force)·F_up, pozitif = burun yukarı; x burundan ölçülür
  Gövde çerçevesine (x-ileri, y-sancak, z-aşağı, CG orijinli) dönüşüm TEK YERDE:
  src/dynamics/model.py (F_z_body = −F_up, x_body = x_cg − x_nose).

Usage Example:
  >>> from src.hydrodynamics import constants, cavity, cavitator, fins, planing
  >>> sigma = 0.02  # Cavitation number
  >>> Lc, Dc, Cx = cavity.cavity_geometry(sigma, Dn=0.05, model="savchenko")
  >>> forces = cavitator.compute_cavitator_forces(V=40, Dn=0.05, sigma=sigma,
  ...                                             alpha_aoa=-0.017, delta_cav=0.035,
  ...                                             x_cg=1.0)
  >>> print(f"F_drag = {forces['F_drag']:.0f} N")
================================================================================
"""

# Public API — import main functions for convenient access
from .constants import (
    # SI constants
    RHO, G, P_ATM, P_VAP,
    # Safety guards
    SIGMA_MIN, V_MIN, CAVITY_MIN,
    # Cavitator coefficients
    CX0_DISK, K_G_CAVITY_DEFAULT, K_V_CONE_DEFAULT, K_SLENDER_DEFAULT,
    # Fin coefficients
    FIN_CL_ALPHA_2D, FIN_CD0, FIN_OSWALD_E, FIN_STALL_ANGLE,
    # Ventilation
    A_V_DEFAULT, TAU_PC, K_LEAK,
    # Body coefficients
    CL_ALPHA_BODY_DEFAULT, CDC_BODY_DEFAULT,
    # Calibration factors
    K_DC_DEFAULT, K_LC_DEFAULT, K_DEV_DEFAULT,
    # Legacy test case
    LEGACY_CASE_1, LEGACY_CASE_1_TOL,
    # Helpers
    compute_sigma_vapor, compute_ambient_pressure,
)

from .cavity import (
    cavity_geometry,
    compute_ventilation_sigma,
    update_cavity_pressure,
    compute_hybrid_sigma,
    compute_pc_target,
    cavity_axis_offset,
)

from .cavitator import (
    cavitator_normal_force,
    cavitator_drag,
    cavitator_lift_coefficient,
    cavitator_lift,
    cavitator_moment,
    conical_cavitator_Cx,
    compute_cavitator_forces,
)

from .fins import (
    fin_3d_lift_slope,
    fin_drag_coefficient,
    fin_immersion_ratio,
    fin_aoa_effective,
    fin_lift_and_drag,
    fin_vertical_component,
    fin_moment_contribution,
    compute_fin_forces,
)

from .planing import (
    planing_force_dzielski_kurdila,
    planing_angle_from_cavity_slope,
    planing_moment_contribution,
    planing_immersion_depth,
    compute_planing_force,
)

from .body import (
    vehicle_radius,
    body_surface_area,
    body_volume,
    cavity_opening_length,
    cavity_radius_logvinovich,
    cavity_radius_at_section,
    wetted_arc,
    compute_body_forces,
)

__all__ = [
    # Constants
    "RHO", "G", "P_ATM", "P_VAP",
    "SIGMA_MIN", "V_MIN", "CAVITY_MIN",
    "CX0_DISK", "K_G_CAVITY_DEFAULT", "K_V_CONE_DEFAULT", "K_SLENDER_DEFAULT",
    "FIN_CL_ALPHA_2D", "FIN_CD0", "FIN_OSWALD_E", "FIN_STALL_ANGLE",
    "A_V_DEFAULT", "TAU_PC", "K_LEAK",
    "CL_ALPHA_BODY_DEFAULT", "CDC_BODY_DEFAULT",
    "K_DC_DEFAULT", "K_LC_DEFAULT", "K_DEV_DEFAULT",
    "LEGACY_CASE_1", "LEGACY_CASE_1_TOL",
    "compute_sigma_vapor", "compute_ambient_pressure",
    # Cavity
    "cavity_geometry",
    "compute_ventilation_sigma",
    "update_cavity_pressure",
    "compute_hybrid_sigma",
    "compute_pc_target",
    "cavity_axis_offset",
    # Cavitator
    "cavitator_normal_force",
    "cavitator_drag",
    "cavitator_lift_coefficient",
    "cavitator_lift",
    "cavitator_moment",
    "conical_cavitator_Cx",
    "compute_cavitator_forces",
    # Fins
    "fin_3d_lift_slope",
    "fin_drag_coefficient",
    "fin_immersion_ratio",
    "fin_aoa_effective",
    "fin_lift_and_drag",
    "fin_vertical_component",
    "fin_moment_contribution",
    "compute_fin_forces",
    # Planing
    "planing_force_dzielski_kurdila",
    "planing_angle_from_cavity_slope",
    "planing_moment_contribution",
    "planing_immersion_depth",
    "compute_planing_force",
    # Body wetting
    "vehicle_radius",
    "body_surface_area",
    "body_volume",
    "cavity_opening_length",
    "cavity_radius_logvinovich",
    "cavity_radius_at_section",
    "wetted_arc",
    "compute_body_forces",
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Supercavitation Research Group"
__license__ = "MIT"
