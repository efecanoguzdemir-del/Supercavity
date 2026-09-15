"""
================================================================================
 Supercavitation — Cavitator Forces (Drag and Lift)
================================================================================
Kavitatör tarafından üretilen kuvvetler (normal force, drag, lift).

May 1975 (Birkhoff slender body) ve Logvinovich klasik teori.
Disk ve konik kavitatörler için viskoz düzeltme.

İşaret Konvansiyonu:
  - Fx: forward drag (positive = retarding, akışa karşı)
  - Fz: vertical (positive = downward in body frame, lift up = negative Fz)
  - My: pitching moment (positive = nose-up, sağ-el kuralı y-ekseni etraf)
================================================================================
"""

import numpy as np
from .constants import (
    V_MIN, SIGMA_MIN,
    CX0_DISK, K_SLENDER_DEFAULT, RHO
)


def cavitator_normal_force(V: float, Dn: float, Cx: float, alpha_eff: float) -> float:
    """
    Compute normal force on cavitator (disk or conical).

    F_n = q · Sn · Cx · cos²(α_eff)

    where:
      q = ½ρV²  — dynamic pressure
      Sn = π(Dn/2)²  — cavitator frontal area
      Cx  — drag coefficient (including (1+σ) factor and angle correction)
      α_eff  — effective cavitator angle (α_aoa + δ_cav)

    Fiziksel anlam:
      - Cx normal force / dynamic pressure / area — cavitator-aligned force
      - cos²(α_eff) — yaw effect (eğik kavitatör drag'ı azaltır)

    Args:
        V: Velocity [m/s]
        Dn: Cavitator diameter [m]
        Cx: Drag coefficient (from cavity_geometry, = Cx0·(1+σ))
        alpha_eff: Effective cavitator angle (α + δ_c) [rad]

    Returns:
        F_n: Normal force [N] (always positive or zero)

    Sign Convention:
      - F_n > 0 — normal force magnitude
      - Applied to forward direction; cos²(α) gives drag component
    """

    V_safe = max(V, V_MIN)
    q = 0.5 * RHO * V_safe * V_safe
    Sn = np.pi * (Dn / 2.0)**2

    cos_a = np.cos(alpha_eff)
    F_n = q * Sn * Cx * cos_a * cos_a

    return max(F_n, 0.0)


def cavitator_drag(V: float, Dn: float, Cx: float, alpha_eff: float) -> float:
    """
    Compute cavitator drag force (horizontal component).

    F_drag = q · Sn · Cx · cos²(α_eff)

    This is the horizontal (forward-opposing) component of normal force.
    For small α_eff, ≈ q·Sn·Cx.

    Args:
        V: Velocity [m/s]
        Dn: Cavitator diameter [m]
        Cx: Drag coefficient
        alpha_eff: Effective cavitator angle [rad]

    Returns:
        F_drag: Drag force [N] (positive = opposing motion)

    Sign Convention:
      - Positive = slowing force (retarding)
      - Applied in -x direction (body-fixed)
    """

    F_n = cavitator_normal_force(V, Dn, Cx, alpha_eff)
    # Normal force fully aligned with flow for cos²(α) already included
    return F_n  # = q·Sn·Cx·cos²(α_eff)


def cavitator_lift_coefficient(sigma: float, cav_type: str = "disk",
                                cone_apex_rad: float = None,
                                K_slender: float = K_SLENDER_DEFAULT,
                                K_cone_lift: float = 0.0) -> float:
    """
    Compute cavitator lift coefficient (May-Birkhoff / Logvinovich hybrid).

    Disk kavitatör (β=180°):
      CL_α = π/(1+σ)  — flat-plate theory

    Konik kavitatör (β<180°):
      CL_α = (π/(1+σ))·sin²(β/2) + K_slender·cos²(β/2)  — hybrid
      + lateral surface area correction

    Fiziksel anlam:
      - Disk (flat): Logvinovich-May klasik (π/(1+σ))
      - Cone (slender): Munk slender body (K_slender·2.0 typically)
      - Interpolation ensures smooth transition disk ↔ cone
      - Lateral factor: sivri konik yan yüzey momentum değişim etkisi

    Args:
        sigma: Cavitation number (σ = 2(p∞−pc)/(ρV²))
        cav_type: "disk" or "cone"
        cone_apex_rad: Cone apex angle [rad] (full angle, β) — if None assumes disk
        K_slender: Slender body cross-flow lift factor (0-2, default 1.0)
        K_cone_lift: Lateral surface effect factor (0-2, default 0.0)

    Returns:
        CL_α: Cavitator lift coefficient per unit angle [1/rad]
               = dF_L / (dα · q · Sn)

    Sign Convention:
      - CL_α > 0 always (lift increases with angle magnitude)
      - CL_α [1/rad] — multiply by α [rad] to get force coefficient
      - Lift force direction: orthogonal to cavitator surface, upward for +α
    """

    sigma_safe = max(sigma, SIGMA_MIN)

    # Base disk coefficient (Logvinovich-May 1975)
    CL_base = np.pi / (1.0 + sigma_safe)  # ≈ 3.0 for σ→0, ≈ 1.6 for σ=1

    if cav_type == "cone" and cone_apex_rad is not None:
        # Hybrid: disk + slender body interpolation
        beta = cone_apex_rad
        beta_half = beta / 2.0

        sin2_b = np.sin(beta_half)**2
        cos2_b = np.cos(beta_half)**2

        # Logvinovich-Birkhoff hybrid
        CL_hybrid = (CL_base * sin2_b + K_slender * cos2_b)

        # Lateral surface area correction (cone yan yüzey)
        # K_lateral = 1 + (K_cone_lift) · (1/sin(β/2) − 1)
        if sin2_b > 0.0:
            K_lateral = 1.0 + K_cone_lift * (1.0 / np.sqrt(max(sin2_b, 0.01)) - 1.0)
        else:
            K_lateral = 1.0

        CL_α = CL_hybrid * K_lateral
    else:
        # Disk (β=180°, sin²(β/2)=1, cos²(β/2)=0)
        CL_α = CL_base

    return max(CL_α, 0.0)  # Must be non-negative


def cavitator_lift(V: float, Dn: float, CL_alpha: float, alpha_eff: float) -> float:
    """
    Compute cavitator lift force (vertical component due to pitch angle).

    F_L = q · Sn · CL_α · α_eff

    Fiziksel anlam:
      - CL_α [1/rad] × α_eff [rad] — linear lift coefficient
      - Positive α_eff (nose-up) → positive F_L (upward lift)
      - Applied at cavitator location (x=0, forward)

    Args:
        V: Velocity [m/s]
        Dn: Cavitator diameter [m]
        CL_alpha: Lift coefficient [1/rad] (from cavitator_lift_coefficient)
        alpha_eff: Effective cavitator angle [rad]

    Returns:
        F_L: Lift force [N]
             Positive = upward (reduces weight)
             Negative = downward (in body-fixed z convention)

    Sign Convention:
      - Body-fixed z (positive = down)
      - Aeronautical lift (positive = up) → F_L > 0 gives negative F_z (up)
      - But this function returns raw lift (positive up, opposite to z-axis)
      - Caller converts to F_z = -F_L for compatibility

      Actually, using standard convention:
      - F_L > 0: upward lift (reduces gravitational load)
      - In body-fixed z (down): F_L > 0 means negative contribution to F_z_net

      For consistency with GUI reference (Fz = total vertical in body frame):
      - Positive F_L (physical upward) → negative contribution to F_z
      - Converter in dynamics: My moment includes (x_cg - x_force)·F_L_signed
    """

    V_safe = max(V, V_MIN)
    q = 0.5 * RHO * V_safe * V_safe
    Sn = np.pi * (Dn / 2.0)**2

    F_L = q * Sn * CL_alpha * alpha_eff

    return F_L


def cavitator_moment(F_L: float, x_cg: float) -> float:
    """
    Compute pitching moment contribution from cavitator lift.

    M_y = (x_cg - x_cav) · F_L

    where x_cav = 0 (cavitator at nose).

    Moment arm sign convention (right-hand rule about y-axis):
      - Cavitator at x=0, c.g. at x=x_cg > 0 (aft)
      - Upward lift (+F_L) at x=0 < x_cg (before c.g.)
      - Moment = (x_cg - 0) · (+F_L) > 0  — nose-up pitch ✓

    Args:
        F_L: Cavitator lift force [N] (positive = upward)
        x_cg: Center of gravity location from nose [m]

    Returns:
        M_y: Pitching moment [N·m]
             Positive = nose-up rotation

    Sign Convention:
      - Consistent with right-hand rule (fingers curl from +x to +F_z, thumb = y)
      - M_y > 0: burun yukarı (nose-up pitch)
      - Applied about y-axis (starboard direction)
    """

    # Cavitator at nose: x_cav = 0
    moment_arm = x_cg  # (x_cg - 0)
    M_y = moment_arm * F_L

    return M_y


# ============================================================================
# CONICAL CAVITATOR DRAG COEFFICIENT (May 1975, Logvinovich-Serebryakov)
# ============================================================================

def conical_cavitator_Cx(Cx_disk: float, cone_apex_rad: float,
                        K_v: float = 0.35) -> float:
    """
    Compute drag coefficient for conical cavitator (viscous correction).

    Sivri konik kavitatör drag'ı teorik slender-body değerinden daha yüksek
    (viskoz boundary layer, separation, cavity wall friction).

    Model: Cx_cone = Cx_disk · [sin²(β/2) + K_v·cos²(β/2)]

    where β = cone full apex angle.

    K_v (viscous correction factor):
      = 0.0  : Pure theory (underestimates ≈50% vs. experiment)
      = 0.15 : May 1975 experimental data fit
      = 0.35 : CFD-compatible, high-Re viscous correction (default)
      = 0.50 : High viscous effect

    Sınırlar:
      β = 180° (disk): Cx_cone = Cx_disk·1 = Cx_disk ✓
      β = 90°  (right cone): Cx_cone = Cx_disk·(0.5 + 0.5·K_v)
      β = 60°  (sharp): Cx_cone = Cx_disk·(0.25 + 0.75·K_v)
      β = 0°   (infinitely sharp): Cx_cone → Cx_disk·K_v (minimum)

    Args:
        Cx_disk: Reference disk drag coefficient (≈0.82)
        cone_apex_rad: Cone apex angle [rad] (full angle β, not half)
        K_v: Viscous correction factor (0-0.5, default 0.35)

    Returns:
        Cx_cone: Conical cavitator drag coefficient

    Sign Convention:
      - Cx_cone > 0 always (drag opposes flow)
      - Typically Cx_disk < Cx_cone for real flow (viscosity)
    """

    beta_half = cone_apex_rad / 2.0
    sin2_b = np.sin(beta_half)**2
    cos2_b = np.cos(beta_half)**2

    Cx_cone = Cx_disk * (sin2_b + K_v * cos2_b)

    return max(Cx_cone, 0.0)


# ============================================================================
# INTEGRATION WITH CAVITY.PY
# ============================================================================

def compute_cavitator_forces(V: float, Dn: float, sigma: float,
                            alpha_aoa: float, delta_cav: float, x_cg: float,
                            cav_type: str = "disk", cone_apex_rad: float = None,
                            K_cone_lift: float = 0.0,
                            Cx0: float = None, k_g: float = None) -> dict:
    """
    Comprehensive cavitator force computation (convenience wrapper).

    Combines cavity geometry, drag coefficient, and lift calculations.

    Args:
        V: Velocity [m/s]
        Dn: Cavitator diameter [m]
        sigma: Cavitation number (efektif, havalandırma dahil)
        alpha_aoa: Angle of attack [rad]
        delta_cav: Cavitator pitch [rad]
        x_cg: Center of gravity location [m]
        cav_type: "disk" or "cone"
        cone_apex_rad: Cone apex angle [rad] (if cone)
        K_cone_lift: Lateral surface effect (0-2)
        Cx0: Base drag coefficient — if None use CX0_DISK
        k_g: Reichardt constant — if None use default

    Returns:
        dict with keys:
          - "F_drag": Cavitator drag [N]
          - "F_lift": Cavitator lift [N]
          - "M_y": Cavitator pitching moment [N·m]
          - "Cx": Final drag coefficient used
          - "CL_alpha": Lift coefficient [1/rad]
          - "alpha_eff": Effective angle [rad]

    Sign Convention:
      - F_drag > 0: retarding force
      - F_lift > 0: upward force (positive in aeronautical convention)
      - M_y > 0: nose-up pitch
    """

    from .cavity import cavity_geometry  # Avoid circular import

    if Cx0 is None:
        Cx0 = CX0_DISK
    if k_g is None:
        from .constants import K_G_CAVITY_DEFAULT
        k_g = K_G_CAVITY_DEFAULT

    # Effective cavitator angle
    alpha_eff = alpha_aoa + delta_cav

    # Cavity geometry (Cx includes (1+σ) factor)
    Lc, Dc, Cx = cavity_geometry(sigma, Dn, model="savchenko",
                                 Cx0=Cx0, k_g_val=k_g)

    # Drag
    F_drag = cavitator_drag(V, Dn, Cx, alpha_eff)

    # Lift coefficient (hybrid disk/cone)
    CL_α = cavitator_lift_coefficient(sigma, cav_type, cone_apex_rad,
                                      K_cone_lift=K_cone_lift)

    # Lift force
    F_lift = cavitator_lift(V, Dn, CL_α, alpha_eff)

    # Moment
    M_y = cavitator_moment(F_lift, x_cg)

    return {
        "F_drag": F_drag,
        "F_lift": F_lift,
        "M_y": M_y,
        "Cx": Cx,
        "CL_alpha": CL_α,
        "alpha_eff": alpha_eff,
    }
