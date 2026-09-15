"""
Kinematik — kuvvetten bağımsız 6-DOF katı cisim kinematiği.

Konvansiyonlar:
  - Gövde ekseni: x-ileri, y-sancak, z-aşağı. Orijin kütle merkezinde (CG).
  - Eylemsiz çerçeve: NED (Z pozitif = derinlik).
  - Euler ZYX (phi, theta, psi) [rad].
  - Konum türevi:  [Xdot, Ydot, Zdot] = R(phi,theta,psi) @ [u, v, w]
  - Euler türevi:  [phidot, thetadot, psidot] = T(phi,theta) @ [p, q, r]

Burun-referanslı konum dönüşümü:
  x_body = x_cg - x_from_nose   (burundaki nokta +x yönünde, CG'nin önünde)
"""

import warnings

import numpy as np

from src.hydrodynamics.constants import G
from .state import SL_NU_LIN, SL_NU_ANG, IDX_PHI, IDX_THETA, IDX_PSI

# Case 1 CG konumu (burundan geriye) [m]
X_CG_DEFAULT = 2.4

# cos(theta) alt sınırı (T matrisinde sıfıra bölme koruması)
COS_THETA_MIN = 1e-6

# |theta| uyarı eşiği
THETA_WARN_LIMIT = np.radians(80.0)

# V küçükken akış açıları için koruma [m/s]
V_FLOW_MIN = 1e-6

_theta_warned = False


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def nose_to_body_x(x_from_nose, x_cg=X_CG_DEFAULT):
    """Burundan geriye ölçülen x konumunu CG-orijinli gövde x'ine çevirir.

    x_body = x_cg - x_from_nose. Burun (x_from_nose=0) -> x_body = +x_cg.
    Skaler veya numpy dizisi kabul eder.
    """
    if np.ndim(x_from_nose):
        return x_cg - np.asarray(x_from_nose, dtype=float)
    return float(x_cg - x_from_nose)


def nose_to_body(r_from_nose, x_cg=X_CG_DEFAULT):
    """Burun-referanslı nokta [x_from_nose, y, z] -> CG-orijinli gövde vektörü.

    Yalnızca x ekseni dönüşür (y, z değişmez; burun ekseni gövde ekseniyle çakışık
    varsayılır).
    """
    r = np.asarray(r_from_nose, dtype=float).copy()
    r[..., 0] = x_cg - r[..., 0]
    return r


def skew(a):
    """S(a): S(a) @ b = a x b."""
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


# ---------------------------------------------------------------------------
# Dönüşüm matrisleri
# ---------------------------------------------------------------------------
def rotation_body_to_ned(phi, theta, psi):
    """R_b^n (ZYX): v_ned = R @ v_body."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi],
    ])


def _safe_cos(theta):
    c = np.cos(theta)
    if abs(c) < COS_THETA_MIN:
        c = COS_THETA_MIN if c >= 0.0 else -COS_THETA_MIN
    return c


def euler_rate_matrix(phi, theta):
    """T(phi, theta): eta_dot = T @ omega.

    |cos(theta)| < COS_THETA_MIN ise cos(theta) işaretini koruyarak bu değere
    kırpılır (gimbal lock civarında sonlu ama büyük değerler üretir, NaN/inf yok).
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = _safe_cos(theta)
    sth = np.sin(theta)
    tth = sth / cth
    return np.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi, -sphi],
        [0.0, sphi / cth, cphi / cth],
    ])


# ---------------------------------------------------------------------------
# Uyarı mekanizması
# ---------------------------------------------------------------------------
def check_pitch_limit(theta):
    """|theta| > 80° ise (süreç başına) yalnızca bir kez RuntimeWarning verir.

    Dönüş: eşik aşıldıysa True.
    """
    global _theta_warned
    exceeded = abs(theta) > THETA_WARN_LIMIT
    if exceeded and not _theta_warned:
        _theta_warned = True
        warnings.warn(
            f"|theta| = {np.degrees(abs(theta)):.1f} deg > 80 deg: Euler ZYX "
            "gimbal lock bölgesine yaklaşılıyor (T matrisi kötü koşullu).",
            RuntimeWarning, stacklevel=2)
    return exceeded


def reset_pitch_warning():
    """Tek seferlik theta uyarısını yeniden silahlandırır (yeni koşu için)."""
    global _theta_warned
    _theta_warned = False


# ---------------------------------------------------------------------------
# Türevler
# ---------------------------------------------------------------------------
def kinematics_derivative(x):
    """Durum vektöründen (15 elemanlı) kinematik türevler.

    Dönüş: (pos_dot [Xdot,Ydot,Zdot] NED, euler_dot [phidot,thetadot,psidot]).
    """
    phi, theta, psi = x[IDX_PHI], x[IDX_THETA], x[IDX_PSI]
    check_pitch_limit(theta)
    v_lin = np.asarray(x[SL_NU_LIN], dtype=float)
    omega = np.asarray(x[SL_NU_ANG], dtype=float)
    pos_dot = rotation_body_to_ned(phi, theta, psi) @ v_lin
    euler_dot = euler_rate_matrix(phi, theta) @ omega
    return pos_dot, euler_dot


def gravity_body(m, phi, theta):
    """Ağırlığın gövde çerçevesindeki kuvveti [Fx, Fy, Fz] (CG'de, moment yok).

    F_b = R^T @ [0, 0, m g] = m g [-sin(theta), cos(theta) sin(phi), cos(theta) cos(phi)]
    """
    W = m * G
    cth = np.cos(theta)
    return np.array([-W * np.sin(theta), W * cth * np.sin(phi), W * cth * np.cos(phi)])


def flow_angles(u, v, w):
    """(V, alpha, beta): V=|[u,v,w]|, alpha=atan2(w,u), beta=asin(v/V).

    V < V_FLOW_MIN iken beta=0 döner; v/V [-1, 1] aralığına kırpılır.
    """
    V = float(np.sqrt(u * u + v * v + w * w))
    alpha = float(np.arctan2(w, u))
    if V < V_FLOW_MIN:
        return V, alpha, 0.0
    beta = float(np.arcsin(np.clip(v / V, -1.0, 1.0)))
    return V, alpha, beta
