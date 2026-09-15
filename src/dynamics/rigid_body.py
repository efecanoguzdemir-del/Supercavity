"""
Katı cisim dinamiği (Fossen), gövde orijini CG'de, diyagonal inertia.

  (M_RB + M_A) nu_dot + C_RB(nu) nu + C_A(nu) nu = tau (+ roll sönümü)

nu  = [u, v, w, p, q, r]
tau = [X, Y, Z, K, M, N]  (gövde çerçevesi, CG etrafında)

Kuvvetten bağımsızdır; hidrodinamik/yerçekimi/itki tau'yu model.py toplar.
"""

import numpy as np

from .kinematics import skew


# ---------------------------------------------------------------------------
# Kütle / inertia
# ---------------------------------------------------------------------------
def cylinder_inertia(m, L, D, Ixx=None, Iyy=None, Izz=None):
    """Homojen dolu silindir inertiası (kendi merkezi etrafında), CG'de varsayılır.

      Ixx = m D^2 / 8
      Iyy = Izz = m (3 (D/2)^2 + L^2) / 12

    Not: gerçek kütle dağılımı bilinmediğinden (CG geometrik merkezde değil) bu
    bir yaklaşıklıktır; Ixx/Iyy/Izz argümanları (config) verilirse onlar kullanılır.
    Dönüş: (Ixx, Iyy, Izz)
    """
    R = 0.5 * D
    ixx = m * R * R / 2.0
    iyy = m * (3.0 * R * R + L * L) / 12.0
    return (float(Ixx) if Ixx is not None else ixx,
            float(Iyy) if Iyy is not None else iyy,
            float(Izz) if Izz is not None else iyy)


def inertia_from_config(cfg):
    """dict'ten (m, L, D, opsiyonel Ixx/Iyy/Izz) -> (Ixx, Iyy, Izz)."""
    return cylinder_inertia(cfg["m"], cfg["L"], cfg["D"],
                            Ixx=cfg.get("Ixx"), Iyy=cfg.get("Iyy"), Izz=cfg.get("Izz"))


def _inertia_matrix(I):
    """(Ixx,Iyy,Izz) veya 3x3 -> 3x3."""
    I = np.asarray(I, dtype=float)
    if I.shape == (3,):
        return np.diag(I)
    if I.shape == (3, 3):
        return I
    raise ValueError(f"I şekli (3,) veya (3,3) olmalı, geldi: {I.shape}")


def M_RB(m, I):
    """6x6 katı cisim kütle matrisi (CG orijinli): diag(m, m, m, I)."""
    M = np.zeros((6, 6))
    M[:3, :3] = m * np.eye(3)
    M[3:, 3:] = _inertia_matrix(I)
    return M


def C_RB(m, I, nu):
    """6x6 Coriolis-merkezcil matris (Fossen, r_g = 0, hıza-bağımsız lineer form).

      C_RB = [[ m S(nu2),        0      ],
              [    0,     -S(I_g nu2)  ]]

    C_RB nu = [m (omega x v), omega x (I omega)]
    """
    nu = np.asarray(nu, dtype=float)
    Ig = _inertia_matrix(I)
    nu2 = nu[3:6]
    C = np.zeros((6, 6))
    C[:3, :3] = m * skew(nu2)
    C[3:, 3:] = -skew(Ig @ nu2)
    return C


def C_A(M_A, nu):
    """Sabit eklenmiş kütle M_A için Coriolis matrisi (Fossen 2011, eş. 6.43).

      C_A = [[     0,        -S(A11 nu1 + A12 nu2)],
             [-S(A11 nu1 + A12 nu2), -S(A21 nu1 + A22 nu2)]]
    """
    M_A = np.asarray(M_A, dtype=float)
    nu = np.asarray(nu, dtype=float)
    nu1, nu2 = nu[:3], nu[3:6]
    a1 = M_A[:3, :3] @ nu1 + M_A[:3, 3:] @ nu2
    a2 = M_A[3:, :3] @ nu1 + M_A[3:, 3:] @ nu2
    C = np.zeros((6, 6))
    C[:3, 3:] = -skew(a1)
    C[3:, :3] = -skew(a1)
    C[3:, 3:] = -skew(a2)
    return C


# ---------------------------------------------------------------------------
# Basit kuvvet yardımcıları
# ---------------------------------------------------------------------------
def roll_damping(p, c_p):
    """Pasif roll sönümü: tau = [0,0,0, -c_p p, 0, 0]."""
    tau = np.zeros(6)
    tau[3] = -c_p * p
    return tau


def thrust_body(T):
    """Gövde +x yönünde itki (CG hattında, moment yok): [T,0,0,0,0,0]."""
    tau = np.zeros(6)
    tau[0] = T
    return tau


# ---------------------------------------------------------------------------
# Türev
# ---------------------------------------------------------------------------
def rigid_body_derivative(nu, tau, m, I, M_A=None, c_p=0.0):
    """nu_dot = (M_RB + M_A)^-1 (tau + tau_damp - C_RB nu - C_A nu).

    nu  : [u,v,w,p,q,r]
    tau : [X,Y,Z,K,M,N] gövde çerçevesi, CG etrafında (tüm dış kuvvetler)
    I   : (Ixx,Iyy,Izz) veya 3x3
    M_A : opsiyonel sabit 6x6 eklenmiş kütle (None/sıfır -> C_A eklenmez)
    c_p : pasif roll sönüm katsayısı [N m s/rad] (K_damp = -c_p p)
    """
    nu = np.asarray(nu, dtype=float)
    tau = np.asarray(tau, dtype=float)
    M = M_RB(m, I)
    rhs = tau - C_RB(m, I, nu) @ nu
    if c_p:
        rhs = rhs + roll_damping(nu[3], c_p)
    if M_A is not None:
        M_A = np.asarray(M_A, dtype=float)
        if np.any(M_A):
            M = M + M_A
            rhs = rhs - C_A(M_A, nu) @ nu
    return np.linalg.solve(M, rhs)
