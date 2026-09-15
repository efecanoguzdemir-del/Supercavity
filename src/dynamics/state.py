"""
Durum vektörü tanımı — TEK KAYNAK.

x = [u, v, w, p, q, r, phi, theta, psi, X, Y, Z, Lc, Dc, pc]   (15 durum)

  u, v, w      gövde ekseni lineer hızlar [m/s]  (x-ileri, y-sancak, z-aşağı)
  p, q, r      gövde ekseni açısal hızlar [rad/s]
  phi, theta, psi  Euler açıları (ZYX, roll-pitch-yaw) [rad]
  X, Y, Z      NED eylemsiz konum [m]  (Z pozitif = derinlik artışı)
  Lc, Dc       kavite uzunluğu / maksimum çapı [m] (tau gecikmeli)
  pc           kavite basıncı [Pa] (tau_pc gecikmeli)

Karar gerekçesi (14 -> 15): legacy simulate() Lc ile birlikte Dc'yi de
aynı tau=0.15 s gecikmesiyle entegre ediyor (satır 650-651). Dc cebirsel
alınırsa gövde ıslanması / planing (Dc'ye bağlı) legacy'den sapar. Bu yüzden
Dc ayrı bir ODE durumu olarak taşınır.

Tüm modüller indeksleri buradan almalı; sabit sayı (6, 7, 8 ...) kullanmayın.
"""

import numpy as np

IDX_U = 0
IDX_V = 1
IDX_W = 2
IDX_P = 3
IDX_Q = 4
IDX_R = 5
IDX_PHI = 6
IDX_THETA = 7
IDX_PSI = 8
IDX_X = 9
IDX_Y = 10
IDX_Z = 11
IDX_LC = 12
IDX_DC = 13
IDX_PC = 14

N_STATES = 15

STATE_NAMES = ("u", "v", "w", "p", "q", "r", "phi", "theta", "psi",
               "X", "Y", "Z", "Lc", "Dc", "pc")

# Dilimler
SL_NU_LIN = slice(IDX_U, IDX_W + 1)      # u, v, w
SL_NU_ANG = slice(IDX_P, IDX_R + 1)      # p, q, r
SL_NU = slice(IDX_U, IDX_R + 1)          # 6-DOF hız vektörü
SL_EULER = slice(IDX_PHI, IDX_PSI + 1)   # phi, theta, psi
SL_POS = slice(IDX_X, IDX_Z + 1)         # X, Y, Z
SL_CAVITY = slice(IDX_LC, IDX_PC + 1)    # Lc, Dc, pc


def make_state(u=0.0, v=0.0, w=0.0, p=0.0, q=0.0, r=0.0,
               phi=0.0, theta=0.0, psi=0.0, X=0.0, Y=0.0, Z=0.0,
               Lc=0.0, Dc=0.0, pc=2340.0):
    """İsimli argümanlardan durum vektörü üretir (açılar rad)."""
    x = np.zeros(N_STATES)
    x[IDX_U], x[IDX_V], x[IDX_W] = u, v, w
    x[IDX_P], x[IDX_Q], x[IDX_R] = p, q, r
    x[IDX_PHI], x[IDX_THETA], x[IDX_PSI] = phi, theta, psi
    x[IDX_X], x[IDX_Y], x[IDX_Z] = X, Y, Z
    x[IDX_LC], x[IDX_DC], x[IDX_PC] = Lc, Dc, pc
    return x


def state_to_dict(x):
    """Durum vektörünü {isim: değer} sözlüğüne çevirir."""
    return {name: float(x[i]) for i, name in enumerate(STATE_NAMES)}
