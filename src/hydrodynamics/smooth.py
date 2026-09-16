"""
Rejim geçişlerini sürekli yapmak için yardımcılar.

Legacy modeldeki sert eşikler (kavite var/yok, kesit kapanması, transom teması,
σ = 1 kavite oluşumu) kuvvetlerde sıçrama üretir. Tasarım kriteri: geçişlerde kuvvet
eğrisi sürekli olmalı, sınırda ihmal edilebilir değere yakınsamalı. Bu modül tek bir
C¹ geçiş fonksiyonu sağlar; eşik genişlikleri constants.py'dedir.
"""

import numpy as np


def smoothstep(x):
    """C¹ geçiş: x ≤ 0 → 0, x ≥ 1 → 1, arada 3x² − 2x³."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
