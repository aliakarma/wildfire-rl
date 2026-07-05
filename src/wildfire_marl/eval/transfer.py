"""Domain adaptation and transfer metrics including Transfer Robustness Score (TRS) and asymmetry.
"""

from __future__ import annotations

import math


def transfer_robustness_score(native: float, transfer: float) -> float:
    """TRS = transfer / native performance ratio. Returns float('nan') if native is 0."""
    if native == 0.0:
        return float('nan')
    return float(transfer / native)


def cross_domain_gap(native: float, transfer: float) -> float:
    """Gap = native - transfer."""
    return float(native - transfer)


def adaptation_asymmetry(trs_a_to_b: float, trs_b_to_a: float) -> float:
    """Asymmetry = TRS(A -> B) - TRS(B -> A)."""
    return float(trs_a_to_b - trs_b_to_a)
