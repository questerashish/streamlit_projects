"""Position sizing helpers."""
from __future__ import annotations

from dataclasses import dataclass


def fixed_fractional(account_size: float, risk_pct: float, stop_distance: float) -> float:
    risk_amount = account_size * risk_pct
    if stop_distance <= 0:
        return 0.0
    return risk_amount / stop_distance


def fixed_units(units: float) -> float:
    return units


def fixed_risk(risk_amount: float, stop_distance: float) -> float:
    if stop_distance <= 0:
        return 0.0
    return risk_amount / stop_distance


def atr_based(account_size: float, atr: float, multiplier: float, atr_value: float) -> float:
    if atr <= 0:
        return 0.0
    return account_size * multiplier / atr_value if atr_value else 0.0


def kelly_fraction(win_rate: float, payoff: float) -> float:
    edge = win_rate - (1 - win_rate) / payoff if payoff else 0.0
    if edge <= 0:
        return 0.0
    return edge / payoff


@dataclass
class RiskSummary:
    position_size: float
    r_multiple: float
    projected_profit: float
    projected_loss: float
