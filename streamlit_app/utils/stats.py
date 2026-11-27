"""Statistical helper functions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free / len(returns)
    return float(np.sqrt(252) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    excess = returns - risk_free / len(returns)
    return float(np.sqrt(252) * excess.mean() / downside.std())


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    drawdowns = equity / cummax - 1
    return float(drawdowns.min())


def expectancy(avg_win: float, avg_loss: float, win_rate: float, loss_rate: float | None = None) -> float:
    loss_rate = loss_rate if loss_rate is not None else 1 - win_rate
    return avg_win * win_rate + avg_loss * loss_rate


def payoff_ratio(avg_win: float, avg_loss: float) -> float:
    return abs(avg_win / avg_loss) if avg_loss else 0.0


def risk_of_ruin(win_rate: float, payoff: float, capital_r_multiple: float) -> float:
    edge = win_rate - (1 - win_rate) / payoff if payoff else 0
    if edge <= 0:
        return 1.0
    return float(((1 - edge) / (1 + edge)) ** capital_r_multiple)
