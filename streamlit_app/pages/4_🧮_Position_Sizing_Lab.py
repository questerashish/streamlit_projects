"""Position sizing lab."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import sizing, stats  # noqa: E402


def main() -> None:
    st.title("🧮 Position Sizing Lab")
    st.caption("Compute position sizes across multiple risk models.")

    account_size = st.number_input("Account size", value=50000.0)
    entry = st.number_input("Entry price", value=100.0)
    stop = st.number_input("Stop loss", value=95.0)
    target = st.number_input("Target", value=110.0)
    risk_pct = st.slider("Risk %", min_value=0.1, max_value=5.0, value=1.0)
    atr = st.number_input("ATR", value=2.5)
    multiplier = st.slider("ATR multiplier", 0.5, 5.0, value=1.0)
    win_rate = st.slider("Win rate", 0.1, 0.9, value=0.45)
    payoff = st.slider("Payoff ratio", 0.5, 5.0, value=1.8)

    stop_distance = abs(entry - stop)
    risk_amount = account_size * (risk_pct / 100)
    fixed_fraction = sizing.fixed_fractional(account_size, risk_pct / 100, stop_distance)
    fixed_risk_size = sizing.fixed_risk(risk_amount, stop_distance)
    atr_based_size = sizing.atr_based(account_size, atr, multiplier, atr)
    kelly_fraction = sizing.kelly_fraction(win_rate, payoff)

    st.metric("Fixed Fractional size", f"{fixed_fraction:.2f} units")
    st.metric("Fixed Risk size", f"{fixed_risk_size:.2f} units")
    st.metric("ATR-based size", f"{atr_based_size:.2f} units")
    st.metric("Kelly fraction", f"{kelly_fraction:.2%}")

    projected_profit = (target - entry) * fixed_fraction
    projected_loss = (entry - stop) * fixed_fraction
    st.write("Projected P&L ladder")
    ladder = pd.DataFrame(
        {
            "R Multiple": [-1, -0.5, 0, 1, 2, 3],
            "P&L": [projected_loss, projected_loss / 2, 0, projected_profit, projected_profit * 2, projected_profit * 3],
        }
    )
    st.table(ladder)

    st.subheader("Risk of Ruin")
    st.write(stats.risk_of_ruin(win_rate, payoff, capital_r_multiple=100))

    st.subheader("Modes")
    st.write("Fixed-fractional, fixed-risk, fixed-units, ATR-based, Kelly — toggle between them above.")


if __name__ == "__main__":
    main()
