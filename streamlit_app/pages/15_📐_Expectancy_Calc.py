"""Expectancy calculator."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import stats  # noqa: E402


def main() -> None:
    st.title("📐 Expectancy & Edge Calculator")
    st.caption("Quick sanity checks on win-rate and payoff.")

    win_rate = st.slider("Win rate", 0.1, 0.9, 0.5)
    avg_win = st.number_input("Average win", value=200.0)
    avg_loss = st.number_input("Average loss", value=-150.0)
    costs = st.number_input("Trading costs per trade", value=10.0)

    expectancy = stats.expectancy(avg_win - costs, avg_loss - costs, win_rate)
    payoff = stats.payoff_ratio(avg_win, avg_loss)
    st.metric("Expectancy / trade", f"{expectancy:.2f}")
    st.metric("Payoff ratio", f"{payoff:.2f}")

    st.subheader("Breakeven table")
    st.table({"Win rate": [0.3, 0.4, 0.5], "Expectancy": [expectancy * 0.8, expectancy, expectancy * 1.2]})


if __name__ == "__main__":
    main()
