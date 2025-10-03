"""Offline backtest runner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import charts, rules  # noqa: E402


def main() -> None:
    st.title("🔁 Backtest Runner")
    st.caption("Vectorised rule testing on uploaded OHLCV data.")

    uploaded = st.file_uploader("Upload OHLCV", type=["csv", "parquet"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            data = pd.read_csv(uploaded, parse_dates=["date"], infer_datetime_format=True)
        else:
            data = pd.read_parquet(uploaded)
        st.dataframe(data.head())
    else:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        data = pd.DataFrame({"date": dates, "close": np.linspace(100, 120, 100) + np.random.randn(100)})

    entry_rule = st.text_input("Entry rule", value="close > close.shift(1)")
    exit_rule = st.text_input("Exit rule", value="close < close.shift(1)")

    block = rules.RuleBlock("entry", "Simple momentum", entry_rule)
    signals = block.evaluate(data.set_index("date"))
    st.write("Sample signals", signals.head())

    st.subheader("Results")
    st.line_chart(data.set_index("date")["close"], height=200)
    st.info("Extend with trade list, equity curve, parameter sweeps, and walk-forward splits.")


if __name__ == "__main__":
    main()
