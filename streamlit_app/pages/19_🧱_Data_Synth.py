"""Manual OHLCV data builder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import io  # noqa: E402


def main() -> None:
    st.title("🧱 Manual Data Builder")
    st.caption("Generate synthetic OHLCV data for sandbox backtests.")

    length = st.slider("Length", 50, 1000, 200)
    volatility = st.slider("Volatility", 0.1, 5.0, 1.0)
    drift = st.slider("Drift", -1.0, 1.0, 0.2)
    spikes = st.slider("Spikes", 0, 10, 2)

    base = np.cumsum(np.random.normal(drift / length, volatility / length, length))
    price = 100 + base
    high = price + np.random.uniform(0, 1, length)
    low = price - np.random.uniform(0, 1, length)
    open_price = price + np.random.uniform(-0.5, 0.5, length)
    close = price
    volume = np.random.randint(100, 1000, length)
    df = pd.DataFrame({"open": open_price, "high": high, "low": low, "close": close, "volume": volume})

    st.line_chart(df[["close"]])
    st.dataframe(df.head())

    if st.button("Save to daily_bars table"):
        st.success("Persist generated data to SQLite via utils.db (implement).")

    st.download_button("Export CSV", df.to_csv(index=False), file_name="synthetic_bars.csv")


if __name__ == "__main__":
    main()
