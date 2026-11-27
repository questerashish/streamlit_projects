"""Performance analytics dashboards."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import charts, stats  # noqa: E402


def demo_equity() -> pd.Series:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=250)
    returns = np.random.normal(0.001, 0.01, len(dates))
    equity = (1 + pd.Series(returns, index=dates)).cumprod() * 10000
    return equity


def main() -> None:
    st.title("📈 Performance Analytics & Reports")
    st.caption("Deep statistics, rolling metrics, and exportable packs.")

    equity = demo_equity()
    returns = equity.pct_change().dropna()

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("CAGR", f"{stats.cagr(equity):.2%}")
    metrics_col2.metric("Sharpe", f"{stats.sharpe_ratio(returns):.2f}")
    metrics_col3.metric("Max DD", f"{stats.max_drawdown(equity):.2%}")

    st.plotly_chart(charts.histogram(returns, bins=30), use_container_width=True)

    with st.expander("Rolling stats"):
        window = st.slider("Window", min_value=10, max_value=100, value=30)
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std()
        st.line_chart(rolling_sharpe, height=200)
        st.line_chart(returns.cumsum(), height=200)

    st.subheader("Underwater curve")
    dd = equity / equity.cummax() - 1
    st.area_chart(dd, height=200)

    st.subheader("Edge by time")
    hourly = pd.DataFrame({"hour": np.arange(0, 24), "avg_R": np.random.uniform(-0.1, 0.2, 24)})
    st.bar_chart(hourly.set_index("hour"))

    st.subheader("Exports")
    st.button("Generate Daily PDF pack")
    st.button("Generate Weekly PDF pack")


if __name__ == "__main__":
    main()
