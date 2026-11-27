"""Risk dashboard enforcing account heat limits."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import charts  # noqa: E402


def demo_open_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["ES", "NQ", "CL"],
            "entry": [4100, 12500, 72],
            "stop": [4050, 12300, 68],
            "size": [2, 1, 3],
            "risk": [1000, 1000, 1200],
            "tags": ["Index", "Index", "Energy"],
        }
    )


def main() -> None:
    st.title("🚨 Risk Dashboard (Account Heat)")
    st.caption("Stress-test proposed trades against risk policy.")

    profile = st.selectbox("Risk profile", ["Default", "Aggressive", "Conservative"])
    max_risk_trade = st.number_input("Max risk per trade", value=1000.0)
    max_portfolio_heat = st.number_input("Max portfolio heat", value=4000.0)

    st.subheader("Open / Proposed trades")
    trades = demo_open_trades()
    st.dataframe(trades)

    total_risk = trades["risk"].sum()
    breaches = trades[trades["risk"] > max_risk_trade]
    st.metric("Total portfolio heat", f"${total_risk:,.0f}")

    if total_risk > max_portfolio_heat:
        st.error("Portfolio heat exceeds limit. Reduce sizes or skip trades.")
    else:
        st.success("Within heat limits.")

    if not breaches.empty:
        st.warning("Trades breaching per-trade risk:")
        st.dataframe(breaches)

    st.subheader("Segment exposure")
    segment_exposure = trades.groupby("tags")["risk"].sum()
    st.plotly_chart(charts.risk_gauge(segment_exposure.values))

    st.subheader("Actionable deltas")
    st.write("Reduce ES by 0.5 lots, widen CL stop by 0.5, skip NQ if adding new positions.")


if __name__ == "__main__":
    main()
