"""Drawdown and regime analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))


def main() -> None:
    st.title("🌊 Drawdown & Regime Analyzer")
    st.caption("Segment equity curve into regimes and surface pain periods.")

    equity = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.01, 250)) * 10000)
    drawdown = equity / equity.cummax() - 1
    st.area_chart(drawdown, height=200)

    st.subheader("Regime segmentation")
    rolling_vol = equity.pct_change().rolling(20).std()
    regime = pd.cut(rolling_vol, bins=3, labels=["Calm", "Normal", "Wild"]).fillna("Calm")
    summary = pd.DataFrame({"regime": regime, "return": equity.pct_change()}).groupby("regime").mean()
    st.dataframe(summary)

    st.subheader("Drawdown episodes")
    episodes = pd.DataFrame(
        {
            "start": ["2022-01-01", "2022-03-15"],
            "end": ["2022-02-20", "2022-04-01"],
            "depth": [-0.12, -0.08],
            "duration": [30, 20],
        }
    )
    st.table(episodes)


if __name__ == "__main__":
    main()
