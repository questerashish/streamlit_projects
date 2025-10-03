"""Symbol and sector notebook."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))


def main() -> None:
    st.title("📚 Symbol / Sector Notebook")
    st.caption("Personal wiki of instruments and playbooks.")

    st.subheader("Add entry")
    symbol = st.text_input("Symbol", value="ES")
    atr = st.number_input("Typical ATR", value=20.0)
    timeframe = st.selectbox("Preferred timeframe", ["1m", "15m", "1h", "Daily"])
    personal_rules = st.text_area("Personal rules", "Trade only during cash session.")
    tags = st.text_input("Tags", "index,trend")

    if st.button("Save note"):
        st.success("Persist to SQLite levels/notes tables (implement).")

    st.subheader("Notebook")
    data = pd.DataFrame(
        {
            "symbol": ["ES", "NQ"],
            "ATR": [20, 30],
            "tags": ["index", "tech"],
            "rules": ["Avoid Fed days", "Trade after Europe close"],
        }
    )
    st.dataframe(data)
    st.download_button("Export crib sheet", data.to_csv(index=False), file_name="symbol_notebook.csv")


if __name__ == "__main__":
    main()
