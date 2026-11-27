"""Trade importer and cleaner."""
from __future__ import annotations

import io as sysio
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import db, io  # noqa: E402


def regex_helper(column: pd.Series, pattern: str) -> pd.Series:
    compiled = re.compile(pattern)
    return column.astype(str).str.extract(compiled, expand=False)


def main() -> None:
    st.title("📥 Trade Importer & Cleaner")
    st.caption("Standardise messy broker exports before journaling.")

    uploaded = st.file_uploader("Upload raw trade CSV", type=["csv"])
    mapper_col1, mapper_col2 = st.columns(2)
    date_format = mapper_col1.text_input("Date format", value="%Y-%m-%d")
    regex_pattern = mapper_col2.text_input("Regex helper", value=r"([A-Z]{1,5})")

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.write("Preview", raw.head())

        with st.expander("Column mapper"):
            mapping = {}
            for col in raw.columns:
                mapping[col] = st.selectbox(
                    f"Map column `{col}`", ["ignore", "date", "symbol", "qty", "price", "fees", "direction"], key=col
                )
            st.json(mapping)

        if st.checkbox("Apply regex helper"):
            target_col = st.selectbox("Column to parse", raw.columns)
            raw[target_col] = regex_helper(raw[target_col], regex_pattern)
            st.write("Regex applied", raw[target_col].head())

        if st.button("Clean & Download"):
            cleaned = raw.rename(columns={col: dest for col, dest in mapping.items() if dest != "ignore"})
            buffer = sysio.StringIO()
            cleaned.to_csv(buffer, index=False)
            st.download_button("Download cleaned CSV", buffer.getvalue(), file_name="clean_trades.csv")

        if st.button("Write into Journal DB"):
            db.init_db()
            st.success("Would insert cleaned records into trades table (extend as needed).")

    st.subheader("Extras")
    st.write("Deduping, fee normalization, and corporate action adjustment placeholders ready.")


if __name__ == "__main__":
    main()
