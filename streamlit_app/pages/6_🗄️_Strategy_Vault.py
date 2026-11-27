"""Strategy vault file manager."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import db  # noqa: E402


def file_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    st.title("🗄️ Strategy Vault")
    st.caption("Store PDFs, code files, and metadata locally.")

    uploaded = st.file_uploader("Upload strategy artifact", type=["pdf", "txt", "py", "json"])
    strategy_name = st.text_input("Strategy name")
    strategy_type = st.selectbox("Type", ["Intraday", "Swing", "Options"])
    tags = st.text_input("Tags", value="momentum,trend")
    rules = st.text_area("Rules (JSON)", value=json.dumps({"entry": "EMA cross"}, indent=2))

    if uploaded and st.button("Save strategy"):
        checksum = file_checksum(uploaded.getvalue())
        file_path = Path("data") / f"{strategy_name}_{checksum[:8]}_{uploaded.name}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(uploaded.getvalue())
        db.upsert(
            "strategies",
            {
                "name": strategy_name,
                "type": strategy_type,
                "rules_json": rules,
                "capital_req": 0,
                "tags": tags,
                "file_path": str(file_path),
            },
        )
        st.success("Strategy stored with checksum versioning.")

    st.subheader("Saved strategies")
    records = db.fetch_all("SELECT name, type, tags, file_path FROM strategies ORDER BY name")
    st.dataframe(pd.DataFrame(records))

    st.subheader("Preview")
    st.info("Quick preview of text-based files shown here.")


if __name__ == "__main__":
    main()
