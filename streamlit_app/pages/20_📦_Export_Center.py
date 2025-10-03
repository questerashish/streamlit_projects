"""Export center for all artefacts."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from utils import io  # noqa: E402


def main() -> None:
    st.title("📦 Export Center")
    st.caption("One-stop shop for CSV/XLSX/PDF and zipped bundles.")

    st.subheader("Trades & Stats")
    st.button("Export trades to XLSX")
    st.button("Export stats PDF pack")

    st.subheader("Strategy bundle")
    st.button("Zip strategies + rules")

    st.subheader("Playbook & Weekly review")
    st.button("Generate weekly PDF")

    st.info("Leverage utils.io helpers for CSV/XLSX plus pdfkit/weasyprint for PDFs.")


if __name__ == "__main__":
    main()
