"""Levels and playbook planner."""
from __future__ import annotations

import json
from datetime import date

import streamlit as st


def main() -> None:
    st.title("🗺️ Levels & Playbook Planner")
    st.caption("Prepare pre-market plans with key levels and scenarios.")

    st.subheader("Key levels")
    levels = st.text_area("Levels JSON", json.dumps({"HTF": [4300, 4250], "LTF": [4280]}, indent=2))
    st.subheader("Bias & scenarios")
    bias = st.selectbox("Bias", ["Bullish", "Neutral", "Bearish"])
    scenarios = st.text_area("If-Then", "If open gap up, look for OR break -> fade")
    checklist = st.multiselect(
        "Session checklist", ["Review overnight session", "Mark economic events", "Sync with team"], default=["Review overnight session"]
    )

    st.subheader("Playbook PDF")
    st.date_input("Session date", value=date.today())
    st.button("Generate Playbook PDF")
    st.info("HTML-to-PDF render handled locally via weasyprint/pdfkit.")


if __name__ == "__main__":
    main()
