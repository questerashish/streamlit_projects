"""Chart helpers built on matplotlib and plotly."""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def equity_curve_chart(equity: pd.Series):
    fig, ax = plt.subplots()
    equity.plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True)
    return fig


def histogram(series: pd.Series, bins: int = 20):
    fig = px.histogram(series, nbins=bins, title="Distribution")
    fig.update_layout(bargap=0.1)
    return fig


def multi_equity_plot(series_dict: dict[str, pd.Series]):
    fig = px.line(pd.DataFrame(series_dict))
    fig.update_layout(title="Equity Comparison", xaxis_title="Date", yaxis_title="Equity")
    return fig


def calendar_heatmap(df: pd.DataFrame, value_col: str):
    pivot = df.pivot(index="week", columns="weekday", values=value_col).fillna(0)
    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Calendar Heatmap")
    fig.colorbar(im, ax=ax)
    return fig


def risk_gauge(values: Iterable[float]):
    series = pd.Series(list(values))
    fig = px.bar(series, title="Risk Exposure", labels={"index": "Item", "value": "Risk"})
    return fig
