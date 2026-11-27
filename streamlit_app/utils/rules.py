"""Rule building helpers for strategy/backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class RuleBlock:
    name: str
    description: str
    expression: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        local_dict = {**self.parameters, "df": data}
        return data.eval(self.expression, local_dict)


def combine_rules(*rules: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
    def _combined(df: pd.DataFrame) -> pd.Series:
        result = pd.Series(True, index=df.index)
        for rule in rules:
            result &= rule(df)
        return result

    return _combined
