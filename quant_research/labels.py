import pandas as pd

from quant_research.config import FORWARD_DAYS


def create_forward_return_labels(
    prices: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
) -> pd.Series:
    forward_returns = prices.shift(-forward_days) / prices - 1

    labels = (
        forward_returns
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="target")
        .set_index(["date", "ticker"])["target"]
    )

    return labels