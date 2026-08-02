import pandas as pd

from quant_research.config import (
    TICKERS,
    BENCHMARK,
    FORWARD_DAYS,
    TOP_N,
    MIN_TRAINING_ROWS,
    FEATURE_COLUMNS,
)
from quant_research.features import create_feature_matrix
from quant_research.labels import create_forward_return_labels
from quant_research.model import train_model, predict_returns, get_feature_importance
from quant_research.portfolio import (
    build_portfolio_weights,
    calculate_portfolio_returns,
    calculate_benchmark_returns,
)
from quant_research.metrics import calculate_performance_metrics


def get_month_end_trading_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return prices.groupby([prices.index.year, prices.index.month]).tail(1).index


def run_walk_forward_backtest(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feature_matrix = create_feature_matrix(prices)
    labels = create_forward_return_labels(prices, FORWARD_DAYS)

    dataset = feature_matrix.join(labels)
    dataset = dataset.dropna()

    rebalance_dates = get_month_end_trading_dates(prices)

    prediction_records = []
    feature_importance_records = []

    for date in rebalance_dates:
        date_position = prices.index.get_loc(date)

        if date_position < 300:
            continue

        cutoff_position = max(0, date_position - FORWARD_DAYS)
        cutoff_date = prices.index[cutoff_position]

        train_data = dataset[
            dataset.index.get_level_values("date") <= cutoff_date
        ]

        if len(train_data) < MIN_TRAINING_ROWS:
            continue

        try:
            current_features = feature_matrix.xs(date, level="date")
        except KeyError:
            continue

        current_features = current_features.dropna()

        if current_features.empty:
            continue

        x_train = train_data[FEATURE_COLUMNS]
        y_train = train_data["target"]

        model = train_model(x_train, y_train)

        predictions = predict_returns(
            model,
            current_features[FEATURE_COLUMNS],
        )

        for ticker, predicted_return in predictions.items():
            prediction_records.append({
                "date": date,
                "ticker": ticker,
                "predicted_forward_return": predicted_return,
            })

        feature_importance = get_feature_importance(model, FEATURE_COLUMNS)

        for feature_name, importance in feature_importance.items():
            feature_importance_records.append({
                "date": date,
                "feature": feature_name,
                "importance": importance,
            })

    if len(prediction_records) == 0:
        raise ValueError("No predictions generated.")

    predictions_df = pd.DataFrame(prediction_records)
    feature_importance_df = pd.DataFrame(feature_importance_records)

    weights = build_portfolio_weights(
        prices=prices,
        predictions=predictions_df,
        top_n=TOP_N,
    )

    portfolio_result = calculate_portfolio_returns(prices, weights)
    benchmark_result = calculate_benchmark_returns(prices, BENCHMARK)

    combined = pd.concat(
        [portfolio_result, benchmark_result],
        axis=1,
    )

    portfolio_metrics = calculate_performance_metrics(
        returns=combined["portfolio_return"],
        equity_curve=combined["equity_curve"],
        turnover=combined["turnover"],
    )

    benchmark_metrics = calculate_performance_metrics(
        returns=combined["benchmark_return"],
        equity_curve=combined["benchmark_equity"],
    )

    performance_report = pd.DataFrame({
    "Random Forest Factor Portfolio": portfolio_metrics,
    "SPY Benchmark": benchmark_metrics,
})
    return {
        "features": feature_matrix,
        "predictions": predictions_df,
        "feature_importance": feature_importance_df,
        "weights": weights,
        "backtest_results": combined,
        "performance_report": performance_report,
    }