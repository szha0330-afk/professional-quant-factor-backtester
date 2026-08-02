import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quant_research.config import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
)


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(x_train, y_train)

    return model


def predict_returns(
    model: RandomForestRegressor,
    x_current: pd.DataFrame,
) -> pd.Series:
    predictions = model.predict(x_current)

    return pd.Series(
        predictions,
        index=x_current.index,
        name="predicted_forward_return",
    )


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
) -> pd.Series:
    return pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="feature_importance",
    ).sort_values(ascending=False)