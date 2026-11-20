"""Default research settings. All targets are illustrative, not investment advice."""

# Match the ten-ETF universe in the preserved historical input data/prices.csv.
TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM", "VNQ", "XLE", "XLK")
BENCHMARK = "SPY"
HORIZON = 21
VOL_LOOKBACK = 63
TOP_N = 3
MAX_WEIGHT = 0.5
RISK_AVERSION = 8.0
COST_BPS = 10.0

FEATURE_COLUMNS = (
    "return_5", "return_21", "return_63", "return_126",
    "ma_gap_20", "ma_gap_60", "volatility_21", "volatility_63", "drawdown_63",
)
