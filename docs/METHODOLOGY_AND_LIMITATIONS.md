# Methodology and research safeguards

## Timeline audit

At month-end close t:
1. Compute nine price factors for all assets using only closes through t.
2. Build an expanding training set containing only rows dated <= t-21 trading sessions, so their forward 21-day labels have completed by t.
3. Fit Random Forest with fixed hyperparameters to those historical examples and predict 21-day returns using factors dated t.
4. Select top 3 forecasts, then calculate weights with either the inverse-volatility benchmark or constrained MVO. Both use only historical returns through t.
5. The target is implemented for returns from t to the next session's close; under this simplified fill assumption the trade occurs at the previous close before the next day's return. Assess realistic fills separately.
6. Charge proportional cost on absolute changes between target weights and *pretrade drifted weights*. Account for price-induced drift between rebalances.

Both approaches share exactly the same forecast table, trade calendar, asset universe, ranking and market data. Only the weighting rule differs.

## Data quality

Use split/dividend-adjusted closes and verify consistent timestamps and each ETF's inception date. Do not forward-fill suspended assets or create synthetic historical observations to mask missing rows. Survivorship and lookback effects remain possible. The default synthetic dataset is intended exclusively to verify software mechanics.

## Covariance and optimization

Use a 63-day sample covariance \(\widehat\Sigma_1\), approximate \(\widehat\Sigma_{21}=21\widehat\Sigma_1\), and shrink 10% toward its diagonal. No covariance model is guaranteed optimal; test risk aversion, shrinkage, and lookback sensitivity on *strictly held-out data*.

## What the report means

- `total_turnover` is total sum of one-way *absolute weight changes*, counting the initial investment as 1. It is not half-turnover.
- `cost_fraction` is turnover times cost rate (bps / 10,000) at each simulated execution.
- Sharpe is annualized mean daily return / annualized sample standard deviation, with risk-free rate 0.
- Max drawdown is the maximum percent fall in equity from a preceding high.
- The synthetic example can show either optimization method as better by chance; it is not an empirical claim.
- MVO weights can be unstable because forecast returns are noisy; test sensitivity, transaction costs, and a true untouched evaluation period before claiming value.
