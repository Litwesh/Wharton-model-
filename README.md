# Quant Portfolio Intelligence v4

This version is a modular research platform rather than a simple stock screener.

## Main systems
- Fundamental Agent: peer-relative Quality / Growth / Valuation / Balance Sheet scoring, FCF and earnings quality.
- Technical Agent: 70+ price/technical features.
- Quant Agent: Logistic Regression + Extra Trees + Random Forest + HistGradientBoosting + XGBoost ensemble.
- Indicator learning: rolling information-coefficient evidence determines indicator weights instead of fixed weights.
- Walk-forward validation: time-ordered out-of-sample accuracy, AUC, Brier score and prediction coverage.
- Portfolio Agent: live holdings, covariance, correlation, risk contribution, Sharpe, Sortino, drawdown, diversification ratio, efficient frontier and maximum-Sharpe allocation.
- Client-fit Agent: matches the security's public business description and quantitative profile against the Wharton case objectives.
- Data Hub: yfinance primary, Alpha Vantage secondary if API key is configured.

## 80% accuracy
No responsible market model can guarantee 80% future accuracy. The application measures the target on unseen historical data and reports coverage alongside accuracy. This matters because selective predictions can otherwise make accuracy look artificially high.

## Install
```bat
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Optional: copy `.env.example` to `.env` and add an Alpha Vantage key.

## Run
```bat
.venv\\Scripts\\activate
streamlit run app.py
```
Stop with Ctrl+C. Next time, activate the environment and run Streamlit again.

## Daily model evolution
```bat
python daily_update.py
```
The application also refreshes training when newer market data is available. Champion/challenger governance prevents a newly trained model from replacing the existing model when its validation is materially worse.

## Important quantitative point
Portfolio variance is determined by the covariance matrix, not individual volatility alone. Lower or negative correlations can reduce covariance contributions and improve diversification; they do not create expected return by themselves.
