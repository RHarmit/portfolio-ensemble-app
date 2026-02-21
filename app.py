import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
import streamlit as st

st.title("Ensemble Portfolio Optimizer")

st.write("Enter tickers and backtest period. The app will run your equal-weight, max-Sharpe, min-vol, risk-parity, and ensemble models.")

# ========= 1. User inputs (DYNAMIC) =========
raw_tickers = st.text_input(
    "Tickers separated by commas",
    "AAPL,MSFT,GOOGL,AMZN"
)

col1, col2 = st.columns(2)
start_year = col1.number_input(
    "Start year (e.g. 2018)",
    min_value=1980,
    max_value=date.today().year,
    value=2018,
    step=1
)
end_year = col2.number_input(
    "End year (e.g. current year)",
    min_value=start_year,
    max_value=date.today().year,
    value=date.today().year,
    step=1
)

risk_free = st.number_input("Annual risk-free rate", value=0.02, step=0.01)
max_weight = st.number_input("Max weight per stock", value=0.35, min_value=0.0, max_value=1.0, step=0.05)

run_button = st.button("Run optimization")

if run_button:
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    start_date = f"{start_year}-01-01"

    today = date.today()
    if end_year == today.year:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = f"{end_year}-12-31"

    st.write(f"Fetching Adj Close data for {tickers} from {start_date} to {end_date}...")

    # ========= 2. Download prices =========
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False
    )

    if "Adj Close" not in data.columns:
        st.error("No Adjusted Close data downloaded. Check tickers or date range.")
        st.stop()

    prices = data["Adj Close"].dropna()

    if prices.empty:
        st.error("No price data after cleaning. Check tickers or date range.")
        st.stop()

    returns = prices.pct_change().dropna()

    # ========= 3. Helper functions =========
    def portfolio_stats(weights, mean_ret, cov, ret_series, rf=0.0):
        w = np.array(list(weights.values()))
        port_ret = float(w @ mean_ret)
        port_vol = float(np.sqrt(w @ cov @ w.T))
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan

        port_daily = (ret_series[list(weights.keys())] @ w)
        var_95 = np.percentile(port_daily, 5)
        cvar_95 = port_daily[port_daily <= var_95].mean() if (port_daily <= var_95).any() else var_95

        return port_ret, port_vol, sharpe, var_95, cvar_95

    def show_portfolio_block(title, weights_dict, stats):
        port_ret, port_vol, sharpe, var_95, cvar_95 = stats
        st.subheader(title)
        df_weights = pd.DataFrame({
            "Ticker": list(weights_dict.keys()),
            "Weight %": [w * 100 for w in weights_dict.values()]
        }).set_index("Ticker")
        st.dataframe(df_weights.style.format({"Weight %": "{:.2f}"}))
        st.write(f"Annual return: {port_ret*100:5.2f}%")
        st.write(f"Annual volatility: {port_vol*100:5.2f}%")
        st.write(f"Sharpe ratio: {sharpe:5.3f}")
        st.write(f"95% daily VaR (approx): {var_95*100:5.2f}%")
        st.write(f"95% daily CVaR (ES): {cvar_95*100:5.2f}%")

    # ========= 4. Baseline equal-weight =========
    mean_ret_hist = returns.mean() * 252
    cov_hist = returns.cov() * 252

    eq_weights = np.array([1 / len(tickers)] * len(tickers))
    eq_dict = dict(zip(tickers, eq_weights))
    eq_stats = portfolio_stats(eq_dict, mean_ret_hist, cov_hist, returns, rf=risk_free)
    show_portfolio_block("Benchmark: Equal-weight portfolio", eq_dict, eq_stats)

    # ========= 5. Robust inputs (EMA returns + shrinkage cov) =========
    mu = expected_returns.ema_historical_return(prices, span=252)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # ========= 6. Model 1 — Constrained Max-Sharpe =========
    ef_ms = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_ms.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef_ms.max_sharpe(risk_free_rate=risk_free)
    w_ms = ef_ms.clean_weights()
    stats_ms = portfolio_stats(w_ms, mean_ret_hist, cov_hist, returns, rf=risk_free)
    show_portfolio_block(
        f"Model 1: Constrained Max-Sharpe ({start_year}-{end_year})",
        w_ms, stats_ms
    )

    # ========= 7. Model 2 — Min-Volatility =========
    ef_mv = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_mv.min_volatility()
    w_mv = ef_mv.clean_weights()
    stats_mv = portfolio_stats(w_mv, mean_ret_hist, cov_hist, returns, rf=risk_free)
    show_portfolio_block(
        f"Model 2: Min-Volatility ({start_year}-{end_year})",
        w_mv, stats_mv
    )

    # ========= 8. Model 3 — Risk-Parity (inverse-vol) =========
    asset_vol = returns.std() * np.sqrt(252)
    inv_vol = 1 / asset_vol.replace(0, np.nan)
    inv_vol = inv_vol / inv_vol.sum()
    inv_vol = np.clip(inv_vol, 0, max_weight)
    inv_vol = inv_vol / inv_vol.sum()
    w_rp = inv_vol.to_dict()
    stats_rp = portfolio_stats(w_rp, mean_ret_hist, cov_hist, returns, rf=risk_free)
    show_portfolio_block(
        "Model 3: Risk-Parity (inverse-vol approximation)",
        w_rp, stats_rp
    )

    # ========= 9. Ensemble of models =========
    all_keys = tickers
    w_ens = {}
    for t in all_keys:
        w_ens[t] = (w_ms.get(t, 0) + w_mv.get(t, 0) + w_rp.get(t, 0)) / 3

    w_arr = np.array(list(w_ens.values()))
    w_arr = np.minimum(w_arr, max_weight)
    w_arr = w_arr / w_arr.sum()
    w_ens = dict(zip(all_keys, w_arr))

    stats_ens = portfolio_stats(w_ens, mean_ret_hist, cov_hist, returns, rf=risk_free)

    st.markdown("---")
    show_portfolio_block(
        "FINAL: Ensemble portfolio (average of Max-Sharpe, Min-Vol, Risk-Parity)",
        w_ens, stats_ens
    )

    st.markdown("### Recommended allocation for NEXT YEAR (ensemble, industry-style):")
    ens_df = pd.DataFrame({
        "Ticker": list(w_ens.keys()),
        "Weight %": [w * 100 for w in w_ens.values()]
    }).set_index("Ticker")
    st.dataframe(ens_df.style.format({"Weight %": "{:.2f}"}))

    ret_ens, vol_ens, sharpe_ens, var_95_ens, cvar_95_ens = stats_ens
    st.write("#### Expected (not guaranteed) performance for next year:")
    st.write(f"Expected annual return: {ret_ens*100:5.2f}%")
    st.write(f"Expected annual volatility: {vol_ens*100:5.2f}%")
    st.write(f"Expected Sharpe ratio: {sharpe_ens:5.3f}")
    st.write(f"95% daily VaR: {var_95_ens*100:5.2f}%")
    st.write(f"95% daily CVaR (ES): {cvar_95_ens*100:5.2f}%")

    # ========= 10. Chart of ensemble allocation =========
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(w_ens.keys(), [w * 100 for w in w_ens.values()],
           color="#1f77b4", edgecolor="black")
    ax.set_ylabel("Portfolio weight (%)")
    ax.set_title(
        f"Ensemble Allocation for Next Year\nData: {start_year}-{end_year} up to {end_date}"
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
