import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
import streamlit as st

st.title("Ensemble Portfolio Optimizer")

# === DYNAMIC INPUTS ===
raw_tickers = st.text_input(
    "Enter tickers separated by commas (e.g. AAPL,MSFT,GOOGL,AMZN):",
    "BMNR,PLTR,HOOD,GOOGL,EWZ"
)
start_year = st.number_input("Start year (e.g. 2018):", value=2024, step=1)
end_year   = st.number_input("End year (e.g. 2025 or current year):", value=2026, step=1)

risk_free = st.number_input("Annual risk-free rate (e.g. 0.02 for 2%):", value=0.02, step=0.01)
max_weight = st.number_input("Max weight per stock (e.g. 0.35 = 35%):", value=0.35, min_value=0.0, max_value=1.0, step=0.05)

if st.button("Run"):
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
        st.stop()

    start_date = f"{start_year}-01-01"
    today = date.today()
    if end_year == today.year:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = f"{end_year}-12-31"

    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False
    )["Adj Close"].dropna()

    if prices.empty:
        st.error("No price data downloaded. Check tickers or date range.")
        st.stop()

    returns = prices.pct_change().dropna()

    def portfolio_stats(weights, mean_ret, cov, ret_series, rf=0.0):
        w = np.array(list(weights.values()))
        port_ret = float(w @ mean_ret)
        port_vol = float(np.sqrt(w @ cov @ w.T))
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
        port_daily = (ret_series[list(weights.keys())] @ w)
        var_95 = np.percentile(port_daily, 5)
        cvar_95 = port_daily[port_daily <= var_95].mean() if (port_daily <= var_95).any() else var_95
        return port_ret, port_vol, sharpe, var_95, cvar_95

    def print_portfolio(title, weights_dict, stats):
        port_ret, port_vol, sharpe, var_95, cvar_95 = stats
        st.text(f"\n=== {title} ===")
        st.text("Weights:")
        for t, w in weights_dict.items():
            st.text(f"  {t}: {w*100:5.2f}%")
        st.text(f"Annual return        : {port_ret*100:5.2f}%")
        st.text(f"Annual volatility    : {port_vol*100:5.2f}%")
        st.text(f"Sharpe ratio         : {sharpe:5.3f}")
        st.text(f"95% daily VaR (approx): {var_95*100:5.2f}%")
        st.text(f"95% daily CVaR (ES)   : {cvar_95*100:5.2f}%")

    # === Benchmark equal-weight ===
    mean_ret_hist = returns.mean() * 252
    cov_hist = returns.cov() * 252

    eq_weights = np.array([1 / len(tickers)] * len(tickers))
    eq_dict = dict(zip(tickers, eq_weights))
    eq_stats = portfolio_stats(eq_dict, mean_ret_hist, cov_hist, returns, rf=risk_free)
    print_portfolio("Benchmark: Equal‑weight portfolio", eq_dict, eq_stats)

    # === Robust inputs ===
    mu = expected_returns.ema_historical_return(prices, span=252)
    S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # === Model 1: Max‑Sharpe ===
    ef_ms = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_ms.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef_ms.max_sharpe(risk_free_rate=risk_free)
    w_ms = ef_ms.clean_weights()
    stats_ms = portfolio_stats(w_ms, mean_ret_hist, cov_hist, returns, rf=risk_free)
    print_portfolio(f"Model 1: Constrained Max‑Sharpe ({start_year}-{end_year})", w_ms, stats_ms)

    # === Model 2: Min‑Vol ===
    ef_mv = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_mv.min_volatility()
    w_mv = ef_mv.clean_weights()
    stats_mv = portfolio_stats(w_mv, mean_ret_hist, cov_hist, returns, rf=risk_free)
    print_portfolio(f"Model 2: Min‑Volatility ({start_year}-{end_year})", w_mv, stats_mv)

    # === Model 3: Risk‑Parity ===
    asset_vol = returns.std() * np.sqrt(252)
    inv_vol = 1 / asset_vol.replace(0, np.nan)
    inv_vol = inv_vol / inv_vol.sum()
    inv_vol = np.clip(inv_vol, 0, max_weight)
    inv_vol = inv_vol / inv_vol.sum()
    w_rp = inv_vol.to_dict()
    stats_rp = portfolio_stats(w_rp, mean_ret_hist, cov_hist, returns, rf=risk_free)
    print_portfolio("Model 3: Risk‑Parity (inverse‑vol approximation)", w_rp, stats_rp)

    # === Ensemble ===
    all_keys = tickers
    w_ens = {}
    for t in all_keys:
        w_ens[t] = (w_ms.get(t, 0) + w_mv.get(t, 0) + w_rp.get(t, 0)) / 3

    w_arr = np.array(list(w_ens.values()))
    w_arr = np.minimum(w_arr, max_weight)
    w_arr = w_arr / w_arr.sum()
    w_ens = dict(zip(all_keys, w_arr))

    stats_ens = portfolio_stats(w_ens, mean_ret_hist, cov_hist, returns, rf=risk_free)

    st.text("\n=== FINAL: Ensemble portfolio (average of Max‑Sharpe, Min‑Vol, Risk‑Parity) ===")
    port_ret, port_vol, sharpe, var_95, cvar_95 = stats_ens
    st.text("Weights:")
    for t, w in w_ens.items():
        st.text(f"  {t}: {w*100:5.2f}%")
    st.text(f"Annual return        : {port_ret*100:5.2f}%")
    st.text(f"Annual volatility    : {port_vol*100:5.2f}%")
    st.text(f"Sharpe ratio         : {sharpe:5.3f}")
    st.text(f"95% daily VaR (approx): {var_95*100:5.2f}%")
    st.text(f"95% daily CVaR (ES)   : {cvar_95*100:5.2f}%")

    st.text("\n>>> Recommended allocation for NEXT YEAR (ensemble, industry‑style):")
    for t, w in w_ens.items():
        st.text(f"  {t}: {w*100:5.2f}%")

    st.text("\nExpected (not guaranteed) performance for next year:")
    st.text(f"  Expected annual return    : {port_ret*100:5.2f}%")
    st.text(f"  Expected annual volatility: {port_vol*100:5.2f}%")
    st.text(f"  Expected Sharpe ratio      : {sharpe:5.3f}")
    st.text(f"  95% daily VaR              : {var_95*100:5.2f}%")
    st.text(f"  95% daily CVaR (ES)        : {cvar_95*100:5.2f}%")
