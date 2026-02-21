import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions, exceptions as ppo_exc
import streamlit as st

st.set_page_config(page_title="Ensemble Portfolio Optimizer", layout="wide")
st.title("Ensemble Portfolio Optimizer")

# ========== Inputs ==========
with st.sidebar:
    st.header("Inputs")
    raw_tickers = st.text_input(
        "Tickers (comma separated)",
        "BMNR,PLTR,HOOD,GOOGL,EWZ"
    )
    start_year = st.number_input("Start year", value=2024, step=1)
    end_year   = st.number_input("End year", value=2026, step=1)
    risk_free  = st.number_input("Risk‑free rate (annual)", value=0.02, step=0.01)
    max_weight = st.number_input("Max weight per stock", value=0.35, min_value=0.0, max_value=1.0, step=0.05)
    run = st.button("Run optimization")

if not run:
    st.info("Set your inputs in the sidebar and click **Run optimization**.")
    st.stop()

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

st.write(f"Using data for **{', '.join(tickers)}** from **{start_date}** to **{end_date}**.")

# ========== Data download ==========
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

# ========== Helpers ==========
def portfolio_stats(weights, mean_ret, cov, ret_series, rf=0.0):
    w = np.array(list(weights.values()))
    port_ret = float(w @ mean_ret)
    port_vol = float(np.sqrt(w @ cov @ w.T))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    port_daily = (ret_series[list(weights.keys())] @ w)
    var_95 = np.percentile(port_daily, 5)
    cvar_95 = port_daily[port_daily <= var_95].mean() if (port_daily <= var_95).any() else var_95
    return {
        "Return %": port_ret * 100,
        "Volatility %": port_vol * 100,
        "Sharpe": sharpe,
        "VaR 95% %": var_95 * 100,
        "CVaR 95% %": cvar_95 * 100,
    }

def weights_table(name, weights):
    df = pd.DataFrame(
        {"Model": name, "Ticker": list(weights.keys()),
         "Weight %": [w * 100 for w in weights.values()]}
    )
    return df

# ========== Baseline equal‑weight ==========
mean_ret_hist = returns.mean() * 252
cov_hist = returns.cov() * 252

eq_weights = dict(zip(tickers, [1 / len(tickers)] * len(tickers)))
eq_stats = portfolio_stats(eq_weights, mean_ret_hist, cov_hist, returns, rf=risk_free)

# ========== Robust inputs ==========
mu = expected_returns.ema_historical_return(prices, span=252)
S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# ========== Model 1: Max‑Sharpe ==========
try:
    ef_ms = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_ms.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef_ms.max_sharpe(risk_free_rate=risk_free)
    w_ms = ef_ms.clean_weights()
    ms_note = ""
except ppo_exc.OptimizationError:
    w_ms = eq_weights.copy()
    ms_note = " (fallback to equal‑weight)"

ms_stats = portfolio_stats(w_ms, mean_ret_hist, cov_hist, returns, rf=risk_free)

# ========== Model 2: Min‑Vol ==========
try:
    ef_mv = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_mv.min_volatility()
    w_mv = ef_mv.clean_weights()
    mv_note = ""
except ppo_exc.OptimizationError:
    w_mv = eq_weights.copy()
    mv_note = " (fallback to equal‑weight)"

mv_stats = portfolio_stats(w_mv, mean_ret_hist, cov_hist, returns, rf=risk_free)

# ========== Model 3: Risk‑Parity ==========
asset_vol = returns.std() * np.sqrt(252)
inv_vol = 1 / asset_vol.replace(0, np.nan)
inv_vol = inv_vol / inv_vol.sum()
inv_vol = np.clip(inv_vol, 0, max_weight)
inv_vol = inv_vol / inv_vol.sum()
w_rp = inv_vol.to_dict()
rp_stats = portfolio_stats(w_rp, mean_ret_hist, cov_hist, returns, rf=risk_free)

# ========== Ensemble ==========
all_keys = tickers
w_ens = {}
for t in all_keys:
    w_ens[t] = (w_ms.get(t, 0) + w_mv.get(t, 0) + w_rp.get(t, 0)) / 3

w_arr = np.array(list(w_ens.values()))
w_arr = np.minimum(w_arr, max_weight)
w_arr = w_arr / w_arr.sum()
w_ens = dict(zip(all_keys, w_arr))
ens_stats = portfolio_stats(w_ens, mean_ret_hist, cov_hist, returns, rf=risk_free)

# ========== Show results: weights tables ==========
weights_all = pd.concat([
    weights_table("Equal‑weight", eq_weights),
    weights_table("Max‑Sharpe" + ms_note, w_ms),
    weights_table("Min‑Vol" + mv_note, w_mv),
    weights_table("Risk‑Parity", w_rp),
    weights_table("Ensemble", w_ens),
])

st.subheader("Portfolio Weights (%)")
st.dataframe(
    weights_all.pivot(index="Ticker", columns="Model", values="Weight %")
              .round(2)
)

# ========== Show results: summary metrics ==========
summary = pd.DataFrame([
    {"Model": "Equal‑weight", **eq_stats},
    {"Model": "Max‑Sharpe" + ms_note, **ms_stats},
    {"Model": "Min‑Vol" + mv_note, **mv_stats},
    {"Model": "Risk‑Parity", **rp_stats},
    {"Model": "Ensemble", **ens_stats},
])

st.subheader("Model Performance Summary")
st.dataframe(summary.set_index("Model").round(2))

st.markdown("### Recommended allocation for next year (Ensemble)")
st.dataframe(
    pd.DataFrame(
        {"Ticker": list(w_ens.keys()),
         "Weight %": [w * 100 for w in w_ens.values()]}
    ).set_index("Ticker").round(2)
)
