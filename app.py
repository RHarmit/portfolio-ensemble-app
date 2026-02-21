import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions, exceptions as ppo_exc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

st.set_page_config(page_title="Ensemble Portfolio Optimizer", layout="wide")
st.title("Ensemble Portfolio Optimizer")

# ===== Sidebar inputs =====
with st.sidebar:
    st.header("Inputs")
    raw_tickers = st.text_input("Tickers (comma separated)", "BMNR,PLTR,HOOD,GOOGL,EWZ")
    start_year  = st.number_input("Start year", value=2024, step=1)
    end_year    = st.number_input("End year", value=2026, step=1)
    risk_free   = st.number_input("Risk-free rate (annual, e.g. 0.02 = 2%)", value=0.02, step=0.01)
    max_weight  = st.number_input("Max weight per stock (e.g. 0.35 = 35%)", value=0.35, min_value=0.0, max_value=1.0, step=0.05)
    run = st.button("Run")

if not run:
    st.info("Set your inputs in the sidebar and click **Run**.")
    st.stop()

tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# ===== Dates =====
start_date = f"{start_year}-01-01"
today      = date.today()
end_date   = (today - timedelta(days=1)).strftime("%Y-%m-%d") if end_year == today.year else f"{end_year}-12-31"

# ===== Download prices =====
prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)["Adj Close"].dropna()

if prices.empty:
    st.error("No price data downloaded. Check tickers or date range.")
    st.stop()

returns = prices.pct_change().dropna()

# ===== Helper: portfolio stats =====
def portfolio_stats(weights, mean_ret, cov, ret_series, rf=0.0):
    w          = np.array(list(weights.values()))
    port_ret   = float(w @ mean_ret)
    port_vol   = float(np.sqrt(w @ cov @ w.T))
    sharpe     = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    port_daily = (ret_series[list(weights.keys())] @ w)
    var_95     = np.percentile(port_daily, 5)
    cvar_95    = port_daily[port_daily <= var_95].mean() if (port_daily <= var_95).any() else var_95
    return port_ret, port_vol, sharpe, var_95, cvar_95

# ===== Baseline inputs =====
mean_ret_hist = returns.mean() * 252
cov_hist      = returns.cov() * 252
eq_weights    = dict(zip(tickers, [1 / len(tickers)] * len(tickers)))

# ===== Robust inputs (EMA + Ledoit-Wolf) =====
mu = expected_returns.ema_historical_return(prices, span=252)
S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# ===== Model 1: Max-Sharpe =====
try:
    ef_ms = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_ms.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef_ms.max_sharpe(risk_free_rate=risk_free)
    w_ms = ef_ms.clean_weights()
except ppo_exc.OptimizationError:
    w_ms = eq_weights.copy()

# ===== Model 2: Min-Vol =====
try:
    ef_mv = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_mv.min_volatility()
    w_mv = ef_mv.clean_weights()
except ppo_exc.OptimizationError:
    w_mv = eq_weights.copy()

# ===== Model 3: Risk-Parity (inverse-vol) =====
asset_vol = returns.std() * np.sqrt(252)
inv_vol   = 1 / asset_vol.replace(0, np.nan)
inv_vol   = inv_vol / inv_vol.sum()
inv_vol   = np.clip(inv_vol, 0, max_weight)
inv_vol   = inv_vol / inv_vol.sum()
w_rp      = inv_vol.to_dict()

# ===== Model 4: Momentum (adaptive) =====
try:
    n = len(prices)
    if n >= 42:
        price_start  = prices.iloc[0]
        price_1m_ago = prices.iloc[-21] if n > 21 else prices.iloc[-1]
        momentum     = (price_1m_ago / price_start) - 1
        momentum     = momentum.clip(lower=0)
        if momentum.sum() == 0:
            w_mom = eq_weights.copy()
            st.info("All stocks have negative momentum. Equal weight used for Model 4.")
        else:
            mom_weights = momentum / momentum.sum()
            mom_weights = np.clip(mom_weights, 0, max_weight)
            mom_weights = mom_weights / mom_weights.sum()
            w_mom       = mom_weights.to_dict()
    else:
        w_mom = eq_weights.copy()
except Exception:
    w_mom = eq_weights.copy()

# ===== Ensemble: average of all 4 models =====
w_ens = {}
for t in tickers:
    w_ens[t] = (w_ms.get(t, 0) + w_mv.get(t, 0) + w_rp.get(t, 0) + w_mom.get(t, 0)) / 4

w_arr = np.array(list(w_ens.values()))
w_arr = np.minimum(w_arr, max_weight)
w_arr = w_arr / w_arr.sum()
w_ens = dict(zip(tickers, w_arr))

ret_ens, vol_ens, sharpe_ens, var_95_ens, cvar_95_ens = portfolio_stats(
    w_ens, mean_ret_hist, cov_hist, returns, rf=risk_free
)

# ===== News Sentiment (VADER) =====
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "No news found", "Neutral", 0.0, []

        headlines = []
        for item in news[:10]:   # use latest 10 headlines
            # handle both old and new yfinance news format
            title = (
                item.get("title") or
                item.get("content", {}).get("title", "") or
                ""
            )
            if title:
                headlines.append(title)

        if not headlines:
            return "No headlines found", "Neutral", 0.0, []

        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg_score = round(np.mean(scores), 3)

        if avg_score >= 0.05:
            label = "Positive"
        elif avg_score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return f"{len(headlines)} headlines analyzed", label, avg_score, headlines

    except Exception as e:
        return f"Error: {e}", "Neutral", 0.0, []

# ===== Seasonality =====
monthly_prices  = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna()
monthly_returns["Month"] = monthly_returns.index.month

month_names = {
    1:"January",  2:"February", 3:"March",     4:"April",
    5:"May",      6:"June",     7:"July",       8:"August",
    9:"September",10:"October", 11:"November",  12:"December"
}

avg_by_month = monthly_returns.groupby("Month")[tickers].mean() * 100
avg_by_month.index = avg_by_month.index.map(month_names)
avg_by_month.index.name = "Month"
avg_by_month = avg_by_month.round(2)

best_month_per_ticker = pd.DataFrame({
    "Best Month": avg_by_month.idxmax(),
    "Avg Return % in that month": avg_by_month.max().round(2)
})
best_month_per_ticker.index.name = "Ticker"

# ===================================================
# =================== OUTPUTS ======================
# ===================================================

# ===== Output 1: Bar chart of ensemble weights =====
st.subheader("Recommended allocation for next year (Ensemble of 4 models)")
weights_df = pd.DataFrame({
    "Ticker":   list(w_ens.keys()),
    "Weight %": [w * 100 for w in w_ens.values()]
}).set_index("Ticker")
st.bar_chart(weights_df["Weight %"])

# ===== Output 2: Weight breakdown by model =====
st.subheader("Weight breakdown by model (%)")
model_weights = pd.DataFrame({
    "Max-Sharpe":  pd.Series(w_ms),
    "Min-Vol":     pd.Series(w_mv),
    "Risk-Parity": pd.Series(w_rp),
    "Momentum":    pd.Series(w_mom),
    "Ensemble":    pd.Series(w_ens),
}) * 100
model_weights.index.name = "Ticker"
st.dataframe(model_weights.round(2))

# ===== Output 3: Expected performance =====
st.subheader("Expected (not guaranteed) performance for next year")
perf_df = pd.DataFrame({
    "Metric": [
        "Expected annual return",
        "Expected annual volatility",
        "Expected Sharpe ratio",
        "95% daily VaR",
        "95% daily CVaR (ES)",
    ],
    "Value": [
        f"{ret_ens*100:5.2f}%",
        f"{vol_ens*100:5.2f}%",
        f"{sharpe_ens:5.3f}",
        f"{var_95_ens*100:5.2f}%",
        f"{cvar_95_ens*100:5.2f}%",
    ],
})
st.table(perf_df)

# ===== Output 4: News Sentiment =====
st.subheader("News Sentiment per stock (latest headlines)")
st.write("Sentiment scored using VADER on the most recent Yahoo Finance headlines for each ticker.")

sentiment_rows = []
all_headlines  = {}

for ticker in tickers:
    info, label, score, headlines = get_sentiment(ticker)
    sentiment_rows.append({
        "Ticker": ticker,
        "Sentiment": label,
        "Avg Score (-1 to 1)": score,
        "Headlines Analyzed": info
    })
    all_headlines[ticker] = headlines

sentiment_df = pd.DataFrame(sentiment_rows).set_index("Ticker")

# color the sentiment column
def color_sentiment(val):
    if val == "Positive":
        return "background-color: #d4edda; color: #155724"
    elif val == "Negative":
        return "background-color: #f8d7da; color: #721c24"
    else:
        return "background-color: #fff3cd; color: #856404"

st.dataframe(sentiment_df.style.applymap(color_sentiment, subset=["Sentiment"]))

# show headlines per ticker in expanders
for ticker in tickers:
    headlines = all_headlines.get(ticker, [])
    if headlines:
        with st.expander(f"Headlines for {ticker}"):
            for i, h in enumerate(headlines, 1):
                st.write(f"{i}. {h}")

# ===== Output 5: Best calendar month per stock =====
st.subheader("Best calendar month per stock (historical average)")
st.write("Which month of the year has historically delivered the highest average return for each stock.")
st.table(best_month_per_ticker)

# ===== Output 6: Full seasonality heatmap =====
st.subheader("Average monthly return % by stock (seasonality table)")
st.write("Each cell = average return % for that stock in that calendar month.")
st.dataframe(avg_by_month.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}%"))
