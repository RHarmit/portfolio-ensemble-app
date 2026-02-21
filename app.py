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

tickers_input = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
if not tickers_input:
    st.error("Please enter at least one ticker.")
    st.stop()

# ===== Dates =====
start_date = f"{start_year}-01-01"
today      = date.today()
end_date   = (today - timedelta(days=1)).strftime("%Y-%m-%d") if end_year == today.year else f"{end_year}-12-31"

# ===== Download prices (smart — drop bad tickers, keep good ones) =====
raw_prices = yf.download(
    tickers_input,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=False
)["Adj Close"]

# handle single ticker (returns Series, not DataFrame)
if isinstance(raw_prices, pd.Series):
    raw_prices = raw_prices.to_frame(name=tickers_input[0])

# find which tickers have enough data (at least 30 rows)
valid_tickers   = [t for t in tickers_input if t in raw_prices.columns and raw_prices[t].dropna().shape[0] >= 30]
invalid_tickers = [t for t in tickers_input if t not in valid_tickers]

if invalid_tickers:
    st.warning(
        f"The following tickers have no or insufficient data for {start_year}-{end_year} "
        f"and have been removed: **{', '.join(invalid_tickers)}**. "
        f"They may not have existed yet or were delisted in that period."
    )

if not valid_tickers:
    st.error("No valid tickers found for the selected date range. Try different tickers or a more recent date range.")
    st.stop()

# use only valid tickers
prices = raw_prices[valid_tickers].dropna()
tickers = valid_tickers

if prices.empty:
    st.error("No price data after cleaning. Try a different date range.")
    st.stop()

st.success(f"Running analysis for: **{', '.join(tickers)}** from {start_date} to {end_date}")

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

# ===== Technical analysis =====
def compute_technicals(price_series):
    s = price_series.dropna()
    score = 0
    details = {}

    # RSI
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1] if not rsi.empty else 50
    if rsi_val < 30:
        score += 1
        details["RSI"] = f"{rsi_val:.1f} (Oversold)"
    elif rsi_val > 70:
        score -= 1
        details["RSI"] = f"{rsi_val:.1f} (Overbought)"
    else:
        details["RSI"] = f"{rsi_val:.1f} (Neutral)"

    # 50MA
    if len(s) >= 50:
        ma50 = s.rolling(50).mean().iloc[-1]
        if s.iloc[-1] > ma50:
            score += 1
            details["50MA"] = "Price above (Bullish)"
        else:
            score -= 1
            details["50MA"] = "Price below (Bearish)"
    else:
        details["50MA"] = "Not enough data"

    # MACD
    if len(s) >= 35:
        ema12  = s.ewm(span=12, adjust=False).mean()
        ema26  = s.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        if macd.iloc[-1] > signal.iloc[-1]:
            score += 1
            details["MACD"] = "Bullish crossover"
        else:
            score -= 1
            details["MACD"] = "Bearish crossover"
    else:
        details["MACD"] = "Not enough data"

    if score >= 2:
        combined = "Strong"
    elif score <= -2:
        combined = "Weak"
    else:
        combined = "Neutral"

    return combined, score, details

# ===== Baseline inputs =====
mean_ret_hist = returns.mean() * 252
cov_hist      = returns.cov() * 252
eq_weights    = dict(zip(tickers, [1 / len(tickers)] * len(tickers)))

# ===== Robust inputs =====
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

# ===== Model 3: Risk-Parity =====
asset_vol = returns.std() * np.sqrt(252)
inv_vol   = 1 / asset_vol.replace(0, np.nan)
inv_vol   = inv_vol / inv_vol.sum()
inv_vol   = np.clip(inv_vol, 0, max_weight)
inv_vol   = inv_vol / inv_vol.sum()
w_rp      = inv_vol.to_dict()

# ===== Model 4: Momentum =====
try:
    n = len(prices)
    if n >= 42:
        price_start  = prices.iloc[0]
        price_1m_ago = prices.iloc[-21] if n > 21 else prices.iloc[-1]
        momentum     = (price_1m_ago / price_start) - 1
        momentum     = momentum.clip(lower=0)
        if momentum.sum() == 0:
            w_mom = eq_weights.copy()
        else:
            mom_weights = momentum / momentum.sum()
            mom_weights = np.clip(mom_weights, 0, max_weight)
            mom_weights = mom_weights / mom_weights.sum()
            w_mom       = mom_weights.to_dict()
    else:
        w_mom = eq_weights.copy()
except Exception:
    w_mom = eq_weights.copy()

# ===== Ensemble =====
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

# ===== News Sentiment =====
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "No news", "Neutral", 0.0, []
        headlines = []
        for item in news[:10]:
            title = (
                item.get("title") or
                item.get("content", {}).get("title", "") or ""
            )
            if title:
                headlines.append(title)
        if not headlines:
            return "No headlines", "Neutral", 0.0, []
        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg    = round(np.mean(scores), 3)
        label  = "Positive" if avg >= 0.05 else ("Negative" if avg <= -0.05 else "Neutral")
        return f"{len(headlines)} headlines", label, avg, headlines
    except Exception as e:
        return f"Error", "Neutral", 0.0, []

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

# ===================== OUTPUTS =====================

# Output 1: Ensemble bar chart
st.subheader("Recommended allocation for next year (Ensemble of 4 models)")
weights_df = pd.DataFrame({
    "Ticker":   list(w_ens.keys()),
    "Weight %": [w * 100 for w in w_ens.values()]
}).set_index("Ticker")
st.bar_chart(weights_df["Weight %"])

# Output 2: Weight breakdown
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

# Output 3: Expected performance
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

# Output 4: Technical + Sentiment
st.subheader("Technical and Sentiment Overview per stock")
st.write("Technical signal combines RSI, 50-day MA, and MACD. Sentiment scores latest Yahoo Finance headlines.")

overview_rows = []
all_headlines = {}

for ticker in tickers:
    price_series            = prices[ticker] if ticker in prices.columns else prices.squeeze()
    combined, score, detail = compute_technicals(price_series)
    info, sent_label, sent_score, headlines = get_sentiment(ticker)
    all_headlines[ticker]   = headlines

    overview_rows.append({
        "Ticker":             ticker,
        "Technical Signal":   combined,
        "Tech Score (-3/+3)": score,
        "RSI":                detail.get("RSI", "N/A"),
        "50MA":               detail.get("50MA", "N/A"),
        "MACD":               detail.get("MACD", "N/A"),
        "News Sentiment":     sent_label,
        "Sentiment Score":    sent_score,
    })

overview_df = pd.DataFrame(overview_rows).set_index("Ticker")

def color_signal(val):
    if val in ["Strong", "Positive"]:
        return "background-color: #d4edda; color: #155724"
    elif val in ["Weak", "Negative"]:
        return "background-color: #f8d7da; color: #721c24"
    else:
        return "background-color: #fff3cd; color: #856404"

st.dataframe(
    overview_df.style.applymap(color_signal, subset=["Technical Signal", "News Sentiment"])
)

for ticker in tickers:
    headlines = all_headlines.get(ticker, [])
    if headlines:
        with st.expander(f"Headlines for {ticker}"):
            for i, h in enumerate(headlines, 1):
                st.write(f"{i}. {h}")

# Output 5: Best calendar month
st.subheader("Best calendar month per stock (historical average)")
st.table(best_month_per_ticker)

# Output 6: Seasonality heatmap
st.subheader("Average monthly return % by stock (seasonality table)")
st.dataframe(avg_by_month.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}%"))
