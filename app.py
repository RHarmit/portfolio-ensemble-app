import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions, exceptions as ppo_exc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

st.set_page_config(page_title="Ensemble Portfolio Optimizer", layout="wide")
st.title("Ensemble Portfolio Optimizer")

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.header("Inputs")
    raw_tickers = st.text_input("Tickers (comma separated)", "BMNR,PLTR,HOOD,GOOGL,EWZ")
    start_year  = st.number_input("Start year",  value=2024, step=1)
    end_year    = st.number_input("End year",    value=2026, step=1)
    risk_free   = st.number_input("Risk-free rate (e.g. 0.02 = 2%)", value=0.02, step=0.01)
    max_weight  = st.number_input("Max weight per stock (e.g. 0.35 = 35%)", value=0.35, min_value=0.0, max_value=1.0, step=0.05)
    run = st.button("Run")

if not run:
    st.info("Set your inputs in the sidebar and click **Run**.")
    st.stop()

tickers_input = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
if not tickers_input:
    st.error("Please enter at least one ticker.")
    st.stop()

# ══════════════════════════════════════════
# DATES
# ══════════════════════════════════════════
start_date = f"{start_year}-01-01"
today      = date.today()
end_date   = (today - timedelta(days=1)).strftime("%Y-%m-%d") if end_year == today.year else f"{end_year}-12-31"
total_trading_days_expected = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days * 5 / 7)

# ══════════════════════════════════════════
# DOWNLOAD PRICES
# ══════════════════════════════════════════
with st.spinner("Downloading price data..."):
    raw_prices = yf.download(
        tickers_input,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )["Adj Close"]

if isinstance(raw_prices, pd.Series):
    raw_prices = raw_prices.to_frame(name=tickers_input[0])

# ══════════════════════════════════════════
# CLASSIFY TICKERS BY DATA COVERAGE
# ══════════════════════════════════════════
MIN_ROWS          = 60    # min ~3 months of trading days
COVERAGE_THRESHOLD = 90.0 # % of requested range needed to be included

full_tickers    = []
partial_tickers = []
missing_tickers = []
ticker_meta     = {}

for t in tickers_input:
    if t not in raw_prices.columns:
        missing_tickers.append(t)
        ticker_meta[t] = {"rows": 0, "first_date": "N/A", "last_date": "N/A", "coverage": 0.0}
        continue

    col        = raw_prices[t].dropna()
    rows       = len(col)
    first_date = col.index[0].strftime("%Y-%m-%d") if rows > 0 else "N/A"
    last_date  = col.index[-1].strftime("%Y-%m-%d") if rows > 0 else "N/A"
    coverage   = min(round(rows / max(total_trading_days_expected, 1) * 100, 1), 100.0)

    ticker_meta[t] = {
        "rows":       rows,
        "first_date": first_date,
        "last_date":  last_date,
        "coverage":   coverage
    }

    if rows < MIN_ROWS:
        missing_tickers.append(t)
    elif coverage < COVERAGE_THRESHOLD:
        partial_tickers.append(t)
    else:
        full_tickers.append(t)

# ── Coverage table ──
st.markdown("## Data Coverage Check")
coverage_rows = []
for t in tickers_input:
    meta = ticker_meta[t]
    if t in full_tickers:
        status = "✅ Included"
        reason = "Full history — used in all models"
    elif t in partial_tickers:
        status = "⚠️ Excluded"
        reason = f"Only {meta['coverage']}% of range covered. Partial history biases optimizer (post-IPO rally effect)."
    else:
        status = "❌ Excluded"
        reason = f"No data or fewer than {MIN_ROWS} trading days. Ticker may not have existed in {start_year}–{end_year}."

    coverage_rows.append({
        "Ticker":          t,
        "Status":          status,
        "First Date":      meta["first_date"],
        "Last Date":       meta["last_date"],
        "Trading Days":    meta["rows"],
        "Coverage %":      f"{meta['coverage']}%",
        "Reason":          reason,
    })

coverage_df = pd.DataFrame(coverage_rows).set_index("Ticker")
st.dataframe(coverage_df)

if partial_tickers:
    st.warning(
        f"**Excluded (partial history):** {', '.join(partial_tickers)}  \n"
        f"These tickers only have data for part of {start_year}–{end_year}. "
        f"Their short window (often just post-IPO) makes the optimizer "
        f"think they are better than they are. Excluded for a fair comparison."
    )

if missing_tickers:
    st.error(
        f"**Excluded (no data):** {', '.join(missing_tickers)}  \n"
        f"These tickers returned no usable price data for {start_year}–{end_year}. "
        f"They either did not exist yet or were delisted."
    )

if not full_tickers:
    st.error(
        f"No tickers have sufficient data for {start_year}–{end_year}. "
        f"Try a more recent start year or different tickers."
    )
    st.stop()

# ── Use only full-coverage tickers ──
tickers = full_tickers
prices  = raw_prices[tickers].dropna()
returns = prices.pct_change().dropna()

st.success(f"Running analysis on: **{', '.join(tickers)}**  |  {start_date} → {end_date}  |  {len(prices)} trading days")

# ══════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════
def portfolio_stats(weights, mean_ret, cov, ret_series, rf=0.0):
    w          = np.array(list(weights.values()))
    port_ret   = float(w @ mean_ret)
    port_vol   = float(np.sqrt(w @ cov @ w.T))
    sharpe     = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    port_daily = ret_series[list(weights.keys())] @ w
    var_95     = np.percentile(port_daily, 5)
    cvar_95    = port_daily[port_daily <= var_95].mean() if (port_daily <= var_95).any() else var_95
    return port_ret, port_vol, sharpe, var_95, cvar_95


def compute_technicals(price_series):
    s     = price_series.dropna()
    score = 0
    details = {}

    # RSI (14-day)
    delta   = s.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / loss.replace(0, np.nan)
    rsi     = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1] if not rsi.empty else 50
    if rsi_val < 30:
        score += 1
        details["RSI"] = f"{rsi_val:.1f} — Oversold ↑"
    elif rsi_val > 70:
        score -= 1
        details["RSI"] = f"{rsi_val:.1f} — Overbought ↓"
    else:
        details["RSI"] = f"{rsi_val:.1f} — Neutral"

    # 50-day MA
    if len(s) >= 50:
        ma50 = s.rolling(50).mean().iloc[-1]
        if s.iloc[-1] > ma50:
            score += 1
            details["50MA"] = "Price above — Bullish ↑"
        else:
            score -= 1
            details["50MA"] = "Price below — Bearish ↓"
    else:
        details["50MA"] = "Not enough data"

    # MACD (12/26/9)
    if len(s) >= 35:
        ema12  = s.ewm(span=12, adjust=False).mean()
        ema26  = s.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        if macd.iloc[-1] > signal.iloc[-1]:
            score += 1
            details["MACD"] = "Bullish crossover ↑"
        else:
            score -= 1
            details["MACD"] = "Bearish crossover ↓"
    else:
        details["MACD"] = "Not enough data"

    if score >= 2:
        combined = "🟢 Strong"
    elif score <= -2:
        combined = "🔴 Weak"
    else:
        combined = "🟡 Neutral"

    return combined, score, details


def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "Neutral", 0.0, []
        headlines = []
        for item in news[:10]:
            title = item.get("title") or item.get("content", {}).get("title", "") or ""
            if title:
                headlines.append(title)
        if not headlines:
            return "Neutral", 0.0, []
        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg    = round(np.mean(scores), 3)
        label  = "Positive" if avg >= 0.05 else ("Negative" if avg <= -0.05 else "Neutral")
        return label, avg, headlines
    except Exception:
        return "Neutral", 0.0, []


def rank_stocks(overview_rows):
    ranked = []
    for row in overview_rows:
        tech_score        = row["Tech Score"]
        sent_label        = row["Sentiment"]
        sent_pts          = 1 if sent_label == "Positive" else (-1 if sent_label == "Negative" else 0)
        final_score       = tech_score + sent_pts

        if final_score >= 3:
            verdict = "🟢 Strong Buy"
        elif final_score == 2:
            verdict = "🟢 Buy"
        elif final_score == 1:
            verdict = "🟡 Mild Buy"
        elif final_score == 0:
            verdict = "🟡 Neutral"
        elif final_score == -1:
            verdict = "🟠 Mild Avoid"
        elif final_score == -2:
            verdict = "🔴 Avoid"
        else:
            verdict = "🔴 Strong Avoid"

        ranked.append({
            "Ticker":           row["Ticker"],
            "Technical Signal": row["Technical Signal"],
            "Tech Score":       tech_score,
            "Sentiment":        sent_label,
            "Sentiment Points": sent_pts,
            "Final Score":      final_score,
            "Verdict":          verdict,
        })

    return pd.DataFrame(ranked).set_index("Ticker").sort_values("Final Score", ascending=False)

# ══════════════════════════════════════════
# OPTIMIZATION MODELS
# ══════════════════════════════════════════
mean_ret_hist = returns.mean() * 252
cov_hist      = returns.cov() * 252
eq_weights    = dict(zip(tickers, [1 / len(tickers)] * len(tickers)))

mu = expected_returns.ema_historical_return(prices, span=252)
S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# Model 1: Max-Sharpe
try:
    ef_ms = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_ms.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef_ms.max_sharpe(risk_free_rate=risk_free)
    w_ms = ef_ms.clean_weights()
except ppo_exc.OptimizationError:
    w_ms = eq_weights.copy()

# Model 2: Min-Vol
try:
    ef_mv = EfficientFrontier(mu, S, weight_bounds=(0.0, max_weight))
    ef_mv.min_volatility()
    w_mv = ef_mv.clean_weights()
except ppo_exc.OptimizationError:
    w_mv = eq_weights.copy()

# Model 3: Risk-Parity
asset_vol = returns.std() * np.sqrt(252)
inv_vol   = 1 / asset_vol.replace(0, np.nan)
inv_vol   = inv_vol / inv_vol.sum()
inv_vol   = np.clip(inv_vol, 0, max_weight)
inv_vol   = inv_vol / inv_vol.sum()
w_rp      = inv_vol.to_dict()

# Model 4: Momentum
try:
    n = len(prices)
    if n >= 42:
        momentum = (prices.iloc[-21] / prices.iloc[0]) - 1
        momentum = momentum.clip(lower=0)
        if momentum.sum() == 0:
            w_mom = eq_weights.copy()
        else:
            mom_w = momentum / momentum.sum()
            mom_w = np.clip(mom_w, 0, max_weight)
            mom_w = mom_w / mom_w.sum()
            w_mom = mom_w.to_dict()
    else:
        w_mom = eq_weights.copy()
except Exception:
    w_mom = eq_weights.copy()

# Ensemble
w_ens = {t: (w_ms.get(t,0) + w_mv.get(t,0) + w_rp.get(t,0) + w_mom.get(t,0)) / 4 for t in tickers}
w_arr = np.array(list(w_ens.values()))
w_arr = np.minimum(w_arr, max_weight)
w_arr = w_arr / w_arr.sum()
w_ens = dict(zip(tickers, w_arr))

ret_ens, vol_ens, sharpe_ens, var_95_ens, cvar_95_ens = portfolio_stats(
    w_ens, mean_ret_hist, cov_hist, returns, rf=risk_free
)

# ══════════════════════════════════════════
# SENTIMENT
# ══════════════════════════════════════════
analyzer     = SentimentIntensityAnalyzer()
all_headlines = {}
overview_rows = []

for ticker in tickers:
    combined, score, detail    = compute_technicals(prices[ticker])
    sent_label, sent_score, hl = get_sentiment(ticker)
    all_headlines[ticker]      = hl
    overview_rows.append({
        "Ticker":           ticker,
        "Technical Signal": combined,
        "Tech Score":       score,
        "RSI":              detail.get("RSI",  "N/A"),
        "50MA":             detail.get("50MA", "N/A"),
        "MACD":             detail.get("MACD", "N/A"),
        "Sentiment":        sent_label,
        "Sentiment Score":  sent_score,
    })

# ══════════════════════════════════════════
# SEASONALITY
# ══════════════════════════════════════════
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
best_month_df = pd.DataFrame({
    "Best Month":               avg_by_month.idxmax(),
    "Avg Return % in that month": avg_by_month.max().round(2),
})

# ══════════════════════════════════════════════════════
# ═══════════════ OUTPUT SECTIONS ═════════════════════
# ══════════════════════════════════════════════════════

# ── Section 1: Ensemble allocation ──
st.markdown("---")
st.markdown("## Portfolio Allocation — Ensemble of 4 Models")
weights_df = pd.DataFrame({"Weight %": [w * 100 for w in w_ens.values()]}, index=tickers)
st.bar_chart(weights_df["Weight %"])

# ── Section 2: Weight breakdown ──
st.markdown("## Weight Breakdown by Model (%)")
model_weights = pd.DataFrame({
    "Max-Sharpe":  pd.Series(w_ms),
    "Min-Vol":     pd.Series(w_mv),
    "Risk-Parity": pd.Series(w_rp),
    "Momentum":    pd.Series(w_mom),
    "Ensemble":    pd.Series(w_ens),
}) * 100
model_weights.index.name = "Ticker"
st.dataframe(model_weights.round(2))

# ── Section 3: Expected performance ──
st.markdown("## Expected Performance (not guaranteed)")
perf_df = pd.DataFrame({
    "Metric": [
        "Expected annual return",
        "Expected annual volatility",
        "Expected Sharpe ratio",
        "95% daily VaR",
        "95% daily CVaR (ES)",
    ],
    "Value": [
        f"{ret_ens*100:.2f}%",
        f"{vol_ens*100:.2f}%",
        f"{sharpe_ens:.3f}",
        f"{var_95_ens*100:.2f}%",
        f"{cvar_95_ens*100:.2f}%",
    ],
})
st.table(perf_df)

# ── Section 4: Technical + Sentiment per stock ──
st.markdown("---")
st.markdown("## Technical & Sentiment Overview")
st.caption("Technical = RSI + 50MA + MACD combined. Sentiment = latest Yahoo Finance headlines.")

overview_df = pd.DataFrame([
    {k: v for k, v in row.items() if k != "Ticker"} for row in overview_rows
], index=[row["Ticker"] for row in overview_rows])
overview_df.index.name = "Ticker"

def color_signal(val):
    v = str(val)
    if "Strong" in v or val == "Positive":
        return "background-color: #d4edda; color: #155724"
    elif "Weak" in v or val == "Negative":
        return "background-color: #f8d7da; color: #721c24"
    elif "Neutral" in v:
        return "background-color: #fff3cd; color: #856404"
    return ""

st.dataframe(
    overview_df.style.applymap(color_signal, subset=["Technical Signal", "Sentiment"])
)

for ticker in tickers:
    if all_headlines.get(ticker):
        with st.expander(f"Headlines — {ticker}"):
            for i, h in enumerate(all_headlines[ticker], 1):
                st.write(f"{i}. {h}")

# ── Section 5: Stock Ranking & Top 3 ──
st.markdown("---")
st.markdown("## Stock Ranking — Technical + Sentiment Score")
st.caption("Final Score = Tech Score (-3 to +3) + Sentiment Points (-1, 0, +1). Range: -4 to +4.")

ranked_df = rank_stocks(overview_rows)

def highlight_top3(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    top3   = df["Final Score"].nlargest(3).index
    for ticker in top3:
        styles.loc[ticker] = "background-color: #d4edda; color: #155724; font-weight: bold"
    return styles

st.dataframe(ranked_df.style.apply(highlight_top3, axis=None))

# ── Top 3 cards ──
st.markdown("### Top 3 Recommended Stocks")
top3 = ranked_df["Final Score"].nlargest(3).index.tolist()
cols = st.columns(3)

for i, ticker in enumerate(top3):
    row = ranked_df.loc[ticker]
    with cols[i]:
        st.metric(
            label = f"#{i+1}  {ticker}",
            value = row["Verdict"],
            delta = f"Score: {int(row['Final Score'])} / 4"
        )
        st.write(f"**Technical:**  {row['Technical Signal']}  ({row['Tech Score']} / 3)")
        st.write(f"**Sentiment:**  {row['Sentiment']}  ({'+' if row['Sentiment Points'] > 0 else ''}{row['Sentiment Points']})")
        # show ensemble weight for context
        ens_w = w_ens.get(ticker, 0) * 100
        st.write(f"**Model weight:** {ens_w:.1f}%")

# ── Section 6: Seasonality ──
st.markdown("---")
st.markdown("## Best Calendar Month per Stock")
st.table(best_month_df)

st.markdown("## Seasonality Heatmap — Avg Monthly Return %")
st.dataframe(
    avg_by_month.style
    .background_gradient(cmap="RdYlGn", axis=None)
    .format("{:.2f}%")
)
