import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from arch import arch_model
from scipy.stats import norm
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

st_autorefresh(interval=60000)  # refresh every 60 seconds

plt.style.use("seaborn-v0_8")

st.set_page_config(
    page_title="Options Analytics Dashboard",
    layout="wide"
)

st.markdown("""
<style>
/* TradingView Dark Theme Colors */
:root {
    --tv-bg: #131722;
    --tv-panel-bg: #1e222d;
    --tv-border: #2a2e39;
    --tv-text: #d1d4dc;
    --tv-text-muted: #8a919e;
    --tv-up: #26a69a;
    --tv-down: #ef5350;
    --tv-blue: #2962ff;
    --tv-hover: #2a2e39;
}

/* Base App Colors */
[data-testid="stAppViewContainer"] {
    background-color: var(--tv-bg);
    color: var(--tv-text);
}
[data-testid="stHeader"] {
    background-color: var(--tv-bg);
}
[data-testid="stSidebar"] {
    background-color: var(--tv-panel-bg) !important;
    border-right: 1px solid var(--tv-border) !important;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}

/* Custom padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 98% !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6, p, span, div {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Streamlit Metrics (Make them look like TV data windows) */
[data-testid="stMetric"] {
    background-color: var(--tv-panel-bg);
    padding: 15px 20px;
    border-radius: 8px;
    border: 1px solid var(--tv-border);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}
[data-testid="stMetricLabel"] {
    color: var(--tv-text-muted) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* Tabs like TradingView */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background-color: var(--tv-panel-bg);
    border-bottom: 1px solid var(--tv-border);
    border-radius: 8px 8px 0 0;
    padding: 0 10px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 20px;
    background-color: transparent;
    border-radius: 0px;
    color: var(--tv-text-muted);
    font-weight: 600;
    font-size: 14px;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: var(--tv-blue) !important;
    border-bottom: 3px solid var(--tv-blue) !important;
    background-color: transparent !important;
}

/* Inputs & Buttons */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    background-color: var(--tv-bg) !important;
    color: white !important;
    border: 1px solid var(--tv-border) !important;
}
.stButton>button {
    background-color: var(--tv-blue) !important;
    color: white !important;
    border-radius: 4px !important;
    border: none !important;
    width: 100%;
    font-weight: bold;
    height: 40px;
    transition: background-color 0.2s ease;
}
.stButton>button:hover {
    background-color: #1e4bd8 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: var(--tv-panel-bg); padding: 15px 25px; border-radius: 8px; border: 1px solid var(--tv-border); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;'>
    <div style='display: flex; align-items: center; gap: 15px;'>
        <div style='background-color: var(--tv-blue); width: 45px; height: 45px; border-radius: 8px; display: flex; justify-content: center; align-items: center; font-size: 22px;'>📈</div>
        <div>
            <h1 style='margin: 0; font-size: 22px;'>Options Analytics Dashboard</h1>
            <p style='margin: 0; color: var(--tv-text-muted); font-size: 14px;'>Real-time pricing • Monte Carlo simulation • Risk analysis</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


st.caption("Volatility Modeling and Option Pricing")

# -----------------------
# SIDEBAR NAVIGATION
# -----------------------

popular_stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

def get_live_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not data.empty:
            close_vals = data["Close"]
            if isinstance(close_vals, pd.DataFrame):
                return float(close_vals.iloc[-1, 0])
            return float(close_vals.iloc[-1])
    except Exception:
        pass
    return None



with st.sidebar:

    selected = option_menu(
        "Navigation",
        ["Options Dashboard", "Volatility Comparison"],
        icons=["graph-up", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )
    st.sidebar.markdown("### 📊 Live Market")

    for stock in popular_stocks:
        price = get_live_price(stock)
        
        if price:
            st.sidebar.markdown(f"""
            <div style="
                background-color:#1e222d;
                padding:10px;
                border-radius:8px;
                margin-bottom:8px;
                border: 1px solid #2a2e39;
            ">
                <b style="color:white;">{stock}</b><br>
                <span style="color:#26a69a;">${price:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        

# -----------------------
# OPTIONS DASHBOARD PAGE
# -----------------------

if selected == "Options Dashboard":

    # -----------------------
    # SIDEBAR ORGANIZATION
    # -----------------------
    
    st.sidebar.header("⚙️ Model Controls")
    
    st.sidebar.subheader("📊 Stock Selection")
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        st.session_state.get("selected_stock", "TSLA")
    )
    
    st.sidebar.subheader("📈 Model Parameters")
    time_to_maturity = st.sidebar.number_input(
        "Time to Maturity (years)",
        value=0.5
    )

    risk_free_rate = st.sidebar.number_input(
        "Risk Free Rate",
        value=0.05
    )

    run_model = st.sidebar.button("Run Model")

    # -----------------------
    # RUN MODEL
    # -----------------------

    if run_model:

        data = yf.download(ticker, start="2020-01-01", progress=False)
        # st.write("Columns:", data.columns)

# Fix column issue (important for Streamlit Cloud)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

# Retry if empty (cloud fix)
        if data.empty:
            time.sleep(2)
            data = yf.download(ticker, start="2020-01-01", progress=False)

# Final check
        if data.empty or "Close" not in data.columns:
            st.error("Failed to fetch data. Try AAPL, TSLA, RELIANCE.NS")
            st.stop()

        returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

        model = arch_model(returns * 100, vol="Garch", p=1, q=1)
        result = model.fit(disp="off")

        forecast = result.forecast(horizon=1)

        predicted_vol_daily = np.sqrt(forecast.variance.values[-1, :][0]) / 100
        sigma = predicted_vol_daily * np.sqrt(252)

        try:
            close_vals = data["Close"].dropna()
            if isinstance(close_vals, pd.DataFrame):
                S = float(close_vals.iloc[-1, 0])
            else:
                S = float(close_vals.iloc[-1])
        except Exception as e:
            st.error(f"Error reading stock price: {e}")
            st.stop()

        # store results so UI can update without rerunning model
        st.session_state["data"] = data
        st.session_state["sigma"] = sigma
        st.session_state["S"] = S
        st.session_state["model_run"] = True

    # -----------------------
    # IF MODEL HAS BEEN RUN
    # -----------------------

    if "model_run" in st.session_state:

        data = st.session_state["data"]
        sigma = st.session_state["sigma"]
        S = st.session_state["S"]

        st.metric("Current Stock Price", round(S,2))

        strike_price = st.slider(
            "Strike Price",
            min_value=float(S * 0.5),
            max_value=float(S * 1.5),
            value=float(S),
            step=1.0
        )

        # -----------------------
        # BLACK SCHOLES
        # -----------------------

        def black_scholes_call(S, K, T, r, sigma):

            d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

        def black_scholes_put(S, K, T, r, sigma):

            d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

        call_price = black_scholes_call(
            S, strike_price, time_to_maturity, risk_free_rate, sigma
        )

        put_price = black_scholes_put(
            S, strike_price, time_to_maturity, risk_free_rate, sigma
        )

        # -----------------------
        # TABS UI
        # -----------------------

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Pricing",
            "📉 Volatility",
            "⚙ Greeks",
            "💰 Payoff",
            "🔍 Implied Volatility",
            "🎲 Monte Carlo"
        ])

        # -----------------------
        # PRICING TAB
        # -----------------------

        with tab1:

            fig = go.Figure(data=[go.Candlestick(x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
                            decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350')])
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#2a2e39',
                    tickfont=dict(color='#d1d4dc'),
                    title=''
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#2a2e39',
                    tickfont=dict(color='#d1d4dc'),
                    title='',
                    side='right'
                ),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)

            c1.metric("Stock Price", round(S,2))
            c2.metric("Call Price", round(call_price,2))
            c3.metric("Put Price", round(put_price,2))

            st.metric("Predicted Annual Volatility", round(sigma,4))

        # -----------------------
        # VOLATILITY TAB
        # -----------------------

        with tab2:

            volatility = arch_model(
                np.log(data["Close"]/data["Close"].shift(1)).dropna()*100,
                vol="Garch",
                p=1,
                q=1
            ).fit(disp="off").conditional_volatility

            fig2 = go.Figure(data=go.Scatter(x=volatility.index, y=volatility, mode='lines', line=dict(color='#2962ff', width=2)))
            fig2.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#d1d4dc'), title='Time'),
                yaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#d1d4dc'), title='Volatility', side='right'),
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

        # -----------------------
        # GREEKS TAB
        # -----------------------

        with tab3:

            d1 = (
                np.log(S/strike_price) +
                (risk_free_rate + sigma**2/2)*time_to_maturity
            ) / (sigma*np.sqrt(time_to_maturity))

            d2 = d1 - sigma*np.sqrt(time_to_maturity)

            delta = norm.cdf(d1)

            gamma = norm.pdf(d1)/(S*sigma*np.sqrt(time_to_maturity))

            vega = S*norm.pdf(d1)*np.sqrt(time_to_maturity)

            theta = (
                -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(time_to_maturity))
                - risk_free_rate*strike_price
                * np.exp(-risk_free_rate*time_to_maturity)
                * norm.cdf(d2)
            )

            g1,g2,g3,g4 = st.columns(4)

            g1.metric("Delta",round(delta,4))
            g2.metric("Gamma",round(gamma,4))
            g3.metric("Vega",round(vega,4))
            g4.metric("Theta",round(theta,4))

        # -----------------------
        # PAYOFF TAB (LIVE UPDATE)
        # -----------------------

        with tab4:

            stock_prices = np.linspace(S*0.5, S*1.5, 100)

            call_payoff = np.maximum(stock_prices-strike_price,0)-call_price
            put_payoff = np.maximum(strike_price-stock_prices,0)-put_price

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=stock_prices, y=call_payoff, mode='lines', name='Call Payoff', line=dict(color='#26a69a', width=2)))
            fig3.add_trace(go.Scatter(x=stock_prices, y=put_payoff, mode='lines', name='Put Payoff', line=dict(color='#ef5350', width=2)))
            
            fig3.add_hline(y=0, line_color="#666666", line_width=1)
            fig3.add_vline(x=strike_price, line_dash="dash", line_color="#2962ff", line_width=1)

            fig3.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#d1d4dc'), title='Stock Price at Expiration'),
                yaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#d1d4dc'), title='Profit / Loss', side='right'),
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            )
            st.plotly_chart(fig3, use_container_width=True)

        # -----------------------
        # IMPLIED VOL TAB
        # -----------------------

        with tab5:

            market_option_price = st.number_input(
                "Observed Market Option Price",
                value=10.0
            )

            def implied_volatility(S,K,T,r,market_price):

                sigma_iv=0.2

                for i in range(100):

                    d1=(np.log(S/K)+(r+sigma_iv**2/2)*T)/(sigma_iv*np.sqrt(T))
                    d2=d1-sigma_iv*np.sqrt(T)

                    price=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

                    vega=S*np.sqrt(T)*norm.pdf(d1)

                    sigma_iv=sigma_iv-(price-market_price)/vega

                return sigma_iv

            if st.button("Calculate Implied Volatility"):

                iv=implied_volatility(
                    S,
                    strike_price,
                    time_to_maturity,
                    risk_free_rate,
                    market_option_price
                )

                st.write("Implied Volatility:",iv)
        
        # -----------------------
        # MONTE CARLO TAB
        # -----------------------

        with tab6:

            st.markdown("<h2 style='color: white; margin-bottom: 0px;'>🎲 Monte Carlo Simulation Analysis</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #8a919e; font-size: 14px;'>Simulate thousands of possible future price paths using Geometric Brownian Motion.</p>", unsafe_allow_html=True)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            # Top Controls Row
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
            with ctrl_col1:
                n_simulations = st.slider("Number of Simulations", 10, 2000, 500, step=50)
            
            # --- Calculation ---
            steps = 252  # daily steps for 1 year
            dt = time_to_maturity / steps

            paths = []
            for _ in range(n_simulations // 2):
                prices1 = [S]
                prices2 = [S]
                for _ in range(steps):
                    shock = np.random.normal(0, 1)
                    shock_antithetic = -shock
                    price1 = prices1[-1] * np.exp((risk_free_rate - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
                    price2 = prices2[-1] * np.exp((risk_free_rate - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock_antithetic)
                    prices1.append(price1)
                    prices2.append(price2)
                paths.append(prices1)
                paths.append(prices2)

            if len(paths) < n_simulations:
                paths.append(paths[-1])
            
            final_prices = np.array([path[-1] for path in paths])
            
            mc_prices = []
            for i in range(50, len(final_prices), 50):
                temp_payoffs = np.maximum(final_prices[:i] - strike_price, 0)
                temp_price = np.exp(-risk_free_rate*time_to_maturity) * np.mean(temp_payoffs)
                mc_prices.append(temp_price)

            call_payoffs = np.maximum(final_prices - strike_price, 0)
            put_payoffs = np.maximum(strike_price - final_prices, 0)

            mc_call_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(call_payoffs)
            mc_put_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(put_payoffs)

            std_error = np.std(call_payoffs) / np.sqrt(len(final_prices))
            ci_lower = mc_call_price - 1.96 * std_error
            ci_upper = mc_call_price + 1.96 * std_error

            prob_above_strike = np.mean(final_prices > strike_price)

            with ctrl_col2:
                st.markdown(f"""
                <div style="background-color:#1e222d; padding:15px; border-radius:8px; border: 1px solid #2a2e39;">
                    <div style="color:#8a919e; font-size:12px; font-weight:600; text-transform:uppercase;">MC Call Price</div>
                    <div style="color:#26a69a; font-size:24px; font-weight:bold;">${mc_call_price:.2f}</div>
                    <div style="color:#d1d4dc; font-size:12px;">95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]</div>
                </div>
                """, unsafe_allow_html=True)

            with ctrl_col3:
                st.markdown(f"""
                <div style="background-color:#1e222d; padding:15px; border-radius:8px; border: 1px solid #2a2e39;">
                    <div style="color:#8a919e; font-size:12px; font-weight:600; text-transform:uppercase;">MC Put Price</div>
                    <div style="color:#ef5350; font-size:24px; font-weight:bold;">${mc_put_price:.2f}</div>
                    <div style="color:#d1d4dc; font-size:12px;">Diff vs BS: ${(mc_put_price - put_price):.4f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Chart Row 1
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown("<h4 style='color: #d1d4dc; margin-bottom: 10px; font-size: 16px;'>🎯 Simulation Paths</h4>", unsafe_allow_html=True)
                fig_mc = go.Figure()
                # Limit paths for performance/visibility
                display_paths = paths[:200] if len(paths) > 200 else paths
                for path in display_paths: 
                    fig_mc.add_trace(go.Scatter(y=path, mode='lines', line=dict(color='#2962ff', width=1), opacity=0.05, showlegend=False))
                
                # Add strike price line
                fig_mc.add_hline(y=strike_price, line_dash="dash", line_color="#ef5350", annotation_text="Strike", annotation_position="bottom right", annotation_font_color="#ef5350")
                
                fig_mc.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#8a919e'), title='Trading Days'),
                    yaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#8a919e'), title='Stock Price', side='right'),
                    height=350
                )
                st.plotly_chart(fig_mc, use_container_width=True)

            with c2:
                st.markdown("<h4 style='color: #d1d4dc; margin-bottom: 10px; font-size: 16px;'>📊 Price Distribution</h4>", unsafe_allow_html=True)
                fig_hist = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=40, marker_color='#26a69a', opacity=0.8, marker_line_width=1, marker_line_color='#1e222d')])
                fig_hist.add_vline(x=strike_price, line_dash="dash", line_color="#ef5350", line_width=2)
                fig_hist.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#8a919e'), title='Final Price'),
                    yaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#8a919e'), title='', showticklabels=False),
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Chart Row 2
            c3, c4 = st.columns([2, 1])

            with c3:
                st.markdown("<h4 style='color: #d1d4dc; margin-bottom: 10px; font-size: 16px;'>🔄 Convergence Analysis</h4>", unsafe_allow_html=True)
                fig_conv = go.Figure(data=[go.Scatter(x=list(range(50, len(final_prices), 50)), y=mc_prices, mode='lines', line=dict(color='#2962ff', width=2))])
                # Add BS Price line for reference
                fig_conv.add_hline(y=call_price, line_dash="dash", line_color="#26a69a", annotation_text="BS Call Price", annotation_position="top right", annotation_font_color="#26a69a")
                
                fig_conv.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#8a919e'), title='Simulations'),
                    yaxis=dict(showgrid=True, gridcolor='#2a2e39', tickfont=dict(color='#8a919e'), title='Option Price', side='right'),
                    height=300
                )
                st.plotly_chart(fig_conv, use_container_width=True)

            with c4:
                st.markdown("<h4 style='color: #d1d4dc; margin-bottom: 10px; font-size: 16px;'>🧠 Insights & Stats</h4>", unsafe_allow_html=True)
                
                if prob_above_strike > 0.65:
                    insight_msg = "High probability of profit at expiration."
                    insight_color = "#26a69a"
                    bg_color = "rgba(38, 166, 154, 0.1)"
                elif prob_above_strike > 0.45:
                    insight_msg = "Moderate probability of profit."
                    insight_color = "#FFD700"
                    bg_color = "rgba(255, 215, 0, 0.1)"
                else:
                    insight_msg = "Low probability of profit at expiration."
                    insight_color = "#ef5350"
                    bg_color = "rgba(239, 83, 80, 0.1)"

                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 15px; border-radius: 8px; border-left: 4px solid {insight_color}; border-top: 1px solid #2a2e39; border-right: 1px solid #2a2e39; border-bottom: 1px solid #2a2e39; margin-bottom: 15px;'>
                    <div style='color: {insight_color}; font-weight: bold; font-size: 14px; margin-bottom: 5px;'>Probability ITM</div>
                    <div style='color: white; font-size: 24px; font-weight: bold;'>{prob_above_strike * 100:.1f}%</div>
                    <div style='color: #8a919e; font-size: 12px; margin-top: 5px;'>{insight_msg}</div>
                </div>
                
                <div style="background-color:#1e222d; padding:15px; border-radius:8px; border: 1px solid #2a2e39;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color:#8a919e; font-size:13px;">Avg Final Price</span>
                        <span style="color:white; font-size:13px; font-weight:bold;">${np.mean(final_prices):.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color:#8a919e; font-size:13px;">Call Diff (vs BS)</span>
                        <span style="color:{'#26a69a' if (mc_call_price - call_price) >= 0 else '#ef5350'}; font-size:13px; font-weight:bold;">${(mc_call_price - call_price):.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color:#8a919e; font-size:13px;">Put Diff (vs BS)</span>
                        <span style="color:{'#26a69a' if (mc_put_price - put_price) >= 0 else '#ef5350'}; font-size:13px; font-weight:bold;">${(mc_put_price - put_price):.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------
# VOLATILITY COMPARISON PAGE
# -----------------------

if selected == "Volatility Comparison":

    st.title("📊 Volatility Comparison Across Stocks")

    compare_tickers=st.text_input(
        "Enter tickers separated by comma",
        "TSLA,AAPL,MSFT"
    )

    if st.button("Compare Volatility"):

        tickers=[t.strip() for t in compare_tickers.split(",")]

        vol_dict={}

        for t in tickers:

            data=yf.download(t,start="2020-01-01")

            if not data.empty:

                returns=np.log(data["Close"]/data["Close"].shift(1)).dropna()

                model=arch_model(returns*100,vol="Garch",p=1,q=1)
                res=model.fit(disp="off")

                forecast=res.forecast(horizon=1)

                vol=np.sqrt(forecast.variance.values[-1,:][0])/100
                vol=vol*np.sqrt(252)

                vol_dict[t]=vol

        st.write("Annual Volatility Estimates")
        st.write(vol_dict)