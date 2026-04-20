import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from arch import arch_model
from scipy.stats import norm
import pandas as pd
import time

plt.style.use("seaborn-v0_8")

st.set_page_config(
    page_title="Options Analytics Dashboard",
    layout="wide"
)

st.title("📈 Options Analytics Dashboard")
st.caption("Volatility Modeling and Option Pricing")

# -----------------------
# SIDEBAR NAVIGATION
# -----------------------

with st.sidebar:

    selected = option_menu(
        "Navigation",
        ["Options Dashboard", "Volatility Comparison"],
        icons=["graph-up", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# -----------------------
# OPTIONS DASHBOARD PAGE
# -----------------------

if selected == "Options Dashboard":

    st.sidebar.header("Model Inputs")

    ticker = st.sidebar.text_input("Stock Ticker", "TSLA")

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
        st.write("Columns:", data.columns)

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
            S = float(data["Close"].dropna().iloc[-1])
        except:
            st.error("Error reading stock price")
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

            fig, ax = plt.subplots()

            ax.plot(data["Close"])

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")

            st.pyplot(fig)

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

            fig2, ax2 = plt.subplots()

            ax2.plot(volatility)

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Volatility")

            st.pyplot(fig2)

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

            fig3, ax3 = plt.subplots()

            ax3.plot(stock_prices,call_payoff,label="Call Payoff")
            ax3.plot(stock_prices,put_payoff,label="Put Payoff")

            ax3.axhline(0)
            ax3.axvline(strike_price,linestyle="--")

            ax3.set_xlabel("Stock Price at Expiration")
            ax3.set_ylabel("Profit / Loss")

            ax3.legend()

            st.pyplot(fig3)

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

            st.subheader("Monte Carlo Simulation (Future Price Paths)")

            n_simulations = st.slider("Number of Simulations", 100, 2000, 300)
            steps = 252  # daily steps for 1 year
            dt = time_to_maturity / steps

            paths = []

            for _ in range(n_simulations // 2):

                prices1 = [S]
                prices2 = [S]

                for _ in range(steps):

                    shock = np.random.normal(0, 1)

                    shock_antithetic = -shock

                    price1 = prices1[-1] * np.exp(
                        (risk_free_rate - 0.5 * sigma**2) * dt
                        + sigma * np.sqrt(dt) * shock
                    )

                    price2 = prices2[-1] * np.exp(
                        (risk_free_rate - 0.5 * sigma**2) * dt
                        + sigma * np.sqrt(dt) * shock_antithetic
                    )

                    prices1.append(price1)
                    prices2.append(price2)

                paths.append(prices1)
                paths.append(prices2)

            if len(paths) < n_simulations:
                paths.append(paths[-1])
            # -----------------------
            # FINAL PRICE DISTRIBUTION
            # -----------------------
            
            final_prices = np.array([path[-1] for path in paths])
            
            mc_prices = []

            for i in range(50, len(final_prices), 50):
                temp_payoffs = np.maximum(final_prices[:i] - strike_price, 0)
                temp_price = np.exp(-risk_free_rate*time_to_maturity) * np.mean(temp_payoffs)
                mc_prices.append(temp_price)

            fig_conv, ax_conv = plt.subplots()
            ax_conv.plot(range(50, len(final_prices), 50), mc_prices)

            ax_conv.set_title("Monte Carlo Convergence")
            ax_conv.set_xlabel("Number of Simulations")
            ax_conv.set_ylabel("Option Price")

            st.pyplot(fig_conv)
            
            

            # -----------------------
            # PLOT SIMULATED PATHS
            # -----------------------

            fig_mc, ax_mc = plt.subplots()

            for path in paths:
                ax_mc.plot(path, alpha=0.3)

            ax_mc.set_title("Simulated Future Price Paths")
            ax_mc.set_xlabel("Time Steps")
            ax_mc.set_ylabel("Stock Price")

            st.pyplot(fig_mc)

            

            # -----------------------
            # MONTE CARLO OPTION PRICING
            # -----------------------

            call_payoffs = np.maximum(final_prices - strike_price, 0)
            put_payoffs = np.maximum(strike_price - final_prices, 0)

            mc_call_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(call_payoffs)
            mc_put_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(put_payoffs)

            st.markdown("### 💰 Monte Carlo Pricing")

            c1, c2 = st.columns(2)
            c1.metric("MC Call Price", round(mc_call_price, 2))
            c2.metric("MC Put Price", round(mc_put_price, 2))
            
            # -----------------------
            # CONFIDENCE INTERVAL
            # -----------------------

            std_error = np.std(call_payoffs) / np.sqrt(len(final_prices))

            ci_lower = mc_call_price - 1.96 * std_error
            ci_upper = mc_call_price + 1.96 * std_error

            st.markdown("### 📊 Confidence Interval (95%)")

            st.write(f"Call Price Range: [{ci_lower:.2f}, {ci_upper:.2f}]")
            
            # -----------------------
            # COMPARISON WITH BLACK-SCHOLES
            # -----------------------

            st.markdown("### ⚖️ Comparison with Black-Scholes")

            c3, c4 = st.columns(2)

            c3.metric(
            "Call Price Difference",
            round(mc_call_price - call_price, 4)
            )

            c4.metric(
            "Put Price Difference",
            round(mc_put_price - put_price, 4)
            )
            
            # -----------------------
            # INTERPRETATION
            # -----------------------

            st.markdown("### 🧠 Interpretation")

            prob_above_strike = np.mean(final_prices > strike_price)

            if prob_above_strike > 0.65:
                st.success("High probability of profit (Call option favorable)")

            elif prob_above_strike > 0.45:
                st.warning("Moderate probability (Risk involved)")

            else:
                st.error("Low probability of profit")

            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(final_prices, bins=30)

            ax_hist.set_title("Distribution of Final Prices")
            ax_hist.set_xlabel("Final Price")
            ax_hist.set_ylabel("Frequency")

            st.pyplot(fig_hist)

            # -----------------------
            # METRICS
            # -----------------------

            st.metric("Average Final Price", round(np.mean(final_prices), 2))

            prob_above_strike = np.mean(final_prices > strike_price)
            st.metric("Probability Price > Strike", round(prob_above_strike, 2))

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
