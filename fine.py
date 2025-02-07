import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from nsetools import Nse
import plotly.graph_objects as go

class IndianStockScreener:
    def __init__(self):
        self.indices = {
            'SENSEX': '^BSESN',
            'NIFTY 50': '^NSEI'
        }
        
        try:
            self.nse = Nse()
        except Exception as e:
            st.error("Error initializing NSE API: " + str(e))
        
        # Extended list of established stocks
        self.established_stocks = {
            'TCS.BO': 'Tata Consultancy Services',
            'RELIANCE.BO': 'Reliance Industries',
            'HDFCBANK.BO': 'HDFC Bank',
            'INFY.BO': 'Infosys',
            'HINDUNILVR.BO': 'Hindustan Unilever',
            'ICICIBANK.BO': 'ICICI Bank',
            'ITC.BO': 'ITC Limited',
            'KOTAKBANK.BO': 'Kotak Mahindra Bank',
            'BHARTIARTL.BO': 'Bharti Airtel',
            'ASIANPAINT.BO': 'Asian Paints',
            'MARUTI.BO': 'Maruti Suzuki',
            'ULTRACEMCO.BO': 'UltraTech Cement',
            'TITAN.BO': 'Titan Company',
            'BAJFINANCE.BO': 'Bajaj Finance',
            'LT.BO': 'Larsen & Toubro',
            'WIPRO.BO': 'Wipro Limited',
            'HCLTECH.BO': 'HCL Technologies',
            'NESTLEIND.BO': 'Nestle India',
            'BAJAJFINSV.BO': 'Bajaj Finserv',
            'SUNPHARMA.BO': 'Sun Pharmaceutical',
            'AXISBANK.BO': 'Axis Bank',
            'ADANIENT.BO': 'Adani Enterprises',
            'TATASTEEL.BO': 'Tata Steel',
            'NTPC.BO': 'NTPC Limited',
            'M&M.BO': 'Mahindra & Mahindra'
        }
        
        # Extended list of emerging stocks
        self.emerging_stocks = {
            'ZOMATO.BO': 'Zomato',
            'NYKAA.BO': 'FSN E-Commerce',
            'PAYTM.BO': 'Paytm',
            'POLICYBZR.BO': 'PB Fintech',
            'DELHIVERY.BO': 'Delhivery',
            'STARHEALTH.BO': 'Star Health Insurance',
            'MEDANTA.BO': 'Global Health',
            'CAMPUS.BO': 'Campus Activewear',
            'DREAMFOLKS.BO': 'Dreamfolks Services',
            'SULA.BO': 'Sula Vineyards',
            'LANDMARK.BO': 'Landmark Cars',
            'KAYNES.BO': 'Kaynes Technology',
            'SAPPHIRE.BO': 'Sapphire Foods',
            'RAINBOW.BO': 'Rainbow Children',
            'LATENTVIEW.BO': 'LatentView Analytics',
            'METRO.BO': 'Metro Brands',
            'DATAPATTERNS.BO': 'Data Patterns',
            'BIKAJI.BO': 'Bikaji Foods',
            'FINOPB.BO': 'Fino Payments Bank',
            'SYRMA.BO': 'Syrma SGS Technology',
            'NAZARA.BO': 'Nazara Technologies',
            'EASEMYTRIP.BO': 'Easy Trip Planners',
            'CARTRADE.BO': 'CarTrade Tech',
            'AETHER.BO': 'Aether Industries',
            'JUSTDIAL.BO': 'Just Dial'
        }
        
    
    def predict_stock_price(self, symbol, timeline_months):
        try:
            stock = yf.Ticker(symbol)
            history = stock.history(period='2y')
            
            if len(history) < 24:
                return None, None, None
            
            X = np.arange(len(history)).reshape(-1, 1)
            y = history['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_days = timeline_months * 30
            future_price = model.predict([[len(history) + future_days]])[0]
            
            confidence = 0.95
            n = len(history)
            std_err = np.sqrt(np.sum((y - model.predict(X))**2) / (n-2))
            confidence_interval = stats.t.ppf((1 + confidence) / 2, n-2) * std_err
            
            # Calculate R-squared for prediction quality
            r_squared = model.score(X, y)
            
            return future_price, confidence_interval, r_squared
        except Exception as e:
            return None, None, None

    def generate_investment_plans(self, investment_amount, timeline_months):
        plans = {
            "Conservative": {"stocks": {}, "expected_return": 0, "final_value": 0, "value_gained": 0, "monthly_growth": []},
            "Balanced": {"stocks": {}, "expected_return": 0, "final_value": 0, "value_gained": 0, "monthly_growth": []},
            "Aggressive": {"stocks": {}, "expected_return": 0, "final_value": 0, "value_gained": 0, "monthly_growth": []}
        }
        
        plan_allocations = {
            "Conservative": (self.established_stocks, 0.8, 0.2, 8, 2),
            "Balanced": (self.established_stocks, 0.5, 0.5, 5, 5),
            "Aggressive": (self.emerging_stocks, 0.3, 0.7, 3, 7)
        }
        
        for plan_name, (stock_list, est_weight, emg_weight, est_count, emg_count) in plan_allocations.items():
            established_sample = dict(list(self.established_stocks.items())[:est_count])
            emerging_sample = dict(list(self.emerging_stocks.items())[:emg_count])
            
            valid_stocks = {}
            monthly_values = [investment_amount]
            
            # Filter and sort stocks by expected return
            potential_stocks = []
            
            for symbol, name in {**established_sample, **emerging_sample}.items():
                future_price, confidence, r_squared = self.predict_stock_price(symbol, timeline_months)
                stock = yf.Ticker(symbol)
                current_price = stock.info.get('currentPrice', None)
                
                if future_price and current_price and r_squared > 0.6:  # Only include stocks with good prediction quality
                    expected_return = ((future_price - current_price) / current_price) * 100
                    if expected_return > 0:  # Only include stocks with positive expected returns
                        potential_stocks.append((symbol, name, current_price, future_price, confidence, expected_return, r_squared))
            
            # Sort by expected return and take top stocks
            potential_stocks.sort(key=lambda x: x[5], reverse=True)
            selected_stocks = potential_stocks[:est_count + emg_count]
            
            if selected_stocks:
                investment_per_stock = investment_amount / len(selected_stocks)
                
                for symbol, name, current_price, future_price, confidence, expected_return, r_squared in selected_stocks:
                    final_stock_value = investment_per_stock * (1 + expected_return / 100)
                    
                    valid_stocks[symbol] = {
                        "name": name,
                        "current_price": current_price,
                        "predicted_price": future_price,
                        "confidence_interval": confidence,
                        "expected_return": expected_return,
                        "prediction_quality": r_squared * 100,
                        "symbol": symbol,
                        "investment_amount": investment_per_stock,
                        "final_stock_value": final_stock_value
                    }
                    
                    # Calculate monthly growth trajectory
                    for month in range(1, timeline_months + 1):
                        monthly_value = investment_per_stock * (1 + (expected_return / 100) * (month / timeline_months))
                        if len(monthly_values) <= month:
                            monthly_values.append(monthly_value)
                        else:
                            monthly_values[month] += monthly_value

            plans[plan_name]["stocks"] = valid_stocks
            if valid_stocks:
                plans[plan_name]["expected_return"] = sum(stock["expected_return"] for stock in valid_stocks.values()) / len(valid_stocks)
                plans[plan_name]["final_value"] = sum(stock["final_stock_value"] for stock in valid_stocks.values())
                plans[plan_name]["value_gained"] = plans[plan_name]["final_value"] - investment_amount
                plans[plan_name]["monthly_growth"] = monthly_values
        
        return plans

def plot_growth_trajectory(monthly_values, investment_amount, plan_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(monthly_values))),
        y=monthly_values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#00b341', width=2)  # Changed to green
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, len(monthly_values)-1],
        y=[investment_amount] * 2,
        mode='lines',
        name='Initial Investment',
        line=dict(color="#00b341", dash='dash', width=2)  # Changed to black
    ))
    
    fig.update_layout(
        title=f"{plan_name} Growth Trajectory",
        xaxis_title="Months",
        yaxis_title="Portfolio Value (₹)",
        showlegend=True,
        height=400,
        template="plotly_white",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black')
    )
    
    return fig

def main():
    st.set_page_config(page_title="Indian Stock Recommender", layout="wide")
    
    # Custom CSS for light theme with black and green accents
    st.markdown("""
        <style>
        .stApp {
            background-color: black;
            color: white;
        }
        .stButton>button {
            background-color: #00b341;
            color: white;
        }
        .stMetric {
            background-color: black;
            padding: 10px;
            border-radius: 5px;
        }
        h1, h2, h3 {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Investment Recommendation System")
    st.write("Get personalized investment plans with positive return predictions!")
    
    screener = IndianStockScreener()

    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input("Enter amount to invest (₹)", min_value=1000, step=1000, value=10000)
    with col2:
        timeline_months = st.number_input("Enter investment timeline in months", min_value=1, max_value=24, step=1, value=12)

    if st.button("Generate Investment Plans 🚀"):
        with st.spinner("Analyzing market data and generating predictions..."):
            plans = screener.generate_investment_plans(investment_amount, timeline_months)

            st.header("💼 Investment Plans")
            
            tabs = st.tabs(["Conservative", "Balanced", "Aggressive"])
            
            for tab, (plan_name, plan_data) in zip(tabs, plans.items()):
                with tab:
                    if not plan_data["stocks"]:
                        st.warning(f"No suitable stocks found for {plan_name} plan based on current predictions.")
                        continue
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{plan_data['expected_return']:.2f}%",
                                 delta=f"{plan_data['expected_return']:.1f}%")
                    with col2:
                        st.metric("Final Value", f"₹{plan_data['final_value']:,.2f}",
                                 delta=f"₹{plan_data['value_gained']:,.2f}")
                    with col3:
                        st.metric("Number of Stocks", f"{len(plan_data['stocks'])}")
                    
                    st.subheader("📈 Growth Trajectory")
                    if plan_data.get("monthly_growth"):
                        fig = plot_growth_trajectory(plan_data["monthly_growth"], investment_amount, plan_name)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📊 Stock Allocation")
                    if plan_data["stocks"]:
                        df = pd.DataFrame(plan_data["stocks"]).T
                        df = df[["name", "current_price", "predicted_price", "expected_return", "prediction_quality", "investment_amount"]]
                        df.columns = ["Company", "Current Price (₹)", "Predicted Price (₹)", "Expected Return (%)", "Prediction Quality (%)", "Investment Amount (₹)"]
                        st.dataframe(df.style.format({
                            "Current Price (₹)": "{:,.2f}",
                            "Predicted Price (₹)": "{:,.2f}",
                            "Expected Return (%)": "{:.2f}",
                            "Prediction Quality (%)": "{:.1f}",
                            "Investment Amount (₹)": "{:,.2f}"
                        }), use_container_width=True)

if __name__ == "__main__":
    main()