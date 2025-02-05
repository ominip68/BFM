import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests

st.set_page_config(layout="wide") # Make the content fit the entire screen

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css("styles.css")

def fetch_stock_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2025-01-25")
    return df

def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    
    dates = pd.date_range(start="2020-01-01", end="2025-01-25", freq='D')
    fundamental_data = []
    
    for date in dates:
        try:
            total_revenue = financials.loc["Total Revenue"].get(date.strftime("%Y-%m-%d"), None) if "Total Revenue" in financials.index else None
            debt_to_equity = (balance_sheet.loc["Total Debt"].get(date.strftime("%Y-%m-%d"), None) / balance_sheet.loc["Total Equity"].get(date.strftime("%Y-%m-%d"), None)) if ("Total Debt" in balance_sheet.index and "Total Equity" in balance_sheet.index) else None
            net_cashflow = cashflow.loc["Total Cash From Operating Activities"].get(date.strftime("%Y-%m-%d"), None) if "Total Cash From Operating Activities" in cashflow.index else None
        except Exception:
            total_revenue, debt_to_equity, net_cashflow = None, None, None
        
        data = {
            "Date": date,
            "Market Cap": info.get("marketCap"),
            "Enterprise Value": info.get("enterpriseValue"),
            "P/E Ratio": info.get("trailingPE"),
            "Debt-to-Equity Ratio": debt_to_equity,
            "Total Revenue": total_revenue,
            "Net Cash Flow": net_cashflow
        }
        fundamental_data.append(data)
    
    return pd.DataFrame(fundamental_data)

def fetch_live_news(api_key, query):
    url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles'] if 'articles' in news_data else []

def fetch_eps_pe_ipo_kpi(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    data = {
        "EPS": info.get("trailingEps"),
        "PE Ratio": info.get("trailingPE"),
        "IPO Date": info.get("ipoDate"),
        "KPI": info.get("kpi"),
        "Current Price": info.get("regularMarketPrice")
    }
    return data

def fetch_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = info.get("longBusinessSummary", "No information available.")
    return summary

# Load opening price data for prediction
def load_opening_price_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    hist = hist[hist.index <= '2025-01-25']  # Limit data till 25th January 2025
    hist.reset_index(inplace=True)
    hist['Year'] = hist['Date'].dt.year
    hist['Month'] = hist['Date'].dt.month
    hist['Day'] = hist['Date'].dt.day
    hist['Opening Price'] = hist['Open']  # Assuming 'Open' prices as 'Opening Price'
    return hist

# Normalize Data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Opening Price']])
    return data_scaled, scaler

# Split Data
def split_data(data_scaled):
    train_size = int(len(data_scaled) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]
    return train_data, test_data

# Create Sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train LSTM Model
def train_lstm_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    return model

# Make Predictions
def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Plot predictions
def plot_predictions(actual_prices, predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual_prices.flatten(), mode="lines", name="Actual Price"))
    fig.add_trace(go.Scatter(y=predictions.flatten(), mode="lines", name="Predicted Price", line=dict(color="red")))
    fig.update_layout(title=title, xaxis_title="Days", yaxis_title="Price (₹)", template="plotly_dark")
    st.plotly_chart(fig)

# Main function
def main():
    st.title("📈 Stock Market Dashboard")

    # Sidebar
    st.sidebar.header("Select Company")
    companies = {
        "Adani Energy": "ADANIGREEN.NS",
        "Tata Power": "TATAPOWER.NS",
        "Jsw Energy": "JSWENERGY.NS",
        "NTPC": "NTPC.NS",
        "Power Grid Corp": "POWERGRID.NS",
        "NHPC": "NHPC.NS"
    }
    company = st.sidebar.selectbox("Choose a company", list(companies.keys()))
    ticker = companies[company]

    col1, col2, col3 = st.columns([3, 1.5, 1.5])

    with col1:
        st.subheader("Opening Price Prediction")

        opening_price_data = load_opening_price_data(ticker)
        data_scaled, scaler = normalize_data(opening_price_data)
        train_data, test_data = split_data(data_scaled)
        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        model = build_lstm_model((X_train.shape[1], 1))
        model = train_lstm_model(model, X_train, y_train, X_test, y_test)
        predictions = make_predictions(model, X_test, scaler)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        plot_predictions(actual_prices, predictions, "Daily Opening Price Prediction")
        
        st.subheader("Opening Price Data")
        filtered_data = opening_price_data[opening_price_data['Year'] == 2025]
        st.dataframe(filtered_data, height=200)

    with col2:
        st.subheader(f"About {company}")
        company_info = fetch_company_info(ticker)
        st.write(company_info)

        st.subheader(f"{company} Performance")
        df_stock = fetch_stock_data(ticker)
        year_data = df_stock[df_stock.index.year == 2025]
        st.slider("Volume Traded", min_value=int(year_data['Volume'].min()), max_value=int(year_data['Volume'].max()), value=int(year_data['Volume'].mean()), step=1)

    with col3:
        st.subheader("Live News")
        news_api_key = "31739ed855eb4759908a898ab99a43e7"
        query = company
        news_articles = fetch_live_news(news_api_key, query)
        news_text = ""
        for article in news_articles:
            news_text += f"{article['title']}\n\n{article['description']}\n\n[Read more]({article['url']})\n\n\n"
        st.text_area("Live News", news_text, height=150)

        st.subheader(f"{company} EPS, PE, IPO KPI")
        eps_pe_ipo_kpi = fetch_eps_pe_ipo_kpi(ticker)
        kpi_info = f"*EPS: {eps_pe_ipo_kpi['EPS']}  |  **PE Ratio: {eps_pe_ipo_kpi['PE Ratio']}  |  **IPO Date: {eps_pe_ipo_kpi['IPO Date']}  |  **KPI*: {eps_pe_ipo_kpi['KPI']}  |  **Current Price*: {eps_pe_ipo_kpi['Current Price']}"
        st.write(kpi_info)

    st.write("Data fetched successfully! Use this for further analysis and prediction.")

if __name__ == "__main__":
    main()
