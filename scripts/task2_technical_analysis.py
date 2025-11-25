import pandas as pd
import yfinance as yf
import talib
import matplotlib
matplotlib.use('Agg') # Fix for Windows backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= CONFIGURATION =================
INPUT_PATH = 'data/raw_analyst_ratings.csv' 
OUTPUT_FOLDER = 'reports/figures_task2'
TOP_N = 20 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
# =================================================

def calculate_financial_metrics(df):
    """
    Implements PyNance-style financial metrics using Pandas.
    Calculates: Daily Returns, Cumulative Returns, and Volatility.
    """
    # 1. Daily Returns (PyNance: pn.metrics.daily_returns)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 2. Cumulative Returns (PyNance: pn.metrics.cumulative_returns)
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
    
    # 3. Volatility (PyNance: pn.metrics.volatility)
    # Using a 20-day rolling window for annualized volatility approximation
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    return df

def analyze_top_stocks():
    print(f"--- Starting Task 2: Technical & Financial Analysis (Top {TOP_N}) ---")
    
    # 1. Load Data & Find Top Stocks
    try:
        df = pd.read_csv(INPUT_PATH)
        df['stock'] = df['stock'].astype(str).str.strip().str.upper()
        top_tickers = df['stock'].value_counts().head(TOP_N).index.tolist()
        print(f"   Top Tickers: {top_tickers}")
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        return

    # 2. Process Each Stock
    for ticker in top_tickers:
        print(f"   Processing {ticker}...", end=" ")
        
        try:
            # Map Ticker
            search_ticker = 'META' if ticker == 'FB' else ticker
            
            # Fetch Data
            stock_df = yf.download(search_ticker, period="2y", progress=False)
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)
            
            if len(stock_df) < 50:
                print("Insufficient data.")
                continue

            # --- A. TECHNICAL ANALYSIS (TA-Lib) ---
            stock_df['SMA_20'] = talib.SMA(stock_df['Close'], timeperiod=20)
            stock_df['SMA_50'] = talib.SMA(stock_df['Close'], timeperiod=50)
            stock_df['RSI'] = talib.RSI(stock_df['Close'], timeperiod=14)
            macd, signal, hist = talib.MACD(stock_df['Close'])
            stock_df['MACD'] = macd
            stock_df['MACD_Signal'] = signal

            # --- B. FINANCIAL METRICS (PyNance Logic) ---
            # Explicitly calculating returns and volatility as requested
            stock_df = calculate_financial_metrics(stock_df)

            # --- C. VISUALIZATION ---
            plt.figure(figsize=(12, 12))
            
            # Plot 1: Price & SMA
            plt.subplot(4, 1, 1) # Changed to 4 rows to show Volatility
            plt.plot(stock_df.index, stock_df['Close'], label='Close')
            plt.plot(stock_df.index, stock_df['SMA_50'], label='SMA 50')
            plt.title(f'{ticker} Technical Analysis')
            plt.legend()
            
            # Plot 2: RSI
            plt.subplot(4, 1, 2)
            plt.plot(stock_df.index, stock_df['RSI'], color='purple', label='RSI')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.legend()
            
            # Plot 3: MACD
            plt.subplot(4, 1, 3)
            plt.plot(stock_df.index, stock_df['MACD'], label='MACD')
            plt.plot(stock_df.index, stock_df['MACD_Signal'], label='Signal')
            plt.legend()

            # Plot 4: Volatility (Financial Metric)
            plt.subplot(4, 1, 4)
            plt.plot(stock_df.index, stock_df['Volatility'], color='orange', label='Annualized Volatility')
            plt.title('Volatility (Risk Metric)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_FOLDER}/{ticker}_analysis.png")
            plt.close('all')
            print("Done.")
            
        except Exception as e:
            print(f"Error: {e}")

    print("\nTask 2 Complete. Financial Metrics & TA calculated.")

if __name__ == "__main__":
    analyze_top_stocks()