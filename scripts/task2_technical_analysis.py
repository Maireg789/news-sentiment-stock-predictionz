import pandas as pd
import yfinance as yf
import talib
import matplotlib
matplotlib.use('Agg') # Prevents crashes
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ================= CONFIGURATION =================
INPUT_PATH = 'data/raw_analyst_ratings.csv' 
OUTPUT_FOLDER = 'reports/figures_task2'
TOP_N = 20 # Analyze top 20 most active stocks

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
# =================================================

def analyze_top_stocks():
    print(f"--- Starting Task 2: Analysis of Top {TOP_N} Stocks ---")
    
    # 1. Identify Top Stocks from News
    print("1. Reading News Data to find active stocks...")
    try:
        df = pd.read_csv(INPUT_PATH)
        # Clean ticker names
        df['stock'] = df['stock'].astype(str).str.strip().str.upper()
        
        # Count frequency
        top_tickers = df['stock'].value_counts().head(TOP_N).index.tolist()
        print(f"   Top {TOP_N} Stocks found: {top_tickers}")
        
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        return

    # 2. Loop and Analyze
    print("2. Downloading Data & Running TA-Lib...")
    
    for i, ticker in enumerate(top_tickers):
        print(f"   [{i+1}/{TOP_N}] Analyzing {ticker}...")
        
        try:
            # Handle Ticker Mappings
            search_ticker = ticker
            if ticker == 'FB': search_ticker = 'META' # Handle old Facebook ticker
            
            # Download MAX history to ensure we have data
            stock_df = yf.download(search_ticker, period="max", progress=False)
            
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)
            
            if len(stock_df) < 200:
                print(f"      Skipping {ticker} (Insufficient data)")
                continue

            # --- TA-Lib Indicators ---
            # SMA
            stock_df['SMA_20'] = talib.SMA(stock_df['Close'], timeperiod=20)
            stock_df['SMA_50'] = talib.SMA(stock_df['Close'], timeperiod=50)
            # RSI
            stock_df['RSI'] = talib.RSI(stock_df['Close'], timeperiod=14)
            # MACD
            macd, signal, hist = talib.MACD(stock_df['Close'])
            stock_df['MACD'] = macd
            stock_df['MACD_Signal'] = signal

            # --- Visualization ---
            # We take the last 2 years for the plot so it's readable
            plot_df = stock_df.tail(365*2)
            
            plt.figure(figsize=(12, 10))
            
            # Price
            plt.subplot(3, 1, 1)
            plt.plot(plot_df.index, plot_df['Close'], label='Close Price')
            plt.plot(plot_df.index, plot_df['SMA_50'], label='SMA 50')
            plt.title(f'{ticker} Technical Analysis')
            plt.legend()
            
            # RSI
            plt.subplot(3, 1, 2)
            plt.plot(plot_df.index, plot_df['RSI'], color='purple', label='RSI')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.legend()
            
            # MACD
            plt.subplot(3, 1, 3)
            plt.plot(plot_df.index, plot_df['MACD'], label='MACD')
            plt.plot(plot_df.index, plot_df['MACD_Signal'], label='Signal')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_FOLDER}/{ticker}_TA.png")
            plt.close('all')
            
        except Exception as e:
            print(f"      Error: {e}")
            
    print("\nTask 2 Complete. Charts saved.")

if __name__ == "__main__":
    analyze_top_stocks()