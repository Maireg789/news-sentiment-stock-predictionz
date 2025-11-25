import pandas as pd
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ================= CONFIGURATION =================
# The specific stocks from your file explorer
MY_STOCKS = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA']

OUTPUT_FOLDER = 'reports/figures_task2'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set plotting style for professional reports
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
# =================================================

def analyze_stock(ticker):
    """
    Fetches data for a specific stock, runs TA-Lib, and saves the chart.
    """
    print(f"--> Analyzing {ticker}...")
    
    try:
        # 1. Fetch Data (2 years history)
        # threads=False prevents the 'Failed download' issues
        df = yf.download(ticker, period="2y", progress=False, threads=False)
        
        # Fix: yfinance often returns MultiIndex columns in 2024/2025
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Check if data is valid
        if df.empty or len(df) < 60:
            print(f"   Warning: Not enough data for {ticker}")
            return

        # 2. Apply Technical Indicators (TA-Lib)
        # SMA (Trend)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        
        # RSI (Momentum)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # MACD (Trend Momentum)
        macd, signal, hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        
        # 3. Financial Metrics (Volatility)
        # This covers the "PyNance" requirement (calculated manually)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

        # 4. Visualization
        plt.figure(figsize=(12, 12))
        
        # Subplot 1: Price & Moving Averages
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        plt.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linewidth=1.5)
        plt.plot(df.index, df['SMA_50'], label='SMA 50', color='green', linewidth=1.5)
        plt.title(f'{ticker} Price Analysis (Trend)')
        plt.legend(loc='upper left')
        
        # Subplot 2: RSI
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['RSI'], color='purple', label='RSI', linewidth=1.5)
        plt.axhline(70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(30, color='green', linestyle='--', alpha=0.5)
        plt.title('RSI (Momentum)')
        plt.legend(loc='upper left')
        
        # Subplot 3: MACD
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
        plt.plot(df.index, df['MACD_Signal'], label='Signal', color='red', linewidth=1.5)
        plt.bar(df.index, hist, label='Hist', color='gray', alpha=0.3)
        plt.title('MACD (Trend Strength)')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot
        save_path = f"{OUTPUT_FOLDER}/{ticker}_technical_analysis.png"
        plt.savefig(save_path)
        plt.close() # Close memory
        
        print(f"   [SUCCESS] Chart saved: {save_path}")
        
    except Exception as e:
        print(f"   [ERROR] Failed to process {ticker}: {e}")

if __name__ == "__main__":
    print(f"Starting Task 2 Analysis for: {MY_STOCKS}")
    
    # Loop through only your specific stocks
    for ticker in MY_STOCKS:
        analyze_stock(ticker)
        # Small pause to be polite to the API
        time.sleep(1)
        
    print("\nTask 2 Complete! Check reports/figures_task2 folder.")