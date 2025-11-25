import pandas as pd
import yfinance as yf
from textblob import TextBlob
import matplotlib
# Use 'Agg' backend to prevent crashes on Windows during loops
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
INPUT_PATH = 'data/raw_analyst_ratings.csv' 
OUTPUT_FOLDER = 'reports/figures_task3'
# HOW MANY STOCKS TO ANALYZE? (Increase to 30 or 50 if you want even more)
TOP_N_STOCKS = 20 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_style('whitegrid')
# =================================================

def run_deep_scan():
    print(f"--- Starting Task 3: Deep Scan (Top {TOP_N_STOCKS} Stocks) ---")
    
    # 1. LOAD DATA
    print("1. Loading and Indexing News Data...")
    try:
        news_df = pd.read_csv(INPUT_PATH)
        # Clean names
        news_df['stock'] = news_df['stock'].astype(str).str.strip().str.upper()
        
        # FIND THE MOST ACTIVE STOCKS
        # This counts which stocks appear most often in the file
        top_stocks = news_df['stock'].value_counts().head(TOP_N_STOCKS).index.tolist()
        print(f"   Top {TOP_N_STOCKS} most active stocks found: {top_stocks}")
        
        # Filter dataset to just these top stocks
        news_df = news_df[news_df['stock'].isin(top_stocks)].copy()
        
    except Exception as e:
        print(f"   Error: {e}")
        return

    # 2. DATES & SENTIMENT
    print("2. Calculating Sentiment & Dates (Batch Processing)...")
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True)
    news_df['date'] = news_df['date'].dt.tz_localize(None).dt.date
    
    news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    daily_sentiment = news_df.groupby(['stock', 'date'])['sentiment'].mean().reset_index()

    # 3. FETCH PRICES & CORRELATE
    print("3. Fetching Historical Prices for Top Stocks...")
    results = []

    for i, ticker in enumerate(top_stocks):
        # Progress Bar
        print(f"   [{i+1}/{TOP_N_STOCKS}] Processing {ticker}...", end=" ")
        
        stock_sent = daily_sentiment[daily_sentiment['stock'] == ticker].set_index('date')
        if stock_sent.empty:
            print("No daily data.")
            continue
            
        start_date = stock_sent.index.min()
        
        try:
            # Handle Ticker Mapping (Common issues)
            search_ticker = ticker
            if ticker == 'FB': search_ticker = 'META'
            if ticker == 'GOOGL': search_ticker = 'GOOG'

            # Download
            price_df = yf.download(search_ticker, start=start_date, end=pd.Timestamp.now().date(), progress=False)
            
            if price_df.empty:
                print("No price data found.")
                continue

            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)

            price_df['Returns'] = price_df['Close'].pct_change()
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None).date
            
            # MERGE
            merged_df = stock_sent.join(price_df[['Returns']], how='inner')
            
            count = len(merged_df)
            
            # ACCEPT ANY OVERLAP > 1
            if count > 1:
                corr = merged_df['sentiment'].corr(merged_df['Returns'])
                if pd.isna(corr): corr = 0.0
                
                results.append({'Stock': ticker, 'Correlation': corr, 'Matches': count})
                print(f"DONE. Matched {count} days. Corr: {corr:.3f}")
                
                # Plot
                plt.figure(figsize=(10, 6))
                sns.regplot(data=merged_df, x='sentiment', y='Returns', 
                           scatter_kws={'alpha':0.6, 'color':'teal', 's':80})
                plt.title(f'{ticker}: Sentiment vs Returns (n={count}, corr={corr:.2f})')
                plt.xlabel('Sentiment')
                plt.ylabel('Daily Return')
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_FOLDER}/{ticker}_correlation.png")
                plt.close('all') # Clear memory
            else:
                print(f"Low overlap ({count}).")

        except Exception as e:
            print(f"Error: {e}")
            
        # Sleep slightly to prevent Yahoo blocking
        time.sleep(0.5)

    # 4. SAVE
    if results:
        res_df = pd.DataFrame(results).sort_values(by='Matches', ascending=False)
        res_df.to_csv(f"{OUTPUT_FOLDER}/correlation_summary.csv", index=False)
        print(f"\n--- Analysis Complete ---")
        print(f"Processed {len(results)} stocks successfully.")
        print(f"Results saved to {OUTPUT_FOLDER}")
        print("\nTop Correlations Found:")
        print(res_df.head())
    else:
        print("\nNo correlations found.")

if __name__ == "__main__":
    run_deep_scan()