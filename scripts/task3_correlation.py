import pandas as pd
import yfinance as yf
from textblob import TextBlob
import matplotlib
# Use 'Agg' backend to prevent Windows GUI crashes during loops
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
INPUT_PATH = 'data/raw_analyst_ratings.csv' 
OUTPUT_FOLDER = 'reports/figures_task3'
TOP_N = 20 # Analyze the top 20 most active stocks

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_style('whitegrid')
# =================================================

def run_correlation_scan():
    print(f"--- Starting Task 3: Deep Scan Correlation (Top {TOP_N}) ---")
    
    # 1. LOAD AND FILTER
    print("1. Loading News Data...")
    try:
        news_df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: {INPUT_PATH} not found.")
        return

    # Clean ticker names (remove spaces)
    news_df['stock'] = news_df['stock'].astype(str).str.strip().str.upper()
    
    # Identify Top Tickers automatically based on news volume
    top_tickers = news_df['stock'].value_counts().head(TOP_N).index.tolist()
    print(f"   Top Active Tickers: {top_tickers}")
    
    # Filter DataFrame to only these stocks to save processing time
    news_df = news_df[news_df['stock'].isin(top_tickers)].copy()
    
    # 2. DATE & SENTIMENT
    print("2. Processing Dates and Sentiment...")
    # Convert to YYYY-MM-DD string to avoid timezone mismatch bugs
    news_df['date_str'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    
    # Calculate Sentiment
    news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Aggregate (Average sentiment per stock per day)
    daily_sentiment = news_df.groupby(['stock', 'date_str'])['sentiment'].mean().reset_index()

    # 3. MATCH WITH HISTORY
    print("3. Fetching Historical Data & Correlating...")
    results = []

    for i, ticker in enumerate(top_tickers):
        print(f"   [{i+1}/{TOP_N}] Processing {ticker}...", end=" ")
        
        # Get news for this ticker
        stock_news = daily_sentiment[daily_sentiment['stock'] == ticker]
        if stock_news.empty: 
            print("No news.")
            continue
            
        # Determine the date range of the NEWS (Start to End)
        news_dates = pd.to_datetime(stock_news['date_str'])
        min_date = news_dates.min()
        
        # Handle FB -> META mapping for older datasets
        search_ticker = ticker
        if ticker == 'FB': search_ticker = 'META'
        
        try:
            # DOWNLOAD HISTORICAL DATA MATCHING THE NEWS YEARS
            # We fetch data starting from the first news article found
            price_df = yf.download(search_ticker, start=min_date, end=pd.Timestamp.now().date(), progress=False)
            
            if price_df.empty:
                print("No price data.")
                continue
                
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
                
            # Calculate Daily Returns
            price_df['Returns'] = price_df['Close'].pct_change()
            
            # Convert Price Index to String YYYY-MM-DD for merging
            price_df['date_str'] = price_df.index.strftime('%Y-%m-%d')
            
            # MERGE on the string date
            merged = pd.merge(stock_news, price_df, on='date_str', how='inner')
            
            count = len(merged)
            
            # We require > 5 matches to calculate a valid correlation
            if count > 5:
                corr = merged['sentiment'].corr(merged['Returns'])
                if pd.isna(corr): corr = 0.0
                
                results.append({'Stock': ticker, 'Correlation': corr, 'Matches': count})
                print(f"DONE. Matches: {count}. Corr: {corr:.4f}")
                
                # Plot Regression Chart
                plt.figure(figsize=(8, 6))
                sns.regplot(data=merged, x='sentiment', y='Returns', scatter_kws={'alpha':0.4, 'color':'teal'})
                plt.title(f'{ticker}: Sentiment vs Returns (n={count})')
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_FOLDER}/{ticker}_correlation.png")
                plt.close('all') # Clear memory
            else:
                print(f"Low overlap ({count}).")
                
        except Exception as e:
            print(f"Error: {e}")

    # 4. SAVE SUMMARY
    if results:
        res_df = pd.DataFrame(results).sort_values(by='Matches', ascending=False)
        res_df.to_csv(f"{OUTPUT_FOLDER}/correlation_summary.csv", index=False)
        print(f"\nTask 3 Complete. Processed {len(results)} stocks.")
        print(f"Summary saved to {OUTPUT_FOLDER}/correlation_summary.csv")
    else:
        print("\nNo significant correlations found.")

if __name__ == "__main__":
    run_correlation_scan()