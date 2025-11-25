import pandas as pd
import yfinance as yf
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg') # Prevents Windows crashes
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
INPUT_PATH = 'data/raw_analyst_ratings.csv' 
OUTPUT_FOLDER = 'reports/figures_task3'
TOP_N = 20 # Analyze the top 20 most active stocks

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_style('whitegrid')
# =================================================

def run_correlation_scan():
    print(f"--- Starting Task 3: Correlation on Top {TOP_N} Stocks ---")
    
    # 1. Load and Filter
    print("1. Loading News to find active stocks...")
    try:
        news_df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: {INPUT_PATH} not found.")
        return

    # Clean ticker names
    news_df['stock'] = news_df['stock'].astype(str).str.strip().str.upper()
    
    # Find Top Tickers automatically
    top_tickers = news_df['stock'].value_counts().head(TOP_N).index.tolist()
    print(f"   Top Tickers found: {top_tickers}")
    
    # Filter DataFrame to only these stocks
    news_df = news_df[news_df['stock'].isin(top_tickers)].copy()
    
    # 2. Date & Sentiment
    print("2. Processing Dates and Sentiment...")
    # Convert to YYYY-MM-DD string to avoid timezone issues entirely
    news_df['date_str'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    
    # Vectorized Sentiment Calc
    news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Group by Stock and Date
    daily_sentiment = news_df.groupby(['stock', 'date_str'])['sentiment'].mean().reset_index()

    # 3. Match with History
    print("3. Fetching Historical Data & Correlating...")
    results = []

    for i, ticker in enumerate(top_tickers):
        print(f"   [{i+1}/{TOP_N}] Processing {ticker}...", end=" ")
        
        # Get news for this ticker
        stock_news = daily_sentiment[daily_sentiment['stock'] == ticker]
        if stock_news.empty: 
            print("No news.")
            continue
            
        # Determine the date range of the NEWS
        news_dates = pd.to_datetime(stock_news['date_str'])
        min_date = news_dates.min()
        
        # Handle FB -> META mapping
        search_ticker = ticker
        if ticker == 'FB': search_ticker = 'META'
        
        try:
            # DOWNLOAD HISTORICAL DATA MATCHING THE NEWS YEARS
            # We look from the first news article until today
            price_df = yf.download(search_ticker, start=min_date, end=pd.Timestamp.now().date(), progress=False)
            
            if price_df.empty:
                print("No price data.")
                continue
                
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
                
            price_df['Returns'] = price_df['Close'].pct_change()
            
            # Convert Price Index to String YYYY-MM-DD
            price_df['date_str'] = price_df.index.strftime('%Y-%m-%d')
            
            # MERGE on the string date
            merged = pd.merge(stock_news, price_df, on='date_str', how='inner')
            
            count = len(merged)
            
            # We accept > 5 matches for statistical relevance
            if count > 5:
                corr = merged['sentiment'].corr(merged['Returns'])
                if pd.isna(corr): corr = 0.0
                
                results.append({'Stock': ticker, 'Correlation': corr, 'Matches': count})
                print(f"DONE. Matches: {count}. Corr: {corr:.4f}")
                
                # Plot
                plt.figure(figsize=(8, 6))
                sns.regplot(data=merged, x='sentiment', y='Returns', scatter_kws={'alpha':0.4, 'color':'teal'})
                plt.title(f'{ticker}: Sentiment vs Returns (n={count})')
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_FOLDER}/{ticker}_correlation.png")
                plt.close('all')
            else:
                print(f"Low overlap ({count}).")
                
        except Exception as e:
            print(f"Error: {e}")

    # 4. Save Summary
    if results:
        res_df = pd.DataFrame(results).sort_values(by='Matches', ascending=False)
        res_df.to_csv(f"{OUTPUT_FOLDER}/correlation_summary.csv", index=False)
        print(f"\nTask 3 Complete. Processed {len(results)} stocks.")
    else:
        print("\nNo correlations found.")

if __name__ == "__main__":
    run_correlation_scan()