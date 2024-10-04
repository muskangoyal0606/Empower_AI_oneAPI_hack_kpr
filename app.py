import streamlit as st
import yfinance as yf
import requests
from googleapiclient.discovery import build
from transformers import pipeline
import numpy as np

# YouTube API setup
def youtube_api_setup():
    api_key = 'AIzaSyB-pW8FME-a7KMRqwEeQJStTxDPqvQNMm0'  # Replace with your YouTube API key
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

# Fetch YouTube sentiment
def fetch_youtube_sentiment(symbol, youtube, sentiment_model):
    search_response = youtube.search().list(q=symbol, part='snippet', maxResults=10).execute()
    video_ids = [item['id']['videoId'] for item in search_response['items'] if 'videoId' in item['id']]
    
    comments = []
    for video_id in video_ids:
        comment_response = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=50).execute()
        for comment in comment_response['items']:
            comment_text = comment['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment_text)
    
    sentiments = sentiment_model(comments)
    sentiment_scores = [s['label'] for s in sentiments]
    positive = sentiment_scores.count('POSITIVE')
    negative = sentiment_scores.count('NEGATIVE')
    return positive, negative

# Moving Average (technical analysis)
def calculate_moving_average(stock_data, window_size):
    prices = stock_data['Close'].to_numpy()
    moving_avg = np.convolve(prices, np.ones(window_size)/window_size, mode='valid')
    return moving_avg

# Fetch stock data using Yahoo Finance
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period="1y")
    return stock_data

# Main app
def main():
    st.title("Empower_AI: powered by intel ONEapi")
    
    # Input for stock symbol
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")
    
    if st.button("Analyze"):
        # Fetch stock data
        stock_data = fetch_stock_data(stock_symbol)
        
        # Display stock data overview
        st.subheader(f"Stock Overview - {stock_symbol}")
        st.write(stock_data.tail())

        # Sentiment analysis
        sentiment_model = pipeline("sentiment-analysis")

        # YouTube Sentiment
        st.subheader("YouTube Sentiment Analysis")
        youtube = youtube_api_setup()
        positive_youtube, negative_youtube = fetch_youtube_sentiment(stock_symbol, youtube, sentiment_model)
        st.write(f"Positive Comments: {positive_youtube}, Negative Comments: {negative_youtube}")
        
        # Technical analysis (Moving Average)
        st.subheader("Technical Analysis (Moving Average)")
        window_size = st.slider("Select Moving Average Window Size:", 5, 100, 20)
        moving_avg = calculate_moving_average(stock_data, window_size)
        st.line_chart(moving_avg)

        # Fundamental analysis
        st.subheader("Fundamental Analysis")
        st.write("Market Cap:", stock_data['Close'].iloc[-1] * stock_data['Volume'].mean())
        st.write("Price-to-Earnings Ratio (P/E):", stock_data['Close'].iloc[-1] / (stock_data['Close'].mean()))
        
        # Recommendation based on sentiment analysis
        st.subheader("Stock Recommendation")
        total_positive = positive_youtube
        total_negative = negative_youtube
        if total_positive > total_negative:
            st.write(f"Recommendation: **BUY** {stock_symbol}")
        elif total_negative > total_positive:
            st.write(f"Recommendation: **SELL** {stock_symbol}")
        else:
            st.write(f"Recommendation: **HOLD** {stock_symbol}")

if __name__ == "__main__":
    main()



