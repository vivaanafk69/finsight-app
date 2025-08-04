import streamlit as st
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="📰 Live News & Sentiment", layout="wide")
st.title("🗞️ Live News Dashboard")

# -------------------------------
# Sidebar filters
# -------------------------------
topics = ["Stock Market", "Technology", "Reliance", "Infosys", "Banking", "Cryptocurrency", "Energy", "Gold", "Inflation"]
selected_topic = st.sidebar.selectbox("Choose News Topic", topics)
refresh_rate = st.sidebar.slider("⏱️ Auto Refresh Interval (sec)", 30, 300, 60)

# -------------------------------
# RSS Fetch Function
# -------------------------------
def fetch_google_news(topic):
    query = topic.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")
    items = soup.find_all("item")
    
    articles = []
    for item in items[:15]:
        title = item.title.text
        link = item.link.text
        pub_date = item.pubDate.text
        source = item.source.text if item.source else "Unknown"
        articles.append({
            "title": title,
            "link": link,
            "pub_date": pub_date,
            "source": source
        })
    return articles

# -------------------------------
# Sentiment Analyzer
# -------------------------------
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.3:
        return "🟢 Positive", score
    elif score <= -0.3:
        return "🔴 Negative", score
    else:
        return "🟡 Neutral", score

# -------------------------------
# Display Function
# -------------------------------
def show_news(articles):
    for article in articles:
        sentiment_label, sentiment_score = analyze_sentiment(article["title"])
        with st.container():
            st.markdown(f"### [{article['title']}]({article['link']})")
            st.markdown(f"- ⏰ **Published:** {article['pub_date']}")
            st.markdown(f"- 📰 **Source:** {article['source']}")
            st.markdown(f"- 📈 **Sentiment:** {sentiment_label} *(Score: {sentiment_score:.2f})*")
            st.markdown("---")

# -------------------------------
# Live Update Loop (Optional)
# -------------------------------
if "last_run" not in st.session_state or time.time() - st.session_state.last_run > refresh_rate:
    st.session_state.last_run = time.time()
    news_data = fetch_google_news(selected_topic)
    st.session_state.news_data = news_data

# -------------------------------
# Display News
# -------------------------------
if "news_data" in st.session_state and st.session_state.news_data:
    show_news(st.session_state.news_data)
else:
    st.warning("⚠️ No news articles found. Try another topic or check your connection.")
