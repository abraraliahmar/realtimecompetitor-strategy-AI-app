import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openai import AzureOpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

API_KEY = "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"
SLACK_WEBHOOK= "https://hooks.slack.com/services/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"

def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("price.csv")
    print(data.head())
    return data


def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews.csv")
    return reviews


def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)


def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["Price"] = data["Price"].astype(int)
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

    X = data[["Price", "Discount"]]
    y = data["Predicted_Discount"]
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

import numpy as np
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    data = data.sort_index()
    print(product_data.index)

    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")
    data = data.dropna(subset=["Discount"])

    discount_series = data["Discount"]
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(
                "Index must be datetime or convertible to datetime."
            ) from e

    model = ARIMA(discount_series, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df



def send_to_slack(data):
    """ """
    payload = {"text": data}
    response = requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

def truncate_text(text, max_length):
    """Truncates text to a maximum length."""
    if len(text) > max_length:
        return text[:max_length] + "..."  # Add ellipsis to indicate truncation
    return text


def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

1. *Product Name*: {product_name}

2. *Competitor Data* (including current prices, discounts, and predicted discounts):
{competitor_data}

3. *Sentiment Analysis*:
{sentiment}


5. *Today's Date*: {str(date)}

### Task:
- Analyze the competitor data and identify key pricing trends.
- Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
- Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
- Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
- Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and outperforming competitors.

Provide your recommendations in a structured format:
1. *Pricing Strategy*
2. *Promotional Campaign Ideas*
3. *Customer Satisfaction Recommendations*
    """

    messages = [{"role": "user", "content": prompt}]

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )
    res = res.json()
    response = res["choices"][0]["message"]["content"]
    return response

st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")


st.title("E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

products = [
    "boAt Bassheads 100 in Ear Wired Earphones with Mic(Black)",
    "ZEBRONICS Zeb-Bro in Ear Wired Earphones with Mic, 3.5mm Audio Jack, 10mm Drivers, Phone/Tablet Compatible(Blue) (Green)",
    "JBL C100SI Wired In Ear Headphones with Mic, JBL Pure Bass Sound, One Button Multi-function Remote, Premium Metallic Finish, Angled Buds for Comfort fit (Black)",
    "Kreo Hydra C Type Dual Dynamic Driver In-Ear Wired Gaming Earphones with Microphone, Tangle-free Gaming Earphones with Noise Isolating Technology and EZ Volume Control Slider, Wired Headset for Gaming",
    "Kreo Hydra C Type Dual Dynamic Driver In-Ear Wired Gaming Earphones with Microphone, Tangle-free Gaming Earphones with Noise Isolating Technology and EZ Volume Control Slider, Wired Headset for Gaming",
    "ZEBRONICS Zeb-Bro in Ear Wired Earphones with Mic, 3.5mm Audio Jack, 10mm Drivers, Phone/Tablet Compatible(Blue) (Black)",
    "boAt Bassheads 225 Wired in Ear Earphone with Mic (Red)",
    "boAt Bassheads 100 C Wired Earphones with Type-C Jack, in-Line Microphone, in Ear, 10Mm Drivers, Signature Sound, Integrated Controls & Multi-Os Compatibility(Black)",
    "boAt Bassheads 152 in Ear Wired Earphones with Mic(Active Black)",
    "boAt Bassheads 100 in Ear Wired Earphones with Mic(Furious Red)",
]
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)


competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

product_data = competitor_data[competitor_data["title"] == selected_product]
product_reviews = reviews_data[reviews_data["title"] == selected_product]

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)
    )
    reviews = product_reviews["reviews"].tolist()
    sentiments = analyze_sentiment(reviews)

    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")


# Preprocessing

product_data["date"] = pd.to_datetime(product_data["date"], errors="coerce")
product_data = product_data.dropna(subset=["date"])
product_data.set_index("date", inplace=True)
product_data = product_data.sort_index()

product_data["selling_price"] = pd.to_numeric(product_data["selling_price"], errors="coerce")
product_data = product_data.dropna(subset=["selling_price"])

# Forecasting Model
product_data_with_predictions = forecast_discounts_arima(product_data)


st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))

recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)
st.subheader("Strategic Recommendations")
st.write(recommendations)
send_to_slack(recommendations)
