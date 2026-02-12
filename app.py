import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from supabase import create_client

st.set_page_config(layout="wide", page_title="Quantara - Quantitative Intelligence Reimagined", page_icon="🚀")

# ================== PREMIUM CSS ==================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.sidebar .sidebar-content {
    padding-top: 20px;
}
.stSidebar {
    background-color: #111827;
}
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    border: none;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
}
.footer {
    text-align:center;
    padding: 20px;
    color: gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ================= SUPABASE =================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if "user" not in st.session_state:
    st.session_state.user = None
if "name" not in st.session_state:
    st.session_state.name = None

def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user = res.user
        st.session_state.name = user_name
        st.success("Login successful")
    except:
        st.error("Invalid credentials")

def signup(email, password):
    try:
        supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })
        st.success("Signup successful! You can login now.")
    except Exception as e:
        st.error(f"Signup failed: {e}")

# ================= LOGIN PAGE =================
if st.session_state.user is None:
    st.markdown("<h1 style='text-align:center;'>🚀 Quantara</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Quantitative Intelligence Reimagined - A initiative by Shravan Shidruk and Rushik Kokate</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])

        with tab1:
            user_name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                login(email, password)

        with tab2:
            new_name = st.text_input("Full Name ")
            new_email = st.text_input("New Email")
            new_password = st.text_input("New Password", type="password")
            if st.button("Signup"):
                signup(new_email, new_password)

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ================= SIDEBAR =================
st.sidebar.markdown("## 📈🌱 Quantara")
st.sidebar.markdown("### 👤 User Profile")
st.sidebar.info(f"""
**Name:** {st.session_state.name}  
**Email:** {st.session_state.user.email}
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Dashboard Control")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.name = None
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### 👨‍💻 Project Contributions")
st.sidebar.markdown("""
### 👨‍💻 Project Contributions

**1️⃣ Shravan Shidruk**  
_TECOMP, GSMCOE, SPPU_  
- UI Design  
- Model Integration  
- Analysis Engine  

---

**2️⃣ Rushik Kokate**  
_TECOMP, GSMCOE, SPPU_  
- Data Scraping Pipeline  
- Data Cleaning  
- CSV Preparation  
""")


st.sidebar.markdown("---")

# ================= IMAGE BANNER =================
import base64

# ========= FIXED WORKING BANNER =========
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("assets/crypto_banner.png")

st.markdown(f"""
<div style="
position:relative;
width:100%;
height:250px;
margin-bottom:30px;
">
    <img src="data:image/png;base64,{img_base64}"
         style="width:100%;height:250px;object-fit:cover;border-radius:15px;">
    <div style="
    position:absolute;
    bottom:25px;
    left:40px;
    font-size:38px;
    font-weight:800;
    letter-spacing:1px;
    color:#111111;
    text-transform:uppercase;
    text-shadow:
        0px 0px 2px rgba(0,0,0,0.6),
        0px 3px 8px rgba(0,0,0,0.4),
        0px 6px 20px rgba(0,0,0,0.3);
    font-family:'Poppins', sans-serif;"> 🌱 QUANTARA - Quantitative Intelligence Reimagined
    </div>
</div>
""", unsafe_allow_html=True)




# ================= DATA =================
DATASETS = {
    "BTC": "Data/BTC_historical_INR.csv",
    "ETH": "Data/ETH_historical_INR.csv",
    "SOL": "Data/SOL_historical_INR.csv",
    "USDC": "Data/USDC_historical_INR.csv",
    "USDT": "Data/USDT_historical_INR.csv"
}

@st.cache_data
def load_price(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

@st.cache_data
def load_news():
    url = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv"
    return pd.read_csv(url)

@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

coin = st.sidebar.selectbox("Select Coin", list(DATASETS.keys()))
data = load_price(DATASETS[coin])
news = load_news()
finbert = load_finbert()

st.markdown(f"<h1>{coin} Forecast Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ================= SENTIMENT =================
st.subheader("📰 Market Sentiment Analysis")
sia = SentimentIntensityAnalyzer()

news['vader'] = news['title'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
news['finbert_label'] = news['title'].apply(lambda x: finbert(str(x))[0]['label'])
label_map = {"positive":1,"negative":-1,"neutral":0}
news['finbert_score'] = news['finbert_label'].str.lower().map(label_map)
news['final_score'] = (news['vader'] + news['finbert_score']) / 2

col1, col2 = st.columns([3,1])
col1.line_chart(news['final_score'])

avg_sent = news['final_score'].mean()
with col2:
    st.metric("Avg Sentiment", round(avg_sent,2))

if avg_sent > 0.1:
    st.success("Bullish Market 🟢")
elif avg_sent < -0.1:
    st.error("Bearish Market 🔴")
else:
    st.warning("Neutral Market 🟡")

st.markdown("<hr>", unsafe_allow_html=True)

# ================= FORECAST SECTION =================
st.subheader("📈 AI Forecast Models")

st.markdown("### Historical Price")
st.line_chart(data['Close'])

@st.cache_resource
def run_arima(series):
    return ARIMA(series, order=(2,1,2)).fit().forecast(30)

st.markdown("### ARIMA Forecast")
st.line_chart(run_arima(data['Close']))

@st.cache_resource
def run_sarima(series):
    return SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit().forecast(30)

st.markdown("### SARIMA Forecast")
st.line_chart(run_sarima(data['Close']))

@st.cache_resource
def run_prophet(df):
    p_df = df.reset_index()[['Date','Close']]
    p_df.columns = ['ds','y']
    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(p_df)
    future = model.make_future_dataframe(periods=30)
    return model.predict(future)[['ds','yhat']].set_index('ds').tail(30)

st.markdown("### Prophet Forecast")
st.line_chart(run_prophet(data))

@st.cache_resource
def run_lstm(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([LSTM(64, return_sequences=True), Dropout(0.2), LSTM(64), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    pred_input = scaled[-60:].reshape(1,60,1)
    preds = []
    for _ in range(30):
        p = model.predict(pred_input, verbose=0)[0][0]
        preds.append(p)
        pred_input = np.append(pred_input[:,1:,:], [[[p]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    future_dates = pd.date_range(series.index[-1], periods=30)
    return pd.DataFrame(preds, index=future_dates)

st.markdown("### LSTM Deep Learning Forecast")
st.line_chart(run_lstm(data['Close']))

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2026 Quantara (Quant + Tara) - Quantitative Intelligence Reimagined  
Built by Team Shravan & Rushik | All Rights Reserved
</div>
""", unsafe_allow_html=True)
