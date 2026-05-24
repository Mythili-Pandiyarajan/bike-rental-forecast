import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="BikeIQ AI Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# ADVANCED UI CSS
# ======================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #050816;
    color: white;
}

.main {
    background: linear-gradient(135deg,#050816,#0f172a);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#111827);
    border-right: 1px solid rgba(255,255,255,0.05);
}

.title-glow {
    font-family: 'Orbitron', sans-serif;
    font-size: 55px;
    font-weight: bold;
    background: linear-gradient(90deg,#00f5ff,#8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-card {
    background: linear-gradient(135deg,#0b1023,#1e1b4b);
    padding: 35px;
    border-radius: 24px;
    border: 1px solid rgba(0,255,255,0.15);
    box-shadow: 0 0 30px rgba(0,255,255,0.08);
}

.metric-card {
    background: rgba(255,255,255,0.03);
    padding: 30px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
}

.big-number {
    font-size: 68px;
    font-weight: bold;
    color: #38bdf8;
}

.small-label {
    color: #94a3b8;
    letter-spacing: 2px;
    font-size: 14px;
}

.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#8b5cf6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.5rem;
    font-weight: bold;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():
    df = pd.read_csv("day.csv")
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

try:
    df = load_data()
except:
    st.error("Upload day.csv in same folder.")
    st.stop()

# ======================================================
# PROPHET MODEL
# ======================================================

prophet_df = df[['dteday', 'cnt']].rename(
    columns={
        'dteday': 'ds',
        'cnt': 'y'
    }
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(prophet_df)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.markdown("""
    <h1 style='font-family:Orbitron;color:#00f5ff;'>
    🚴 BikeIQ
    </h1>
    """, unsafe_allow_html=True)

    selected = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Predict",
            "Forecast",
            "Analysis"
        ]
    )

    st.markdown("---")

    st.markdown("""
    ### AI Model

    Prophet Forecasting AI

    #### Metrics

    - Forecast till 2045
    - Interactive analytics
    - Future trend prediction
    - Real-time visualization
    """)

# ======================================================
# HERO SECTION
# ======================================================

st.markdown("""
<div class="hero-card">

<div class="title-glow">
Smart Bike Rental Forecasting AI
</div>

<br>

<p style='font-size:18px;color:#cbd5e1;'>

Advanced bike demand prediction using Prophet AI.
Analyze trends, predict rentals and future forecasting till 2045.

</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ======================================================
# DASHBOARD
# ======================================================

if selected == "Dashboard":

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="metric-card">

        <div class="small-label">
        TOTAL RENTALS
        </div>

        <div class="big-number">
        {df['cnt'].sum():,}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric-card">

        <div class="small-label">
        AVG DAILY DEMAND
        </div>

        <div class="big-number">
        {int(df['cnt'].mean()):,}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="metric-card">

        <div class="small-label">
        PEAK DEMAND
        </div>

        <div class="big-number">
        {df['cnt'].max():,}
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    fig = px.line(
        df,
        x='dteday',
        y='cnt',
        title="Bike Rental Trends"
    )

    fig.update_layout(
        paper_bgcolor="#0b1023",
        plot_bgcolor="#0b1023",
        font_color="white",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PREDICT
# ======================================================

elif selected == "Predict":

    st.subheader("Single Day Prediction")

    col1, col2 = st.columns([1,1])

    with col1:

        selected_date = st.date_input(
            "Select Date",
            datetime(2026,1,1)
        )

        temp = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.5
        )

        wind = st.slider(
            "Wind Speed",
            0.0,
            1.0,
            0.2
        )

        humidity = st.slider(
            "Humidity",
            0.0,
            1.0,
            0.4
        )

    with col2:

        future = pd.DataFrame({
            'ds': [pd.to_datetime(selected_date)]
        })

        forecast = model.predict(future)

        pred = int(forecast['yhat'].values[0])

        st.markdown(f"""
        <div class="metric-card" style='height:420px;text-align:center;'>

            <div class="small-label">
                PREDICTED RENTALS
            </div>

            <div class="big-number">
                {pred:,}
            </div>

            <div style='font-size:22px;color:#cbd5e1;'>
                Bikes / Day
            </div>

            <br>

            <div style='color:#22c55e;font-size:20px;'>
                🚴 Ideal Cycling Conditions
            </div>

        </div>
        """, unsafe_allow_html=True)

# ======================================================
# FORECAST
# ======================================================

elif selected == "Forecast":

    st.subheader("Future Forecast till 2045")

    future = model.make_future_dataframe(
        periods=365*20
    )

    forecast = model.predict(future)

    future_data = forecast[
        forecast['ds'] >= '2026-01-01'
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=future_data['ds'],
            y=future_data['yhat'],
            mode='lines',
            name='Forecast'
        )
    )

    fig.update_layout(
        title="Bike Demand Forecast (2026 - 2045)",
        paper_bgcolor="#0b1023",
        plot_bgcolor="#0b1023",
        font_color="white",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        future_data[['ds','yhat']].tail(100)
    )

# ======================================================
# ANALYSIS
# ======================================================

elif selected == "Analysis":

    st.subheader("Advanced Analytics")

    c1, c2 = st.columns(2)

    with c1:

        fig1 = px.box(
            df,
            x='season',
            y='cnt',
            title='Season vs Rentals'
        )

        fig1.update_layout(
            paper_bgcolor="#0b1023",
            plot_bgcolor="#0b1023",
            font_color="white"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with c2:

        fig2 = px.scatter(
            df,
            x='temp',
            y='cnt',
            color='windspeed',
            title='Temperature vs Rentals'
        )

        fig2.update_layout(
            paper_bgcolor="#0b1023",
            plot_bgcolor="#0b1023",
            font_color="white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    corr = df.corr(numeric_only=True)

    heatmap = px.imshow(
        corr,
        text_auto=True,
        aspect='auto',
        title="Feature Correlation Heatmap"
    )

    heatmap.update_layout(
        paper_bgcolor="#0b1023",
        plot_bgcolor="#0b1023",
        font_color="white",
        height=700
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )
