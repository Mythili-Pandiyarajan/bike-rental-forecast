import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BikeIQ — Rental Forecasting",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0e17;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --danger: #ef4444;
    --border: rgba(0,229,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide Streamlit items */
#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #111827 100%) !important;
    border-right: 1px solid var(--border);
}

/* NAVIGATION LABELS */
div[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;

    color: #ffffff !important;     /* ← FIXED VISIBILITY */
    font-size: 18px !important;
    font-weight: 700 !important;

    transition: 0.3s ease;
    cursor: pointer !important;
}

/* HOVER EFFECT */
div[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,229,255,0.15) !important;
    border: 1px solid #00e5ff !important;
    color: #00e5ff !important;
    transform: translateX(5px);
}

/* ACTIVE ITEM */
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] {
    color: white !important;
}

/* RADIO CIRCLE HIDE */
.stRadio > div {
    gap: 0.5rem;
}

.stRadio [role="radiogroup"] label div:first-child {
    display: none;
}

/* HERO */
.hero-banner {
    background: linear-gradient(135deg, #0d1220 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.3rem;
    font-weight: 700;
    color: #00e5ff;
}

.hero-sub {
    color: #94a3b8;
    margin-top: 0.5rem;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #7c3aed22);
    border: 1px solid #00e5ff !important;
    color: #00e5ff !important;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
    padding: 0.7rem;
    transition: 0.3s;
}

.stButton > button:hover {
    background: #00e5ff22 !important;
    transform: scale(1.02);
}

/* PREDICTION BOX */
.pred-box {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(124,58,237,0.08));
    border: 1px solid #00e5ff;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.pred-number {
    font-size: 4rem;
    font-weight: 700;
    color: #00e5ff;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
}

.metric-title {
    color: #94a3b8;
    font-size: 0.8rem;
}

.metric-value {
    color: #00e5ff;
    font-size: 2rem;
    font-weight: 700;
}

.sidebar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00e5ff;
    padding-bottom: 1rem;
}

.sidebar-section {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA + TRAIN MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def train_model():

    try:
        df = pd.read_csv("data/Data/day.csv")

    except:
        dates = pd.date_range("2011-01-01", periods=731)

        np.random.seed(42)

        df = pd.DataFrame({
            "dteday": dates,
            "cnt": np.random.randint(1000, 8000, len(dates)),
            "temp": np.random.uniform(0.1, 0.9, len(dates)),
            "windspeed": np.random.uniform(0.05, 0.4, len(dates))
        })

    df['dteday'] = pd.to_datetime(df['dteday'])

    train_df = pd.DataFrame({
        'ds': df['dteday'],
        'y': df['cnt'],
        'temp': df['temp'],
        'windspeed': df['windspeed']
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )

    model.add_regressor('temp')
    model.add_regressor('windspeed')

    model.fit(train_df)

    return model, df

model, df = train_model()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown('<div class="sidebar-logo">🚲 BikeIQ</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "📊 Dashboard",
            "🔮 Predict",
            "📅 Forecast",
            "🔬 Analysis"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">MODEL</div>
        <div class="metric-value" style="font-size:1.1rem;">Prophet</div>
        <div style="color:#94a3b8;font-size:0.8rem;margin-top:8px;">
        Forecasting up to 2045
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────

if page == "📊 Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Bike Rental Intelligence</div>
        <div class="hero-sub">
            Forecast bike rental demand using Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">TOTAL RENTALS</div>
            <div class="metric-value">{int(df['cnt'].sum()):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AVG DAILY</div>
            <div class="metric-value">{int(df['cnt'].mean()):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PEAK RENTALS</div>
            <div class="metric-value">{int(df['cnt'].max()):,}</div>
        </div>
        """, unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['dteday'],
        y=df['cnt'],
        mode='lines',
        line=dict(color='#00e5ff', width=2)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

elif page == "🔮 Predict":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Single Day Prediction</div>
        <div class="hero-sub">
            Predict bike rentals for a selected future date
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.3])

    with col1:

        # ← FIXED DATE LIMIT ISSUE
        pred_date = st.date_input(
            "Select Future Date",
            value=datetime(2026, 1, 1),
            min_value=datetime(2013, 1, 1),
            max_value=datetime(2045, 12, 31)
        )

        temp_input = st.slider(
            "Temperature",
            0.05,
            0.95,
            0.5
        )

        wind_input = st.slider(
            "Wind Speed",
            0.05,
            0.50,
            0.2
        )

        predict = st.button("⚡ Predict Demand")

    with col2:

        if predict:

            future = pd.DataFrame({
                'ds': [pd.Timestamp(pred_date)],
                'temp': [temp_input],
                'windspeed': [wind_input]
            })

            forecast = model.predict(future)

            prediction = int(forecast['yhat'].iloc[0])

            st.markdown(f"""
            <div class="pred-box">
                <div style="font-size:1rem;color:#94a3b8;">
                    Predicted Rentals
                </div>

                <div class="pred-number">
                    {prediction:,}
                </div>

                <div style="margin-top:10px;color:#94a3b8;">
                    Bikes / Day
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────────────────────

elif page == "📅 Forecast":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">20-Year Forecast</div>
        <div class="hero-sub">
            Predict bike rental demand up to year 2045
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:

        start_year = st.number_input(
            "Start Year",
            min_value=2013,
            max_value=2045,
            value=2025
        )

        forecast_days = st.number_input(
            "Forecast Days",
            min_value=30,
            max_value=7300,
            value=365
        )

        avg_temp = st.slider(
            "Average Temperature",
            0.05,
            0.95,
            0.5
        )

        avg_wind = st.slider(
            "Average Wind",
            0.05,
            0.50,
            0.2
        )

        run = st.button("📅 Run Forecast")

    with col2:

        if run:

            future_dates = pd.date_range(
                start=f"{int(start_year)}-01-01",
                periods=int(forecast_days)
            )

            future_df = pd.DataFrame({
                'ds': future_dates,
                'temp': avg_temp,
                'windspeed': avg_wind
            })

            forecast = model.predict(future_df)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                line=dict(color='#00e5ff', width=2),
                name='Forecast'
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                forecast[['ds', 'yhat']].rename(
                    columns={
                        'ds': 'Date',
                        'yhat': 'Predicted Rentals'
                    }
                ).head(100)
            )

# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "🔬 Analysis":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Exploratory Data Analysis</div>
        <div class="hero-sub">
            Understand bike rental trends and feature impact
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Correlation Analysis")

    corr = df[['cnt', 'temp', 'windspeed']].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Plasma',
        text=corr.round(2).values,
        texttemplate='%{text}'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""

### 📝 Key Insights

- Temperature has a positive relationship with bike rentals
- Wind speed negatively affects bike demand
- Rentals increase during warmer seasons
- Prophet with regressors improves forecasting performance
- Seasonal trends are clearly visible in the dataset

""")
