import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="BikeIQ — Chromatic Hazard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────
# CUSTOM CSS — ANIME / DYSTOPIAN UI
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Noto+Sans+JP:wght@300;400;700&display=swap');

:root {
    --bg: #05070d;
    --panel: rgba(15,15,20,0.88);
    --panel2: rgba(25,25,35,0.92);
    --border: rgba(255,255,255,0.08);
    --text: #e8e8ea;
    --muted: #8a8a95;
    --purple: #b56cff;
    --red: #ff3b3b;
    --green: #59ff7a;
    --blue: #74b9ff;
    --gold: #ffb347;
    --pink: #ff8fd8;
}

html, body, [class*="css"] {
    background: #05070d !important;
    color: var(--text) !important;
    font-family: 'Noto Sans JP', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding: 1rem 2rem !important;
    max-width: 100% !important;
}

.stApp {
    background:
        linear-gradient(rgba(5,7,13,0.92), rgba(5,7,13,0.95)),
        url('https://images.unsplash.com/photo-1519608487953-e999c86e7455?q=80&w=1920&auto=format&fit=crop') center/cover fixed;
}

/* HERO */
.hero {
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(135deg, rgba(20,20,30,0.92), rgba(5,5,10,0.95));
    padding: 2rem;
    border-radius: 22px;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.03) 50%, transparent 100%);
    animation: shine 8s linear infinite;
}

@keyframes shine {
    0% {transform: translateX(-100%);}
    100% {transform: translateX(100%);}
}

.hero-title {
    font-size: 3rem;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
    background: linear-gradient(90deg, #ffffff, #9f7aea, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #9ea3b5;
    font-size: 1rem;
    letter-spacing: 0.08em;
}

.jp-text {
    margin-top: 1rem;
    color: rgba(255,255,255,0.75);
    font-size: 0.9rem;
    letter-spacing: 0.18em;
}

/* CHARACTER STYLE CARDS */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: rgba(10,10,18,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.3rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(180deg, var(--purple), transparent);
}

.stat-label {
    color: #9ea3b5;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
}

.stat-value {
    font-size: 2rem;
    font-family: 'Orbitron', sans-serif;
    color: white;
    margin-top: 0.4rem;
}

.stat-desc {
    color: #7b8197;
    font-size: 0.78rem;
    margin-top: 0.4rem;
}

/* PANELS */
.panel {
    background: rgba(10,10,18,0.88);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}

.section-title {
    font-size: 1rem;
    letter-spacing: 0.2em;
    margin-bottom: 1rem;
    font-family: 'Orbitron', sans-serif;
    color: white;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, rgba(181,108,255,0.25), rgba(255,59,59,0.2)) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.1em !important;
    transition: 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(181,108,255,0.6) !important;
    box-shadow: 0 0 25px rgba(181,108,255,0.35);
}

/* INPUTS */
.stSlider label,
.stSelectbox label,
.stDateInput label {
    color: #cfcfe7 !important;
    letter-spacing: 0.08em;
}

.stSelectbox div[data-baseweb="select"],
.stDateInput input {
    background: rgba(20,20,30,0.95) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(8,8,12,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.sidebar-logo {
    text-align: center;
    font-size: 1.6rem;
    font-family: 'Orbitron', sans-serif;
    color: white;
    margin-bottom: 2rem;
    letter-spacing: 0.15em;
}

/* PREDICTION BOX */
.prediction-box {
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    background:
        linear-gradient(135deg, rgba(181,108,255,0.15), rgba(255,59,59,0.08));
    border: 1px solid rgba(255,255,255,0.08);
}

.prediction-number {
    font-size: 4rem;
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(90deg, #ffffff, #b56cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.prediction-label {
    color: #8f93a8;
    letter-spacing: 0.18em;
    margin-top: 0.5rem;
}

/* TABLE */
[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    overflow: hidden !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────────────
@st.cache_resource

def train_model():
    dates = pd.date_range("2011-01-01", periods=731, freq="D")

    np.random.seed(42)

    rentals = (
        4500
        + 2000 * np.sin(np.linspace(0, 4*np.pi, 731))
        + np.random.normal(0, 400, 731)
    )

    rentals = np.maximum(rentals, 500)

    temp = np.clip(np.random.normal(0.5, 0.18, 731), 0.05, 0.95)
    wind = np.clip(np.random.normal(0.2, 0.08, 731), 0.05, 0.5)

    df = pd.DataFrame({
        'ds': dates,
        'y': rentals,
        'temp': temp,
        'windspeed': wind
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )

    model.add_regressor('temp')
    model.add_regressor('windspeed')

    model.fit(df)

    return model, df

model, df = train_model()

# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">BIKEIQ</div>', unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "⚔ Dashboard",
            "🔮 Prediction",
            "📅 Forecast",
            "📊 Analysis"
        ],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">CHROMATIC HAZARD</div>
    <div class="hero-sub">Bike Rental Intelligence • Dystopian Forecast System</div>
    <div class="jp-text">退廃的 × ディストピア / 管理された美しさ</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────
if page == "⚔ Dashboard":

    total = int(df['y'].sum())
    avg = int(df['y'].mean())
    peak = int(df['y'].max())
    low = int(df['y'].min())

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">TOTAL RENTALS</div>
            <div class="stat-value">{total:,}</div>
            <div class="stat-desc">Network-wide usage</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">DAILY AVERAGE</div>
            <div class="stat-value">{avg:,}</div>
            <div class="stat-desc">Projected mean</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">PEAK DEMAND</div>
            <div class="stat-value">{peak:,}</div>
            <div class="stat-desc">Maximum detected</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">LOWEST DAY</div>
            <div class="stat-value">{low:,}</div>
            <div class="stat-desc">Minimum observed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">SYSTEM TIMELINE</div>', unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        line=dict(color='#b56cff', width=2),
        fill='tozeroy',
        fillcolor='rgba(181,108,255,0.08)'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────
elif page == "🔮 Prediction":

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">INPUT PARAMETERS</div>', unsafe_allow_html=True)

        pred_date = st.date_input(
            "Date",
            value=datetime.now()
        )

        temp = st.slider(
            "Temperature",
            0.05,
            0.95,
            0.5,
            0.01
        )

        wind = st.slider(
            "Wind Speed",
            0.05,
            0.5,
            0.2,
            0.01
        )

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Cloudy",
                "Rain"
            ]
        )

        run = st.button("EXECUTE FORECAST")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:

        if run:
            input_df = pd.DataFrame({
                'ds': [pd.Timestamp(pred_date)],
                'temp': [temp],
                'windspeed': [wind]
            })

            forecast = model.predict(input_df)
            pred = int(max(0, forecast['yhat'].iloc[0]))

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-number">{pred:,}</div>
                <div class="prediction-label">EXPECTED RENTALS</div>
            </div>
            """, unsafe_allow_html=True)

            fig2 = go.Figure(go.Indicator(
                mode='gauge+number',
                value=pred,
                title={'text': 'Demand Level'},
                gauge={
                    'axis': {'range': [0, 9000]},
                    'bar': {'color': '#b56cff'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0
                }
            ))

            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350
            )

            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────────────
elif page == "📅 Forecast":

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">LONG RANGE FORECAST</div>', unsafe_allow_html=True)

    days = st.slider(
        "Forecast Days",
        7,
        365,
        90
    )

    run_forecast = st.button("GENERATE TIMELINE")

    if run_forecast:

        future = pd.DataFrame({
            'ds': pd.date_range(datetime.now(), periods=days),
            'temp': np.random.uniform(0.3, 0.7, days),
            'windspeed': np.random.uniform(0.1, 0.3, days)
        })

        fc = model.predict(future)

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=fc['ds'],
            y=fc['yhat_upper'],
            line=dict(width=0),
            showlegend=False
        ))

        fig3.add_trace(go.Scatter(
            x=fc['ds'],
            y=fc['yhat_lower'],
            fill='tonexty',
            fillcolor='rgba(181,108,255,0.12)',
            line=dict(width=0),
            name='Confidence'
        ))

        fig3.add_trace(go.Scatter(
            x=fc['ds'],
            y=fc['yhat'],
            line=dict(color='#ff6b6b', width=3),
            name='Prediction'
        ))

        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )

        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────
elif page == "📊 Analysis":

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">TEMPERATURE IMPACT</div>', unsafe_allow_html=True)

        fig4 = go.Figure(go.Scatter(
            x=df['temp'],
            y=df['y'],
            mode='markers',
            marker=dict(
                color=df['y'],
                colorscale='plasma',
                size=5,
                opacity=0.7
            )
        ))

        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350
        )

        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">SYSTEM DISTRIBUTION</div>', unsafe_allow_html=True)

        fig5 = go.Figure(go.Histogram(
            x=df['y'],
            nbinsx=40,
            marker=dict(color='#ff6b6b')
        ))

        fig5.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350
        )

        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
