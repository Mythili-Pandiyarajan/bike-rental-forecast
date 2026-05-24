import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="BikeIQ — Rental Forecasting",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0e17;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --accent3: #f59e0b;
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

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── SIDEBAR RADIO NAV — IMPROVED ── */
div[data-testid="stSidebar"] .stRadio > div {
    gap: 8px !important;
    flex-direction: column !important;
}
/* Hide default radio circle */
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Navigation buttons */
div[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: 0.7rem !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 1rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.05) !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    user-select: none !important;
}
/* Hover effect */
div[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,229,255,0.15) !important;
    color: #00e5ff !important;
    border-color: rgba(0,229,255,0.45) !important;
    transform: translateX(5px);
}
/* Active selected page */
div[data-testid="stSidebar"] .stRadio [aria-checked="true"] {
    background: rgba(0,229,255,0.18) !important;
    border-radius: 12px !important;
}
/* Remove default baseweb padding */
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    background: transparent !important;
    padding: 0 !important;
    width: 100% !important;
}
/* Click hint */
.nav-hint {
    font-size: 0.72rem;
    color: #94a3b8;
    text-align: center;
    margin-top: 0.5rem;
    font-style: italic;
}

/* ── HEADER BANNER ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1220 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-top: 0.75rem;
}

/* ── METRIC CARDS ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
    margin: 0.25rem 0;
}
.metric-delta {
    font-size: 0.78rem;
    color: var(--success);
}

/* ── SECTION HEADERS ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--text);
    font-weight: 700;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
}

/* ── PANELS ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.panel-dark {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* ── INPUT WIDGETS ── */
.stSlider > div > div > div > div { background: var(--accent) !important; }
.stSlider label { color: var(--text) !important; font-size: 0.85rem !important; }

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label { color: var(--muted) !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #7c3aed22) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff44, #7c3aed44) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.3) !important;
}

/* ── PREDICTION BOX ── */
.pred-box {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(124,58,237,0.08));
    border: 1px solid var(--accent);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,229,255,0.1);
}
.pred-number {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.pred-label { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.1em; }
.pred-range { color: var(--text); font-size: 0.95rem; margin-top: 0.75rem; }

/* ── INSIGHT TAGS ── */
.tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 0.2rem;
}
.tag-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.tag-yellow { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.tag-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.tag-blue { background: rgba(0,229,255,0.1); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.2), rgba(124,58,237,0.2)) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
}

/* ── TABLE ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; }

/* ── SIDEBAR LABEL ── */
.sidebar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 1.5rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── HELPER: TRAIN MODEL (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(data_path="data/Data/day.csv"):
    """Train Prophet model on the bike rental dataset."""
    try:
        df = pd.read_csv(data_path)
    except:
        # Generate synthetic data if CSV not available
        dates = pd.date_range("2011-01-01", periods=731, freq="D")
        np.random.seed(42)
        base = 4500 + 2000 * np.sin(np.linspace(0, 4 * np.pi, 731))
        noise = np.random.normal(0, 400, 731)
        trend = np.linspace(0, 1500, 731)
        cnt = np.maximum(500, base + noise + trend).astype(int)
        temp = 0.3 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 731)) + np.random.normal(0, 0.05, 731)
        wind = 0.2 + np.random.normal(0, 0.07, 731)
        df = pd.DataFrame({"dteday": dates.strftime("%Y-%m-%d"), "cnt": cnt,
                           "temp": np.clip(temp, 0.05, 0.95), "windspeed": np.clip(wind, 0.05, 0.5)})

    df['dteday'] = pd.to_datetime(df['dteday'])
    Q1 = df['windspeed'].quantile(0.25)
    Q3 = df['windspeed'].quantile(0.75)
    df['windspeed'] = df['windspeed'].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

    train_df = pd.DataFrame({
        'ds': df['dteday'],
        'y': df['cnt'],
        'temp': df['temp'],
        'windspeed': df['windspeed']
    })

    model = Prophet(
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=10,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.add_regressor('temp')
    model.add_regressor('windspeed')
    model.fit(train_df)
    return model, df


@st.cache_data(show_spinner=False)
def get_analysis_data(data_path="data/Data/day.csv"):
    try:
        df = pd.read_csv(data_path)
        df['dteday'] = pd.to_datetime(df['dteday'])
    except:
        dates = pd.date_range("2011-01-01", periods=731, freq="D")
        np.random.seed(42)
        base = 4500 + 2000 * np.sin(np.linspace(0, 4 * np.pi, 731))
        noise = np.random.normal(0, 400, 731)
        trend = np.linspace(0, 1500, 731)
        cnt = np.maximum(500, base + noise + trend).astype(int)
        temp = 0.3 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 731)) + np.random.normal(0, 0.05, 731)
        wind = 0.2 + np.random.normal(0, 0.07, 731)
        season = ((pd.DatetimeIndex(dates).month % 12) // 3 + 1)
        weathersit = np.random.choice([1, 2, 3], size=731, p=[0.6, 0.3, 0.1])
        df = pd.DataFrame({"dteday": dates, "cnt": cnt, "temp": np.clip(temp, 0.05, 0.95),
                           "windspeed": np.clip(wind, 0.05, 0.5), "hum": np.random.uniform(0.4, 0.9, 731),
                           "season": season, "weathersit": weathersit,
                           "casual": (cnt * 0.2).astype(int), "registered": (cnt * 0.8).astype(int),
                           "yr": (pd.DatetimeIndex(dates).year == 2012).astype(int),
                           "mnth": pd.DatetimeIndex(dates).month,
                           "weekday": pd.DatetimeIndex(dates).dayofweek})
    return df


def get_weather_insight(temp_norm, wind_norm):
    temp_c = temp_norm * 47 - 8
    wind_kmh = wind_norm * 67
    insights = []
    if temp_c < 5:
        insights.append(("❄️ Cold weather", "red"))
    elif temp_c < 15:
        insights.append(("🌤 Cool weather", "yellow"))
    elif temp_c < 28:
        insights.append(("☀️ Ideal cycling", "green"))
    else:
        insights.append(("🌡 Hot day", "yellow"))

    if wind_kmh < 15:
        insights.append(("🌬 Calm winds", "green"))
    elif wind_kmh < 30:
        insights.append(("💨 Moderate wind", "yellow"))
    else:
        insights.append(("🌪 Strong winds", "red"))
    return insights


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🚲 BikeIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "",
        ["📊 Dashboard", "🔮 Predict", "📅 Forecast", "🔬 Analysis"],
        label_visibility="collapsed"
    )

    # Small hint so users know items are clickable
    st.markdown('<div class="nav-hint">👆 Click menu items to navigate</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-dark">
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#00e5ff;">Prophet + Regressors</div>
        <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem;">R² Score: 0.46</div>
        <div style="font-size:0.8rem;color:#64748b;">RMSE: 1371.53</div>
        <div style="font-size:0.8rem;color:#64748b;">MAE: 1023.93</div>
        <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem;">Trained on 2011–2012</div>
    </div>
    """, unsafe_allow_html=True)

    data_path = "data/Data/day.csv"

# ── LOAD MODEL & DATA ─────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, raw_df = train_model(data_path)
    df = get_analysis_data(data_path)


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Bike Rental Intelligence</div>
        <div class="hero-sub">Real-time analysis · Demand forecasting · Environmental insights</div>
        <div class="hero-badge">● LIVE DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ──
    total = int(df['cnt'].sum())
    avg_daily = int(df['cnt'].mean())
    peak_day = int(df['cnt'].max())
    yoy = int(df[df['yr'] == 1]['cnt'].mean()) - int(df[df['yr'] == 0]['cnt'].mean())

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Rentals</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-delta">▲ All time</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Daily Average</div>
            <div class="metric-value">{avg_daily:,}</div>
            <div class="metric-delta">▲ Across 731 days</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Peak Day</div>
            <div class="metric-value">{peak_day:,}</div>
            <div class="metric-delta">▲ Single day record</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">YoY Growth</div>
            <div class="metric-value">+{yoy:,}</div>
            <div class="metric-delta">▲ 2012 vs 2011</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── MAIN TREND CHART ──
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Rental Trend — Full Timeline</div></div>', unsafe_allow_html=True)

        df_plot = df.copy()
        df_plot['7d_avg'] = df_plot['cnt'].rolling(7).mean()
        df_plot['30d_avg'] = df_plot['cnt'].rolling(30).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['dteday'], y=df_plot['cnt'],
            mode='lines', name='Daily',
            line=dict(color='rgba(0,229,255,0.3)', width=1),
            fill='tozeroy', fillcolor='rgba(0,229,255,0.04)'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['dteday'], y=df_plot['7d_avg'],
            mode='lines', name='7-Day Avg',
            line=dict(color='#00e5ff', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['dteday'], y=df_plot['30d_avg'],
            mode='lines', name='30-Day Avg',
            line=dict(color='#7c3aed', width=2, dash='dot')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='DM Sans'),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">By Season</div></div>', unsafe_allow_html=True)
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        season_avg = df.groupby('season')['cnt'].mean().reset_index()
        season_avg['season_name'] = season_avg['season'].map(season_map)
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        fig2 = go.Figure(go.Bar(
            x=season_avg['cnt'], y=season_avg['season_name'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=0)),
            text=season_avg['cnt'].round(0).astype(int),
            textposition='inside', textfont=dict(color='white', size=10)
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='DM Sans'),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showticklabels=False),
            yaxis=dict(gridcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── MONTHLY + WEATHER ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Monthly Pattern</div></div>', unsafe_allow_html=True)
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly = df.groupby('mnth')['cnt'].mean().reset_index()
        fig3 = go.Figure(go.Bar(
            x=[months[m-1] for m in monthly['mnth']],
            y=monthly['cnt'],
            marker=dict(
                color=monthly['cnt'],
                colorscale=[[0,'#1a2235'],[0.5,'#00e5ff'],[1,'#7c3aed']],
                showscale=False
            )
        ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='DM Sans'),
            height=240, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Temp vs Rentals</div></div>', unsafe_allow_html=True)
        fig4 = go.Figure(go.Scatter(
            x=df['temp'], y=df['cnt'],
            mode='markers',
            marker=dict(
                color=df['cnt'], colorscale='plasma',
                size=4, opacity=0.6, showscale=False
            ),
            text=df['dteday'].dt.strftime('%b %Y'),
            hovertemplate='<b>%{text}</b><br>Temp: %{x:.2f}<br>Rentals: %{y}<extra></extra>'
        ))
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='DM Sans'),
            height=240, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='Normalized Temperature', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Daily Rentals', gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE DAY PREDICT
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Single Day Prediction</div>
        <div class="hero-sub">Enter environmental conditions to predict daily bike rental demand</div>
        <div class="hero-badge">● PROPHET + REGRESSORS</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Input Conditions</div></div>', unsafe_allow_html=True)

        pred_date = st.date_input(
            "📅 Select Date",
            value=datetime(2026, 1, 1),
            min_value=datetime(2013, 1, 1),
            max_value=datetime(2045, 12, 31)
        )

        st.markdown("**🌡 Temperature** (Normalized 0–1)")
        temp_input = st.slider("", min_value=0.05, max_value=0.95, value=0.5, step=0.01, key="temp_slider", label_visibility="collapsed")
        temp_c = round(temp_input * 47 - 8, 1)
        st.caption(f"≈ {temp_c}°C actual temperature")

        st.markdown("**💨 Wind Speed** (Normalized 0–1)")
        wind_input = st.slider("", min_value=0.05, max_value=0.5, value=0.2, step=0.01, key="wind_slider", label_visibility="collapsed")
        wind_kmh = round(wind_input * 67, 1)
        st.caption(f"≈ {wind_kmh} km/h actual wind speed")

        st.markdown("**🌤 Weather Condition**")
        weather_label = st.selectbox("", ["Clear / Partly Cloudy", "Misty / Cloudy", "Light Rain or Snow"], label_visibility="collapsed")
        weather_map = {"Clear / Partly Cloudy": 1, "Misty / Cloudy": 2, "Light Rain or Snow": 3}
        weather_val = weather_map[weather_label]

        weather_penalty = {1: 1.0, 2: 0.92, 3: 0.72}

        st.markdown("")
        predict_btn = st.button("⚡ PREDICT DEMAND", key="predict_btn")

    with col2:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Prediction Result</div></div>', unsafe_allow_html=True)

        if predict_btn:
            input_df = pd.DataFrame({
                'ds': [pd.Timestamp(pred_date)],
                'temp': [temp_input],
                'windspeed': [wind_input]
            })
            forecast = model.predict(input_df)
            yhat = max(0, int(forecast['yhat'].iloc[0] * weather_penalty[weather_val]))
            yhat_lower = max(0, int(forecast['yhat_lower'].iloc[0] * weather_penalty[weather_val]))
            yhat_upper = max(0, int(forecast['yhat_upper'].iloc[0] * weather_penalty[weather_val]))

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Predicted Rentals</div>
                <div class="pred-number">{yhat:,}</div>
                <div class="pred-label">bikes / day</div>
                <div class="pred-range">Range: <b>{yhat_lower:,}</b> — <b>{yhat_upper:,}</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            # Insights
            insights = get_weather_insight(temp_input, wind_input)
            tag_html = ""
            for label, color in insights:
                tag_html += f'<span class="tag tag-{color}">{label}</span>'
            if weather_val == 3:
                tag_html += '<span class="tag tag-red">🌧 Rainy day penalty</span>'
            elif weather_val == 2:
                tag_html += '<span class="tag tag-yellow">🌫 Overcast penalty</span>'
            st.markdown(f'<div style="margin-top:1rem">{tag_html}</div>', unsafe_allow_html=True)

            # Mini gauge
            avg_rental = int(df['cnt'].mean())
            pct = min(100, int(yhat / df['cnt'].max() * 100))
            diff = yhat - avg_rental
            diff_str = f"▲ +{diff:,}" if diff > 0 else f"▼ {diff:,}"
            diff_color = "#10b981" if diff > 0 else "#ef4444"

            st.markdown(f"""
            <div class="panel" style="margin-top:1rem">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.75rem">
                    <span style="font-size:0.8rem;color:var(--muted)">vs Daily Average ({avg_rental:,})</span>
                    <span style="font-size:0.85rem;color:{diff_color};font-family:'Space Mono',monospace">{diff_str}</span>
                </div>
                <div style="background:var(--surface2);border-radius:4px;height:6px;overflow:hidden">
                    <div style="background:linear-gradient(90deg,#00e5ff,#7c3aed);width:{pct}%;height:100%;border-radius:4px;transition:width 0.5s"></div>
                </div>
                <div style="font-size:0.75rem;color:var(--muted);margin-top:0.4rem">{pct}% of peak demand</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:320px;display:flex;align-items:center;justify-content:center;border:1px dashed rgba(0,229,255,0.2);border-radius:12px">
                <div style="text-align:center;color:#64748b">
                    <div style="font-size:2.5rem;margin-bottom:0.5rem">🔮</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem">Set conditions and click<br>PREDICT DEMAND</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — FUTURE FORECAST
# ════════════════════════════════════════════════════════════════════
elif page == "📅 Forecast":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Future Forecast</div>
        <div class="hero-sub">Multi-day demand forecasting with confidence intervals — up to 20 years ahead</div>
        <div class="hero-badge">● TIME SERIES PROJECTION</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Forecast Settings</div></div>', unsafe_allow_html=True)

        st.markdown("**📅 Forecast Start**")
        _sc1, _sc2 = st.columns(2)
        with _sc1:
            start_year = st.number_input("Year", min_value=2013, max_value=2045, value=2026, step=1, key="start_year")
        with _sc2:
            start_month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1, key="start_month")
        start_date = datetime(int(start_year), int(start_month), 1)

        forecast_days = st.number_input(
            "Forecast Days (max 7300 = 20 years)",
            min_value=7,
            max_value=7300,                    # ← 20 years
            value=365,
            step=1
        )
        forecast_days = int(forecast_days)     # ← FIXED: was accidentally outdented before

        st.markdown("**Expected Avg Temp**")
        avg_temp = st.slider("avg_temp", 0.1, 0.9, 0.45, 0.05, label_visibility="collapsed")
        st.caption(f"≈ {round(avg_temp * 47 - 8, 1)}°C")

        st.markdown("**Expected Avg Wind**")
        avg_wind = st.slider("avg_wind", 0.05, 0.45, 0.2, 0.05, label_visibility="collapsed")
        st.caption(f"≈ {round(avg_wind * 67, 1)} km/h")

        season_sel = st.selectbox("Season", ["Auto (by date)", "Winter", "Spring", "Summer", "Fall"])

        run_forecast = st.button("📅 RUN FORECAST", key="forecast_btn")

    with col2:
        if run_forecast:
            future_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')

            # Seasonal temp variation
            if season_sel == "Auto (by date)":
                month_vals = future_dates.month
                temp_seasonal = avg_temp + 0.15 * np.sin((month_vals - 4) * np.pi / 6)
                temp_seasonal = np.clip(temp_seasonal, 0.05, 0.95)
            else:
                season_temps = {"Winter": 0.2, "Spring": 0.5, "Summer": 0.75, "Fall": 0.45}
                base_temp = season_temps[season_sel]
                temp_seasonal = np.full(forecast_days, base_temp) + np.random.normal(0, 0.03, forecast_days)
                temp_seasonal = np.clip(temp_seasonal, 0.05, 0.95)

            wind_var = avg_wind + np.random.normal(0, 0.02, forecast_days)
            wind_var = np.clip(wind_var, 0.05, 0.5)

            future_df = pd.DataFrame({
                'ds': future_dates,
                'temp': temp_seasonal,
                'windspeed': wind_var
            })

            forecast = model.predict(future_df)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

            # ── SUMMARY METRICS ──
            total_pred = int(forecast['yhat'].sum())
            avg_pred = int(forecast['yhat'].mean())
            peak_pred = int(forecast['yhat'].max())
            peak_date = forecast.loc[forecast['yhat'].idxmax(), 'ds'].strftime('%b %d')

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Forecast</div>
                    <div class="metric-value">{total_pred:,}</div>
                    <div class="metric-delta">{forecast_days} days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Daily Average</div>
                    <div class="metric-value">{avg_pred:,}</div>
                    <div class="metric-delta">projected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Peak Day</div>
                    <div class="metric-value">{peak_pred:,}</div>
                    <div class="metric-delta">{peak_date}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">80%</div>
                    <div class="metric-delta">interval</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── FORECAST CHART ──
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'],
                mode='lines', line=dict(width=0), showlegend=False, name='Upper'
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0,229,255,0.08)',
                name='80% Confidence'
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                mode='lines+markers',
                line=dict(color='#00e5ff', width=2.5),
                marker=dict(size=5, color='#00e5ff'),
                name='Forecast',
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>Predicted: %{y:,.0f} rentals<extra></extra>'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='DM Sans'),
                height=340, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Predicted Rentals'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── DOWNLOAD TABLE ──
            st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Forecast Table</div></div>', unsafe_allow_html=True)
            result_df = pd.DataFrame({
                'Date': forecast['ds'].dt.strftime('%Y-%m-%d'),
                'Predicted Rentals': forecast['yhat'].round(0).astype(int),
                'Lower Bound': forecast['yhat_lower'].round(0).astype(int),
                'Upper Bound': forecast['yhat_upper'].round(0).astype(int),
                'Temp (norm)': future_df['temp'].round(3).values,
                'Wind (norm)': future_df['windspeed'].round(3).values
            })
            st.dataframe(result_df, use_container_width=True, height=250)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Forecast CSV", csv, f"bike_forecast_{forecast_days}d.csv", "text/csv")

        else:
            st.markdown("""
            <div style="height:400px;display:flex;align-items:center;justify-content:center;border:1px dashed rgba(0,229,255,0.2);border-radius:12px">
                <div style="text-align:center;color:#64748b">
                    <div style="font-size:3rem;margin-bottom:0.75rem">📅</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem">Configure settings and click<br>RUN FORECAST</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSIS
# ════════════════════════════════════════════════════════════════════
elif page == "🔬 Analysis":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Deep Analysis</div>
        <div class="hero-sub">Correlation analysis · Feature impact · Model performance · Distribution insights</div>
        <div class="hero-badge">● EXPLORATORY DATA ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "🌦 Weather Impact", "🏆 Model Performance"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['cnt'], nbinsx=40,
                marker=dict(color='#00e5ff', opacity=0.7, line=dict(color='#0a0e17', width=0.5)),
                name='cnt'
            ))
            fig.update_layout(
                title=dict(text='Daily Rental Distribution', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=260,
                margin=dict(l=0, r=0, t=35, b=0),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Box(
                x=df['cnt'], marker_color='#7c3aed',
                line=dict(color='#7c3aed'), fillcolor='rgba(124,58,237,0.2)',
                boxmean='sd', name='Distribution'
            ))
            fig2.update_layout(
                title=dict(text='Box Plot with Std Dev', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=260,
                margin=dict(l=0, r=0, t=35, b=0),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
            weekday_avg = df.groupby('weekday')['cnt'].mean().reset_index()
            fig3 = go.Figure(go.Bar(
                x=[days[d] for d in weekday_avg['weekday']],
                y=weekday_avg['cnt'],
                marker=dict(color='#f59e0b', opacity=0.8)
            ))
            fig3.update_layout(
                title=dict(text='Avg Rentals by Weekday', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=260,
                margin=dict(l=0, r=0, t=35, b=0),
                xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            yr_data = df.groupby(['yr','mnth'])['cnt'].mean().reset_index()
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig4 = go.Figure()
            for yr, color, name in [(0,'#3b82f6','2011'),(1,'#00e5ff','2012')]:
                d = yr_data[yr_data['yr']==yr]
                fig4.add_trace(go.Scatter(
                    x=[months[m-1] for m in d['mnth']], y=d['cnt'],
                    mode='lines+markers', name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color)
                ))
            fig4.update_layout(
                title=dict(text='Year-over-Year Comparison', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=260,
                margin=dict(l=0, r=0, t=35, b=0),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        num_cols = ['cnt','temp','hum','windspeed']
        if all(c in df.columns for c in num_cols):
            corr = df[num_cols].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0,'#ef4444'],[0.5,'#1a2235'],[1,'#00e5ff']],
                zmid=0, text=corr.round(2).values,
                texttemplate='%{text}', textfont=dict(size=12),
                zmin=-1, zmax=1
            ))
            fig.update_layout(
                title=dict(text='Feature Correlation Matrix', font=dict(color='#e2e8f0', size=14)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=380,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Feature vs Target Scatter</div></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, feature, color in [(col1,'temp','#00e5ff'),(col2,'hum','#7c3aed'),(col3,'windspeed','#f59e0b')]:
            if feature in df.columns:
                with col:
                    fig = go.Figure(go.Scatter(
                        x=df[feature], y=df['cnt'], mode='markers',
                        marker=dict(color=color, size=3, opacity=0.5)
                    ))
                    fig.update_layout(
                        title=dict(text=f'{feature} vs cnt', font=dict(color='#e2e8f0', size=12)),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'), height=220,
                        margin=dict(l=0, r=0, t=35, b=0),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            season_box = {1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'}
            df['season_name'] = df['season'].map(season_box)
            fig = go.Figure()
            colors = {'Winter':'#3b82f6','Spring':'#10b981','Summer':'#f59e0b','Fall':'#ef4444'}
            for s in ['Winter','Spring','Summer','Fall']:
                d = df[df['season_name']==s]['cnt']
                fig.add_trace(go.Box(y=d, name=s, marker_color=colors[s],
                                     line=dict(color=colors[s]),
                                     fillcolor=colors[s]+'33'))
            fig.update_layout(
                title=dict(text='Rentals by Season', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'weathersit' in df.columns:
                weather_labels = {1:'Clear', 2:'Misty', 3:'Light Rain/Snow'}
                df['weather_label'] = df['weathersit'].map(weather_labels)
                weather_avg = df.groupby('weather_label')['cnt'].mean().reset_index()
                fig2 = go.Figure(go.Bar(
                    x=weather_avg['weather_label'], y=weather_avg['cnt'],
                    marker=dict(color=['#10b981','#f59e0b','#ef4444'], opacity=0.85)
                ))
                fig2.update_layout(
                    title=dict(text='Rentals by Weather Condition', font=dict(color='#e2e8f0', size=13)),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title">Model Comparison — All Models</div></div>', unsafe_allow_html=True)

        results = pd.DataFrame({
            'Model': ['AR','ARIMA','SARIMA(4,1,1)(1,1,1,7)','SARIMA(1,1,1)',
                      'Prophet','Prophet+Temp','Prophet+Temp+Wind ✅','Prophet+All Features'],
            'RMSE': [1806.04, 2112.30, 2867.74, 2091.37, 1495.69, 1405.34, 1371.53, 1779.54],
            'MAE':  [1322.05, 1522.11, 2042.82, 1514.71, 1088.27, 1055.06, 1023.93, 1278.97],
            'R2':   [0.07, -0.27, -1.34, -0.24, 0.36, 0.44, 0.46, 0.10]
        })

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=results['Model'], y=results['RMSE'],
                marker=dict(
                    color=['#ef4444' if r > 1500 else '#10b981' for r in results['RMSE']],
                    opacity=0.8
                ),
                text=results['RMSE'], textposition='outside',
                textfont=dict(size=10, color='#94a3b8')
            ))
            fig.update_layout(
                title=dict(text='RMSE (Lower = Better)', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=320,
                margin=dict(l=0, r=0, t=40, b=80),
                xaxis=dict(tickangle=45, gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Bar(
                x=results['Model'], y=results['R2'],
                marker=dict(
                    color=['#10b981' if r > 0.3 else ('#f59e0b' if r > 0 else '#ef4444') for r in results['R2']],
                    opacity=0.8
                ),
                text=results['R2'], textposition='outside',
                textfont=dict(size=10, color='#94a3b8')
            ))
            fig2.update_layout(
                title=dict(text='R² Score (Higher = Better)', font=dict(color='#e2e8f0', size=13)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=320,
                margin=dict(l=0, r=0, t=40, b=80),
                xaxis=dict(tickangle=45, gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(results.style.highlight_max(subset=['R2'], color='rgba(16,185,129,0.3)')
                                   .highlight_min(subset=['RMSE','MAE'], color='rgba(16,185,129,0.3)'),
                     use_container_width=True)
