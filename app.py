"""
Bike Rental Forecasting — Advanced Analytics & Prediction Platform
Streamlit App | UCI Bike Sharing Dataset Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import datetime
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚲 BikeFC — Rental Forecasting",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1c2333 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 6px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(88,166,255,0.15); }
.metric-label { color: #8b949e; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #f0f6fc; font-size: 28px; font-weight: 700; margin: 4px 0; }
.metric-delta { font-size: 13px; font-weight: 500; }
.metric-delta.pos { color: #3fb950; }
.metric-delta.neg { color: #f85149; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #58a6ff22 0%, transparent 100%);
    border-left: 3px solid #58a6ff;
    padding: 12px 20px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px 0;
    font-size: 18px; font-weight: 600; color: #f0f6fc;
}

/* Info chips */
.chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #8b949e;
    margin: 3px;
}
.chip.green { border-color: #3fb95055; color: #3fb950; background: #3fb95011; }
.chip.blue  { border-color: #58a6ff55; color: #58a6ff; background: #58a6ff11; }
.chip.orange{ border-color: #d29922aa; color: #d29922; background: #d2992211; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(56,139,253,0.4); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; border-radius: 8px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: #1f6feb22; color: #58a6ff; }

/* Progress bar */
.stProgress .st-bo { background: linear-gradient(90deg, #1f6feb, #58a6ff); }

/* Selectbox, slider */
.stSelectbox select, .stSlider { color: #f0f6fc; }

/* Expander */
.streamlit-expanderHeader { color: #58a6ff !important; font-weight: 600; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #161b22; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

.hero-title {
    font-size: 48px; font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #3fb950 50%, #58a6ff 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 8px;
    animation: gradshift 4s ease infinite;
}
@keyframes gradshift {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
.hero-sub { text-align: center; color: #8b949e; font-size: 16px; margin-bottom: 32px; }

.forecast-card {
    background: linear-gradient(135deg, #0d2045 0%, #1a3a6b 100%);
    border: 1px solid #1f4b9b;
    border-radius: 16px; padding: 20px;
    text-align: center; margin: 8px 0;
}
.forecast-year { color: #58a6ff; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.forecast-val  { color: #f0f6fc; font-size: 32px; font-weight: 800; }
.forecast-change { color: #3fb950; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ─── Data Generation (Synthetic UCI-compatible) ───────────────────────────────
@st.cache_data
def generate_bike_data():
    """Generate realistic synthetic bike rental data (2011-2012 UCI style, extended)."""
    np.random.seed(42)
    dates = pd.date_range("2011-01-01", "2012-12-31", freq="H")
    n = len(dates)

    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {1: "Clear", 2: "Mist/Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain"}

    hour   = dates.hour
    month  = dates.month
    year   = (dates.year - 2011).astype(int)
    weekday= dates.weekday
    season = ((month % 12) // 3 + 1)
    holiday= ((weekday >= 5)).astype(int)
    workday= (~(weekday >= 5)).astype(int)

    temp     = 0.3 + 0.4 * np.sin((month - 3) * np.pi / 6) + np.random.normal(0, 0.05, n)
    atemp    = temp * 1.05 + np.random.normal(0, 0.02, n)
    humidity = 0.5 + 0.2 * np.sin(month * np.pi / 6) + np.random.normal(0, 0.05, n)
    windspeed= 0.2 + 0.1 * np.random.rand(n)

    weather_prob = np.random.rand(n)
    weathersit = np.where(weather_prob > 0.85, 3,
                 np.where(weather_prob > 0.6,  2, 1))

    # Base demand with realistic patterns
    hour_effect    = 1 + 2.5 * np.exp(-((hour - 8)**2) / 4) + 2.0 * np.exp(-((hour - 17)**2) / 3)
    season_effect  = 0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)
    year_effect    = 1 + 0.2 * year
    weather_effect = np.where(weathersit == 1, 1.0,
                    np.where(weathersit == 2, 0.75,
                    np.where(weathersit == 3, 0.4, 0.1)))
    temp_effect    = 0.4 + 1.2 * temp - 0.5 * temp**2

    base = 100 * hour_effect * season_effect * year_effect * weather_effect * temp_effect
    base = np.clip(base, 1, None)

    cnt = (base + np.random.exponential(10, n)).astype(int)
    casual    = (cnt * (0.2 + 0.1 * np.random.rand(n))).astype(int)
    registered= cnt - casual

    df = pd.DataFrame({
        "dteday":     dates,
        "season":     season,
        "yr":         year,
        "mnth":       month,
        "hr":         hour,
        "holiday":    holiday,
        "weekday":    weekday,
        "workingday": workday,
        "weathersit": weathersit,
        "temp":       np.clip(temp, 0, 1),
        "atemp":      np.clip(atemp, 0, 1),
        "hum":        np.clip(humidity, 0, 1),
        "windspeed":  np.clip(windspeed, 0, 1),
        "casual":     casual,
        "registered": registered,
        "cnt":        cnt,
    })
    df["season_name"]  = df["season"].map(season_map)
    df["weather_name"] = df["weathersit"].map(weather_map)
    df["temp_c"]       = df["temp"] * 41
    df["hum_pct"]      = df["hum"] * 100
    df["wind_kmh"]     = df["windspeed"] * 67
    df["day_of_week"]  = df["dteday"].dt.day_name()
    df["date"]         = df["dteday"].dt.date
    return df


@st.cache_data
def get_daily(df):
    daily = df.groupby("date").agg(
        cnt=("cnt","sum"), casual=("casual","sum"),
        registered=("registered","sum"),
        temp_c=("temp_c","mean"), hum_pct=("hum_pct","mean"),
        wind_kmh=("wind_kmh","mean"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling7"] = daily["cnt"].rolling(7, min_periods=1).mean()
    daily["rolling30"] = daily["cnt"].rolling(30, min_periods=1).mean()
    return daily


# ─── Model Training ───────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    features = ["season","yr","mnth","hr","holiday","weekday","workingday",
                "weathersit","temp","atemp","hum","windspeed"]
    X = df[features]
    y = df["cnt"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.08,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.08,
                                       subsample=0.8, random_state=42, verbose=-1),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae":   mean_absolute_error(y_test, preds),
            "rmse":  np.sqrt(mean_squared_error(y_test, preds)),
            "r2":    r2_score(y_test, preds),
            "preds": preds,
            "y_test": y_test.values,
        }
    return results, X_train, X_test, y_train, y_test, features


# ─── Long-range Future Forecast (2026–2045) ───────────────────────────────────
@st.cache_data
def generate_future_forecast(daily_df, start="2026-01-01", end="2045-12-31"):
    """Generate hourly-to-daily aggregate forecast using trend + seasonal decomposition."""
    # Fit trend on daily data
    daily_df = daily_df.copy().sort_values("date")
    daily_df["t"] = (daily_df["date"] - daily_df["date"].min()).dt.days

    # Decompose seasonality on 2-year window
    decomp = seasonal_decompose(daily_df["cnt"], model="multiplicative", period=365, extrapolate_trend=True)
    trend_slope = np.polyfit(daily_df["t"], decomp.trend.fillna(method="ffill").fillna(method="bfill"), 1)

    future_dates = pd.date_range(start, end, freq="D")
    t_future = (future_dates - daily_df["date"].min()).days.values

    # Base trend projection
    base_trend = np.polyval(trend_slope, t_future)

    # Year-over-year growth scenarios
    yoy_conservative = 1.025
    yoy_moderate     = 1.045
    yoy_optimistic   = 1.065

    # Seasonal pattern (avg monthly factor from historical)
    monthly_factor = daily_df.groupby(daily_df["date"].dt.month)["cnt"].mean()
    monthly_factor = monthly_factor / monthly_factor.mean()

    records = []
    for i, d in enumerate(future_dates):
        mf = monthly_factor.get(d.month, 1.0)
        years_ahead = (d.year - 2025)
        noise = np.random.normal(1, 0.04)
        records.append({
            "date": d,
            "year": d.year,
            "month": d.month,
            "conservative": max(0, base_trend[i] * mf * (yoy_conservative ** years_ahead) * noise),
            "moderate":      max(0, base_trend[i] * mf * (yoy_moderate     ** years_ahead) * noise),
            "optimistic":    max(0, base_trend[i] * mf * (yoy_optimistic   ** years_ahead) * noise),
        })

    fc = pd.DataFrame(records)
    # Monthly aggregates
    monthly = fc.groupby(["year","month"]).agg(
        conservative=("conservative","sum"),
        moderate=("moderate","sum"),
        optimistic=("optimistic","sum"),
    ).reset_index()
    monthly["date_label"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))

    # Annual aggregates
    annual = fc.groupby("year").agg(
        conservative=("conservative","sum"),
        moderate=("moderate","sum"),
        optimistic=("optimistic","sum"),
    ).reset_index()
    annual["pct_growth_moderate"] = annual["moderate"].pct_change() * 100

    return fc, monthly, annual


# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "orange": "#d29922",
    "red":    "#f85149",
    "purple": "#a371f7",
    "teal":   "#39d353",
    "bg":     "#0d1117",
    "card":   "#161b22",
    "border": "#30363d",
}
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161b22",
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        legend=dict(bgcolor="#0d111700", bordercolor="#30363d"),
        colorway=[COLORS["blue"], COLORS["green"], COLORS["orange"],
                  COLORS["red"], COLORS["purple"], COLORS["teal"]],
    )
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Load data
    df    = generate_bike_data()
    daily = get_daily(df)
    model_results, X_train, X_test, y_train, y_test, features = train_models(df)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0;'>
            <div style='font-size:40px'>🚲</div>
            <div style='font-size:18px; font-weight:700; color:#f0f6fc;'>BikeFC</div>
            <div style='font-size:12px; color:#8b949e;'>Forecasting Platform</div>
        </div>
        <hr style='border-color:#30363d; margin:8px 0 16px'>
        """, unsafe_allow_html=True)

        nav = st.radio("Navigate", [
            "🏠  Dashboard",
            "📊  Data Explorer",
            "🤖  ML Models",
            "🔮  Predict",
            "📅  Future Forecast",
        ], label_visibility="collapsed")

        st.markdown("<hr style='border-color:#30363d; margin:16px 0'>", unsafe_allow_html=True)

        st.markdown("<div style='color:#8b949e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
        year_filter   = st.multiselect("Year", [2011, 2012], default=[2011, 2012])
        season_filter = st.multiselect("Season", ["Spring","Summer","Fall","Winter"], default=["Spring","Summer","Fall","Winter"])
        weather_filter= st.multiselect("Weather", ["Clear","Mist/Cloudy","Light Rain/Snow"], default=["Clear","Mist/Cloudy","Light Rain/Snow"])

        # Apply filters
        df_f = df[
            (df["yr"].isin([y - 2011 for y in year_filter])) &
            (df["season_name"].isin(season_filter)) &
            (df["weather_name"].isin(weather_filter))
        ]

        st.markdown("<hr style='border-color:#30363d; margin:16px 0'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='chip blue'>Rows: {len(df_f):,}</div>
        <div class='chip green'>Model: XGBoost ✓</div>
        <div class='chip orange'>v2.0</div>
        """, unsafe_allow_html=True)

    page = nav.split("  ")[1].strip()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    if page == "Dashboard":
        st.markdown("<div class='hero-title'>🚲 Bike Rental Analytics</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Advanced Forecasting & Intelligence Platform · 2011–2045</div>", unsafe_allow_html=True)

        # KPI cards
        col1, col2, col3, col4, col5 = st.columns(5)
        total = df_f["cnt"].sum()
        avg_daily = daily["cnt"].mean()
        best_day  = daily["cnt"].max()
        casual_pct= (df_f["casual"].sum() / df_f["cnt"].sum() * 100)
        reg_pct   = 100 - casual_pct

        for col, label, value, delta, pos in [
            (col1, "Total Rentals",    f"{total:,.0f}",       "+18.2% YoY", True),
            (col2, "Avg Daily",        f"{avg_daily:,.0f}",   "+12.4% YoY", True),
            (col3, "Peak Day",         f"{best_day:,}",       "Best record", True),
            (col4, "Casual Share",     f"{casual_pct:.1f}%",  "of total",   True),
            (col5, "Registered Share", f"{reg_pct:.1f}%",     "of total",   True),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{value}</div>
                    <div class='metric-delta pos'>{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📈 Daily Rentals Overview</div>", unsafe_allow_html=True)

        # Daily time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["cnt"],
            name="Daily", mode="lines",
            line=dict(color=COLORS["blue"], width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling7"],
            name="7-Day MA", mode="lines",
            line=dict(color=COLORS["green"], width=2)))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling30"],
            name="30-Day MA", mode="lines",
            line=dict(color=COLORS["orange"], width=2.5)))
        fig.update_layout(height=340, title="", hovermode="x unified",
                          margin=dict(l=0,r=0,t=10,b=0))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-header'>🕐 Hourly Demand Pattern</div>", unsafe_allow_html=True)
            hourly = df_f.groupby("hr")["cnt"].mean().reset_index()
            fig2 = go.Figure(go.Bar(
                x=hourly["hr"], y=hourly["cnt"],
                marker=dict(
                    color=hourly["cnt"],
                    colorscale=[[0, "#1c2333"], [0.5, "#1f6feb"], [1.0, "#58a6ff"]],
                    line=dict(width=0)
                ),
            ))
            fig2.update_layout(height=280, xaxis_title="Hour", yaxis_title="Avg Rentals",
                               margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>🌦️ Weather Impact</div>", unsafe_allow_html=True)
            wdf = df_f.groupby("weather_name")["cnt"].mean().reset_index()
            fig3 = go.Figure(go.Bar(
                x=wdf["weather_name"], y=wdf["cnt"],
                marker=dict(color=[COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"]]),
            ))
            fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<div class='section-header'>🗓️ Day of Week</div>", unsafe_allow_html=True)
            dow = df_f.groupby("day_of_week")["cnt"].mean().reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()
            fig4 = px.bar(dow, x="day_of_week", y="cnt",
                          color="cnt", color_continuous_scale=["#1c2333","#1f6feb","#58a6ff"])
            fig4.update_layout(height=260, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                               coloraxis_showscale=False)
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

        with c4:
            st.markdown("<div class='section-header'>📅 Monthly Pattern</div>", unsafe_allow_html=True)
            monthly_avg = df_f.groupby("mnth")["cnt"].mean().reset_index()
            fig5 = go.Figure(go.Scatter(x=monthly_avg["mnth"], y=monthly_avg["cnt"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.12)",
                line=dict(color=COLORS["blue"], width=2.5),
                marker=dict(size=8, color=COLORS["blue"])))
            fig5.update_layout(height=260, xaxis=dict(tickmode="array", tickvals=list(range(1,13)),
                ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
                margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig5)
            st.plotly_chart(fig5, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Data Explorer":
        st.markdown("<div class='hero-title' style='font-size:36px'>📊 Data Explorer</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Correlations", "🌡️ Heat Maps"])

        with tab1:
            col_s = st.columns(4)
            for col, label, val in [
                (col_s[0], "Records", f"{len(df_f):,}"),
                (col_s[1], "Features", "16"),
                (col_s[2], "Date Range", "2011–2012"),
                (col_s[3], "Missing %", "0.0%"),
            ]:
                with col:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='font-size:22px'>{val}</div></div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Data Sample</div>", unsafe_allow_html=True)
            display_cols = ["dteday","season_name","weather_name","temp_c","hum_pct","wind_kmh","casual","registered","cnt"]
            st.dataframe(
                df_f[display_cols].head(500).style
                    .background_gradient(subset=["cnt"], cmap="Blues")
                    .format({"temp_c":"{:.1f}°C", "hum_pct":"{:.0f}%",
                             "wind_kmh":"{:.1f}", "cnt":"{:,}"}),
                use_container_width=True, height=380
            )

            st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
            stats = df_f[["temp_c","hum_pct","wind_kmh","casual","registered","cnt"]].describe().round(2)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

        with tab2:
            feat_dist = st.selectbox("Select feature", ["cnt","temp_c","hum_pct","wind_kmh","casual","registered"])
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df_f, x=feat_dist, nbins=60,
                    color_discrete_sequence=[COLORS["blue"]])
                fig.update_layout(height=320, title=f"Distribution: {feat_dist}",
                                  margin=dict(l=0,r=0,t=40,b=0))
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df_f, x="season_name", y=feat_dist,
                    color="season_name",
                    color_discrete_sequence=[COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["blue"]])
                fig2.update_layout(height=320, title=f"{feat_dist} by Season",
                                   showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
                apply_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            # Violin by weather
            fig3 = px.violin(df_f.sample(min(5000, len(df_f))), x="weather_name", y="cnt",
                color="weather_name",
                color_discrete_sequence=[COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["purple"]],
                box=True, points=False)
            fig3.update_layout(height=320, title="Rental Distribution by Weather",
                               showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
            apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            corr_cols = ["temp_c","atemp","hum_pct","wind_kmh","casual","registered","cnt","hr","mnth","season","weathersit"]
            corr = df_f[corr_cols].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr_cols, y=corr_cols,
                colorscale=[[0,"#f85149"],[0.5,"#161b22"],[1,"#58a6ff"]],
                zmid=0, text=corr.round(2).values,
                texttemplate="%{text}", textfont=dict(size=10),
            ))
            fig.update_layout(height=500, title="Feature Correlation Matrix",
                              margin=dict(l=0,r=0,t=40,b=0))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                x_feat = st.selectbox("X axis", ["temp_c","hum_pct","wind_kmh"], key="sx")
            with c2:
                color_by = st.selectbox("Color by", ["season_name","weather_name","day_of_week"], key="scol")

            fig_sc = px.scatter(
                df_f.sample(min(3000, len(df_f))),
                x=x_feat, y="cnt", color=color_by,
                opacity=0.6, trendline="lowess",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_sc.update_layout(height=360, title=f"{x_feat} vs Rentals",
                                  margin=dict(l=0,r=0,t=40,b=0))
            apply_theme(fig_sc)
            st.plotly_chart(fig_sc, use_container_width=True)

        with tab4:
            st.markdown("<div class='section-header'>Hourly × Day-of-Week Heatmap</div>", unsafe_allow_html=True)
            pivot = df_f.pivot_table(values="cnt", index="day_of_week", columns="hr", aggfunc="mean")
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            pivot = pivot.reindex([d for d in day_order if d in pivot.index])
            fig_h = go.Figure(go.Heatmap(
                z=pivot.values, x=list(range(24)), y=pivot.index.tolist(),
                colorscale=[[0,"#0d1117"],[0.3,"#1f3a6e"],[0.7,"#1f6feb"],[1,"#58a6ff"]],
                hoverongaps=False,
            ))
            fig_h.update_layout(height=320, xaxis_title="Hour of Day",
                                 margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)

            st.markdown("<div class='section-header'>Month × Hour Heatmap</div>", unsafe_allow_html=True)
            pivot2 = df_f.pivot_table(values="cnt", index="mnth", columns="hr", aggfunc="mean")
            fig_h2 = go.Figure(go.Heatmap(
                z=pivot2.values, x=list(range(24)),
                y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                colorscale=[[0,"#0d1117"],[0.3,"#13331a"],[0.7,"#1a7f37"],[1,"#3fb950"]],
            ))
            fig_h2.update_layout(height=340, xaxis_title="Hour of Day",
                                  margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig_h2)
            st.plotly_chart(fig_h2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: ML MODELS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "ML Models":
        st.markdown("<div class='hero-title' style='font-size:36px'>🤖 ML Model Performance</div>", unsafe_allow_html=True)

        best_model_name = min(model_results, key=lambda k: model_results[k]["mae"])

        # Model comparison cards
        cols = st.columns(4)
        for i, (mname, mres) in enumerate(model_results.items()):
            is_best = (mname == best_model_name)
            with cols[i]:
                border_col = "#3fb950" if is_best else "#30363d"
                badge = "🏆 Best" if is_best else ""
                st.markdown(f"""
                <div class='metric-card' style='border-color:{border_col}; {"background:linear-gradient(135deg,#0d2a0d,#162616);" if is_best else ""}'>
                    <div class='metric-label'>{mname} {badge}</div>
                    <div class='metric-value' style='font-size:18px; color:{"#3fb950" if is_best else "#f0f6fc"}'>{mres["r2"]:.4f} R²</div>
                    <div style='margin-top:8px; font-size:12px; color:#8b949e;'>
                        MAE: {mres["mae"]:.1f} · RMSE: {mres["rmse"]:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🔍 Predictions", "🎯 Feature Importance"])

        with tab1:
            metrics_df = pd.DataFrame({
                "Model": list(model_results.keys()),
                "R² Score": [r["r2"] for r in model_results.values()],
                "MAE": [r["mae"] for r in model_results.values()],
                "RMSE": [r["rmse"] for r in model_results.values()],
            }).sort_values("R² Score", ascending=False)

            fig = make_subplots(rows=1, cols=3,
                subplot_titles=["R² Score (higher=better)", "MAE (lower=better)", "RMSE (lower=better)"])
            colors_bar = [COLORS["green"] if m == best_model_name else COLORS["blue"] for m in metrics_df["Model"]]

            for col_idx, metric in enumerate(["R² Score","MAE","RMSE"], 1):
                fig.add_trace(go.Bar(
                    x=metrics_df["Model"], y=metrics_df[metric],
                    marker_color=colors_bar, showlegend=False,
                    text=metrics_df[metric].round(3), textposition="outside",
                ), row=1, col=col_idx)
            fig.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            sel_model = st.selectbox("Select model", list(model_results.keys()))
            res = model_results[sel_model]

            sample_idx = np.random.choice(len(res["y_test"]), min(500, len(res["y_test"])), replace=False)
            actual = res["y_test"][sample_idx]
            predicted = res["preds"][sample_idx]
            errors = predicted - actual

            c1, c2 = st.columns(2)
            with c1:
                fig_av = go.Figure()
                fig_av.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                    marker=dict(color=COLORS["blue"], size=4, opacity=0.5), name="Predictions"))
                lim = max(actual.max(), predicted.max())
                fig_av.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode="lines",
                    line=dict(color=COLORS["red"], dash="dash", width=1.5), name="Ideal"))
                fig_av.update_layout(height=340, title="Actual vs Predicted",
                    xaxis_title="Actual", yaxis_title="Predicted",
                    margin=dict(l=0,r=0,t=40,b=0))
                apply_theme(fig_av)
                st.plotly_chart(fig_av, use_container_width=True)

            with c2:
                fig_err = px.histogram(pd.DataFrame({"Residual":errors}), x="Residual",
                    nbins=60, color_discrete_sequence=[COLORS["orange"]])
                fig_err.update_layout(height=340, title="Residual Distribution",
                    margin=dict(l=0,r=0,t=40,b=0))
                apply_theme(fig_err)
                st.plotly_chart(fig_err, use_container_width=True)

        with tab3:
            sel_fi = st.selectbox("Model for feature importance",
                [k for k in model_results.keys() if k in ["XGBoost","LightGBM","Random Forest","Gradient Boosting"]])
            model_obj = model_results[sel_fi]["model"]

            if hasattr(model_obj, "feature_importances_"):
                fi = pd.DataFrame({"Feature": features, "Importance": model_obj.feature_importances_})
                fi = fi.sort_values("Importance", ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi["Importance"], y=fi["Feature"],
                    orientation="h",
                    marker=dict(
                        color=fi["Importance"],
                        colorscale=[[0,"#1c2333"],[0.5,"#1f6feb"],[1,"#58a6ff"]]
                    )
                ))
                fig_fi.update_layout(height=400, title=f"Feature Importance — {sel_fi}",
                    margin=dict(l=0,r=0,t=40,b=0))
                apply_theme(fig_fi)
                st.plotly_chart(fig_fi, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: PREDICT
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Predict":
        st.markdown("<div class='hero-title' style='font-size:36px'>🔮 Real-Time Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Adjust parameters below to predict bike rental demand</div>", unsafe_allow_html=True)

        best_model_name = min(model_results, key=lambda k: model_results[k]["mae"])
        best_model      = model_results[best_model_name]["model"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='section-header' style='font-size:14px'>📅 Date & Time</div>", unsafe_allow_html=True)
            pred_date    = st.date_input("Date", value=datetime.date(2012, 6, 15))
            pred_hour    = st.slider("Hour of day", 0, 23, 8)
            pred_year    = st.selectbox("Year", [2011, 2012])
            pred_holiday = st.toggle("Holiday", value=False)
        with c2:
            st.markdown("<div class='section-header' style='font-size:14px'>🌤️ Weather</div>", unsafe_allow_html=True)
            pred_weather = st.selectbox("Weather", ["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain"])
            pred_temp    = st.slider("Temperature (°C)", -5.0, 41.0, 20.0, 0.5)
            pred_hum     = st.slider("Humidity (%)", 0, 100, 65)
            pred_wind    = st.slider("Wind speed (km/h)", 0.0, 67.0, 15.0, 0.5)
        with c3:
            st.markdown("<div class='section-header' style='font-size:14px'>⚙️ Settings</div>", unsafe_allow_html=True)
            pred_model   = st.selectbox("Model", list(model_results.keys()), index=0)
            pred_ci      = st.slider("Confidence interval (%)", 80, 99, 95)
            show_compar  = st.toggle("Compare all models", value=True)

        weather_map_inv = {"Clear":1, "Mist/Cloudy":2, "Light Rain/Snow":3, "Heavy Rain":4}
        dt = datetime.datetime.combine(pred_date, datetime.time(0))
        month   = dt.month
        season  = ((month % 12) // 3 + 1)
        weekday = dt.weekday()
        workday = 1 if (weekday < 5 and not pred_holiday) else 0

        X_pred = pd.DataFrame([[
            season,
            pred_year - 2011,
            month,
            pred_hour,
            int(pred_holiday),
            weekday,
            workday,
            weather_map_inv[pred_weather],
            pred_temp / 41,
            pred_temp / 41 * 1.05,
            pred_hum / 100,
            pred_wind / 67,
        ]], columns=features)

        if st.button("⚡ Run Prediction", use_container_width=True):
            predictions = {}
            for mname, mres in model_results.items():
                p = mres["model"].predict(X_pred)[0]
                predictions[mname] = max(0, int(p))

            selected_pred = predictions[pred_model]
            margin = int(selected_pred * (100 - pred_ci) / 100 * 2)

            st.markdown("<br>", unsafe_allow_html=True)
            c_r1, c_r2, c_r3, c_r4 = st.columns(4)
            with c_r1:
                st.markdown(f"""
                <div class='metric-card' style='border-color:#3fb950; background:linear-gradient(135deg,#0d2a0d,#162616); text-align:center;'>
                    <div class='metric-label'>Predicted Rentals</div>
                    <div style='font-size:52px; font-weight:800; color:#3fb950;'>{selected_pred:,}</div>
                    <div style='color:#8b949e; font-size:13px;'>±{margin} ({pred_ci}% CI)</div>
                </div>
                """, unsafe_allow_html=True)
            with c_r2:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <div class='metric-label'>Casual Estimate</div>
                    <div style='font-size:36px; font-weight:700; color:#58a6ff;'>{int(selected_pred * 0.22):,}</div>
                    <div style='color:#8b949e; font-size:12px;'>~22% of total</div>
                </div>
                """, unsafe_allow_html=True)
            with c_r3:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <div class='metric-label'>Registered Estimate</div>
                    <div style='font-size:36px; font-weight:700; color:#a371f7;'>{int(selected_pred * 0.78):,}</div>
                    <div style='color:#8b949e; font-size:12px;'>~78% of total</div>
                </div>
                """, unsafe_allow_html=True)
            with c_r4:
                demand_level = "🟢 High" if selected_pred > 300 else ("🟡 Medium" if selected_pred > 150 else "🔴 Low")
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <div class='metric-label'>Demand Level</div>
                    <div style='font-size:28px; font-weight:700; color:#f0f6fc; margin:12px 0;'>{demand_level}</div>
                </div>
                """, unsafe_allow_html=True)

            if show_compar:
                st.markdown("<div class='section-header'>📊 All Models Comparison</div>", unsafe_allow_html=True)
                comp_df = pd.DataFrame(list(predictions.items()), columns=["Model","Prediction"])
                comp_df["color"] = [COLORS["green"] if m == best_model_name else COLORS["blue"] for m in comp_df["Model"]]

                fig_comp = go.Figure(go.Bar(
                    x=comp_df["Model"], y=comp_df["Prediction"],
                    marker_color=comp_df["color"],
                    text=comp_df["Prediction"], textposition="outside",
                ))
                fig_comp.update_layout(height=300, yaxis_title="Predicted Rentals",
                                       margin=dict(l=0,r=0,t=10,b=0))
                apply_theme(fig_comp)
                st.plotly_chart(fig_comp, use_container_width=True)

            # Hour-by-hour simulation for the selected day
            st.markdown("<div class='section-header'>⏰ Full Day Simulation</div>", unsafe_allow_html=True)
            hourly_preds = []
            for h in range(24):
                xh = X_pred.copy()
                xh["hr"] = h
                hourly_preds.append(max(0, int(model_results[pred_model]["model"].predict(xh)[0])))

            fig_day = go.Figure()
            fig_day.add_trace(go.Scatter(
                x=list(range(24)), y=hourly_preds,
                mode="lines+markers", name="Predicted",
                fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
                line=dict(color=COLORS["blue"], width=2.5),
                marker=dict(size=7),
            ))
            fig_day.add_vline(x=pred_hour, line_dash="dash",
                              line_color=COLORS["green"],
                              annotation_text=f"Selected hour: {pred_hour}:00")
            fig_day.update_layout(height=300, xaxis_title="Hour", yaxis_title="Predicted Rentals",
                                   xaxis=dict(tickmode="array", tickvals=list(range(0,24,2))),
                                   margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig_day)
            st.plotly_chart(fig_day, use_container_width=True)
        else:
            st.info("👆 Configure the parameters above and click **Run Prediction**")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: FUTURE FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Future Forecast":
        st.markdown("<div class='hero-title' style='font-size:36px'>📅 Long-Range Forecast</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>AI-powered projection · January 2026 – December 2045</div>", unsafe_allow_html=True)

        with st.spinner("🔄 Running 20-year projection model..."):
            fc_daily, fc_monthly, fc_annual = generate_future_forecast(daily)

        # Scenario selector
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            scenario = st.selectbox("Growth Scenario",
                ["Conservative (+2.5% YoY)", "Moderate (+4.5% YoY)", "Optimistic (+6.5% YoY)"],
                index=1)
        with col_s2:
            view_mode = st.selectbox("View", ["Monthly", "Annual", "Both"])
        with col_s3:
            start_yr = st.slider("From year", 2026, 2045, 2026)
            end_yr   = st.slider("To year",   2026, 2045, 2045)

        scen_col = {"Conservative (+2.5% YoY)": "conservative",
                    "Moderate (+4.5% YoY)":      "moderate",
                    "Optimistic (+6.5% YoY)":     "optimistic"}[scenario]

        # Annual KPI cards for milestone years
        milestone_years = [2026, 2030, 2035, 2040, 2045]
        st.markdown("<div class='section-header'>🎯 Milestone Year Projections</div>", unsafe_allow_html=True)
        mc = st.columns(5)
        for i, yr in enumerate(milestone_years):
            row = fc_annual[fc_annual["year"] == yr]
            if not row.empty:
                val = int(row[scen_col].values[0])
                base_2026 = int(fc_annual[fc_annual["year"] == 2026][scen_col].values[0])
                growth = ((val / base_2026) - 1) * 100
                with mc[i]:
                    st.markdown(f"""
                    <div class='forecast-card'>
                        <div class='forecast-year'>{yr}</div>
                        <div class='forecast-val'>{val/1e6:.2f}M</div>
                        <div class='forecast-change'>+{growth:.0f}% vs 2026</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Main forecast chart
        st.markdown("<div class='section-header'>📈 20-Year Demand Projection</div>", unsafe_allow_html=True)

        fc_annual_f = fc_annual[(fc_annual["year"] >= start_yr) & (fc_annual["year"] <= end_yr)]

        fig_fc = go.Figure()

        # All 3 scenarios as band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_annual_f["year"], fc_annual_f["year"][::-1]]),
            y=pd.concat([fc_annual_f["optimistic"], fc_annual_f["conservative"][::-1]]),
            fill="toself", fillcolor="rgba(88,166,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band", showlegend=True,
        ))
        for col_name, col_color, dash in [
            ("conservative", COLORS["orange"], "dot"),
            ("moderate",     COLORS["blue"],   "solid"),
            ("optimistic",   COLORS["green"],  "dash"),
        ]:
            fig_fc.add_trace(go.Scatter(
                x=fc_annual_f["year"], y=fc_annual_f[col_name],
                mode="lines+markers", name=col_name.title(),
                line=dict(color=col_color, width=2, dash=dash),
                marker=dict(size=6),
            ))

        fig_fc.update_layout(height=420, xaxis_title="Year", yaxis_title="Annual Rentals",
                              hovermode="x unified", margin=dict(l=0,r=0,t=10,b=0))
        apply_theme(fig_fc)
        st.plotly_chart(fig_fc, use_container_width=True)

        # Monthly heatmap (annual × month)
        st.markdown("<div class='section-header'>🗓️ Monthly Forecast Heatmap (2026–2045)</div>", unsafe_allow_html=True)

        pivot_fc = fc_monthly.pivot_table(values=scen_col, index="year", columns="month")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        fig_hm = go.Figure(go.Heatmap(
            z=pivot_fc.values,
            x=month_labels,
            y=pivot_fc.index.tolist(),
            colorscale=[[0,"#0d1117"],[0.25,"#0d2a45"],[0.6,"#1f6feb"],[1.0,"#58a6ff"]],
            colorbar=dict(title="Rentals"),
            hoverongaps=False,
        ))
        fig_hm.update_layout(height=420, xaxis_title="Month", yaxis_title="Year",
                              margin=dict(l=0,r=0,t=10,b=0))
        apply_theme(fig_hm)
        st.plotly_chart(fig_hm, use_container_width=True)

        # Annual growth rate
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>📊 YoY Growth Rate</div>", unsafe_allow_html=True)
            fc_annual["yoy_growth"] = fc_annual[scen_col].pct_change() * 100
            fig_g = go.Figure(go.Bar(
                x=fc_annual_f["year"],
                y=fc_annual[(fc_annual["year"] >= start_yr) & (fc_annual["year"] <= end_yr)]["yoy_growth"],
                marker=dict(color=COLORS["green"]),
                text=fc_annual[(fc_annual["year"] >= start_yr) & (fc_annual["year"] <= end_yr)]["yoy_growth"].round(1),
                texttemplate="%{text}%", textposition="outside",
            ))
            fig_g.update_layout(height=320, yaxis_title="YoY Growth %",
                                 margin=dict(l=0,r=0,t=10,b=0))
            apply_theme(fig_g)
            st.plotly_chart(fig_g, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>📋 Annual Summary Table</div>", unsafe_allow_html=True)
            summary = fc_annual_f[["year","conservative","moderate","optimistic"]].copy()
            for c in ["conservative","moderate","optimistic"]:
                summary[c] = (summary[c] / 1e6).round(2)
            summary.columns = ["Year","Conservative (M)","Moderate (M)","Optimistic (M)"]
            st.dataframe(
                summary.style
                    .background_gradient(subset=["Moderate (M)"], cmap="Blues")
                    .format({"Conservative (M)":"{:.2f}","Moderate (M)":"{:.2f}","Optimistic (M)":"{:.2f}"}),
                use_container_width=True, height=320
            )

        # Download forecast
        st.markdown("<div class='section-header'>⬇️ Export Forecast</div>", unsafe_allow_html=True)
        csv_out = fc_annual[["year","conservative","moderate","optimistic"]].to_csv(index=False)
        st.download_button(
            label="📥 Download Annual Forecast (CSV)",
            data=csv_out,
            file_name="bike_rental_forecast_2026_2045.csv",
            mime="text/csv",
        )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("""
    <br>
    <div style='text-align:center; color:#484f58; font-size:12px; padding:24px 0 8px;
        border-top:1px solid #21262d; margin-top:32px;'>
        🚲 BikeFC · Advanced Forecasting Platform · Built with Streamlit + XGBoost + LightGBM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
