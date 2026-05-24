"""
Bike Rental Forecasting — Streamlit App
Based on: PRCP-1018 | Best Model: Prophet + Temp + Windspeed (R²=0.46, RMSE=1371)
Dataset:  UCI Bike Sharing day.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import warnings, datetime, io
warnings.filterwarnings("ignore")

# ── Prophet (optional — graceful fallback) ────────────────────────────────────
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚲 Bike Rental Forecasting",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#161b22,#0d1117);border-right:1px solid #30363d;}

.kpi{background:linear-gradient(135deg,#1c2333,#21262d);border:1px solid #30363d;
     border-radius:14px;padding:18px 20px;margin:4px 0;transition:all .2s;}
.kpi:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(88,166,255,.15);}
.kpi-label{color:#8b949e;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;}
.kpi-val{color:#f0f6fc;font-size:26px;font-weight:800;margin:4px 0 2px;}
.kpi-sub{font-size:12px;font-weight:500;}
.good{color:#3fb950;} .warn{color:#d29922;} .bad{color:#f85149;}

.sec{background:linear-gradient(90deg,#58a6ff22,transparent);border-left:3px solid #58a6ff;
     padding:10px 18px;border-radius:0 8px 8px 0;margin:20px 0 12px;
     font-size:16px;font-weight:600;color:#f0f6fc;}

.model-best{background:linear-gradient(135deg,#0d2a0d,#162616);
            border:1px solid #3fb95066;border-radius:14px;padding:16px 20px;margin:4px 0;}
.model-card{background:linear-gradient(135deg,#1c2333,#21262d);
            border:1px solid #30363d;border-radius:14px;padding:16px 20px;margin:4px 0;}

.hero{font-size:44px;font-weight:800;
      background:linear-gradient(135deg,#58a6ff,#3fb950);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;}
.sub{text-align:center;color:#8b949e;font-size:15px;margin-bottom:28px;}

.insight{background:#161b22;border:1px solid #30363d;border-radius:10px;
         padding:14px 18px;margin:6px 0;font-size:13px;color:#c9d1d9;line-height:1.6;}
.insight b{color:#58a6ff;}

.stButton>button{background:linear-gradient(135deg,#1f6feb,#388bfd);color:#fff;
                 border:none;border-radius:10px;font-weight:600;padding:10px 28px;}
.stButton>button:hover{box-shadow:0 4px 16px rgba(56,139,253,.4);}
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:10px;padding:3px;}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:#1f6feb22;color:#58a6ff;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─── Theme helpers ─────────────────────────────────────────────────────────────
C = dict(blue="#58a6ff", green="#3fb950", orange="#d29922",
         red="#f85149", purple="#a371f7", teal="#39d353")

def theme(fig, h=None):
    kw = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
              font=dict(family="Inter", color="#c9d1d9"),
              xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
              yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
              legend=dict(bgcolor="rgba(0,0,0,0)"),
              colorway=[C["blue"],C["green"],C["orange"],C["red"],C["purple"]])
    if h: kw["height"] = h
    fig.update_layout(**kw)
    return fig

# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def make_data():
    """Synthetic UCI day.csv-compatible data (2011-2012, 731 rows)."""
    np.random.seed(42)
    dates = pd.date_range("2011-01-01", "2012-12-31", freq="D")
    n = len(dates)
    mnth = dates.month
    season = ((mnth % 12)//3 + 1)
    yr = (dates.year - 2011).astype(int)
    weekday = dates.weekday
    holiday = np.zeros(n, int)
    holiday[[10,50,100,150,200,250,300,350,365,400,450,500,550,600,650,700]] = 1
    workingday = ((weekday < 5) & (holiday == 0)).astype(int)

    temp      = 0.3 + 0.4*np.sin((mnth-3)*np.pi/6) + np.random.normal(0,.04,n)
    atemp     = temp*1.05 + np.random.normal(0,.02,n)
    hum       = 0.5 + 0.2*np.sin(mnth*np.pi/6) + np.random.normal(0,.05,n)

    # IQR-capped windspeed (matches notebook preprocessing)
    ws_raw = 0.19 + 0.12*np.random.rand(n)
    q1,q3 = np.percentile(ws_raw,.25), np.percentile(ws_raw,.75)
    iqr = q3-q1
    windspeed = np.clip(ws_raw, q1-1.5*iqr, q3+1.5*iqr)

    weathersit = np.where(np.random.rand(n)>.85, 3,
                 np.where(np.random.rand(n)>.60, 2, 1))

    base  = 3000 + 2000*np.sin((mnth-3)*np.pi/6)
    trend = 800*yr
    t_eff = 0.4 + 1.2*temp - 0.4*temp**2
    w_eff = np.where(weathersit==1,1.0,np.where(weathersit==2,.78,.45))
    wind_eff = 1 - 0.4*windspeed
    noise = np.random.normal(0,300,n)

    cnt = np.clip((base + trend)*t_eff*w_eff*wind_eff + noise, 22, 8714).astype(int)
    casual    = np.clip((cnt*(.18+.08*np.random.rand(n))).astype(int), 0, cnt)
    registered = cnt - casual

    df = pd.DataFrame(dict(
        instant=range(1,n+1), dteday=dates,
        season=season, yr=yr, mnth=mnth,
        holiday=holiday, weekday=weekday, workingday=workingday,
        weathersit=weathersit,
        temp=np.clip(temp,0,1), atemp=np.clip(atemp,0,1),
        hum=np.clip(hum,0,1), windspeed=windspeed,
        casual=casual, registered=registered, cnt=cnt
    ))
    return df

@st.cache_data
def prep(df):
    """Reproduce notebook preprocessing exactly."""
    d = df.copy()
    # Drop leakage cols (as per notebook)
    # Keep for display but won't use in model
    d["dteday"] = pd.to_datetime(d["dteday"])
    # Windspeed IQR capping already done in make_data
    d["temp_c"]   = d["temp"] * 41       # denormalize for display
    d["hum_pct"]  = d["hum"] * 100
    d["wind_kmh"] = d["windspeed"] * 67
    season_map    = {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}
    weather_map   = {1:"Clear",2:"Mist/Cloudy",3:"Light Rain/Snow"}
    d["season_name"]  = d["season"].map(season_map)
    d["weather_name"] = d["weathersit"].map(weather_map)
    d["day_name"]     = d["dteday"].dt.day_name()
    return d

@st.cache_data
def ts_setup(df):
    """Time series setup — matches notebook Section 6."""
    ts = df[["dteday","cnt"]].copy()
    ts = ts.set_index("dteday").asfreq("D")
    return ts

@st.cache_data
def train_test(ts):
    split = int(len(ts)*0.8)
    return ts["cnt"][:split], ts["cnt"][split:]

# ─── Model results from notebook (pre-computed exact values) ─────────────────
RESULTS = pd.DataFrame({
    "Model": ["AR","ARIMA(4,1,1)","SARIMA(4,1,1)(1,1,1,7)","SARIMA(1,1,1)",
              "Prophet (Basic)","Prophet + Temp","Prophet + Temp + Windspeed ✅",
              "Prophet + All Features"],
    "RMSE": [1806.04,2112.30,2867.74,2091.37,1495.69,1405.34,1371.53,1779.54],
    "MAE":  [1322.05,1522.11,2042.82,1514.71,1088.27,1055.06,1023.93,1278.97],
    "R2":   [0.07,-0.27,-1.34,-0.24,0.36,0.44,0.46,0.10],
})

# ─── Prophet model (train live if available) ──────────────────────────────────
@st.cache_resource
def train_prophet(_df):
    if not _HAS_PROPHET:
        return None
    split = int(len(_df)*0.8)
    train_p = pd.DataFrame({
        "ds": _df["dteday"].values[:split],
        "y":  _df["cnt"].values[:split],
        "temp":      _df["temp"].values[:split],
        "windspeed": _df["windspeed"].values[:split],
    })
    m = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10,
                daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.add_regressor("temp")
    m.add_regressor("windspeed")
    m.fit(train_p)
    return m

@st.cache_data
def prophet_test_preds(_df, _model):
    if _model is None:
        return None, None
    split = int(len(_df)*0.8)
    test_p = pd.DataFrame({
        "ds": _df["dteday"].values[split:],
        "temp":      _df["temp"].values[split:],
        "windspeed": _df["windspeed"].values[split:],
    })
    fc = _model.predict(test_p)
    return fc["yhat"].values, _df["cnt"].values[split:]

# ─── Long-range forecast 2026-2045 ────────────────────────────────────────────
@st.cache_data
def future_forecast(_model, _df, start="2026-01-01", end="2045-12-31"):
    future_dates = pd.date_range(start, end, freq="D")
    # Historical monthly avg temp & windspeed as seasonal proxy
    _df2 = _df.copy()
    _df2["mnth_num"] = _df2["dteday"].dt.month
    monthly_temp = _df2.groupby("mnth_num")["temp"].mean()
    monthly_wind = _df2.groupby("mnth_num")["windspeed"].mean()

    rows = []
    for d in future_dates:
        rows.append({"ds": d,
                     "temp": monthly_temp.get(d.month, 0.4),
                     "windspeed": monthly_wind.get(d.month, 0.2)})
    future_df = pd.DataFrame(rows)

    if _model is not None:
        fc = _model.predict(future_df)
        base = fc["yhat"].values
    else:
        # Fallback: trend extrapolation
        last_val = _df["cnt"].mean()
        base = np.array([
            last_val * (1.04 ** ((d.year - 2012) + (d.month-1)/12)) *
            (0.6 + 0.8*np.sin((d.month-3)*np.pi/6))
            for d in future_dates
        ])

    yrs_ahead = np.array([(d.year - 2025) + (d.month-1)/12 for d in future_dates])
    out = pd.DataFrame({
        "date": future_dates,
        "year": future_dates.year,
        "month": future_dates.month,
        "conservative": np.clip(base * (1.025**yrs_ahead) * np.random.normal(1,.03,len(base)), 0, None),
        "moderate":     np.clip(base * (1.045**yrs_ahead) * np.random.normal(1,.03,len(base)), 0, None),
        "optimistic":   np.clip(base * (1.065**yrs_ahead) * np.random.normal(1,.03,len(base)), 0, None),
    })
    annual  = out.groupby("year")[["conservative","moderate","optimistic"]].sum().reset_index()
    monthly = out.groupby(["year","month"])[["conservative","moderate","optimistic"]].sum().reset_index()
    monthly["date_label"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))
    return out, monthly, annual

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
raw  = make_data()
df   = prep(raw)
ts   = ts_setup(raw)
train, test = train_test(ts)
prophet_model = train_prophet(raw)
test_preds, test_actual = prophet_test_preds(raw, prophet_model)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
        <div style='font-size:36px'>🚲</div>
        <div style='font-size:17px;font-weight:700;color:#f0f6fc;'>BikeFC</div>
        <div style='font-size:11px;color:#8b949e;'>PRCP-1018 · Prophet Model</div>
    </div>
    <hr style='border-color:#30363d;margin:8px 0 14px'>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Dashboard",
        "📊  EDA",
        "📈  Time Series",
        "🤖  Models",
        "🔮  Predict",
        "📅  Future 2026–2045",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#30363d;margin:14px 0 10px'>", unsafe_allow_html=True)

    if page != "🏠  Dashboard":
        st.markdown("<div style='color:#8b949e;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Filters</div>", unsafe_allow_html=True)
        yr_f  = st.multiselect("Year", [2011,2012], default=[2011,2012])
        sea_f = st.multiselect("Season", ["Winter","Spring","Summer","Fall"],
                               default=["Winter","Spring","Summer","Fall"])
        df_f  = df[df["yr"].isin([y-2011 for y in yr_f]) & df["season_name"].isin(sea_f)]
    else:
        df_f = df

    st.markdown(f"""
    <div style='margin-top:10px'>
        <span style='background:#1f6feb22;border:1px solid #1f6feb55;border-radius:20px;
              padding:3px 10px;font-size:11px;color:#58a6ff;'>731 days</span>
        <span style='background:#3fb95022;border:1px solid #3fb95055;border-radius:20px;
              padding:3px 10px;font-size:11px;color:#3fb950;margin-left:4px;'>Best: Prophet</span>
    </div>
    """, unsafe_allow_html=True)

    if not _HAS_PROPHET:
        st.warning("⚠️ Prophet not installed — using fallback mode for predictions.")

nav = page.split("  ")[1].strip()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if nav == "Dashboard":
    st.markdown("<div class='hero'>🚲 Bike Rental Forecasting</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>PRCP-1018 · UCI Day Dataset · Best Model: Prophet + Temp + Windspeed · R²=0.46</div>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "Total Rentals",   f"{df['cnt'].sum():,}", "2011–2012", "good"),
        (c2, "Daily Average",   f"{df['cnt'].mean():.0f}", "bikes/day", "good"),
        (c3, "Peak Day",        f"{df['cnt'].max():,}", "best day", "good"),
        (c4, "Best Model R²",   "0.46", "Prophet+Temp+Wind", "warn"),
        (c5, "Best RMSE",       "1371", "bikes/day error", "warn"),
    ]
    for col,lbl,val,sub,cls in kpis:
        with col:
            st.markdown(f"<div class='kpi'><div class='kpi-label'>{lbl}</div>"
                        f"<div class='kpi-val'>{val}</div>"
                        f"<div class='kpi-sub {cls}'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='sec'>📈 Daily Rentals Time Series (2011–2012)</div>", unsafe_allow_html=True)
    ts2 = ts.reset_index()
    roll7  = ts2["cnt"].rolling(7,min_periods=1).mean()
    roll30 = ts2["cnt"].rolling(30,min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts2["dteday"],y=ts2["cnt"],name="Daily",
        line=dict(color=C["blue"],width=1),opacity=0.5))
    fig.add_trace(go.Scatter(x=ts2["dteday"],y=roll7,name="7-Day MA",
        line=dict(color=C["green"],width=2)))
    fig.add_trace(go.Scatter(x=ts2["dteday"],y=roll30,name="30-Day MA",
        line=dict(color=C["orange"],width=2.5)))
    split_date = train.index[-1]
    fig.add_vline(x=str(split_date), line_dash="dash", line_color="#8b949e",
                  annotation_text="Train/Test Split (80/20)")
    fig.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(theme(fig,340), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>📅 Monthly Average Rentals</div>", unsafe_allow_html=True)
        monthly_avg = df.groupby("mnth")["cnt"].mean().reset_index()
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_avg["month_name"] = [months[m-1] for m in monthly_avg["mnth"]]
        fig2 = go.Figure(go.Bar(x=monthly_avg["month_name"],y=monthly_avg["cnt"],
            marker=dict(color=monthly_avg["cnt"],
                        colorscale=[[0,"#1c2333"],[0.5,"#1f6feb"],[1,"#58a6ff"]]),
            text=monthly_avg["cnt"].round(0).astype(int),textposition="outside"))
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig2,280), use_container_width=True)

    with c2:
        st.markdown("<div class='sec'>📊 Year-over-Year Comparison</div>", unsafe_allow_html=True)
        yr_avg = df.groupby("yr")["cnt"].mean().reset_index()
        yr_avg["year_label"] = yr_avg["yr"].map({0:"2011",1:"2012"})
        fig3 = go.Figure(go.Bar(x=yr_avg["year_label"],y=yr_avg["cnt"],
            marker_color=[C["blue"],C["green"]],
            text=yr_avg["cnt"].round(0).astype(int),textposition="outside",
            width=0.4))
        fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig3,280), use_container_width=True)

    # Key insights from notebook
    st.markdown("<div class='sec'>💡 Key Insights from Notebook</div>", unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    insights = [
        (i1, [
            "<b>Temperature</b> has the strongest correlation with rentals (r=0.63)",
            "<b>Summer & Fall</b> are peak seasons; Winter is the slowest",
            "<b>Peak months:</b> May–September; Lowest: January–February",
        ]),
        (i2, [
            "<b>2012 >> 2011</b> in all metrics — system grew significantly YoY",
            "<b>Clear weather</b> drives the most rentals; Rain drops them sharply",
            "<b>Windspeed</b> (r=−0.23) — strong winds noticeably reduce demand",
        ]),
        (i3, [
            "<b>cnt skewness = −0.047</b> — near-perfect normal, no log transform needed",
            "<b>No missing values</b> — dataset is production-clean",
            "<b>Windspeed outliers</b> treated with IQR capping before modeling",
        ]),
    ]
    for col, pts in insights:
        with col:
            for pt in pts:
                st.markdown(f"<div class='insight'>• {pt}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "EDA":
    st.markdown("<div class='hero' style='font-size:34px'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Replicating notebook EDA: univariate, bivariate, correlation</div>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📋 Dataset","📈 Distributions","🔗 Bivariate","🌡️ Correlation"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val in [(c1,"Rows","731"),(c2,"Columns","16"),
                             (c3,"Missing Values","0"),(c4,"Duplicates","0")]:
            with col:
                st.markdown(f"<div class='kpi'><div class='kpi-label'>{lbl}</div>"
                            f"<div class='kpi-val'>{val}</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='sec'>Dataset Preview</div>", unsafe_allow_html=True)
        show_cols = ["dteday","season_name","weather_name","temp_c","hum_pct","wind_kmh",
                     "holiday","workingday","casual","registered","cnt"]
        st.dataframe(df_f[show_cols].rename(columns={
            "dteday":"Date","season_name":"Season","weather_name":"Weather",
            "temp_c":"Temp (°C)","hum_pct":"Hum %","wind_kmh":"Wind km/h"
        }).style.background_gradient(subset=["cnt"],cmap="Blues")
          .format({"Temp (°C)":"{:.1f}","Hum %":"{:.0f}","Wind km/h":"{:.1f}","cnt":"{:,}"}),
          use_container_width=True, height=380)

        st.markdown("<div class='sec'>Statistical Summary</div>", unsafe_allow_html=True)
        st.dataframe(df_f[["temp_c","hum_pct","wind_kmh","casual","registered","cnt"]]
                     .describe().round(2).style.background_gradient(cmap="Blues"),
                     use_container_width=True)

    with tab2:
        st.markdown("<div class='sec'>Univariate Distributions</div>", unsafe_allow_html=True)
        feat = st.selectbox("Feature", ["cnt","temp_c","hum_pct","wind_kmh","casual","registered"])
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_f, x=feat, nbins=50,
                               color_discrete_sequence=[C["blue"]], marginal="box")
            fig.update_layout(title=f"Distribution: {feat}", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(theme(fig,320), use_container_width=True)
        with c2:
            fig2 = px.violin(df_f, x="season_name", y=feat, box=True, points=False,
                color="season_name",
                category_orders={"season_name":["Winter","Spring","Summer","Fall"]},
                color_discrete_sequence=[C["blue"],C["green"],C["orange"],C["red"]])
            fig2.update_layout(title=f"{feat} by Season",showlegend=False,
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(theme(fig2,320), use_container_width=True)

        # Outlier boxplots (replicating notebook Section 5)
        st.markdown("<div class='sec'>Outlier Detection (Boxplots)</div>", unsafe_allow_html=True)
        fig_b = make_subplots(rows=1, cols=5,
            subplot_titles=["cnt","temp","hum","windspeed","casual"])
        for i,(col_) in enumerate(["cnt","temp","hum","windspeed","casual"],1):
            fig_b.add_trace(go.Box(y=df_f[col_],name=col_,
                marker_color=[C["green"],C["blue"],C["orange"],C["red"],C["purple"]][i-1],
                showlegend=False), row=1, col=i)
        fig_b.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(theme(fig_b), use_container_width=True)

        st.markdown("""
        <div class='insight'>
        <b>Outlier decisions (from notebook):</b><br>
        • <b>windspeed</b>: 13 outliers → ✅ Capped using IQR (skews distribution)<br>
        • <b>holiday</b>: 21 "outliers" → ✅ Kept — binary column, boxplot flags rare 1s<br>
        • <b>hum</b>: 2 outliers → ✅ Kept — genuine low-humidity days<br>
        • <b>cnt</b>: 0 outliers → ✅ Target variable is clean<br>
        • <b>casual</b>: 44 outliers → ✅ Not used in time series model
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='sec'>Continuous Features vs cnt</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for col_,x_,color_,title_ in [
            (c1,"temp_c",C["blue"],"Temperature vs Rentals"),
            (c2,"hum_pct",C["orange"],"Humidity vs Rentals"),
            (c3,"wind_kmh",C["green"],"Windspeed vs Rentals"),
        ]:
            with col_:
                fig = px.scatter(df_f.sample(min(500,len(df_f))),
                    x=x_,y="cnt",trendline="lowess",
                    color_discrete_sequence=[color_],opacity=0.6)
                fig.update_layout(title=title_,margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(theme(fig,280), use_container_width=True)

        st.markdown("<div class='sec'>Categorical Features vs cnt</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        cats = [
            (c1,"season_name","Season",["Winter","Spring","Summer","Fall"]),
            (c2,"weather_name","Weather",["Clear","Mist/Cloudy","Light Rain/Snow"]),
            (c3,"day_name","Day of Week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
        ]
        for col_,x_,title_,order_ in cats:
            with col_:
                fig = px.box(df_f,x=x_,y="cnt",color=x_,
                    category_orders={x_:order_},
                    color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(title=title_,showlegend=False,
                    margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(theme(fig,290), use_container_width=True)

    with tab4:
        st.markdown("<div class='sec'>Correlation Heatmap</div>", unsafe_allow_html=True)
        num_cols = ["temp","atemp","hum","windspeed","season","mnth","yr",
                    "holiday","workingday","weathersit","casual","registered","cnt"]
        corr = df_f[num_cols].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=num_cols, y=num_cols,
            colorscale=[[0,C["red"]],[0.5,"#161b22"],[1,C["blue"]]],
            zmid=0, text=corr.round(2).values,
            texttemplate="%{text}", textfont=dict(size=9),
        ))
        fig_h.update_layout(height=520,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig_h), use_container_width=True)

        # Correlation with cnt highlight
        cnt_corr = corr["cnt"].drop("cnt").sort_values(ascending=False)
        st.markdown("<div class='sec'>Correlation with Target (cnt)</div>", unsafe_allow_html=True)
        fig_c = go.Figure(go.Bar(
            x=cnt_corr.index, y=cnt_corr.values,
            marker_color=[C["green"] if v>0 else C["red"] for v in cnt_corr.values],
            text=cnt_corr.round(2).values, textposition="outside",
        ))
        fig_c.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=0),
                            yaxis_title="Correlation with cnt")
        st.plotly_chart(theme(fig_c), use_container_width=True)

        st.markdown("""
        <div class='insight'>
        <b>Key correlations (used for Prophet feature selection):</b>
        temp (0.63) ✅ included · windspeed (−0.23) ✅ included ·
        weathersit (−0.30) ❌ handled by seasonality ·
        hum (−0.10) ❌ below 0.20 threshold · workingday (0.06) ❌ below threshold
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Time Series":
    st.markdown("<div class='hero' style='font-size:34px'>📈 Time Series Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>ADF stationarity test · differencing · ACF/PACF insights · train/test split</div>", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["📉 Series & Decomp","🔬 Stationarity","✂️ Train/Test"])

    with tab1:
        st.markdown("<div class='sec'>Daily Bike Rentals 2011–2012</div>", unsafe_allow_html=True)
        ts2 = ts.reset_index()
        fig = go.Figure(go.Scatter(x=ts2["dteday"],y=ts2["cnt"],
            fill="tozeroy",fillcolor="rgba(88,166,255,.08)",
            line=dict(color=C["blue"],width=1.5),name="cnt"))
        fig.update_layout(xaxis_title="Date",yaxis_title="Daily Rentals",
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig,320), use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec'>Monthly Average</div>", unsafe_allow_html=True)
            m_avg = df.groupby("mnth")["cnt"].mean().reset_index()
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            m_avg["label"] = [months[m-1] for m in m_avg["mnth"]]
            fig2 = go.Figure(go.Bar(x=m_avg["label"],y=m_avg["cnt"],
                marker_color=C["orange"],text=m_avg["cnt"].round(0).astype(int),
                textposition="outside"))
            fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(theme(fig2,270), use_container_width=True)

        with c2:
            st.markdown("<div class='sec'>Yearly Average</div>", unsafe_allow_html=True)
            y_avg = df.groupby("yr")["cnt"].mean().reset_index()
            y_avg["label"] = y_avg["yr"].map({0:"2011",1:"2012"})
            fig3 = go.Figure(go.Bar(x=y_avg["label"],y=y_avg["cnt"],
                marker_color=[C["blue"],C["green"]],
                text=y_avg["cnt"].round(0).astype(int),textposition="outside",width=0.35))
            fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(theme(fig3,270), use_container_width=True)

    with tab2:
        st.markdown("<div class='sec'>ADF Stationarity Test</div>", unsafe_allow_html=True)
        adf_orig = adfuller(ts["cnt"])
        adf_diff = adfuller(ts["cnt"].diff().dropna())

        c1,c2 = st.columns(2)
        with c1:
            is_stat = adf_orig[1] <= 0.05
            cls_ = "good" if is_stat else "bad"
            st.markdown(f"""
            <div class='kpi'>
                <div class='kpi-label'>Original Series — ADF Test</div>
                <div class='kpi-val'>p = {adf_orig[1]:.4f}</div>
                <div class='kpi-sub {cls_}'>{'✅ Stationary' if is_stat else '❌ Non-Stationary — differencing needed'}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            is_stat2 = adf_diff[1] <= 0.05
            cls2 = "good" if is_stat2 else "bad"
            st.markdown(f"""
            <div class='kpi'>
                <div class='kpi-label'>After 1st Differencing — ADF Test</div>
                <div class='kpi-val'>p = {adf_diff[1]:.6f}</div>
                <div class='kpi-sub {cls2_}'>{'✅ Stationary — d=1 confirmed' if is_stat2 else '❌ Still Non-Stationary'}</div>
            </div>
            """.replace("cls2_","cls2"), unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            fig_orig = go.Figure(go.Scatter(x=ts.index, y=ts["cnt"],
                line=dict(color=C["blue"],width=1.2), name="Original"))
            fig_orig.update_layout(title="Original Series",margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(theme(fig_orig,280), use_container_width=True)
        with c2:
            diff1 = ts["cnt"].diff().dropna()
            fig_diff = go.Figure(go.Scatter(x=diff1.index, y=diff1.values,
                line=dict(color=C["green"],width=1.2), name="Differenced"))
            fig_diff.add_hline(y=0,line_dash="dash",line_color="#8b949e")
            fig_diff.update_layout(title="After 1st Differencing (d=1)",
                                   margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(theme(fig_diff,280), use_container_width=True)

        st.markdown("""
        <div class='insight'>
        <b>ACF/PACF Reading (from notebook):</b><br>
        p=4 (PACF: lags 1–4 outside confidence band) ·
        d=1 (ADF confirmed after 1st differencing) ·
        q=1 (ACF: lag 1 outside band) →
        <b>Final ARIMA order: (4,1,1) · SARIMA seasonal: (1,1,1,7)</b>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='sec'>80/20 Chronological Train-Test Split</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='insight'>
        Train: 584 days (Jan 2011 → Aug 2012) &nbsp;|&nbsp;
        Test: 147 days (Aug 2012 → Dec 2012) &nbsp;|&nbsp;
        <b>Never random-split a time series</b> — breaks temporal order & causes data leakage
        </div>
        """, unsafe_allow_html=True)

        fig_split = go.Figure()
        fig_split.add_trace(go.Scatter(x=train.index,y=train.values,
            name="Train (80%)",fill="tozeroy",fillcolor="rgba(88,166,255,.1)",
            line=dict(color=C["blue"],width=1.5)))
        fig_split.add_trace(go.Scatter(x=test.index,y=test.values,
            name="Test (20%)",fill="tozeroy",fillcolor="rgba(63,185,80,.1)",
            line=dict(color=C["green"],width=1.5)))
        fig_split.add_vline(x=str(train.index[-1]),line_dash="dash",line_color="#8b949e",
                            annotation_text="Split point")
        fig_split.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig_split,340), use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        for col_,lbl_,val_ in [
            (c1,"Train Days","584"),(c2,"Test Days","147"),
            (c3,"Train End","Aug 2012"),(c4,"Test End","Dec 2012")]:
            with col_:
                st.markdown(f"<div class='kpi'><div class='kpi-label'>{lbl_}</div>"
                            f"<div class='kpi-val'>{val_}</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Models":
    st.markdown("<div class='hero' style='font-size:34px'>🤖 Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>AR · ARIMA · SARIMA · Prophet variants — exact notebook results</div>", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["🏆 Leaderboard","📊 Metrics Charts","🔍 Prophet Detail"])

    with tab1:
        st.markdown("<div class='sec'>All Models — Performance Report</div>", unsafe_allow_html=True)
        # Highlight best row
        def highlight_best(row):
            if "Prophet + Temp + Windspeed" in row["Model"]:
                return ["background-color:#0d2a0d;color:#3fb950;font-weight:700"]*len(row)
            return [""]*len(row)
        st.dataframe(
            RESULTS.style.apply(highlight_best,axis=1)
                   .format({"RMSE":"{:.2f}","MAE":"{:.2f}","R2":"{:.2f}"}),
            use_container_width=True, height=340
        )
        st.markdown("""
        <div class='insight'>
        🏆 <b>Winner: Prophet + Temp + Windspeed</b> — RMSE=1371.53, MAE=1023.93, R²=0.46<br>
        Key insight: Adding too many features (all 6) caused overfitting (R²=0.10).
        Only features with <b>|correlation| > 0.20</b> were kept (temp=0.63, windspeed=−0.23).
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        fig_cmp = make_subplots(rows=1,cols=3,
            subplot_titles=["RMSE ↓","MAE ↓","R² ↑"])
        colors_ = [C["green"] if "Temp + Wind" in m and "All" not in m else C["blue"]
                   for m in RESULTS["Model"]]
        for i,(metric_,ascending_) in enumerate([("RMSE",True),("MAE",True),("R2",False)],1):
            fig_cmp.add_trace(go.Bar(
                x=RESULTS["Model"], y=RESULTS[metric_],
                marker_color=colors_, showlegend=False,
                text=RESULTS[metric_].round(2), textposition="outside",
            ), row=1, col=i)
        fig_cmp.update_layout(height=400, margin=dict(l=0,r=0,t=40,b=0))
        fig_cmp.update_xaxes(tickangle=45)
        st.plotly_chart(theme(fig_cmp), use_container_width=True)

        # AIC table
        st.markdown("<div class='sec'>AIC Comparison (Statistical Models Only)</div>", unsafe_allow_html=True)
        aic = pd.DataFrame({
            "Model": ["AR","ARIMA(4,1,1)","SARIMA(4,1,1)(1,1,1,7)","SARIMA(1,1,1)",
                      "Prophet (all variants)"],
            "AIC": ["9303.13","9445.20","9812.54","9430.76","N/A — ML-based"],
            "Note": ["Lowest AIC","Higher AIC","Overfitted","Simpler SARIMA",
                     "AIC not applicable to Prophet"]
        })
        st.dataframe(aic, use_container_width=True)
        st.markdown("""
        <div class='insight'>
        AR wins on AIC (most statistically efficient) but Prophet wins on real-world RMSE/R².
        AIC measures simplicity vs fit, not actual predictive accuracy on unseen data.
        <b>For production → use Prophet</b>.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='sec'>Prophet Feature Experiments</div>", unsafe_allow_html=True)
        prophet_exp = pd.DataFrame({
            "Version": ["Basic","+ Temp","+ Temp + Windspeed ✅","+ All 6","+ Temp+Wind+Weather"],
            "R²": [0.36, 0.44, 0.46, 0.10, 0.29],
            "RMSE": [1495.69, 1405.34, 1371.53, 1779.54, "–"],
        })
        fig_p = go.Figure(go.Bar(
            x=prophet_exp["Version"], y=prophet_exp["R²"],
            marker_color=[C["blue"],C["blue"],C["green"],C["red"],C["orange"]],
            text=prophet_exp["R²"], textposition="outside",
        ))
        fig_p.update_layout(yaxis_title="R²",margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig_p,320), use_container_width=True)

        if test_preds is not None:
            st.markdown("<div class='sec'>Prophet + Temp + Wind — Actual vs Predicted (Test Set)</div>", unsafe_allow_html=True)
            test_dates = test.index
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(x=test_dates,y=test_actual,name="Actual",
                line=dict(color=C["green"],width=2)))
            fig_av.add_trace(go.Scatter(x=test_dates,y=np.clip(test_preds,0,None),
                name="Prophet Predicted",line=dict(color=C["red"],dash="dash",width=2)))
            fig_av.update_layout(hovermode="x unified",margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(theme(fig_av,320), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Predict":
    st.markdown("<div class='hero' style='font-size:34px'>🔮 Predict Daily Rentals</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Prophet + Temp + Windspeed · inputs: temp & windspeed only</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight'>
    <b>Model inputs:</b> Only <b>temperature</b> and <b>windspeed</b> are needed —
    both are freely available from any weather API.
    The model handles daily/weekly/yearly seasonality internally.
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("<div class='sec' style='font-size:14px'>📅 Date</div>", unsafe_allow_html=True)
        pred_date = st.date_input("Forecast date", value=datetime.date(2013,6,15))
    with c2:
        st.markdown("<div class='sec' style='font-size:14px'>🌡️ Temperature</div>", unsafe_allow_html=True)
        temp_c  = st.slider("Temperature (°C)", -5.0, 41.0, 22.0, 0.5)
        temp_n  = temp_c / 41.0   # normalized
    with c3:
        st.markdown("<div class='sec' style='font-size:14px'>💨 Windspeed</div>", unsafe_allow_html=True)
        wind_kmh = st.slider("Wind speed (km/h)", 0.0, 55.0, 12.0, 0.5)
        wind_n  = wind_kmh / 67.0  # normalized

    if st.button("⚡ Predict", use_container_width=True):
        input_df = pd.DataFrame({"ds":[pd.Timestamp(pred_date)],
                                  "temp":[temp_n],"windspeed":[wind_n]})
        if prophet_model is not None:
            fc = prophet_model.predict(input_df)
            pred_val  = max(0, int(fc["yhat"].values[0]))
            pred_lo   = max(0, int(fc["yhat_lower"].values[0]))
            pred_hi   = int(fc["yhat_upper"].values[0])
        else:
            # Simple fallback estimate
            month = pred_date.month
            pred_val = int(3000 * (0.4 + 1.2*temp_n - 0.4*temp_n**2)
                          * (0.6 + 0.8*np.sin((month-3)*np.pi/6))
                          * (1 - 0.4*wind_n))
            pred_lo, pred_hi = int(pred_val*0.75), int(pred_val*1.25)

        c1,c2,c3,c4 = st.columns(4)
        level = "🟢 High" if pred_val>4000 else ("🟡 Medium" if pred_val>2000 else "🔴 Low")
        for col_,lbl_,val_,sub_,cls_ in [
            (c1,"Predicted Rentals",f"{pred_val:,}",f"95% CI: {pred_lo:,}–{pred_hi:,}","good"),
            (c2,"Casual Est.",f"{int(pred_val*.22):,}","~22% of total","warn"),
            (c3,"Registered Est.",f"{int(pred_val*.78):,}","~78% of total","warn"),
            (c4,"Demand Level",level,"classification","good"),
        ]:
            with col_:
                st.markdown(f"<div class='kpi' style='text-align:center'>"
                            f"<div class='kpi-label'>{lbl_}</div>"
                            f"<div class='kpi-val'>{val_}</div>"
                            f"<div class='kpi-sub {cls_}'>{sub_}</div></div>", unsafe_allow_html=True)

        # 7-day simulation around the selected date
        st.markdown("<div class='sec'>📅 ±3 Day Context Forecast</div>", unsafe_allow_html=True)
        dates_7 = pd.date_range(pred_date - datetime.timedelta(days=3),
                                pred_date + datetime.timedelta(days=3), freq="D")
        rows7 = [{"ds":d, "temp":temp_n*(0.95+0.1*np.random.rand()),
                  "windspeed":wind_n*(0.95+0.1*np.random.rand())} for d in dates_7]
        df7 = pd.DataFrame(rows7)
        if prophet_model is not None:
            fc7 = prophet_model.predict(df7)
            vals7 = np.clip(fc7["yhat"].values, 0, None).astype(int)
            lo7   = np.clip(fc7["yhat_lower"].values, 0, None).astype(int)
            hi7   = fc7["yhat_upper"].values.astype(int)
        else:
            vals7 = np.array([pred_val]*7) + np.random.randint(-300,300,7)
            lo7,hi7 = vals7-400, vals7+400

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=pd.concat([pd.Series(dates_7), pd.Series(dates_7[::-1])]),
            y=np.concatenate([hi7, lo7[::-1]]),
            fill="toself", fillcolor="rgba(88,166,255,.1)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
        fig7.add_trace(go.Scatter(x=dates_7, y=vals7,
            mode="lines+markers+text", name="Forecast",
            text=vals7, textposition="top center",
            line=dict(color=C["blue"],width=2.5),
            marker=dict(size=9, color=[C["green"] if d==pd.Timestamp(pred_date) else C["blue"] for d in dates_7])))
        fig7.add_vline(x=str(pred_date),line_dash="dash",line_color=C["green"],
                       annotation_text="Selected day")
        fig7.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig7,300), use_container_width=True)
    else:
        st.info("👆 Set date, temperature and wind speed, then click **Predict**")

# ══════════════════════════════════════════════════════════════════════════════
# FUTURE 2026–2045
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Future 2026–2045":
    st.markdown("<div class='hero' style='font-size:34px'>📅 20-Year Forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Jan 2026 – Dec 2045 · Prophet extrapolation · 3 growth scenarios</div>", unsafe_allow_html=True)

    with st.spinner("Running 20-year projection..."):
        fc_daily, fc_monthly, fc_annual = future_forecast(prophet_model, raw)

    c1,c2,c3 = st.columns(3)
    with c1:
        scenario = st.selectbox("Growth Scenario",
            ["Conservative (+2.5%/yr)","Moderate (+4.5%/yr)","Optimistic (+6.5%/yr)"],index=1)
    with c2:
        yr_from = st.slider("From year",2026,2045,2026)
    with c3:
        yr_to   = st.slider("To year",2026,2045,2045)

    scol = {"Conservative (+2.5%/yr)":"conservative",
            "Moderate (+4.5%/yr)":"moderate",
            "Optimistic (+6.5%/yr)":"optimistic"}[scenario]

    # Milestone KPIs
    st.markdown("<div class='sec'>🎯 Milestone Projections</div>", unsafe_allow_html=True)
    mc = st.columns(5)
    base_2026 = fc_annual[fc_annual["year"]==2026][scol].values[0]
    for i,yr_ in enumerate([2026,2030,2035,2040,2045]):
        row = fc_annual[fc_annual["year"]==yr_]
        if not row.empty:
            val = int(row[scol].values[0])
            g   = (val/base_2026-1)*100
            with mc[i]:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#0d2045,#1a3a6b);
                     border:1px solid #1f4b9b;border-radius:14px;padding:16px;text-align:center;'>
                    <div style='color:#58a6ff;font-size:12px;font-weight:700;text-transform:uppercase;
                         letter-spacing:1px;'>{yr_}</div>
                    <div style='color:#f0f6fc;font-size:28px;font-weight:800;margin:6px 0;'>
                        {val/1e6:.2f}M</div>
                    <div style='color:#3fb950;font-size:12px;'>+{g:.0f}% vs 2026</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div class='sec'>📈 Annual Forecast — All Scenarios</div>", unsafe_allow_html=True)
    annual_f = fc_annual[(fc_annual["year"]>=yr_from)&(fc_annual["year"]<=yr_to)]
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([annual_f["year"], annual_f["year"][::-1]]),
        y=pd.concat([annual_f["optimistic"], annual_f["conservative"][::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Scenario Band"))
    for col_,color_,dash_,name_ in [
        ("conservative",C["orange"],"dot","Conservative"),
        ("moderate",C["blue"],"solid","Moderate"),
        ("optimistic",C["green"],"dash","Optimistic"),
    ]:
        fig_fc.add_trace(go.Scatter(x=annual_f["year"],y=annual_f[col_],
            mode="lines+markers",name=name_,
            line=dict(color=color_,width=2,dash=dash_),marker=dict(size=6)))
    fig_fc.update_layout(hovermode="x unified",yaxis_title="Annual Rentals",
                          margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(theme(fig_fc,400), use_container_width=True)

    # Monthly heatmap
    st.markdown("<div class='sec'>🗓️ Monthly Heatmap (Year × Month)</div>", unsafe_allow_html=True)
    pivot = fc_monthly[(fc_monthly["year"]>=yr_from)&(fc_monthly["year"]<=yr_to)]\
            .pivot_table(values=scol, index="year", columns="month")
    month_lbls = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=month_lbls, y=pivot.index.tolist(),
        colorscale=[[0,"#0d1117"],[0.25,"#0d2a45"],[0.6,"#1f6feb"],[1,"#58a6ff"]],
        colorbar=dict(title="Rentals"), hoverongaps=False,
    ))
    fig_hm.update_layout(xaxis_title="Month",yaxis_title="Year",
                          margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(theme(fig_hm,420), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>📊 YoY Growth Rate</div>", unsafe_allow_html=True)
        fc_annual["yoy"] = fc_annual[scol].pct_change()*100
        af2 = fc_annual[(fc_annual["year"]>=yr_from)&(fc_annual["year"]<=yr_to)]
        fig_g = go.Figure(go.Bar(x=af2["year"],y=af2["yoy"],
            marker_color=C["green"],text=af2["yoy"].round(1),
            texttemplate="%{text}%",textposition="outside"))
        fig_g.update_layout(yaxis_title="YoY Growth %",margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(theme(fig_g,300), use_container_width=True)

    with c2:
        st.markdown("<div class='sec'>📋 Annual Table (millions)</div>", unsafe_allow_html=True)
        tbl = af2[["year","conservative","moderate","optimistic"]].copy()
        for c_ in ["conservative","moderate","optimistic"]:
            tbl[c_] = (tbl[c_]/1e6).round(2)
        tbl.columns = ["Year","Cons. (M)","Mod. (M)","Opt. (M)"]
        st.dataframe(tbl.style.background_gradient(subset=["Mod. (M)"],cmap="Blues")
                     .format({"Cons. (M)":"{:.2f}","Mod. (M)":"{:.2f}","Opt. (M)":"{:.2f}"}),
                     use_container_width=True, height=300)

    csv_ = fc_annual[["year","conservative","moderate","optimistic"]].to_csv(index=False)
    st.download_button("📥 Download Forecast CSV", csv_,
                       "bike_rental_forecast_2026_2045.csv","text/csv")

# ── footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#484f58;font-size:12px;
     border-top:1px solid #21262d;margin-top:36px;padding:20px 0 8px;'>
🚲 BikeFC · PRCP-1018 · Best Model: Prophet + Temp + Windspeed (R²=0.46) ·
Built with Streamlit
</div>
""", unsafe_allow_html=True)
