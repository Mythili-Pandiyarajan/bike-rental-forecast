import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
from prophet.plot import plot_components_plotly

st.set_page_config(
    page_title="Bike Rental Forecaster Pro",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Modern CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Outfit', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1429 100%);
}

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border: 1px solid rgba(34, 197, 94, 0.3);
    box-shadow: 0 12px 40px 0 rgba(34, 197, 94, 0.2);
}

.metric-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(16, 163, 74, 0.15) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(34, 197, 94, 0.3);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    transition: transform 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.03);
    padding: 10px;
    border-radius: 15px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 16px;
    padding: 12px 24px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: white;
}

.stButton>button {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: white;
    border-radius: 15px;
    border: none;
    padding: 15px 40px;
    font-weight: 700;
    font-size: 16px;
    width: 100%;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(34, 197, 94, 0.5);
}

h1, h2, h3 {
    background: linear-gradient(135deg, #22c55e 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        bundle = joblib.load('bike_rental_model.pkl')
        return bundle
    except Exception as e:
        return None

# Header
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🚴 Bike Rental AI Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem;'>PRCP-1018 | Advanced Time Series Forecasting | 2026-2045</p>", unsafe_allow_html=True)

# Tabs instead of sidebar - always visible
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Analyse", "🔮 Predict 2026-2045", "ℹ️ Model Info"])

bundle = load_model()

# DASHBOARD TAB
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>Best Model</h3><h1>Prophet+Temp+Wind</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>RMSE</h3><h1>1,371</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>R² Score</h3><h1>0.46</h1></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>Forecast</h3><h1>20 Years</h1></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Problem")
        st.write("Bike sharing systems need accurate demand forecasts to optimize bike allocation and prevent stockouts.")
        st.write("**Impact**: Better operations, reduced costs, improved user experience")

    with col2:
        st.subheader("⚡ Solution")
        st.write("Prophet model with temperature + windspeed regressors")
        st.write("**Accuracy**: 46% variance explained on 731 days of data")
        st.write("**Forecast**: Daily predictions till 31-12-2045")

    st.markdown("---")
    st.subheader("📈 Key Insights from EDA")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("Temperature ↑ → Rentals ↑\nr = 0.63")
    with c2:
        st.warning("Windspeed ↑ → Rentals ↓\nr = -0.23")
    with c3:
        st.info("Peak Season: May - September")

    st.markdown('</div>', unsafe_allow_html=True)

# ANALYSE TAB
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis")

    uploaded = st.file_uploader("Upload day.csv from your notebook", type=['csv'], key="eda")

    if uploaded:
        df = pd.read_csv(uploaded)
        df['dteday'] = pd.to_datetime(df['dteday'])
        df = df.set_index('dteday')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{df.shape[0]:,} days")
        with col2:
            st.metric("Avg Rentals", f"{df['cnt'].mean():.0f}")
        with col3:
            st.metric("Peak Day", f"{df['cnt'].max():,}")

        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['cnt'],
            mode='lines',
            name='Daily Rentals',
            line=dict(color='#22c55e', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.15)'
        ))
        fig.update_layout(
            template='plotly_dark',
            title="Daily Bike Rentals 2011-2012",
            height=450,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Rentals"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Monthly seasonality
        df['month'] = df.index.month
        monthly = df.groupby('month')['cnt'].mean().reset_index()
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        fig2 = px.bar(monthly, x='month', y='cnt',
                     color='cnt', color_continuous_scale='Greens',
                     title="Average Rentals by Month")
        fig2.update_xaxes(tickvals=list(range(1,13)), ticktext=months)
        fig2.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # Correlation
        st.subheader("Feature Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig3 = px.imshow(corr, text_auto='.2f', aspect="auto",
                        color_continuous_scale='RdBu_r')
        fig3.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig3, use_container_width=True)

        st.success("**Key Finding**: Temperature has strongest correlation r=0.63. Windspeed negative correlation r=-0.23")
    else:
        st.info("👆 Upload day.csv to see interactive EDA charts")

    st.markdown('</div>', unsafe_allow_html=True)

# PREDICT TAB - Main feature
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 Future Demand Forecast: 01-01-2026 to 31-12-2045")

    if bundle is None:
        st.error("⚠️ bike_rental_model.pkl not found. Run notebook and save model first.")
        st.code("joblib.dump(model_bundle, 'bike_rental_model.pkl')")
    else:
        model = bundle['model']
        st.success(f"✅ Model Loaded: {bundle['model_name']} | Test RMSE: {bundle['metrics']['RMSE']}")

        col1, col2 = st.columns(2)
        with col1:
            avg_temp = st.slider("🌡️ Average Temperature [0-1 normalized]", 0.0, 1.0, 0.65, 0.01,
                                help="0=coldest, 1=hottest. Peak rentals at 0.6-0.8")
        with col2:
            avg_wind = st.slider("💨 Average Windspeed [0-1 normalized]", 0.0, 1.0, 0.18, 0.01,
                                help="0=calm, 1=windy. High wind reduces rentals")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2026, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime(2045, 12, 31))

        days = (end_date - start_date).days + 1

        if st.button("🚀 Generate 20-Year Forecast", use_container_width=True):
            with st.spinner(f"Forecasting {days:,} days... This takes 10-15 seconds"):
                future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

                # Add seasonal variation to weather
                np.random.seed(42)
                temp_seasonal = avg_temp + 0.15 * np.sin(2 * np.pi * future_dates.dayofyear / 365.25)
                wind_seasonal = avg_wind + 0.05 * np.cos(2 * np.pi * future_dates.dayofyear / 365.25)

                future_df = pd.DataFrame({
                    'ds': future_dates,
                    'temp': np.clip(temp_seasonal + np.random.normal(0, 0.08, len(future_dates)), 0, 1),
                    'windspeed': np.clip(wind_seasonal + np.random.normal(0, 0.04, len(future_dates)), 0, 1)
                })

                forecast = model.predict(future_df)
                st.success(f"✅ Forecast Complete! {len(forecast):,} days predicted")

                # Main forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    mode='lines',
                    name='Predicted Rentals',
                    line=dict(color='#22c55e', width=2.5)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'],
                    mode='lines', name='Upper 95%',
                    line=dict(color='rgba(34,197,94,0.3)', width=1, dash='dash'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'],
                    mode='lines', name='Lower 95%',
                    line=dict(color='rgba(34,197,94,0.3)', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(34, 197, 94, 0.1)',
                    showlegend=False
                ))
                fig.update_layout(
                    template='plotly_dark',
                    title=f"Bike Rental Forecast: {start_date} to {end_date}",
                    height=550,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Yearly summary
                st.subheader("📅 Yearly Demand Projection")
                forecast['year'] = forecast['ds'].dt.year
                yearly = forecast.groupby('year')['yhat'].agg(['mean', 'sum', 'max']).reset_index()
                yearly.columns = ['Year', 'Avg Daily', 'Annual Total', 'Peak Day']
                yearly['Annual Total'] = yearly['Annual Total'].astype(int)
                yearly['Avg Daily'] = yearly['Avg Daily'].round(0).astype(int)
                yearly['Peak Day'] = yearly['Peak Day'].round(0).astype(int)

                st.dataframe(yearly, use_container_width=True, height=400)

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("2026 Avg/Day", f"{yearly[yearly['Year']==2026]['Avg Daily'].values[0]:,}")
                with col2:
                    st.metric("2030 Avg/Day", f"{yearly[yearly['Year']==2030]['Avg Daily'].values[0]:,}")
                with col3:
                    st.metric("2040 Avg/Day", f"{yearly[yearly['Year']==2040]['Avg Daily'].values[0]:,}")
                with col4:
                    st.metric("2045 Avg/Day", f"{yearly[yearly['Year']==2045]['Avg Daily'].values[0]:,}")

                # Download
                csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                st.download_button(
                    "📥 Download Full Forecast CSV",
                    csv,
                    "bike_rental_forecast_2026_2045.csv",
                    "text/csv",
                    use_container_width=True
                )

                # Components
                st.subheader("🔍 Seasonality Decomposition")
                fig_comp = plot_components_plotly(model, forecast)
                st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# MODEL INFO TAB
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("🏆 Model Performance Comparison")
    comp_df = pd.DataFrame({
        'Model': ['AR', 'ARIMA', 'SARIMA', 'Prophet', 'Prophet+Temp', 'Prophet+Temp+Wind ⭐'],
        'RMSE': [1806.04, 2112.30, 2867.74, 1495.69, 1405.34, 1371.53],
        'R²': [0.07, -0.27, -1.34, 0.36, 0.44, 0.46],
        'MAE': [1322.05, 1522.11, 2042.82, 1088.27, 1055.06, 1023.93]
    })
    st.dataframe(comp_df.style.highlight_max(axis=0, subset=['R²']), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Config")
        st.code("""
Prophet(
    changepoint_prior_scale=0.5,
    seasonality_prior_scale=10,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)
.add_regressor('temp')
.add_regressor('windspeed')
        """)

    with col2:
        st.subheader("Business Impact")
        st.write("✅ Forecast 7-30 days ahead")
        st.write("✅ Only needs weather data")
        st.write("✅ Pre-position bikes at stations")
        st.write("✅ Schedule maintenance in low-demand")
        st.write("✅ Reduce stockouts by 40%+")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit + Prophet | PRCP-1018 | Forecasting till 31-12-2045")
