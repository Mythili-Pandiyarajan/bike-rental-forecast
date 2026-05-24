import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from prophet.plot import plot_components_plotly

st.set_page_config(
    page_title="Bike Rental Forecaster Pro",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS - Bike theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
}

.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.metric-card {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 30px;
    font-weight: 600;
    font-size: 16px;
    width: 100%;
    transition: all 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(34, 197, 94, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Load model bundle
@st.cache_resource
def load_model():
    try:
        bundle = joblib.load('bike_rental_model.pkl')
        return bundle
    except:
        st.error("⚠️ bike_rental_model.pkl not found. Train model in notebook first.")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### 🚴 Bike AI Forecaster")
    st.markdown("PRCP-1018 | Time Series")

    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Analyse", "Predict Future", "Model Info"],
        icons=["house", "bar-chart", "crystal-ball", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"color": "#cbd5e1", "font-size": "16px", "margin": "5px"},
            "nav-link-selected": {"background": "linear-gradient(90deg, #22c55e, #16a34a)"}
        }
    )

# Dashboard
if selected == "Dashboard":
    st.markdown("<h1>🚴 Bike Rental Demand Forecast</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h4>Best Model</h4><h2>Prophet+Temp+Wind</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>RMSE</h4><h2>1371.53</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>R² Score</h4><h2>0.46</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h4>Forecast Range</h4><h2>2026-2045</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Problem Statement")
    st.write("Bike sharing systems need accurate daily demand forecasts to optimize bike allocation and reduce stockouts.")
    st.write("**Solution**: Prophet model with temperature + windspeed regressors predicts demand 20 years ahead.")

    st.subheader("Key Insights from EDA")
    col1, col2 = st.columns(2)
    with col1:
        st.write("✅ Strong seasonal pattern - Summer/Fall peak")
        st.write("✅ Temperature ↑ → Rentals ↑ (r=0.63)")
    with col2:
        st.write("✅ Windspeed ↑ → Rentals ↓ (r=-0.23)")
        st.write("✅ Year-over-year growth 2011→2012")
    st.markdown('</div>', unsafe_allow_html=True)

# Analyse Tab
elif selected == "Analyse":
    st.markdown("<h1>📊 EDA & Time Series Analysis</h1>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload day.csv for analysis", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        df['dteday'] = pd.to_datetime(df['dteday'])
        df = df.set_index('dteday')

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days", df.shape[0])
        with col2:
            st.metric("Avg Daily Rentals", f"{df['cnt'].mean():.0f}")
        with col3:
            st.metric("Max Rentals", f"{df['cnt'].max():,}")

        # Time series plot
        st.subheader("Daily Rentals Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['cnt'],
            mode='lines',
            name='Rentals',
            line=dict(color='#22c55e', width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.1)'
        ))
        fig.update_layout(template='plotly_dark', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Monthly average
        st.subheader("Seasonality Analysis")
        df['month'] = df.index.month
        monthly_avg = df.groupby('month')['cnt'].mean().reset_index()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        fig2 = px.bar(monthly_avg, x='month', y='cnt',
                     title="Average Rentals by Month",
                     color='cnt', color_continuous_scale='Greens')
        fig2.update_xaxes(tickvals=list(range(1,13)), ticktext=month_names)
        fig2.update_layout(template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        st.subheader("Feature Correlation")
        corr = df.corr(numeric_only=True)
        fig3 = px.imshow(corr, text_auto='.2f', aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Heatmap")
        fig3.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig3, use_container_width=True)

        st.info("**Key EDA Findings**: Temperature r=0.63, Windspeed r=-0.23, Peak months May-Sep, Strong yearly growth")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload day.csv to view EDA charts")

# Predict Future Tab - Main functionality
elif selected == "Predict Future":
    st.markdown("<h1>🔮 Forecast till 31-12-2045</h1>", unsafe_allow_html=True)

    bundle = load_model()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if bundle is None:
        st.stop()

    model = bundle['model']
    st.success(f"✅ Loaded model: {bundle['model_name']} | RMSE: {bundle['metrics']['RMSE']}")

    st.subheader("Future Weather Inputs")
    st.caption("Provide temperature and windspeed forecasts for prediction. Values should be normalized 0-1 like training data.")

    col1, col2 = st.columns(2)
    with col1:
        avg_temp = st.slider("Average Temperature (normalized 0-1)", 0.0, 1.0, 0.6, 0.05)
        st.caption("0=coldest, 1=hottest. Peak rentals at 0.6-0.8")
    with col2:
        avg_wind = st.slider("Average Windspeed (normalized 0-1)", 0.0, 1.0, 0.2, 0.05)
        st.caption("0=calm, 1=windy. High wind reduces rentals")

    # Forecast period
    start_date = st.date_input("Start Date", datetime(2026, 1, 1))
    end_date = st.date_input("End Date", datetime(2045, 12, 31))

    days = (end_date - start_date).days + 1

    if st.button("🚀 Generate 20-Year Forecast", use_container_width=True):
        with st.spinner(f"Forecasting {days} days till 2045..."):

            # Create future dataframe
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            future_df = pd.DataFrame({
                'ds': future_dates,
                'temp': np.random.normal(avg_temp, 0.1, len(future_dates)).clip(0, 1),
                'windspeed': np.random.normal(avg_wind, 0.05, len(future_dates)).clip(0, 1)
            })

            # Predict
            forecast = model.predict(future_df)

            st.success(f"✅ Forecast generated for {len(forecast)} days!")

            # Main chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                mode='lines',
                name='Predicted Rentals',
                line=dict(color='#22c55e', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(color='#16a34a', width=1, dash='dash'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                line=dict(color='#16a34a', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(34, 197, 94, 0.2)',
                showlegend=False
            ))
            fig.update_layout(
                template='plotly_dark',
                title=f"Bike Rental Forecast: {start_date} to {end_date}",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Yearly summary
            st.subheader("Yearly Demand Summary")
            forecast['year'] = forecast['ds'].dt.year
            yearly_summary = forecast.groupby('year')['yhat'].agg(['mean', 'sum', 'max']).reset_index()
            yearly_summary.columns = ['Year', 'Avg Daily', 'Annual Total', 'Peak Day']
            yearly_summary['Annual Total'] = yearly_summary['Annual Total'].astype(int)

            st.dataframe(yearly_summary.head(20), use_container_width=True)

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("2026 Avg Daily", f"{yearly_summary[yearly_summary['Year']==2026]['Avg Daily'].values[0]:.0f}")
            with col2:
                st.metric("2035 Avg Daily", f"{yearly_summary[yearly_summary['Year']==2035]['Avg Daily'].values[0]:.0f}")
            with col3:
                st.metric("2045 Avg Daily", f"{yearly_summary[yearly_summary['Year']==2045]['Avg Daily'].values[0]:.0f}")

            # Download
            csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                "📥 Download Forecast CSV",
                csv,
                f"bike_forecast_{start_date}_to_{end_date}.csv",
                "text/csv"
            )

            # Prophet components
            st.subheader("Seasonality Components")
            fig_components = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_components, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Model Info Tab
elif selected == "Model Info":
    st.markdown("<h1>ℹ️ Model Details</h1>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("Model Comparison from Notebook")
    comp_df = pd.DataFrame({
        'Model': ['AR', 'ARIMA', 'SARIMA', 'Prophet', 'Prophet+Temp', 'Prophet+Temp+Wind'],
        'RMSE': [1806.04, 2112.30, 2867.74, 1495.69, 1405.34, 1371.53],
        'R2': [0.07, -0.27, -1.34, 0.36, 0.44, 0.46]
    })
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("Best Model: Prophet + Temp + Windspeed")
    st.code("""
    Prophet(
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=10,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    add_regressor('temp')
    add_regressor('windspeed')
    """)

    st.subheader("Preprocessing from Notebook")
    st.write("1. Used day.csv - daily aggregated data")
    st.write("2. Dropped casual + registered to avoid target leakage")
    st.write("3. IQR capping on windspeed outliers")
    st.write("4. Train-Test split: 80/20 chronological")
    st.write("5. ADF test → 1st differencing for stationarity")
    st.write("6. Feature selection: temp r=0.63, windspeed r=-0.23")

    st.subheader("Business Impact")
    st.write("• **Operations**: Pre-position bikes at high-demand stations")
    st.write("• **Maintenance**: Schedule during low-demand periods")
    st.write("• **Forecast Horizon**: 7-30 days ahead using weather API")
    st.write("• **Inputs Required**: Only temp + windspeed from weather forecast")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("PRCP-1018 | Forecasting Daily Bike Rental Demand | Prophet Model")
