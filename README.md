# 🚲 Bike Rental Demand Forecasting

> Forecasting daily bike rental demand using Time Series Analysis and Machine Learning — helping urban mobility planning, fleet optimisation, and demand-aware transportation systems.

---

# 🔴 Live Demo

https://bike-rental-forecast-mythili.streamlit.app/

---

# 📌 Problem Statement

Bike-sharing systems face fluctuating rental demand depending on seasonality, weather conditions, working days, and temporal patterns.

Accurate demand forecasting helps transportation systems and smart-city planners:

- Optimise bike availability
- Reduce shortages during peak demand
- Improve operational efficiency
- Support sustainable urban transportation

This project builds a Time Series forecasting system to predict future bike rental demand using historical rental and weather data.

---

# 🎯 Problem Type

### Time Series Forecasting / Regression

Target variable — `cnt`

| Variable | Meaning |
|---|---|
| cnt | Total bike rentals |

---

# 📊 Dataset

| Item | Detail |
|---|---|
| Source | UCI Bike Sharing Dataset |
| Granularity | Daily bike rental records |
| Features | Weather + temporal variables |
| Target | `cnt` |

### Important Variables

- Temperature
- Humidity
- Windspeed
- Season
- Weather situation
- Working day
- Date-based temporal patterns

---

# ⚙️ ML Pipeline

| Step | Detail |
|---|---|
| Time Series setup | Datetime indexing |
| Stationarity check | Augmented Dickey-Fuller (ADF) Test |
| Differencing | Applied for stationarity |
| Seasonal decomposition | Trend + seasonality analysis |
| Feature engineering | Weather regressors added to Prophet |
| Models compared | AR, ARIMA, SARIMA, Prophet |
| Best model | Prophet + Temperature + Windspeed |

---

# 📈 Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| AR | 1806.04 | 1322.05 | 0.07 |
| ARIMA | 2112.30 | 1522.11 | -0.27 |
| SARIMA(4,1,1) | 2867.74 | 2042.82 | -1.34 |
| SARIMA(1,1,1) | 2091.37 | 1514.71 | -0.24 |
| Prophet | 1495.69 | 1088.27 | 0.36 |
| Prophet + Temp | 1405.34 | 1055.06 | 0.44 |
| Prophet + Temp + Wind | 1371.53 | 1023.93 | 0.46 |
| Prophet + All Features | 1779.54 | 1278.97 | 0.10 |

---

# 🏆 Best Model

```python
Prophet + Temperature + Windspeed
```

### Best Model Metrics

```python
RMSE = 1371.53
MAE = 1023.93
R² = 0.46
```

---

# 🔍 Key Findings

- Weather variables significantly improve forecasting performance
- Temperature positively correlates with bike rentals
- High windspeed negatively impacts rental demand
- Prophet outperformed ARIMA and SARIMA models
- Seasonal patterns strongly affect bike rental behaviour
- Adding too many regressors reduced model performance

---

# 🚀 Deployment

| Item | Detail |
|---|---|
| Framework | Streamlit |
| Platform | Streamlit Cloud |
| Python version | 3.11 |
| Model serialisation | Pickle |

### App Features

- Interactive bike demand forecasting
- Weather-aware prediction
- Forecast visualisation dashboard
- Trend exploration charts
- Time series analytics

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Prophet
- Statsmodels
- Scikit-learn
- Plotly
- Streamlit

---

# 📁 Project Structure

```text
bike-rental-forecast/
│
├── app.py
├── bike_rental_model.pkl
├── requirements.txt
├── runtime.txt
├── .python-version
└── bike_rental_forecasting.ipynb
```

---

# 🔮 Future Improvements

- LSTM/GRU deep learning forecasting
- Real-time weather API integration
- Hourly demand forecasting
- Multi-city rental prediction
- Advanced anomaly detection

---

# 👩‍💻 Author

### Mythili P · Data Science Portfolio Project
