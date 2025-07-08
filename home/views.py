import os
import json
import datetime
import pandas as pd
import requests
from io import StringIO

from django.shortcuts import render
from django.conf import settings
from catboost import CatBoostRegressor


# --- Basic static views ---

def index(request):
    return render(request, 'home/index.html')

def blog(request):
    return render(request, 'home/blog.html')

def about(request):
    return render(request, 'home/about.html')

def contact(request):
    return render(request, 'home/contact.html')


# --- AQI prediction setup ---

# Paths to models
MODEL_DIR = os.path.join(settings.BASE_DIR, 'home', 'ml_models')
KATHMANDU_MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_kathmandu_model.cbm')
CHANGUNARAYAN_MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_changunarayan_model.cbm')

# Load models once
kathmandu_model = CatBoostRegressor()
kathmandu_model.load_model(KATHMANDU_MODEL_PATH)

changunarayan_model = CatBoostRegressor()
changunarayan_model.load_model(CHANGUNARAYAN_MODEL_PATH)


# AQI breakpoints
pm25_breakpoints = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500)
]

pm10_breakpoints = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500)
]


def calculate_individual_aqi(concentration, breakpoints):
    for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
        if Clow <= concentration <= Chigh:
            return round(((Ihigh - Ilow) / (Chigh - Clow)) * (concentration - Clow) + Ilow)
    return 0


def calculate_aqi(pm25, pm10):
    aqi_pm25 = calculate_individual_aqi(pm25, pm25_breakpoints)
    aqi_pm10 = calculate_individual_aqi(pm10, pm10_breakpoints)
    return (aqi_pm25, 'PM2.5') if aqi_pm25 >= aqi_pm10 else (aqi_pm10, 'PM10')


def get_health_recommendation(aqi):
    if aqi <= 50:
        return "Air quality is good. It's safe to go outside."
    elif aqi <= 100:
        return "Air quality is moderate. Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for sensitive groups. Reduce outdoor activities if you have respiratory issues."
    elif aqi <= 200:
        return "Unhealthy. Everyone may experience health effects. Limit outdoor activities."
    elif aqi <= 300:
        return "Very unhealthy. Avoid outdoor exertion. Stay indoors if possible."
    else:
        return "Hazardous. Everyone should avoid outdoor activities and stay indoors."


def get_health_recommendation_next_hour(aqi):
    if aqi <= 50:
        return "Air quality will remain good in the next hour. Enjoy outdoor activities safely."
    elif aqi <= 100:
        return "Air quality may be moderate next hour. Sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Next hour's air quality could be unhealthy for sensitive groups. Reduce outdoor activities."
    elif aqi <= 200:
        return "Unhealthy air quality expected in the next hour. Limit outdoor exposure."
    elif aqi <= 300:
        return "Very unhealthy air quality anticipated. Avoid outdoor exertion."
    else:
        return "Hazardous air quality forecasted. Stay indoors and avoid all outdoor activities."


def fetch_csv_from_gdrive(url):
    try:
        return pd.read_csv(url)
    except Exception:
        resp = requests.get(url)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))


def preprocess_with_time_features(df):
    df = df.rename(columns={
        'PM2.5': 'pm2.5_atm',
        'PM10': 'pm10.0_atm',
        'Temp': 'temperature',
        'Temperature': 'temperature',
        'Humidity': 'humidity',
        'Pressure': 'pressure',
        'CO': 'co',
        'time_stamp': 'timestamp',
        'Time': 'timestamp',
        'date': 'timestamp',
        'Date': 'timestamp'
    })

    datetime_col_candidates = ['timestamp', 'datetime', 'time', 'date']
    datetime_col = None
    for col in datetime_col_candidates:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    else:
        df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')

    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df.set_index('datetime', inplace=True)

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(full_index)

    cols_to_fill = ['pm2.5_atm', 'pm10.0_atm', 'temperature', 'humidity', 'pressure', 'AQI']
    for col in cols_to_fill:
        if col not in df.columns:
            df[col] = 0
    df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill').fillna(0)

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    target_col = 'AQI'

    for lag in [1, 3, 6]:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
        df[f'pm25_lag{lag}'] = df['pm2.5_atm'].shift(lag)

    for window in [3, 6, 12]:
        df[f'{target_col}_rollmean_{window}'] = df[target_col].rolling(window).mean().shift(1)
        df[f'pm25_rollmean_{window}'] = df['pm2.5_atm'].rolling(window).mean().shift(1)
        df[f'{target_col}_rollstd_{window}'] = df[target_col].rolling(window).std().shift(1)
        df[f'pm25_rollstd_{window}'] = df['pm2.5_atm'].rolling(window).std().shift(1)

    df.fillna(0, inplace=True)
    df = df.reset_index().rename(columns={'index': 'datetime'})

    return df


def predict_aqi(request):
    csv_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSW0EZrtnbwFqDddCDGd_Od7yRIFbSm4m9aEkuLDbPQTV-PYGHs_cHIIy2WWD8mRZTHZjRAsx2PMx8h/pub?output=csv'

    df = fetch_csv_from_gdrive(csv_url)

    if df.empty:
        return render(request, 'home/predict.html', {'error': 'CSV file is empty or inaccessible.'})

    df = preprocess_with_time_features(df)
    latest = df.iloc[-1]

    data_fetched_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    feature_cols = [
        'humidity', 'temperature', 'pressure', 'pm2.5_atm', 'pm10.0_atm',
        'hour', 'dayofweek', 'is_weekend',
        'AQI_lag1', 'AQI_lag3', 'AQI_lag6',
        'pm25_lag1', 'pm25_lag3', 'pm25_lag6',
        'AQI_rollmean_3', 'AQI_rollmean_6', 'AQI_rollmean_12',
        'pm25_rollmean_3', 'pm25_rollmean_6', 'pm25_rollmean_12',
        'AQI_rollstd_3', 'AQI_rollstd_6', 'AQI_rollstd_12',
        'pm25_rollstd_3', 'pm25_rollstd_6', 'pm25_rollstd_12'
    ]

    features = [[float(latest.get(col, 0)) for col in feature_cols]]

    model_choice = request.GET.get('model_choice', 'kathmandu').lower()
    model = kathmandu_model if model_choice == 'kathmandu' else changunarayan_model

    predicted_aqi = model.predict(features)[0]

    pm25 = float(latest.get('pm2.5_atm', 0))
    pm10 = float(latest.get('pm10.0_atm', 0))
    temp = float(latest.get('temperature', 0))
    humidity = float(latest.get('humidity', 0))
    pressure = float(latest.get('pressure', 0))
    co = float(latest.get('co', 0))

    current_aqi, main_pollutant = calculate_aqi(pm25, pm10)

    now = datetime.datetime.now()
    future_timestamps = [(now + datetime.timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
    future_predictions = [min(predicted_aqi + i * 2, 500) for i in range(24)]

    next_hour_aqi = future_predictions[1] if len(future_predictions) > 1 else predicted_aqi

    trend_text = (
        "Predicted to rise in AQI levels in the next hour." if next_hour_aqi > predicted_aqi else
        "Predicted to decrease in AQI levels in the next hour." if next_hour_aqi < predicted_aqi else
        "Predicted to remain stable in the next hour."
    )

    recommended_levels = {
        'PM2_5': '0-12 ug/m³',
        'pm10': '0-54 ug/m³',
        'co': '0-4.4 ppm'
    }

    context = {
        'current_aqi': current_aqi,
        'main_pollutant': main_pollutant,
        'predicted_aqi': round(predicted_aqi, 2),
        'hardware_data': {
            'PM2_5': pm25,
            'pm10': pm10,
            'temperature': temp,
            'humidity': humidity,
            'pressure': pressure,
            'co': co,
        },
        'location': model_choice,
        'model_choice': model_choice,
        'future_timestamps': json.dumps(future_timestamps),
        'future_predictions': json.dumps(future_predictions),
        'health_recommendation': get_health_recommendation(current_aqi),
        'trend_text': trend_text,
        'health_next_hour': get_health_recommendation_next_hour(next_hour_aqi),
        'error': None,
        'data_fetched_at': data_fetched_at,
        'recommended_levels': recommended_levels,
    }

    return render(request, 'home/predict.html', context)
