import json
import pandas as pd
import plotly.graph_objects as go
import plotly
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
app = Flask(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/OSIN')
def OSIN():
    return render_template('OS_index.html')

@app.route('/TP_index')
def TP_index():
    return render_template('TP_index.html')

@app.route('/Con_index')
def Con_index():
    return render_template('Con_index.html')

@app.route('/forecasting')
def forecasting():
    return render_template('time_series.html')

@app.route('/time_series', methods=['GET'])
def time_series():
    country = request.args.get('country', default='India', type=str)
    df = pd.read_excel('C:/Users/Maaz/Desktop/DDDIs/New/data/Investment1948.xlsx')
    year_column = 'Year'
    df = df.loc[df['Country'] == country]
    df[year_column] = pd.to_datetime(df[year_column], format='%Y')
    df.set_index(year_column, inplace=True)
    
    time_series_data = [
        go.Scatter(
            x=df.index,
            y=df['Investment'],
            mode='lines',
            name=country
        )
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Investment'], mode='lines+markers', name='Original Time Series'))
    fig.update_layout(
    title='Annual Military Investment',
    xaxis_title='Year',
    yaxis_title='Investment'
)
    data=str(time_series_data)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Generate a small para based on the time series data of military investment in {country}: {data}")
    report = response.text

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graphJSON=graphJSON, report=report)

@app.route('/decompose', methods=['GET'])
def decompose():
    country = request.args.get('country', default='India', type=str)
    df = pd.read_excel('C:/Users/Maaz/Desktop/DDDIs/New/data/Investment1948.xlsx')
    year_column = 'Year'
    value_column = 'Investment'
    df = df.loc[df['Country'] == country]
    df[year_column] = pd.to_datetime(df[year_column], format='%Y')
    df.set_index(year_column, inplace=True)
    
    decomposition = seasonal_decompose(df[value_column], model='multiplicative')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=df.index, y=trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=df.index, y=seasonal, mode='lines', name='Seasonal'))
    fig.add_trace(go.Scatter(x=df.index, y=residual, mode='lines', name='Residual'))

    fig.update_layout(
        title='Time Series Decomposition',
        height=1600,
        grid=dict(rows=4, columns=1, pattern='independent'),
        showlegend=True,
        xaxis_title='Year',
    yaxis_title='Investment'
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Generate a small para based on the time series data of military investment in {country}: Original:{df[value_column]} ,After decomposition the Trend:{trend}, Seasonal:{seasonal}, Residual:{residual}")
    report = response.text
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify(trendJSON=graphJSON,report=report)


def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')
    return result[1]  # p-value

# Function to make the time series stationary
def make_stationary(series):
    p_value = check_stationarity(series)
    d = 0
    while p_value > 0.05:
        d += 1
        series = series.diff().dropna()
        p_value = check_stationarity(series)
        print(f'Differencing {d} time(s)')
    return series, d



# data_path='data/Investment1948.xlsx'
data_path = 'C:/Users/Maaz/Desktop/DDDIs/New/data/Investment1948.xlsx'
year_column='Year'
value_column='Investment'
@app.route('/predict', methods=['GET'])
def predict():
    country = request.args.get('country', default='India', type=str)
    forecast_years = request.args.get('forecast_years', default=10, type=int)
    percentage = request.args.get('percentage', default=0.9, type=float)
    model=request.args.get('model', default='ARIMA', type=str)
    print(forecast_years)
    
    df = pd.read_excel(data_path)
    df = df.loc[df['Country'] == country]
    df[year_column] = pd.to_datetime(df[year_column], format='%Y')
    df.set_index(year_column, inplace=True)
    if model=='ARIMA':
        
        # 2. Check Stationarity (Important for ARIMA)
        result = adfuller(df[value_column])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] <= 0.05:
            print("Time series is likely stationary.")
        else:
            print("Time series is likely non-stationary. Consider differencing.")
        
        train_size = int(len(df) * percentage)
        train_end = df.index[train_size]  # Get the last date of the training data

        train = df[value_column][:train_end]  # Train data up to train_end (inclusive)
        test = df[value_column][train_end:]   # Test data from train_end (exclusive) onwards

        # 6. ARIMA Model
        model = ARIMA(train, order=(1, 1, 1))  # Example order, adjust based on analysis!
        model_fit = model.fit()

        # 7. Forecast and Accuracy Check
        forecast_steps = len(test) + forecast_years
        forecast = model_fit.get_forecast(steps=forecast_steps)  # Forecast the test set AND additional years
        predictions = forecast.predicted_mean[len(test):]         # Isolate only the out-of-sample forecasts
        mae = mean_absolute_error(test, forecast.predicted_mean[:len(test)])
        mse = mean_squared_error(test, forecast.predicted_mean[:len(test)])
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test, forecast.predicted_mean[:len(test)]) * 100
        

        # 8. Plot Forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines+markers', name='Original Time Series'))
        fig.add_trace(go.Scatter(x=forecast.conf_int().index, y=forecast.predicted_mean, mode='lines+markers', name='Forecast'))  
        fig.update_layout(title='Interactive Forecast Plot', xaxis_title='Year', yaxis_title=value_column)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f" in english Generate a small para based on the forecating of time series data of military investment in {country}: Original:{df[value_column]} ,forecast data:{predictions},and discuss the following metrics: Mean Absolute Error (MAE): {mae}, Mean Absolute Percentage Error (MAPE): {mape} which is got from arima model.")
        report = response.text
        print(report)

    elif model=='Simple_Exponential':
        print(model)
        train_size = int(len(df) * percentage)
        train_end = df.index[train_size]  
        train = df[value_column][:train_end] 
        test = df[value_column][train_end:]   

        # 4. ETS (Simple Exponential Smoothing) Model
        model = SimpleExpSmoothing(train).fit(optimized=True)

        # 5. Forecast and Accuracy Check
        forecast_steps = len(test) + forecast_years
        forecast = model.forecast(steps=forecast_steps)
        predictions = forecast[len(test):]  
        mae = mean_absolute_error(test, forecast[:len(test)])
        mse = mean_squared_error(test, forecast[:len(test)])

        # 6. Plot Forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines+markers', name='Original Time Series'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines+markers', name='Forecast'))  
        fig.update_layout(title='Interactive Forecast Plot (ETS SES)', xaxis_title='Year', yaxis_title=value_column)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f" in english Generate a small para based on the forecating of time series data of military investment in {country}: Original:{df[value_column]} ,forecast data:{predictions},and discuss the following metrics: Mean Absolute Error (MAE): {mae}, which is from Simple  exponential model.")
        report = response.text
        print(report)
        
    elif model=='Double_Exponential':
        print(model)
        result = adfuller(df[value_column])
        if result[1] <= 0.05:
            print("Time series is likely stationary.")
        else:
            print("Time series is likely non-stationary. Consider differencing.")

        # 4. Split Data (for Accuracy Check)
        train_size = int(len(df) * percentage)
        train_end = df.index[train_size]  # Get the last date of the training data

        train = df[value_column][:train_end]  # Train data up to train_end (inclusive)
        test = df[value_column][train_end:]   # Test data from train_end (exclusive) onwards

        # 5. Holt's Exponential Smoothing Model
        model = ExponentialSmoothing(train, trend='add')
        model_fit = model.fit()

        # 6. Forecast and Accuracy Check
        forecast_steps = len(test) + forecast_years
        forecast = model_fit.forecast(steps=forecast_steps)  # Forecast the test set AND additional years
        # Evaluate accuracy on test set
        mae = mean_absolute_error(test, forecast[:len(test)])
        mse = mean_squared_error(test, forecast[:len(test)])

        # 7. Plot Forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines+markers', name='Original Time Series'))
        # forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='Y')[1:]
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines+markers', name='Forecast'))
        fig.update_layout(title='Interactive Forecast Plot', xaxis_title='Year', yaxis_title=value_column)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f" in english Generate a small para based on the forecating of time series data of military investment in {country}: Original:{df[value_column]} ,forecast data:{forecast},and discuss the following metrics: Mean Absolute Error (MAE): {mae},  which is from double exponential model.")
        report = response.text
        print(report)
    
    elif model=='Triple_Exponential':
        print(model)
        result = adfuller(df[value_column])
        
        if result[1] <= 0.05:
            print("Time series is likely stationary.")
        else:
            print("Time series is likely non-stationary. Consider differencing.")

        # 4. Split Data (for Accuracy Check)
        train_size = int(len(df) * percentage)
        train_end = df.index[train_size]  # Get the last date of the training data

        train = df[value_column][:train_end]  # Train data up to train_end (inclusive)
        test = df[value_column][train_end:]   # Test data from train_end (exclusive) onwards

        # 5. Holt-Winters Exponential Smoothing Model
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=3)
        model_fit = model.fit()

        # 6. Forecast and Accuracy Check
        forecast_steps = len(test) + forecast_years
        forecast = model_fit.forecast(steps=forecast_steps)  # Forecast the test set AND additional years
        # Evaluate accuracy on test set
        mae = mean_absolute_error(test, forecast[:len(test)])
        mse = mean_squared_error(test, forecast[:len(test)])

        # 7. Plot Forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines+markers', name='Original Time Series'))
        # forecast_index = pd.date_range(start=df.index[-3], periods=forecast_steps + 1, freq='Y')[0:]
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines+markers', name='Forecast'))
        fig.update_layout(title='Interactive Forecast Plot', xaxis_title='Year', yaxis_title=value_column)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f" in english Generate a small para based on the forecating of time series data of military investment in {country}: Original:{df[value_column]} ,forecast data:{forecast},and discuss the following metrics: Mean Absolute Error (MAE): {mae},  which is from triple exponential model.")
        report = response.text
        print(report)
        
    return jsonify(arimagraph=graphJSON,report=report)

@app.route('/stationary', methods=['GET'])
def stationary():
    # 1. Load and Prepare Data
    country = request.args.get('country', default='India', type=str)
    df = pd.read_excel(data_path)
    df = df.loc[df['Country'] == country]
    df[year_column] = pd.to_datetime(df[year_column], format='%Y')
    df.set_index(year_column, inplace=True)

    # 2. Check Stationarity and make stationary if necessary
    stationary_series, d = make_stationary(df[value_column])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stationary_series.index, y=stationary_series, mode='lines+markers', name='Stationary Series'))
    fig.update_layout(title='Interactive Stationary Series Plot', xaxis_title='Year', yaxis_title=value_column)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Generate a small para based on the forecating of time series data of military investment in {country}: Original:{df[value_column]} ,After making the time series stationary the data:{stationary_series} and the differencing is {d} times.")
    report = response.text
    print(report)
    return jsonify(stationarygraph=graphJSON,report=report)

if __name__ == '__main__':
    app.run(debug=True)
