import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator

st.set_page_config(layout="wide", page_title="Stock Price Predictions", initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.sidebar.title("Stock Analysis")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
    
    is_valid, error_message = validate_dates(start_date, end_date)
    if not is_valid:
        st.error(error_message)
        return

    stock_name = st.sidebar.text_input("Stock Name", "MSFT")
    option = st.sidebar.selectbox("Select Option", ["Visualize", "View Table", "Prediction"])
    data = fetch_stock_data(stock_name, start_date, end_date)
    if data is None:
        return
    if option == "Visualize":
        visualize(start_date, end_date, stock_name)
    elif option == "View Table":
        view_table(start_date, end_date, stock_name)
    elif option == "Prediction":
        test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.2, 0.05)
        predict(start_date, end_date, stock_name, test_size)

@st.cache_data
def fetch_stock_data(stock_name, start_date, end_date):
    try:
        data = yf.download(stock_name, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {stock_name}. Please check the stock symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def validate_dates(start_date, end_date):
    if start_date >= end_date:
        return False, "Start date must be before end date."
    return True, ""

def visualize(start_date, end_date, stock_name):
    
    data = yf.download(stock_name, start=start_date, end=end_date)
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Candlestick', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'], horizontal=True)

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]

    # MACD
    macd = MACD(data.Close).macd()

    # RSI
    rsi = RSIIndicator(data.Close).rsi()

    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()

    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Candlestick':
        # st.markdown('# Candlestick Chart')
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'],
                                             increasing={'line': {'color': '#089981'}},
                                             decreasing={'line': {'color': '#f23645'}})])
        fig.update_layout(xaxis_rangeslider_visible=False, width=800, height=600)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True)

    elif option == 'BB':
        # st.markdown('# Bollinger Bands')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bb.index, y=bb['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=bb.index, y=bb['bb_h'], mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=bb.index, y=bb['bb_l'], mode='lines', name='Lower Band'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif option == 'MACD':
        # st.markdown('# Moving Average Convergence Divergence')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=macd.index, y=macd, mode='lines', name='MACD'))
        fig.update_layout(xaxis_title='Date', yaxis_title='MACD', width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif option == 'RSI':
        # st.markdown('# Relative Strength Indicator')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI'))
        fig.update_layout(xaxis_title='Date', yaxis_title='RSI', width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif option == 'SMA':
        # st.markdown('# Simple Moving Average')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name='SMA'))
        fig.update_layout(xaxis_title='Date', yaxis_title='SMA', width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # st.markdown('# Exponential Moving Average')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name='EMA'))
        fig.update_layout(xaxis_title='Date', yaxis_title='EMA', width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)

def view_table(start_date, end_date, stock_name):
    st.title(f"{stock_name} Historical Data")
    data = yf.download(stock_name, start=start_date, end=end_date)
    st.dataframe(data, width = 1000, height = 550)

def predict(start_date, end_date, stock_name, test_size):
    st.title(f"{stock_name} Prediction")
    data = yf.download(stock_name, start=start_date, end=end_date, period="max")

    model = st.selectbox("Select Model", ["Linear Regression", "Decision Tree Regression", "Random Forest Regression"])

    if model == "Linear Regression":
        predict_linear_regression(data, test_size)
    elif model == "Decision Tree Regression":
        predict_decision_tree_regression(data, test_size)
    elif model == "Random Forest Regression":
        predict_random_forest_regression(data, test_size)

def predict_linear_regression(stock_data, test_size):

    df = stock_data.copy()
    x = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=False)
    reg = LinearRegression(n_jobs=-1)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    df_res = pd.DataFrame({"Real Price": y_test, 'Predicted Price': pred})

    def evaluation_metrics_print():
        st.markdown("# Evaluation Metrics")
        st.write(f'Mean Absolute Error (MAE): {mean_absolute_error(df_res["Real Price"], df_res["Predicted Price"])}')
        st.write(f'Mean Squared Error (MSE) : {mean_squared_error(df_res["Real Price"], df_res["Predicted Price"])}')
        st.write(f'Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(df_res["Real Price"], df_res["Predicted Price"]))}')

    def comparing_chart_print():
        st.markdown("# Pred VS Actual DataFrame")
        st.dataframe(df_res, width=1000)

    def charts_print():
        st.markdown("# Real VS Predicted Chart")
        fig_preds = go.Figure()
        fig_preds.add_trace(go.Scatter(x=stock_data.index[train_size:], y=df_res['Real Price'], mode='lines', name='Actual', line=dict(color='green')))
        fig_preds.add_trace(go.Scatter(x=stock_data.index[train_size:], y=df_res['Predicted Price'], mode='lines', name='Predicted', line=dict(color='red')))
        fig_preds.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_preds, use_container_width=True)

        # Residual Plot with multi-colored markers based on residual values
        st.markdown("# Residual Plot")
        df_res['residual'] = df_res['Real Price'] - df_res['Predicted Price']
        fig = px.scatter(df_res, x='Predicted Price', y='residual', marginal_y='violin', trendline='ols', color='residual', color_continuous_scale=px.colors.diverging.Portland, labels={'x': 'Predicted Values', 'y': 'Residual'})
        fig.update_layout(xaxis_title='Predicted Values', yaxis_title='Residual')
        st.plotly_chart(fig, use_container_width=False)

        # Enhanced Prediction Error Analysis with discrete color scale for scatter plot markers
        st.markdown("# Enhanced Prediction Error Analysis")
        df_res['residual'] = df_res['Real Price'] - df_res['Predicted Price']
        fig = px.scatter(df_res, x='Real Price', y='Predicted Price', marginal_x='histogram', marginal_y='histogram', trendline='ols', height=650, color='residual', color_discrete_sequence=px.colors.diverging.Portland, labels={'x': 'Actual Values', 'y': 'Predicted Values'})
        fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
        fig.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=df_res['Real Price'].min(), y0=df_res['Real Price'].min(), x1=df_res['Real Price'].max(), y1=df_res['Real Price'].max())
        fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values')
        st.plotly_chart(fig, use_container_width=False)

    def forecast_print(df_res, stock_data, n_days):
        last_date = stock_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=n_days+1, freq='D')[1:]
        forecast_features = stock_data[['Open', 'High', 'Low', 'Volume']].iloc[-n_days:].values
        forecast_values = reg.predict(forecast_features)

        st.markdown("# Forecast Chart")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual', line=dict(color='green')))
        fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecast', line=dict(color='red')))
        fig_forecast.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_forecast, use_container_width=True)

    train_size = int(len(df) * (1 - test_size))
    type_eval = st.radio("Select Evaluation Type", options=['Full Report', 'Evaluation Metrics', 'Comparing Table', 'Charts', 'Forecast'], horizontal=True)
    if type_eval == 'Evaluation Metrics':
        evaluation_metrics_print()
    elif type_eval == 'Comparing Table':
        comparing_chart_print()
    elif type_eval == 'Charts':
        charts_print()
    elif type_eval == 'Forecast':
        n_days = st.number_input("Enter the number of days to forecast", min_value=1, step=1)
        forecast_print(df_res, stock_data, n_days)
    else:
        evaluation_metrics_print()
        comparing_chart_print()
        charts_print()
        n_days = st.number_input("Enter the number of days to forecast", min_value=1, step=1)
        forecast_print(df_res, stock_data, n_days)

def predict_decision_tree_regression(stock_data, test_size, max_depth=None, min_samples_split=2, min_samples_leaf=1):

    df = stock_data.copy()
    
    # Feature Engineering
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['Close_shift_1'] = df['Close'].shift(1)
    
    x = df[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_21', 'Volume_MA_7', 'Close_shift_1']]
    y = df['Close']
    
    # Walk-Forward Validation
    train_size = int(len(df) * (1 - test_size))
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    y_test_all = []
    y_pred_all = []
    
    for i in range(train_size, len(df)):
        x_train, x_test = x.iloc[:i], x.iloc[i:i+1]
        y_train, y_test = y.iloc[:i], y.iloc[i:i+1]
        
        tree_reg = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
        tree_reg.fit(x_train, y_train)
        pred = tree_reg.predict(x_test)
        
        mae_scores.append(mean_absolute_error(y_test, pred))
        mse_scores.append(mean_squared_error(y_test, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, pred)))
        y_test_all.append(y_test.values[0])
        y_pred_all.append(pred[0])
    
    df_res = pd.DataFrame({"Real Price": y_test_all, 'Predicted Price': y_pred_all})
    
    def evaluation_metrics_print(df_res):
        st.markdown("# Evaluation Metrics")
        st.write(f'Mean Absolute Error (MAE): {mean_absolute_error(df_res["Real Price"], df_res["Predicted Price"])}')
        st.write(f'Mean Squared Error (MSE): {mean_squared_error(df_res["Real Price"], df_res["Predicted Price"])}')
        st.write(f'Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(df_res["Real Price"], df_res["Predicted Price"]))}')

    def comparing_chart_print(df_res):
        st.markdown("# Pred VS Actual DataFrame")
        st.dataframe(df_res, width=1000)

    def charts_print(df_res, stock_data):
        st.markdown("# Real VS Predicted Chart")
        fig_preds = go.Figure()
        fig_preds.add_trace(go.Scatter(x=stock_data.index[train_size:], y=df_res['Real Price'], mode='lines', name='Actual', line=dict(color='green')))
        fig_preds.add_trace(go.Scatter(x=stock_data.index[train_size:], y=df_res['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='red')))
        fig_preds.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_preds, use_container_width=True)

        # Residual Plot with multi-colored markers based on residual values
        st.markdown("# Residual Plot")
        df_res['residual'] = df_res['Real Price'] - df_res['Predicted Price']
        fig = px.scatter(df_res, x='Predicted Price', y='residual', marginal_y='violin', trendline='ols', color='residual', color_continuous_scale=px.colors.diverging.Portland, labels={'x': 'Predicted Values', 'y': 'Residual'})
        fig.update_layout(xaxis_title='Predicted Values', yaxis_title='Residual')
        st.plotly_chart(fig, use_container_width=False)

        # Enhanced Prediction Error Analysis with discrete color scale for scatter plot markers
        st.markdown("# Enhanced Prediction Error Analysis")
        df_res['residual'] = df_res['Real Price'] - df_res['Predicted Price']
        fig = px.scatter(df_res, x='Real Price', y='Predicted Price', marginal_x='histogram', marginal_y='histogram', trendline='ols', height=650, color='residual', color_discrete_sequence=px.colors.diverging.Portland, labels={'x': 'Actual Values', 'y': 'Predicted Values'})
        fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
        fig.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=df_res['Real Price'].min(), y0=df_res['Real Price'].min(), x1=df_res['Real Price'].max(), y1=df_res['Real Price'].max())
        fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values')
        st.plotly_chart(fig, use_container_width=False)

    def forecast_print(df_res, stock_data, n_days):
        last_date = stock_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=n_days+1, freq='D')[1:]
        forecast_values = tree_reg.predict(x.iloc[-n_days:])

        st.markdown("# Forecast Chart")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual', line=dict(color='green')))
        fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecast', line=dict(color='red')))
        fig_forecast.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_forecast, use_container_width=True)

    type_eval = st.radio("Select Evaluation Type", options=['Evaluation Metrics', 'Comparing Table', 'Charts', 'Forecast', 'All'], horizontal=True)
    if type_eval == 'Evaluation Metrics':
        evaluation_metrics_print(df_res)
    elif type_eval == 'Comparing Table':
        comparing_chart_print(df_res)
    elif type_eval == 'Charts':
        charts_print(df_res, stock_data)
    elif type_eval == 'Forecast':
        n_days = st.number_input("Enter the number of days to forecast", min_value=1, step=1)
        forecast_print(df_res, stock_data, n_days)
    else:
        evaluation_metrics_print(df_res)
        comparing_chart_print(df_res)
        charts_print(df_res, stock_data)
        n_days = st.number_input("Enter the number of days to forecast", min_value=1, step=1)
        forecast_print(df_res, stock_data, n_days)

def predict_random_forest_regression(stock_data, test_size):
    
    df = stock_data.copy()
    
    # Feature Engineering
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['Close_shift_1'] = df['Close'].shift(1)
    
    # Add feature engineering columns to stock_data
    stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()
    stock_data['MA_21'] = stock_data['Close'].rolling(window=21).mean()
    stock_data['Volume_MA_7'] = stock_data['Volume'].rolling(window=7).mean()
    stock_data['Close_shift_1'] = stock_data['Close'].shift(1)
    
    x = df[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_21', 'Volume_MA_7', 'Close_shift_1']]
    y = df['Close']
    
    # Walk-Forward Validation
    train_size = int(len(df) * (1 - test_size))
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    y_test_all = []
    y_pred_all = []
    
    for i in range(train_size, len(df)):
        x_train, x_test = x.iloc[:i], x.iloc[i:i+1]
        y_train, y_test = y.iloc[:i], y.iloc[i:i+1]
        
        rf_reg = RandomForestRegressor(n_estimators=5, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, max_features='sqrt', max_samples=0.8)
        rf_reg.fit(x_train, y_train)
        pred = rf_reg.predict(x_test)
        
        mae_scores.append(mean_absolute_error(y_test, pred))
        mse_scores.append(mean_squared_error(y_test, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, pred)))
        y_test_all.append(y_test.values[0])
        y_pred_all.append(pred[0])
    
    df_res = pd.DataFrame({"Real Price": y_test_all, 'Predicted Price': y_pred_all})
    
    def evaluation_metrics_print(df_res):
        st.markdown("# Evaluation Metrics")
        st.write(f'Mean Absolute Error (MAE): {mean_absolute_error(df_res["Real Price"], df_res["Predicted Price"])}')
        st.write(f'Mean Squared Error (MSE): {mean_squared_error(df_res["Real Price"], df_res["Predicted Price"])}')
        st.write(f'Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(df_res["Real Price"], df_res["Predicted Price"]))}')

    def comparing_chart_print(df_res):
        st.markdown("# Pred VS Actual DataFrame")
        st.dataframe(df_res, width=1000, height=550)

    def charts_print(df_res, stock_data):
        st.markdown("# Real VS Predicted Chart")
        fig_preds = go.Figure()
        fig_preds.add_trace(go.Scatter(x=stock_data.index[train_size:], y=df_res['Real Price'], mode='lines', name='Actual', line=dict(color='green')))
        fig_preds.add_trace(go.Scatter(x=stock_data.index[train_size:], y=df_res['Predicted Price'], mode='lines', name='Predicted', line=dict(color='red')))
        fig_preds.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_preds, use_container_width=True)

        # Residual Plot with multi-colored markers based on residual values
        st.markdown("# Residual Plot")
        df_res['residual'] = df_res['Real Price'] - df_res['Predicted Price']
        fig = px.scatter(df_res, x='Predicted Price', y='residual', marginal_y='violin', trendline='ols', color='residual', color_continuous_scale=px.colors.diverging.Portland, labels={'x': 'Predicted Values', 'y': 'Residual'})
        fig.update_layout(xaxis_title='Predicted Values', yaxis_title='Residual')
        st.plotly_chart(fig, use_container_width=False)

        # Enhanced Prediction Error Analysis with discrete color scale for scatter plot markers
        st.markdown("# Enhanced Prediction Error Analysis")
        df_res['residual'] = df_res['Real Price'] - df_res['Predicted Price']
        fig = px.scatter(df_res, x='Real Price', y='Predicted Price', marginal_x='histogram', marginal_y='histogram', trendline='ols', height=650, color='residual', color_discrete_sequence=px.colors.diverging.Portland, labels={'x': 'Actual Values', 'y': 'Predicted Values'})
        fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
        fig.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=df_res['Real Price'].min(), y0=df_res['Real Price'].min(), x1=df_res['Real Price'].max(), y1=df_res['Real Price'].max())
        fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values')
        st.plotly_chart(fig, use_container_width=False)

    def forecast_print(df_res, stock_data, n_days):
        last_date = stock_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=n_days+1, freq='D')[1:]
        forecast_features = stock_data[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_21', 'Volume_MA_7', 'Close_shift_1']].iloc[-n_days:].values
        forecast_values = rf_reg.predict(forecast_features)

        st.markdown("# Forecast Chart")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual', line=dict(color='green')))
        fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecast', line=dict(color='red')))
        fig_forecast.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_forecast, use_container_width=True)

    type_eval = st.radio("Select Evaluation Type", options=['Evaluation Metrics', 'Comparing Table', 'Charts', 'Forecast', 'Full Report'], horizontal=True)
    if type_eval == 'Evaluation Metrics':
        evaluation_metrics_print(df_res)
    elif type_eval == 'Comparing Table':
        comparing_chart_print(df_res)
    elif type_eval == 'Charts':
        charts_print(df_res, stock_data)
    elif type_eval == 'Forecast':
        n_days = st.number_input("Enter the number of days to forecast", min_value=1, step=1)
        forecast_print(df_res, stock_data, n_days)
    else:
        evaluation_metrics_print(df_res)
        comparing_chart_print(df_res)
        charts_print(df_res, stock_data)
        n_days = st.number_input("Enter the number of days to forecast", min_value=1, step=1)
        forecast_print(df_res, stock_data, n_days)

if __name__ == "__main__":
    main()