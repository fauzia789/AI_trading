import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Streamlit UI setup
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start = '2010-01-01'
end = '2023-12-31'

# Alpha Vantage API Key
api_key = 'your_alpha_vantage_api_key'
ts = TimeSeries(key=api_key, output_format='pandas')

try:
    data, meta_data = ts.get_daily(symbol=user_input, outputsize='full')
    
    if data.empty:
        st.error("No data fetched for the given symbol.")
    else:
        # Sort data by date to ensure it's in ascending order
        data = data.sort_index()

        st.subheader(f"Data for {user_input} from 2010-2023")
        st.write(data.describe())
        
        # Visualizing Closing Price
        st.subheader('Closing Price vs Time chart')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['4. close'], label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Splitting the data
        train_data = data['4. close']['2010-01-01':'2023-12-31']
        test_data = data['4. close']['2024-01-01':]

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(np.array(train_data).reshape(-1, 1))

        # Prepare data for LSTM model
        def create_dataset(dataset, time_step=100):
            x_data, y_data = [], []
            for i in range(time_step, len(dataset)):
                x_data.append(dataset[i-time_step:i, 0])
                y_data.append(dataset[i, 0])
            return np.array(x_data), np.array(y_data)

        # Creating datasets for training
        x_train, y_train = create_dataset(train_scaled)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        # Building the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

        # LSTM Predictions
        past_100_days = train_data[-100:]
        final_df = pd.concat([past_100_days, test_data], ignore_index=True)
        input_data = scaler.transform(final_df.values.reshape(-1, 1))
        
        x_test, y_test = create_dataset(input_data)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        y_pred_lstm = model.predict(x_test)
        
        # Rescale Predictions
        scale_factor = 1 / scaler.scale_[0]
        y_pred_lstm = y_pred_lstm * scale_factor
        y_test = y_test * scale_factor
        
        # Evaluate LSTM
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        rmse_lstm = np.sqrt(mse_lstm)
        mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
        
        # Linear Regression (for test data)
        lin_model = LinearRegression()
        lin_train_X = np.arange(len(train_data)).reshape(-1, 1)
        lin_train_y = np.array(train_data).reshape(-1, 1)
        lin_model.fit(lin_train_X, lin_train_y)
        
        # Adjust Linear Regression prediction to match test period
        lin_test_X = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
        lin_pred = lin_model.predict(lin_test_X)
        
        mse_lin = mean_squared_error(y_test, lin_pred)
        rmse_lin = np.sqrt(mse_lin)
        mae_lin = mean_absolute_error(y_test, lin_pred)
        
        # Random Forest (for test data)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(lin_train_X, lin_train_y.ravel())
        
        rf_pred = rf_model.predict(lin_test_X)
        
        mse_rf = mean_squared_error(y_test, rf_pred)
        rmse_rf = np.sqrt(mse_rf)
        mae_rf = mean_absolute_error(y_test, rf_pred)
        
        # Display errors
        st.subheader('Model Evaluation')
        st.write(f'**LSTM:** MSE={mse_lstm:.2f}, RMSE={rmse_lstm:.2f}, MAE={mae_lstm:.2f}')
        st.write(f'**Linear Regression:** MSE={mse_lin:.2f}, RMSE={rmse_lin:.2f}, MAE={mae_lin:.2f}')
        st.write(f'**Random Forest:** MSE={mse_rf:.2f}, RMSE={rmse_rf:.2f}, MAE={mae_rf:.2f}')
        
        # Visualization
        st.subheader('Predictions vs Original')
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, 'b', label='Original Price')
        ax.plot(y_pred_lstm, 'r', label='LSTM Prediction')
        ax.plot(lin_pred, 'g', label='Linear Regression Prediction')
        ax.plot(rf_pred, 'y', label='Random Forest Prediction')
        plt.legend()
        st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred: {e}")
