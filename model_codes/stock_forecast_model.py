import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.file_handler import load_csv,save_csv
from utils.logger import get_logger
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import shap
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger=get_logger(name='stock model')


def load_data(file_path):
    try:
        df=load_csv(file_path,index=False)
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    
def data_preprocess(df):
    # Drop unnecessary columns
    try:
        df['open_l-1']=df['open'].shift(1)
        df['high_l-1']=df['high'].shift(1)
        df['low_l-1']=df['low'].shift(1)
        df['close_l-1']=df['close'].shift(1)
        df['volume_l-1']=df['volume'].shift(1)
        df['open_l-2']=df['open'].shift(2)
        df['high_l-2']=df['high'].shift(2)
        df['low_l-2']=df['low'].shift(2)
        df['close_l-2']=df['close'].shift(2)
        df['volume_l-2']=df['volume'].shift(2)
        
        df['next_day_open']=df['open'].shift(-1)
        df['next_day_high']=df['high'].shift(-1)
        df['next_day_low']=df['low'].shift(-1)
        df['next_day_close']=df['close'].shift(-1)
        df=df.dropna(inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error in data preprocess: {e}")
        raise
    
def feature_engineering(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure date format
        df.set_index('Date', inplace=True)  # Set Date as index for time-series operations

        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()

        # Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

        # Momentum and Rate of Change
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # MACD Indicator
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # Lag Features for Previous Days (+1, -1, +2)
        df['Close_-1'] = df['Close'].shift(1)   # Previous day close
        df['Close_+1'] = df['Close'].shift(-1)  # Next day close (for predictions)
        df['Close_+2'] = df['Close'].shift(-2)  # 2 days ahead close

        # Ensure no NaN values after feature creation
        df.dropna(inplace=True)

        # Display updated dataset structure
        print(df.head())
        df=df.copy()
        scaler=StandardScaler()
        df=scaler.fit_transform(df)
        return df
    
    except Exception as e:
        logger.error('error in doing feature engineering ')
        
    def split_data(df):
        try:
            x=df.drop(columns=['next_day_open', 'next_day_high', 'next_day_low', 'next_day_close'])
            y=df['next_day_open', 'next_day_high', 'next_day_low', 'next_day_close']
            x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            return x_train,y_train,x_test,y_test
        except Exception as e:
            logger.error('error in splitting data ')
            raise
        
    def model_training(x_train,y_train,x_test,y_test):
        try:
            # Initialize the model
            models = {
                'LSTM': LSTM_model(),
                'GRU': GRU_model(),
                'MLP': MLP_model(),
                'CNN': CNN_model(),
                'RNN': RNN_model(),
                'ARIMA': ARIMA(),
                'SARIMA': SARIMAX(),
                'LSTM_with_attention': LSTM_with_attention_model(),
                'LSTM_with_LSTM': LSTM_with_LSTM_model(),
                'LSTM_with_GRU': LSTM_with_GRU_model(),
                'logistic_regression': LogisticRegression(),
                'linear_regression': LinearRegression(),  # Corrected indentation
                'random_forest_regressor': RandomForestRegressor()
            }
            model=LinearRegression()
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            return model,y_pred
        
            
    
    
            
        
        