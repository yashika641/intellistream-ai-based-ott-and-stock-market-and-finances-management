import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_handler import load_csv
from utils.logger import get_logger
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import shap
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger=get_logger(name='stock model')


def load_data(file_path):
    try:
        df=load_csv(file_path )
        logger.info('data loaded sucessfully')
        
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    
def data_preprocess(df):
    # Drop unnecessary columns
    try:
        logger.info('data preprocessing started')
        df['open_l-1']=df['open'].shift(1)
        df['high_l-1']=df['high'].shift(1)
        df['low_l-1']=df['low'].shift(1)
        df['close_l-1']=df['close'].shift(1)
        df['volume_l-1']=df['volume'].shift(1)
        logger.info('lag features for previous day created')
        df['open_l-2']=df['open'].shift(2)
        df['high_l-2']=df['high'].shift(2)
        df['low_l-2']=df['low'].shift(2)
        df['close_l-2']=df['close'].shift(2)
        df['volume_l-2']=df['volume'].shift(2)
        logger.info('lag features for previous 2 days created')
        df['next_day_open']=df['open'].shift(-1)
        df['next_day_high']=df['high'].shift(-1)
        df['next_day_low']=df['low'].shift(-1)
        df['next_day_close']=df['close'].shift(-1)
        logger.info('next day features created')
        df.dropna(inplace=True)
        logger.info('data preprocessing completed')
        
        
        return df
    except Exception as e:
        logger.error(f"Error in data preprocess: {e}")
        raise
    
def feature_engineering(df):
    try:
        logger.info('feature engineering started')
        # Use lowercase column names for consistency
        df['date'] = pd.to_datetime(df['date'])  # Ensure date format
        df.set_index('date', inplace=True)  # Set date as index for time-series operations
        logger.info('date column converted to datetime and set as index')
        # Moving Averages
        df['MA_5'] = df['close'].rolling(window=5).mean()
        df['MA_20'] = df['close'].rolling(window=20).mean()
        logger.info('moving averages created')
        # Exponential Moving Averages
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        logger.info('exponential moving averages created')
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        logger.info('relative strength index created')
        # Bollinger Bands
        df['BB_upper'] = df['MA_20'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['MA_20'] - 2 * df['close'].rolling(window=20).std()
        logger.info('bollinger bands created')
        # Momentum and Rate of Change
        df['Momentum_10'] = df['close'] - df['close'].shift(10)
        df['ROC_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        logger.info('momentum and rate of change created')
        # MACD Indicator
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        logger.info('MACD indicator created')
        # Lag Features for Previous Days (+1, -1, +2)
        df['close_-1'] = df['close'].shift(1)   # Previous day close
        df['close_+1'] = df['close'].shift(-1)  # Next day close (for predictions)
        df['close_+2'] = df['close'].shift(-2)  # 2 days ahead close
        logger.info('lag features for previous and next days created')
        # Ensure no NaN values after feature creation
        df.dropna(inplace=True)
        logger.info('NaN values dropped after feature creation')
        # Display updated dataset structure
        print(df.head())
        df_copy = df.copy()
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_copy)
        return df_scaled
    
    except Exception as e:
        logger.error('error in doing feature engineering ')
        raise

def split_data(df):
    try:
        logger.info('data splitting started')
        x = df.drop(columns=['next_day_open', 'next_day_high', 'next_day_low', 'next_day_close'])
        y = df[['next_day_open', 'next_day_high', 'next_day_low', 'next_day_close']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logger.info('data splitting completed')
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logger.error('error in splitting data ')
        raise
 

def model_training(x_train, y_train, x_test, y_test):
    results = {}
    logger.info('model training started')
    # === Traditional ML Models ===
    models = {
        'linear_regression': LinearRegression(),
        'multioutput_regressor': MultiOutputRegressor(LinearRegression()),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        logger.info(f'{name} model trained')
        y_pred = model.predict(x_test)
        logger.info(f'{name} model predictions made')
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
    logger.info('traditional ML models trained and evaluated')

    # === ARIMA (univariate y only) ===
    try:
        logger.info('ARIMA model training started')
        arima_model = ARIMA(y_train, order=(5, 1, 0)).fit()
        logger.info('ARIMA model trained')
        arima_pred = arima_model.forecast(steps=len(y_test))
        logger.info('ARIMA model predictions made')
        results['arima'] = {
            'MAE': mean_absolute_error(y_test, arima_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, arima_pred)),
            'R2': r2_score(y_test, arima_pred)
        }
        logger.info('ARIMA model trained and evaluated')
    except Exception as e:
        print("ARIMA failed:", e)

    # === SARIMAX ===
    try:
        logger.info('SARIMAX model training started')
        
        sarimax_model = SARIMAX(y_train, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        logger.info('SARIMAX model trained')
        sarimax_pred = sarimax_model.forecast(steps=len(y_test))
        logger.info('SARIMAX model predictions made')
        results['sarimax'] = {
            'MAE': mean_absolute_error(y_test, sarimax_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, sarimax_pred)),
            'R2': r2_score(y_test, sarimax_pred)
        }
        logger.info('SARIMAX model trained and evaluated')
    except Exception as e:
        print("SARIMAX failed:", e)

    # === Deep Learning Models ===
    x_train_dl = np.array(x_train).reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_dl = np.array(x_test).reshape((x_test.shape[0], x_test.shape[1], 1))
    deep_models = ['lstm', 'gru', 'conv1d']

    for model_name in deep_models:
        model = Sequential()
        if model_name == 'lstm':
            model.add(LSTM(50, activation='relu', input_shape=(x_train_dl.shape[1], 1)))
        elif model_name == 'gru':
            model.add(GRU(50, activation='relu', input_shape=(x_train_dl.shape[1], 1)))
        elif model_name == 'conv1d':
            model.add(Conv1D(64, 2, activation='relu', input_shape=(x_train_dl.shape[1], 1)))
            model.add(Flatten())

        model.add(Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1))
        logger.info(f'{model_name} model architecture created')
        
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
        logger.info(f'{model_name} model compiled')
        model.fit(x_train_dl, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        logger.info(f'{model_name} model trained')
        y_pred_dl = model.predict(x_test_dl)
        logger.info(f'{model_name} model predictions made')
        results[model_name] = {
            'MAE': mean_absolute_error(y_test, y_pred_dl),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dl)),
            'R2': r2_score(y_test, y_pred_dl)
        }
    logger.info('deep learning models trained and evaluated')
    # === Return Results as DataFrame ===
    df_results = pd.DataFrame(results).T.sort_values('RMSE')
    df_results.reset_index(inplace=True)
    # === Optional: Plot ===
    df_results[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 5))
    plt.title('Model Performance (Lower is Better)')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    logger.info('model training completed')
    return df_results

            
    
def model_training_and_eval(x_train, y_train, x_test, y_test):
    try:
        model= LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred
    except Exception as e:

        logger.error(f"Error in model training and evaluation: {e}")
        raise
    
def hyperparameter_tuning(x_train, y_train, x_test, y_test):
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        model= RandomForestRegressor(random_state=42)
        param_grid={
            
        }
        scorer=make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=5, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        results = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Hyperparameter tuning results: {results}")
        return results,grid_search.best_params_,best_model
    
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise
            
def save_model(model,model_name):
    import pickle
    try:
        model_path=f'model_codes/{model_name}.pkl'
        with open(model_path,'wb'):
            pickle.dump(model,open(model_path,'wb'))
            
        logger.info(f"Model saved at {model_path}")
    except Exception as e:
        logger.error(f"Error in saving model: {e}")
        raise   

def main():
    try:
        file_path=(r'C:\Users\Dell\OneDrive\Desktop\intellistream\intellistream-ai-based-ott-and-stock-market-and-finances-management\docs\synthetic_stock_data_enhanced.csv')   
        df=load_data(file_path=file_path)
        df=data_preprocess(df)
        df=feature_engineering(df)
        x_train, y_train, x_test, y_test = split_data(df)
        results = model_training(x_train, y_train, x_test, y_test)
        print(results)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
        
if __name__ == "__main__":
    main()     
            
        
        