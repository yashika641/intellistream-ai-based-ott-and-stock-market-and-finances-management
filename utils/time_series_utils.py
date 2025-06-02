# utils/time_series_utils.py

import pandas as pd
import numpy as np
from typing import Optional

# === Handling Timestamps === #
def convert_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def fill_missing_dates(df: pd.DataFrame, date_col: str, freq: str = 'D') -> pd.DataFrame:
    df = df.set_index(date_col).asfreq(freq).reset_index()
    return df

# === Feature Engineering === #
def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

# === Lag & Rolling Features === #
def add_lag_feature(df: pd.DataFrame, value_col: str, lag: int = 1) -> pd.DataFrame:
    df[f'{value_col}_lag{lag}'] = df[value_col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, value_col: str, windows: list[int]) -> pd.DataFrame:
    for window in windows:
        df[f'{value_col}_roll_mean_{window}'] = df[value_col].rolling(window).mean()
        df[f'{value_col}_roll_std_{window}'] = df[value_col].rolling(window).std()
    return df

# === Anomaly Detection Support === #
def detect_anomalies_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    df['anomaly'] = (np.abs(z_scores) > threshold).astype(int)
    return df

# === Train-Test Split === #
def time_series_train_test_split(df: pd.DataFrame, date_col: str, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df[date_col] = pd.to_datetime(df[date_col])
    train = df[df[date_col] < split_date]
    test = df[df[date_col] >= split_date]
    return train, test
