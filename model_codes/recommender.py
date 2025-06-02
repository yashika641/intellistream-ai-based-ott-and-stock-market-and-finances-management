import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from utils.file_handler import load_csv
from utils.logger import get_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

logger=get_logger(name="recommender model")

def load_data(file_path):
    try:
            
        df=load_csv(file_path)
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        sys.exit(1)
        
def data_preprocess(df):
    try:
        categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols=df.select_dtypes(include=['number']).columns.tolist()
        df[categorical_cols] = df[categorical_cols].astype('category')
        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(inplace=True)
        for col in categorical_cols:
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col])
            
        for col in numerical_cols:
            scaler=StandardScaler()
            df[col]=scaler.fit_transform(df[col])
            
        logger.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        sys.exit(1)
        
def feature_engineering(df):
    try:
        df['watch_completion_ratio'] = df['watch_time_min'] / df['duration_min']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['view_hour'] = df['timestamp'].dt.hour
        df['view_dayofweek'] = df['timestamp'].dt.dayofweek
        bins = [0, 17, 25, 35, 50, 100]
        labels = ['teen', 'young_adult', 'adult', 'middle_age', 'senior']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        df['date'] = df['timestamp'].dt.date
        binge_watch = df.groupby(['user_id', 'date'])['watch_time_min'].sum().reset_index()
        binge_watch['is_binge'] = binge_watch['watch_time_min'] > 120
        df = df.merge(binge_watch[['user_id', 'date', 'is_binge']], on=['user_id', 'date'], how='left')
        user_genre_counts = df.groupby(['user_id', 'genre']).size().unstack(fill_value=0)
        df = df.join(user_genre_counts, on='user_id', rsuffix='_watched')
        df['days_since_watch'] = (pd.Timestamp.now() - df['timestamp']).dt.days
        df.drop(columns=['timestamp', 'age'], inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        logger.info("Feature engineering completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        sys.exit(1)
        
def split_data(df):
    try:
        x=df.drop()



    
