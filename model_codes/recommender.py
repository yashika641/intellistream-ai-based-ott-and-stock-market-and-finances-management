import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from utils.file_handler import load_csv
from utils.logger import get_logger
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.layers import Input ,Dense,Dropout,concatenate,Flatten,Embedding, Concatenate
from keras.optimizers import Adam
from keras.models import Model, Sequential


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
        print(num_users = df['user'].nunique().sum())
        print(num_movies = df['movie'].nunique().sum())
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
        
def fnn_recommender(df,num_users, num_movies):
    x_user = df['user'].values
    x_movie = df['movie'].values
    y = df['watch_time_min'].values
    y = y / df['watch_time_min'].max()
    
    x_train_user, x_test_user, x_train_movie, x_test_movie, y_train, y_test = train_test_split(
    x_user, x_movie, y, test_size=0.2, random_state=42)
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    
    user_embedding = Embedding(num_users, 50)(user_input)
    movie_embedding = Embedding(num_movies, 50)(movie_input)

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    
    concat = Concatenate()([user_vec, movie_vec])
    sequential_block = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    output = sequential_block(concat)

    # Build model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    # Step 5: Train
    model.fit([x_train_user, x_train_movie], y_train,
            validation_data=([x_test_user, x_test_movie], y_test),
            epochs=10, batch_size=64)

    # Step 6: Predict and denormalize
    preds = model.predict([x_test_user, x_test_movie]) * df['watch_time_min'].max()
    print(preds[:5])
            
def cosine_similarity_recommender(df):
    # Combine features for better representation
    df['text_features'] = df['genre'].fillna('') + ' ' + df['title'].fillna('')

    # Step 2: Vectorize using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])

    # Step 3: Cosine similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Step 4: Create mapping from movie_id to index
    movie_id_to_index = pd.Series(df.index, index=df['movie_id'])
    
def bert_sentiment_analysis(df):
    try:
        from transformers import pipeline

        bert_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

        
        result = bert_sentiment(df['title'][:512])[0]
        stars = int(result['label'][0])  # e.g., '4 stars'
        df['title_sentiment'] = df['title'].apply(bert_sentiment_analysis)
        return stars  # Convert to numeric sentiment
    except:
                return 3  # Neutral fallback
def main_scoring_recommender():
    file_path=(r'C:\Users\Dell\OneDrive\Desktop\intellistream\intellistream-ai-based-ott-and-stock-market-and-finances-management\docs\ott_full_dataset.csv')
    # Step 1: Load the data
    df = load_data(file_path)

    # Step 2: Preprocess the data
    df = data_preprocess(df)

    # Step 3: Feature Engineering
    df = feature_engineering(df)

    # Step 4: Run FNN-based recommendation system
    num_users = df['user'].nunique()
    num_movies = df['movie'].nunique()
    fnn_recommender(df, num_users, num_movies)

    # Step 5: Run Cosine Similarity Recommender
    cosine_similarity_recommender(df)

    # Step 6: (Optional) Sentiment Analysis using BERT
    try:
        df['sentiment_score'] = df['title'].apply(lambda x: bert_sentiment_analysis({"title": x}))
    except Exception as e:
        logger.warning(f"Sentiment analysis skipped due to error: {e}")

    logger.info("All recommenders and analysis completed.")

# Entry point
if __name__ == "__main__":
    main_scoring_recommender()


