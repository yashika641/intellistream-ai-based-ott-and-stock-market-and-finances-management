import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        sys.exit(1)
        
def data_preprocess(df):
    try:
        logger.info("Starting data preprocessing...")
        categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols=df.select_dtypes(include=['number']).columns.tolist()
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numerical columns: {numerical_cols}")
        df[categorical_cols] = df[categorical_cols].astype('category')
        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        logger.info(f"Data shape after dropping NA: {df.shape}")
        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(inplace=True)
        logger.info("Data reset index and duplicates removed.")
        for col in categorical_cols:
            le=LabelEncoder()
            df[col]=le.fit_transform(df[col])
        logger.info("Categorical columns encoded successfully.")
        
        scaler=StandardScaler()
        df[numerical_cols]=scaler.fit_transform(df[numerical_cols])
        logger.info("Numerical columns scaled successfully.")
        logger.info("Data preprocessing completed successfully.")
        return df,scaler,le
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        sys.exit(1)
        
def feature_engineering(df):
    try:
        logger.info("Starting feature engineering...")
        df['watch_completion_ratio'] = df['watch_time_min'] / df['duration_min']
        logger.info("Watch completion ratio calculated.")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info("Timestamp converted to datetime.")
        df['view_hour'] = df['timestamp'].dt.hour
        logger.info("View hour extracted from timestamp.")
        df['view_dayofweek'] = df['timestamp'].dt.dayofweek
        bins = [0, 17, 25, 35, 50, 100]
        labels = ['teen', 'young_adult', 'adult', 'middle_age', 'senior']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        logger.info("Age groups created.")
        df['date'] = df['timestamp'].dt.date
        logger.info("Date extracted from timestamp.")
        binge_watch = df.groupby(['user_id', 'date'])['watch_time_min'].sum().reset_index()
        binge_watch['is_binge'] = binge_watch['watch_time_min'] > 120
        binge_watch = binge_watch[['user_id', 'date', 'is_binge']]
        df = df.merge(binge_watch[['user_id', 'date', 'is_binge']], on=['user_id', 'date'], how='left')
        user_genre_counts = df.groupby(['user_id', 'genre']).size().unstack(fill_value=0)
        df = df.join(user_genre_counts, on='user_id', rsuffix='_watched')
        logger.info("User genre counts merged into the main dataframe.")
        df['days_since_watch'] = (pd.Timestamp.now() - df['timestamp']).dt.days
        df.drop(columns=['timestamp', 'age'], inplace=True)
        logger.info("Timestamp and age columns dropped.")
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        logger.info("Feature engineering completed successfully.")
        num_users = df['user_name'].nunique()
        num_movies = df['movie_name'].nunique()
        logger.info(f"Number of users: {num_users}")
        logger.info(f"Number of movies: {num_movies}")
        return df,num_movies,num_users
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        sys.exit(1)
        
def fnn_recommender(df,num_users, num_movies):
    x_user = df['user_id'].values
    x_movie = df['movie_id'].values
    y = df['watch_time_min'].values
    y = y / df['watch_time_min'].max()
    logger.info("Starting FNN recommender model training...")
    
    x_train_user, x_test_user, x_train_movie, x_test_movie, y_train, y_test = train_test_split(
    x_user, x_movie, y, test_size=0.2, random_state=42)
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    
    user_embedding = Embedding(num_users, 50)(user_input)
    movie_embedding = Embedding(num_movies, 50)(movie_input)
    logger.info("Embedding layers created for users and movies.")
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    logger.info("Flatten layers applied to embeddings.")
    
    concat = Concatenate()([user_vec, movie_vec])
    sequential_block = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    logger.info("Sequential block created with Dense layers.")
    output = sequential_block(concat)
    logger.info("Output layer created.")
    # Build model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    logger.info("Model compiled with Adam optimizer and MSE loss.")
    # Step 5: Train
    model.fit([x_train_user, x_train_movie], y_train,
            validation_data=([x_test_user, x_test_movie], y_test),
            epochs=10, batch_size=64)
    logger.info("Model training completed.")
    # Step 6: Predict and denormalize
    preds = model.predict([x_test_user, x_test_movie]) * df['watch_time_min'].max()
    print(preds[:5])
    logger.info("Predictions made and denormalized.")
    return preds,model
    
def cosine_similarity_recommender(df, top_n=5):
    try:
        logger.info("Starting cosine similarity recommendation for all movies.")
        df = df.reset_index(drop=True)
        # Select feature columns
        feature_cols = df.drop(columns=['movie_id', 'title'], errors='ignore').columns.tolist()

        # Convert categorical columns to string first to avoid setitem errors
        for col in feature_cols:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        # Attempt numeric conversion
        # df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        # Show percentage of NaNs introduced in each feature column
        df[feature_cols].apply(lambda col: pd.to_numeric(col, errors='coerce').isna().mean() * 100).sort_values(ascending=False)

        # Option 1: Drop rows with any NaN values in features
        initial_rows = df.shape[0]
        df = df.dropna(subset=feature_cols)
        logger.info("Dropped %d rows due to invalid or missing feature data.", initial_rows - df.shape[0])

        # Option 2: If you prefer to keep all data, use fillna instead (comment out dropna above)
        # df[feature_cols] = df[feature_cols].fillna(0)

        if df.empty:
            logger.error("No valid data remains after preprocessing. Cannot compute recommendations.")
            return {}

        # Select only numeric features
        movie_features = df[feature_cols].select_dtypes(include=[np.number])

        if movie_features.empty:
            logger.error("Feature matrix is empty after selecting numeric types.")
            return {}

        logger.info("Computing cosine similarity matrix.")
        similarity_matrix = cosine_similarity(movie_features)

        movie_indices = df.index
        movie_titles = df['title'].values
        recommendations = {}

        logger.info("Generating top %d similar movies for each movie.", top_n)
        for idx, movie_title in zip(movie_indices, movie_titles):
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = [i for i in sim_scores if i[0] != idx]  # exclude self

            top_similar_indices = [i[0] for i in sim_scores[:top_n]]
            top_similar_titles = df.iloc[top_similar_indices]['title'].tolist()

            recommendations[movie_title] = top_similar_titles
            logger.info("Recommendations for '%s': %s", movie_title, top_similar_titles)

        logger.info("Completed cosine similarity recommendations.")
        return recommendations ,similarity_matrix

    except Exception as e:
        logger.error("Error in generating recommendations: %s", str(e))
        raise
        
def bert_sentiment_analysis(df):
    try:
        from transformers import pipeline
        logger.info("Starting BERT sentiment analysis...")
        bert_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

        logger.info("BERT sentiment analysis pipeline created.")
        # You currently only run on first 512 titles and get first result which is problematic:
        # result = bert_sentiment(df['title'][:512])[0]
        # stars = int(result['label'][0])  # e.g., '4 stars' <-- unused actually

        # Instead, apply sentiment analysis to all titles properly:
        def analyze_sentiment(title):
            try:
                result = bert_sentiment(title)[0]
                stars = int(result['label'][0])  # e.g., '4 stars' -> 4
                return stars
            except Exception as e:
                logger.warning(f"Sentiment analysis failed on title '{title}': {e}")
                return 3  # neutral fallback

        df['title_sentiment'] = df['title'].apply(analyze_sentiment)
        logger.info("Sentiment analysis completed and stars extracted.")
        return df

    except Exception as e:
        logger.error(f"BERT sentiment analysis pipeline error: {e}")
        # Return df unchanged but add a neutral sentiment column
        df['title_sentiment'] = 3
        return df

            
def final_recommendations(df,num_movies,num_users):
    try:
        logger.info("Generating final recommendations...")
        content_recommendations,Model = fnn_recommender(df, num_users, num_movies)
        logger.info("Content-based recommendations generated.")
        user_based_recommendations,similarity_matrix = cosine_similarity_recommender(df, top_n=5)
        top_n =5
        logger.info("User-based recommendations generated.")
        df = bert_sentiment_analysis(df)
        logger.info("BERT sentiment analysis completed.")
        all_recommendations = content_recommendations[:]
        if isinstance(content_recommendations, np.ndarray):
            all_recommendations = content_recommendations.tolist()
        else:
            all_recommendations = list(content_recommendations+user_based_recommendations.values())
        flat_recommendations = []
        for item in all_recommendations:
            if isinstance(item, list):
                flat_recommendations.extend(item)
            else:
                flat_recommendations.append(item)

        # Remove duplicates by converting to set then back to list
        all_recommendations = list(set(flat_recommendations))

        recommended_movies_df = df[df['title'].isin(all_recommendations)]
        recommended_movies_df = recommended_movies_df.sort_values(by="title_sentiment", ascending=False)
        final_recommendations_list = recommended_movies_df['title'].tolist()[:top_n]

        logger.info(f"Final recommendations generated: {final_recommendations_list}")
        return final_recommendations_list, Model,similarity_matrix

    except Exception as e:
        logger.error(f"Error in final recommendations: %s", e)
        raise        
# Save utility at bottom of the script or import from utils/save_model.py

def save_model_components(
    df,
    cosine_sim_matrix,
    fnn_model,
    save_dir="saved_model",
    scalers=None,
    label_encoders=None
    ):
    try:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving model components to {save_dir}")

        movie_id_to_index = pd.Series(df.index, index=df['movie_id']).to_dict()

        with open(os.path.join(save_dir, "recommender_metadata.pkl"), "wb") as f:
            pickle.dump({
                "movie_id_to_index": movie_id_to_index,
                "cosine_similarity_matrix": cosine_sim_matrix,
                "dataframe": df,
                "scalers": scalers,
                "label_encoders": label_encoders
            }, f)
        logger.info("Metadata and similarity matrix saved.")

        # Save Keras model
        fnn_model.save(os.path.join(save_dir, "fnn_model.h5"))
        logger.info("FNN model saved successfully.")

        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

    
def main():
    file_path=(r'C:\Users\Dell\OneDrive\Desktop\intellistream\intellistream-ai-based-ott-and-stock-market-and-finances-management\docs\ott_full_dataset.csv')
    # Step 1: Load the data
    df = load_data(file_path)

    # Step 2: Preprocess the data
    df,scaler,LabelEncoder = data_preprocess(df)

    # Step 3: Feature Engineering
    df,num_movies,num_users = feature_engineering(df)

    final_recommendation,model,similarity_matrix=final_recommendations(df,num_movies,num_users)
    
    logger.info(f"Final Recommendations: {final_recommendation}")
    
    save_model_status = save_model_components(df,similarity_matrix,model,
    save_dir="saved_model",
    scalers=scaler,
    label_encoders=LabelEncoder)
    if save_model_status:
        logger.info("Model saved successfully.")
        
    else:
        logger.error("Failed to save the model.")

# Entry point
if __name__ == "__main__":
    main()


