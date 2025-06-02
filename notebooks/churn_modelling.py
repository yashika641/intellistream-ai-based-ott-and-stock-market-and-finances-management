
#data loading to preprocssing pipeline


import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_handler import load_csv ,save_csv # Importing custom function
from utils.logger import get_logger
logger= get_logger(name='churning model')

def load_data(file_path):

    try:
        
        # Load the CSV using your custom load_csv function
        df = load_csv(file_path)
        logger.info("csv loaded sucessfully")
        return df
    
    except Exception as e:
        logger.error("Error loading CSV: %s", e)


def data_preprocessing(df):
    try:
    
        # Show the first row
        print(df.head(10))

        print(df.isna().sum())
        print(df.describe())
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        print(df.columns)
        logger.info('duplicates removed')
        df.drop(columns=['user_id'],inplace=True)
        logger.info('column user_id is dropped%s')
        print(np.isinf(df).sum())

# Check for very large numbers (e.g., > 1e308)
        print((np.abs(df) > 1e308).sum())
        print("\nMax values:\n", df.max())
        print("\nData types:\n", df.dtypes)
        print("Is Inf in X:\n", np.isinf(df).sum())
        print("Is NaN in X:\n", np.isnan(df).sum())
        return df
    
    except Exception as e:
        logger.error('error occured during data preprocessing and the file is not in a good format%s',e)

def feature_enfineering(df):
    try:
        #feature engineering
        logger.debug('feature engineering started')
        df['watch_time_per_session'] = df['total_watch_time'] / (df['sessions_per_week'].replace(0, np.nan) * 4)
        logger.debug('watch_time_per_session is done')
        df['weekly_login_ratio'] = 7 / df['days_since_last_login'].replace(0, np.nan)
        logger.debug('weekly_login_ratio is done')
        df['subscription_age_ratio'] = df['subscription_length_months'] / df['age'].replace(0, np.nan)
        logger.debug('subscription_age_ratio is done')
        
        df['long_term_user'] = df['subscription_length_months'].apply(lambda x: 1 if x > 12 else 0)
        logger.debug('long_time_user is done')
        df['is_inactive_lately']=df['days_since_last_login'].apply(lambda x: 1 if x>14 else 0)
        logger.debug('is inactive lately is done')
        df['payment_issue']=df['payment_success_rate'].apply(lambda x: 1 if x<0.9 else 0)
        logger.debug('payment issue is done')
        logger.debug('feature engineering has completed sucessfully')
        df.fillna(0, inplace=True)
        # if df.isnull().values.any():
        #     logger.error('there are null values in the data')
        #     df.isna().sum()
        return df

    except Exception as e:
        logger.error('Error occurred during feature engineering: %s', e)
        raise


def split_data(df):
    try:
        logger.info('Starting train-test split')

        # Define features and target
        columns = ['age', 'total_watch_time', 'average_session_length',
                   'sessions_per_week', 'days_since_last_login',
                   'subscription_length_months', 'payment_success_rate',
                   'number_of_tickets', 'watch_time_per_session',
                   'weekly_login_ratio', 'subscription_age_ratio',
                   'long_term_user', 'is_inactive_lately', 'payment_issue']
        target = 'churned'

        x = df[columns]
        y = df[target]

        # Standard scaling only on numerical features (optional if all features are numeric)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=columns)

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

        logger.info('Train-test split successful')
        return x_train, x_test, y_train, y_test

    except Exception as e:
        logger.error('Error during train-test split: %s', e)
        raise
def model_training(x_train,y_train,x_test,y_test):
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'catboostclassifier': CatBoostClassifier(),
        'lightgbmclassifier': LGBMClassifier(),
        'xgboostclassifier': XGBClassifier(),
        'linear_regression': LinearRegression()
    }
    results={}
    for name,model in models.items():
        try:
            logger.debug('model_training starts')
            model.fit(x_train,y_train)
            logger.debug('model_traing ends')
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(x_test)[:, 1]
            else:
                y_probs = model.predict(x_test)
                # If output is regression values, clip to [0, 1]
                if name == 'LinearRegression':
                    y_probs = np.clip(y_probs, 0, 1)
            y_pred = (y_probs >= 0.5).astype(int)

            acc= accuracy_score(y_test,y_pred)
            roc=roc_auc_score(y_test,y_probs)
            f1=f1_score(y_test,y_pred)
            results[name]=f1
            results[name]=roc
            results[name]=acc
            print(f"{name} Accuracy: {acc:.4f}")
            print(f"{name} roc_auc: {acc:.4f}")
            print(f"{name} f1_score: {acc:.4f}")
            logger.debug('scoring done')
    # Final result summary
            print("\nModel Accuracy Comparison:")
            for model, score in results.items():
                print(f"{model}: {score:.4f}")
            logger.debug('results printed')
        except Exception as e:
            logger.error('Error during model training: %s', e)

def model_training_catboost(x_train,y_train,x_test,y_test):
    try:
        logger.debug('model_training_catboost starts')
        model=CatBoostClassifier()
        model.fit(x_train,y_train)
        logger.debug('model_training_catboost ends')
        y_probs = model.predict_proba(x_test)[:, 1]
        y_pred = (y_probs >=0.5).astype(int)
        acc= accuracy_score(y_test,y_pred)
        roc=roc_auc_score(y_test,y_probs)
        f1=f1_score(y_test,y_pred)
        print(f"'accuracy':{acc},'roc':{roc},'f1_score':{f1}")
        logger.debug('scoring done')
        return acc,roc,f1,model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def hyperparameter_tuning():
    try:
        logger.debug('hyperparameter_tuning starts')
        # Define hyperparameter space for GridSearchCV
        param_grid ={

        }

    except Exception as e:
        logger.error('Error during hyperparameter tuning: %s', e)
        raise


def main():
    df = load_data(r'C:\Users\Dell\OneDrive\Desktop\intellistream\intellistream-ai-based-ott-and-stock-market-and-finances-management\docs\synthetic_churn_data.csv')
    
    if df is None:
        logger.error("Data loading failed. Exiting pipeline.")
        return
    
    df = data_preprocessing(df)
    if df is None:
        logger.error("Preprocessing failed. Exiting pipeline.")
        return

    df = feature_enfineering(df)
    if df is None:
        logger.error("Feature engineering failed. Exiting pipeline.")
        return

    x_train, x_test, y_train, y_test = split_data(df)
    logger.info('Data preprocessing and feature engineering completed')
    model_training(x_train, y_train, x_test, y_test)
    logger.info('training catboost model')
    model=model_training_catboost(x_train,y_train,x_test,y_test)
    logger.info('model training completed')
    model=CatBoostClassifier()
    print(model.get_all_params().keys())



if __name__ == "__main__":
    main()