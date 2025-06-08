
#data loading to preprocssing pipeline


import pandas as pd
import numpy as np
import sys
import os
import category_encoders as ce
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler ,LabelEncoder   
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report
from sklearn.metrics import make_scorer,mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error,mean_absolute_percentage_error

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
        # print(np.isinf(df).sum())

# Check for very large numbers (e.g., > 1e308)
        # print((np.abs(df) > 1e308).sum())
        le=LabelEncoder()
        columns=['gender', 'region', 'plan', 'device_type', 'preferred_genre', 'payment_method', 'ad_type', 'peak_hours']
        for col in columns:
            df[col]=le.fit_transform(df[col])
        encoder=ce.BinaryEncoder(cols=['auto_renew'])
        
        print("\nMax values:\n", df.max())
        print("\nData types:\n", df.dtypes)
        # print("Is Inf in X:\n", np.isinf(df).sum())
        # print("Is NaN in X:\n", np.isnan(df).sum())
        return df
    
    except Exception as e:
        logger.error('error occured during data preprocessing and the file is not in a good format%s',e)

def feature_enfineering(df):
    try:
        #feature engineering
        logger.debug('feature engineering starts')
        df['total_revenue'] = df['monthly_charge'] * df['months_subscribed']
        logger.debug('total_revenue done')
        df['watch_time_ratio'] = df['watch_time_per_day'] / 24
        logger.debug('watch time ratio done')
        df['avg_watch_hours_per_month'] = df['total_watch_hours'] / df['months_subscribed']
        binge_threshold = 7.5
        df['binge_indicator'] = (df['binge_score'] > binge_threshold).astype(int)
        logger.debug('binge indicator done')
        heavy_user_threshold = 3

        df['is_heavy_user'] = (df['watch_time_per_day'] > heavy_user_threshold).astype(int)
        logger.debug('is heavy user done')
        # Convert to datetime if not already
        
        if 'last_active_date' in df.columns and 'joined_date' in df.columns:
            df['joined_date'] = pd.to_datetime(df['joined_date'], errors='coerce')
            df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')
            
            df['churn_days_gap'] = (df['last_active_date'] - df['joined_date']).dt.days
            df['tenure'] = df['churn_days_gap'] / 30.0
            df['tenure'] = df['tenure'].round(1)
            logger.debug("churn_days_gap and tenure engineered")
        else:
            logger.warning("joined_date or last_active_date not found in DataFrame — skipping churn_days_gap & tenure")

        df.fillna(0, inplace=True)
        df = df.drop(columns=['joined_date', 'last_active_date','monthly_charge','months_subscribed',
                              'watch_time_per_day','total_watch_hours','binge_score','churn_days_gap'])
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
        columns = ['age', 'gender', 'region', 'plan',
                'device_type', 'preferred_genre',
                'payment_method', 'ad_type', 'auto_renew',
                'discount_used', 'complaints', 'support_calls',
                'rating', 'reviews_written', 'peak_hours',
                'total_revenue', 'watch_time_ratio', 'avg_watch_hours_per_month',
                'binge_indicator', 'is_heavy_user', 'tenure']
        target = 'churn'

        x = df[columns]
        y = df[target]

        # Standard scaling only on numerical features (optional if all features are numeric)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=columns)

        # Train-test split
        model=CatBoostClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)
        rfe=RFE(model,n_features_to_select=5)
        x_train=rfe.fit_transform(x_train,y_train)
        x_test=rfe.transform(x_test)
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
        logger.info('catboost model is training')
        model = CatBoostClassifier(
        iterations=200,            # Number of boosting rounds
        learning_rate=0.1,        # Step size
        depth=4,                   # Depth of the tree
        l2_leaf_reg=3.0,           # L2 regularization
        loss_function='Logloss',   # Loss for binary classification
        eval_metric='AUC', 
        od_type='Iter',# Evaluation metric
        random_seed=42,            # Reproducibility
        verbose=100,
        rsm=1.0 ,
        random_strength=1,
        border_count=128,
        boosting_type='Ordered',
        bootstrap_type='Bayesian',
        bagging_temperature=0.5,
        grow_policy='SymmetricTree',
        class_weights=[1,5405/2595],
        )

        model=model.fit(x_train,y_train,eval_set=(x_test,y_test),use_best_model=True)
        y_pred=model.predict(x_test)
        acc= accuracy_score(y_test,y_pred)
        results=classification_report(y_test,y_pred)
        logger.info('classification report: %s', results)
        roc=roc_auc_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)
        
        return model,acc,f1,roc,results
        
    except Exception as e:
        logger.error('Error during catboost model training: %s', e)
        raise

def model_training_linear_regression(x_train,y_train,x_test,y_test):
    try:
        logger.debug('model_training_linear_regression starts')
        model=LinearRegression()
        model=model.fit(x_train,y_train)
        logger.debug('model_training_linear_regression ends')
        y_pred = model.predict(x_test)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        acc= accuracy_score(y_test,y_pred_binary)
        roc=roc_auc_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred_binary)
        print(f"'accuracy':{acc},'roc':{roc},'f1_score':{f1}")
        logger.debug('scoring done')
        return acc,roc,f1
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def hyperparameter_tuning_linear_regression(x_train,y_train):
    try:
        logger.debug('hyperparameter_tuning starts')
        # Define hyperparameter space for GridSearchCV
        model=LinearRegression()
        param_grid ={
            'fit_intercept':[True,False],
            'copy_X':[True,False],
            'positive': [True,False]

        }

        mse_scorer=make_scorer(mean_squared_error,greater_is_better=False)

        Grid_search=GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=mse_scorer,
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        Grid_search.fit(x_train,y_train)
        print('best parameters found', Grid_search.best_params_)
        print('best CV mse:',-Grid_search.best_score_)

        return Grid_search.best_estimator_


    except Exception as e:
        logger.error('Error during hyperparameter tuning: %s', e)
        raise

def hyperparamter_tuning_catboost(x_train,y_train):
    try:
        model=CatBoostClassifier()
        param_grid = {
            'depth': [4, 6, 8],                        # Controls model complexity
            'learning_rate': [0.01, 0.05, 0.1],        # Step size for weight updates
            'iterations': [100, 200, 300],             # Max trees before stopping
            'l2_leaf_reg': [1, 3, 5, 7, 9],            # Regularization strength
            'border_count': [32, 64, 128],             # Number of bins for continuous features
            'bagging_temperature': [0.1, 0.5, 1],      # For Bayesian bootstrap (subsampling control)
            'random_strength': [1, 5, 10],             # Score noise for feature selection
            'rsm': [0.8, 1.0],                         # Row subsample (Random Subspace Method)
            'boosting_type': ['Plain', 'Ordered'],     # Gradient boosting strategy
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],  # Tree growth method
            'bootstrap_type': ['Bayesian', 'Bernoulli'],
             }
        roc_auc=make_scorer(roc_auc_score,greater_is_better=True)
        from sklearn.model_selection import RandomizedSearchCV

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=1000,           # Change this to 100 for better coverage
            cv=3,
            scoring='roc_auc',
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(x_train, y_train)
        print("Best Parameters:", random_search.best_params_)
        print("Best ROC AUC Score:", random_search.best_score_)

        return random_search.best_estimator_

    except Exception as e:
        logger.error('Error during hyperparameter tuning: %s', e)
        raise

def main():
    df = load_data(r'C:\Users\Dell\OneDrive\Desktop\intellistream\intellistream-ai-based-ott-and-stock-market-and-finances-management\docs\churn_data.csv')
    
    if df is None:
        logger.error("Data loading failed. Exiting pipeline.")
        return
    
    df = data_preprocessing(df)
    if df is None:
        logger.error("Preprocessing failed. Exiting pipeline.")
        return

    df = feature_enfineering(df)  # Keep the name consistent or fix function name
    if df is None:
        logger.error("Feature engineering failed. Exiting pipeline.")
        return

    x_train, x_test, y_train, y_test = split_data(df)
    logger.info('Data preprocessing and feature engineering completed')
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training label distribution: {dict(zip(unique, counts))}")

    # Train and evaluate multiple models
    model_training(x_train, y_train, x_test, y_test)
    
    logger.info('Training CatBoost model')
    model, acc, f1, roc,results = model_training_catboost(x_train, y_train, x_test, y_test)
    logger.info(f"CatBoost - Accuracy: {acc}, F1: {f1}, ROC AUC: {roc}")

    # logger.debug('Hyperparameter tuning started')
    # best_catboost_model = hyperparamter_tuning_catboost(x_train, y_train)
    # logger.debug('Hyperparameter tuning completed')

    # Evaluate best tuned model
    y_pred_best =  model.predict(x_test)
    y_proba_best = model.predict_proba(x_test)[:, 1]
    results_best = classification_report(y_test, y_pred_best)
    
    print(results_best)
    logger.info(f"Best CatBoost tuned model classification report:\n{results_best}")
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    logger.info('SHAP values calculated and summary plot generated')
    # import joblib
    # # Save the best model
    # joblib.dump(best_catboost_model, 'best_model.joblib')
    # logger.info('Best model saved as best_model.joblib')
    
    # You may want to do something with precision, recall, thresholds here

if __name__ == "__main__":
    main()



 