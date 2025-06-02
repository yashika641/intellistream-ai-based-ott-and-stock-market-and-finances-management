#settings for the model
from pydantic import BaseSettings

class settings(BaseSettings):
    PROJECT_NAME:str = 'IntelliStreamAI'
    API_VERSION:str = 'v1'

    ALLOWED_ORIGINS=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8501"
    ]
    DATA_DIR: str = "data/"
    MODEL_DIR: str = "models/"
    SCRIPT_DIR: str = "data/scripts/"
    
    CHURN_MODEL_PATH: str = "models/churn_model/model.pkl"
    RECOMMENDER_MODEL_PATH: str = "models/recommender/model.pt"
    NLP_MODEL_PATH: str = "models/nlp_model/"

    USE_FIREBASE: bool = True
    MONGO_URI: str = "mongodb://localhost:27017/"


settings = Settings()
# ✅ You can then import settings in any FastAPI route or module: 
from config.settings import settings
print(settings.PROJECT_NAME)

