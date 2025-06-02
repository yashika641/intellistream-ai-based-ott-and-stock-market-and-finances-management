from fastapi import FastAPI
from backend.api import recommender,churn,script_nlp,stock_forecast

app= FastAPI(
    title='intellistream API',
    version='1.0'
)

app.include_router(recommender.router,prefix='/api')
app.include_router(churn.router,prefix='/api')
app.include_router(script_nlp.router,prefix='/api')
app.include_router(stock_forecast.router,prefix='/api')