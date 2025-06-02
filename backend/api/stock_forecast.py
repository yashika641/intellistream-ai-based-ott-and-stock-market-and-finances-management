from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.stock_forecast_service import forecast_stock

router = APIRouter()

class StockRequest(BaseModel):
    company_name: str
    days: int = 7  # Number of days to predict

class StockResponse(BaseModel):
    company_name: str
    forecast: list

router = APIRouter()

@router.post("/stock", response_model=StockResponse)
def stock_prediction(request: StockRequest):
    forecast = forecast_stock(request.company_name, request.days)
    return {
        "company_name": request.company_name,
        "forecast": forecast
    }
