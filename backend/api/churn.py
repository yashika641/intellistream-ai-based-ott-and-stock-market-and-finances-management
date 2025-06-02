from fastapi import APIRouter 
from pydantic import BaseModel
from typing import Dict

from backend.services.churn_service import predict_churn

router = APIRouter()

class ChurnRequest(BaseModel):
    user_id:str
    watch_time:float
    num_pauses:int
    day_active:int

class churnresponse(BaseModel):
    user_id:str
    churn_probability:float


@router.post('/churn',response_model=churnresponse)
def churn_prediction (request:ChurnRequest):
    probability=predict_churn(request.model_dump())
    return {
        'user_id':request.user_id,
        'churn_probability':probability
        }
