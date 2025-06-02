from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from backend.services.recommender_service import get_recommendations

router = APIRouter()

class recommendationrequest(BaseModel):
    user_id: str

class recommendationresponse(BaseModel):
    user_id:str
    recommended: list[str]

@router.post('/recommend',response_model=recommendationresponse)
def recommend_content(request:recommendationrequest):
    recommendations= get_recommendations(request.user_id)
    return{
        'user_id':request.user_id,
        'recommended':recommendations
        }