from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from backend.services.nlp_service import analyze_script

router= APIRouter()

class scriptrequest(BaseModel):
    script_txt:str

class scriptresponse(BaseModel):
    sentiment:str
    emotion:str
    success_prediction:bool

@router.post('/script-nlp',response_model=scriptresponse)
def script_nlp(request:scriptrequest):
    result = analyze_script(request.script_txt)
    return result