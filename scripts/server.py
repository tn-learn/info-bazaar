from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bazaar.lem_utils import get_llm

app = FastAPI()


class APIQuery(BaseModel):
    query: str
    model_name: str

class APIResponse(BaseModel):
    response: str

@app.post("/predict", status_code=200)
def get_prediction(query: APIQuery):
    try:
        breakpoint()
        llm = get_llm(query.model_name)
        response = {"ping": "pong"}
    except Exception as detail:
        raise HTTPException(status_code=500, detail=detail)
    return response
