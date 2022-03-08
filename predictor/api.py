import sys
import os

sys.path.insert(1, os.getcwd())

import uvicorn
from fastapi.responses import JSONResponse

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from predict import Predict

app = FastAPI()

# @app.get('/teste', status_code=200)
# def test_endpoint():
#     """
#     Endpoint de teste
#     :return: resposta teste
#     """
#     return JSONResponse({"resposta":"Oi mundo!"})

class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Sex: str
    Age: int


@app.get('/health', status_code=200)
def health_check():
    """teste de sanidade da aplicação"""
    return JSONResponse({"status": "healthy"})

@app.post('/predict')
def predict(passenger: Passenger):
    """
    Use input from request to transform data and make predictions
    :param data: list of passenger used as input to model
    :return: predictions serialized as json
    """
    data = pd.DataFrame([vars(passenger)])
    id = passenger.PassengerId
    del data['PassengerId']
    status, result = predictor.predict(data)
    return JSONResponse({"id":int(id), "score":float(result), "predicao":status})


if __name__ == "__main__":
    predictor = Predict()

    uvicorn.run(
        app=app,
        host='0.0.0.0',
        port=8000
    )