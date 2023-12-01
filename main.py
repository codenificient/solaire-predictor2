from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_demand_pipeline
from model.model import __version__ as model_version
import uvicorn

app = FastAPI(title="Solaire Predictor API")

class TextInput(BaseModel):
    country_code: str
    year: str

class PredictionOut(BaseModel):
    energy_demand: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict-demand", response_model=PredictionOut)
def predict(payload: TextInput):
    energy_demand = predict_demand_pipeline(payload.country_code, payload.year)
    return {"energy_demand": energy_demand}

