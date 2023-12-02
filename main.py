from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_demand_pipeline, predict_gdp_growth
from model.model import __version__ as model_version
import uvicorn

app = FastAPI(title="Solaire Predictor API")

class TextInput(BaseModel):
    country_code: str
    year: str

class PredictionOut(BaseModel):
    predicted_value: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict-demand", response_model=PredictionOut)
def predict(payload: TextInput):
    energy_demand = predict_demand_pipeline(payload.country_code, payload.year)
    return {"predicted_value": energy_demand}

@app.post("/predict-gdp-growth", response_model=PredictionOut)
def predict(payload: TextInput):
    gdp_growth = predict_gdp_growth(payload.country_code, payload.year)
    return {"predicted_value": gdp_growth}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
