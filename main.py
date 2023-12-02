from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_electricity_usage, predict_gdp_growth, predict_population, predict_electrification, predict_population_growth
from model.model import __version__ as model_version
import uvicorn

app = FastAPI(title="Solaire Predictor API")

class TextInput(BaseModel):
    country_code: str
    year: str
    use_linear: bool

class PredictionOut(BaseModel):
    predicted_value: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict-electricity-demand", response_model=PredictionOut)
def predict(payload: TextInput):
    energy_demand = predict_electricity_usage(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": energy_demand}

@app.post("/predict-gdp-growth", response_model=PredictionOut)
def predict(payload: TextInput):
    gdp_growth = predict_gdp_growth(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": gdp_growth}

@app.post("/predict-population", response_model=PredictionOut)
def predict(payload: TextInput):
    population = predict_population(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": population}

@app.post("/predict-population-growth", response_model=PredictionOut)
def predict(payload: TextInput):
    pop_growth = predict_population_growth(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": pop_growth}

@app.post("/predict-electrification", response_model=PredictionOut)
def predict(payload: TextInput):
    rate = predict_electrification(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": rate}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
