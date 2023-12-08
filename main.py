from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_electricity_usage, predict_gdp_growth, predict_population, predict_electrification, predict_population_growth, predict_gdp_total
from model.model import __version__ as model_version
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Solaire Predictor API")


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000/dashboard",
    "https://solaire-lemon.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class UserInput(BaseModel):
    country_code: str
    year: str
    use_linear: bool

class PredictionOut(BaseModel):
    predicted_value: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict-electricity-usage", response_model=PredictionOut)
def predict(payload: UserInput):
    energy_demand = predict_electricity_usage(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": energy_demand}

@app.post("/predict-gdp-total", response_model=PredictionOut)
def predict(payload: UserInput):
    gdp_total = predict_gdp_total(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": gdp_total}

@app.post("/predict-gdp-growth", response_model=PredictionOut)
def predict(payload: UserInput):
    gdp_growth = predict_gdp_growth(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": gdp_growth}

@app.post("/predict-population", response_model=PredictionOut)
def predict(payload: UserInput):
    population = predict_population(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": population}

@app.post("/predict-population-growth", response_model=PredictionOut)
def predict(payload: UserInput):
    pop_growth = predict_population_growth(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": pop_growth}

@app.post("/predict-electrification-rate", response_model=PredictionOut)
def predict(payload: UserInput):
    rate = predict_electrification(payload.country_code, payload.year, payload.use_linear)
    return {"predicted_value": rate}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
