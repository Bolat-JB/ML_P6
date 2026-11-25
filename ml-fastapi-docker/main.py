from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


class AdultIncomeInput(BaseModel):
    """Input schema matching the 12 features used by the trained pipeline."""

    age: int = Field(..., ge=0, le=120)
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., ge=0)
    capital_loss: int = Field(..., ge=0)
    hours_per_week: int = Field(..., ge=1, le=168)
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "Private",
                    "education": "Bachelors",
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 5178,
                    "capital_loss": 0,
                    "hours_per_week": 45,
                    "native_country": "United-States",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    income_class: str
    probability_over_50k: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serialized sklearn pipeline once when the API starts."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Adult Census Income Prediction API",
    description="Predicts whether annual income exceeds $50K using a trained scikit-learn pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: AdultIncomeInput) -> PredictionResponse:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_df = pd.DataFrame([features.model_dump()]).replace("?", np.nan)
    pred = int(model.predict(input_df)[0])
    prob_over_50k = float(model.predict_proba(input_df)[0][1])

    return PredictionResponse(
        prediction=pred,
        income_class=">50K" if pred == 1 else "<=50K",
        probability_over_50k=round(prob_over_50k, 4),
    )
