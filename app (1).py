
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Load the trained model
model = joblib.load("model.pkl")

app = FastAPI(title="Titanic Survival Prediction API")

# Validate passenger input
class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }

# Prediction endpoint
@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive",
        "survival_probability": round(probability, 4)
    }
