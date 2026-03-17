import os
import logging
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.orm import Session

import mlflow.xgboost
from prometheus_fastapi_instrumentator import Instrumentator

from db import get_db, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#--- Config ----#
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detection-model")
 
# Features
FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category",
    "distance_from_home",
    "transaction_count",
    "is_foreign",
]
# Global model variable
model = None

# Lifespane
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/production")
        logger.info(f"Model {MODEL_NAME} loaded successfully from MLflow.")
    except Exception as e:
        logger.warning(f"Failed to load model from MLflow: {e}")
        logger.warning("API will start without a model. Predictions will fail until a model is deployed.")
    yield

# FastAPI app instance
app = FastAPI(title="Fraud Detection API",
              description="API for fraud detection using XGBoost model. Model trained with MLflow, artifacts stored in MinIO (S3-compatible).",
              version="1.0.0",
              lifespan=lifespan)

# Instrumentator for Prometheus metrics - tracks request count, latency, error rate 
Instrumentator().instrument(app).expose(app)

# Pydantic model for request validation
class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None,  description="Optional ID for tracking")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    merchant_category: int = Field(..., ge=0, description="Merchant category code")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km")
    transaction_count: int = Field(..., ge=1, description="Number of transactions today")
    is_foreign: int = Field(..., ge=0, le=1, description="1 if foreign transaction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 2500.00,
                "hour_of_day": 3,
                "day_of_week": 6,
                "merchant_category": 17,
                "distance_from_home": 350.0,
                "transaction_count": 15,
                "is_foreign": 1,
            }
        }
    }

class PredictionResponse(BaseModel):
    is_fraud:           int
    fraud_probability:  float
    verdict:            str

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., min_length=1)

class BatchPredictionResponse(BaseModel):
    index: int
    is_fraud: Optional[int] = None
    fraud_probability: Optional[float] = None
    verdict: Optional[str] = None
    error: Optional[str] = None

# Run the model inference for a single transaction and log the prediction to the database
def run_inference(tx: TransactionRequest, db:Session) -> PredictionResponse:
    features = np.array(
        [[getattr(tx, f) for f in FEATURES]],
        dtype=float
    )

    prob = float(model.predict_proba(features)[0][1])
    is_fraud = int(prob >= 0.5)
    verdict = "FRAUDULENT" if is_fraud else "LEGITIMATE"

    # Log prediction to DB
    try:
        prediction_log = Prediction(
            transaction_id = tx.transaction_id,
            fraud_probability = round(prob, 4),
            decision = verdict,
            model_version = MODEL_NAME,
        )
        db.add(prediction_log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log prediction to database: {e}")
    
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        verdict=verdict
    )

# Health check endpoint. Used by Docker, load balancers, and monitoring tools to check if the API is up and if the model is loaded.
@app.get("/health", tags=["System"])
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "model_name":   MODEL_NAME,
    }

# Predict whether a transaction is fraudulent
# Validate input with Pydantic, run XGBoost inference, log to PostgreSQL
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(transaction: TransactionRequest, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    return run_inference(transaction, db)