from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import os
import pandas as pd

router = APIRouter()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ml", "models", "xgboost_review_model.pkl"
)

class ReviewRequest(BaseModel):
    price               : float
    freight_value       : float
    payment_value       : float
    payment_installments: int
    days_to_deliver     : int
    days_estimated      : int
    delivery_delay_days : int

@router.post("/review-predict")
def predict_review(request: ReviewRequest):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        # Build feature row matching training features
        data = {
            "price"               : request.price,
            "freight_value"       : request.freight_value,
            "payment_value"       : request.payment_value,
            "payment_installments": request.payment_installments,
            "days_to_deliver"     : request.days_to_deliver,
            "days_estimated"      : request.days_estimated,
            "delivery_delay_days" : request.delivery_delay_days,
            "delivered_early"     : int(request.delivery_delay_days < 0),
            "delivery_gap"        : request.days_to_deliver - request.days_estimated,
            "freight_ratio"       : request.freight_value / (request.price + 1),
            "price_to_payment"    : request.price / (request.payment_value + 1),
            "expensive_item"      : int(request.price > 120),
            "high_installments"   : int(request.payment_installments > 3),
        }

        df   = pd.DataFrame([data])

        # Align columns with trained model
        model_features = model.get_booster().feature_names
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        df = df[model_features]

        prediction   = model.predict(df)[0]
        probability  = model.predict_proba(df)[0]

        return {
            "prediction"          : "Good Review" if prediction == 1 else "Bad Review",
            "confidence"          : round(float(max(probability)) * 100, 2),
            "good_review_prob"    : round(float(probability[1]) * 100, 2),
            "bad_review_prob"     : round(float(probability[0]) * 100, 2),
            "delivered_early"     : bool(request.delivery_delay_days < 0),
            "delivery_gap_days"   : request.days_to_deliver - request.days_estimated
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))