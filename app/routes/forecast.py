from fastapi import APIRouter, HTTPException
import pickle
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import run_query

router = APIRouter()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ml", "models", "prophet_model.pkl"
)

@router.get("/forecast")
def forecast(days: int = 30):
    try:
        # Load saved Prophet model
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        # Load recent daily orders from DB to use as regressor
        query = """
            SELECT
                d.full_date                 AS ds,
                COUNT(DISTINCT f.order_id)  AS total_orders
            FROM fact_orders f
            JOIN dim_date d ON f.order_date_key = d.date_key
            WHERE f.order_status = 'delivered'
            GROUP BY d.full_date
            ORDER BY d.full_date
        """
        history     = run_query(query)
        history_df  = pd.DataFrame(history)
        history_df["ds"] = pd.to_datetime(history_df["ds"])

        # Build future dataframe
        future = model.make_future_dataframe(periods=days)

        # Merge known total_orders into future
        future = future.merge(history_df, on="ds", how="left")

        # Fill missing future dates with rolling average of last 14 days
        avg_orders = history_df["total_orders"].tail(14).mean()
        future["total_orders"] = future["total_orders"].fillna(avg_orders)

        # Predict
        forecast_df = model.predict(future)

        # Return only future predictions
        result = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days).copy()
        result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")

        return {
            "forecast_days" : days,
            "total_revenue" : round(result["yhat"].sum(), 2),
            "daily_average" : round(result["yhat"].mean(), 2),
            "peak_day"      : result.loc[result["yhat"].idxmax(), "ds"],
            "peak_revenue"  : round(result["yhat"].max(), 2),
            "predictions"   : result.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))