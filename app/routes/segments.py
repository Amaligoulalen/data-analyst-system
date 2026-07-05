from fastapi import APIRouter, HTTPException
import pickle
import os
import sys
import pandas as pd
from sqlalchemy import text
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import engine

router = APIRouter()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "ml", "models",
    "kmeans_model.pkl"
)

SCALER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "ml", "models",
    "kmeans_scaler.pkl"
)

SEGMENT_LABELS = {
    0: "Standard Buyers",
    1: "Mid-Value Buyers",
    2: "Premium Buyers",
    3: "Frustrated Buyers"
}

FEATURES = [
    "total_spent",
    "avg_order_value",
    "avg_freight",
    "avg_delivery_days",
    "avg_review_score",
    "avg_delay",
    "total_installments",
    "unique_categories",
    "price_per_day",
    "freight_ratio"
]


@router.get("/segments")
def get_segments():

    try:

        # Load model
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        # Query data
        query = """
        SELECT
            c.customer_id,

            COUNT(DISTINCT f.order_id) AS total_orders,
            SUM(f.price)               AS total_spent,
            AVG(f.price)               AS avg_order_value,
            AVG(f.freight_value)       AS avg_freight,
            AVG(f.days_to_deliver)     AS avg_delivery_days,
            AVG(f.review_score)        AS avg_review_score,
            AVG(f.delivery_delay_days) AS avg_delay,
            SUM(f.payment_installments) AS total_installments,
            COUNT(DISTINCT p.category_name_english) AS unique_categories

        FROM fact_orders f
        JOIN dim_customers c
            ON f.customer_key = c.customer_key
        JOIN dim_products p
            ON f.product_key = p.product_key

        WHERE f.order_status = 'delivered'
        AND f.days_to_deliver IS NOT NULL
        AND f.review_score IS NOT NULL

        GROUP BY c.customer_id

        -- FIXED
        HAVING COUNT(DISTINCT f.order_id) >= 1

        LIMIT 500
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        if df.empty:
            raise ValueError("Query returned 0 rows")

        # Feature Engineering (same as training)

        df["price_per_day"] = (
            df["total_spent"] /
            (df["avg_delivery_days"] + 1)
        )

        df["freight_ratio"] = (
            df["avg_freight"] /
            (df["avg_order_value"] + 1)
        )

        # OPTIONAL (only if used in training)
        # log transform
        log_features = [
            "total_spent",
            "avg_order_value",
            "price_per_day"
        ]

        for col in log_features:
            df[col] = np.log1p(df[col])

        # Prepare data
        X = df[FEATURES].fillna(0)

        X_scaled = scaler.transform(X)

        clusters = model.predict(X_scaled)

        df["cluster"] = clusters

        df["segment"] = df["cluster"].map(
            SEGMENT_LABELS
        )

        # Segment summary
        summary = df.groupby(
            ["cluster", "segment"]
        ).agg(
            customer_count=("customer_id", "count"),
            avg_spent=("total_spent", "mean"),
            avg_review=("avg_review_score", "mean"),
            avg_delivery=("avg_delivery_days", "mean")
        ).reset_index().round(2)

        return {
            "total_customers": len(df),
            "segments": summary.to_dict(
                orient="records"
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )