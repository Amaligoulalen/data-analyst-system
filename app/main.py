import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.query    import router as query_router
from routes.forecast import router as forecast_router
from routes.review   import router as review_router
from routes.segments import router as segments_router

# ============================================
# APP INIT
# ============================================
app = FastAPI(
    title       = "AI Data Analyst API",
    description = "Natural language analytics powered by LLM + ML models",
    version     = "1.0.0"
)

# ============================================
# CORS — allows Power BI and browser access
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

# ============================================
# ROUTES
# ============================================
app.include_router(query_router,    tags=["Query"])
app.include_router(forecast_router, tags=["Forecast"])
app.include_router(review_router,   tags=["Review Prediction"])
app.include_router(segments_router, tags=["Customer Segments"])

# ============================================
# HEALTH CHECK
# ============================================
@app.get("/")
def root():
    return {
        "status"   : "running",
        "endpoints": [
            "POST /query           - Ask any business question in natural language",
            "GET  /forecast        - Get sales forecast for next N days",
            "POST /review-predict  - Predict if an order will get a good review",
            "GET  /segments        - Get customer segmentation results"
        ]
    }