from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database           import run_query
from text_to_sql        import generate_sql
from insight_generator  import generate_insight

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query(request: QueryRequest):
    try:
        # Step 1 — Convert question to SQL
        sql = generate_sql(request.question)

        # Step 2 — Run SQL against Data Warehouse
        results = run_query(sql)

        # Step 3 — Generate AI insight
        insight = generate_insight(request.question, sql, results)

        return {
            "question" : request.question,
            "sql"      : sql,
            "results"  : results,
            "insight"  : insight,
            "row_count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))