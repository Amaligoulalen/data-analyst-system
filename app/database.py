import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', '')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'ai_analytics')}"
)

engine = create_engine(DATABASE_URL)

def get_connection():
    return engine.connect()

def run_query(sql: str):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows    = result.fetchall()
        columns = result.keys()
        return [dict(zip(columns, row)) for row in rows]